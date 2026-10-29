"""Region-held-out evaluation; test opening requires a frozen validation decision."""
from pathlib import Path
import shutil
import numpy as np
from affine import Affine
from .common import read_json,write_json,sha256,utcnow,identity
from .inputs import Observation
from .registry import load_model
from .benchmark import confusion_metrics

ROOT=Path(__file__).resolve().parents[2]
AREA_KEYS=('true_water_ha','false_water_ha','missed_water_ha','true_nonwater_ha')


def load_scene(spec):
    path=Path(spec['path'])
    if sha256(path)!=spec['sha256']:raise ValueError('Scene hash mismatch: '+spec['id'])
    sidecar=read_json(path.with_suffix('.json'))
    meta=dict(sidecar['source_observation']);meta['source_observation_id']=meta.get('id');meta['id']=spec['id']
    meta['data_origin']='observed';meta['is_demo']=False
    with np.load(path,allow_pickle=False) as z:
        o=Observation(z['power'],z['valid'],Affine(*z['transform']),int(z['epsg']),meta,z['aoi_weights'])
        labels,evaluate=z['labels'].copy(),z['evaluate'].copy()
    return o,labels,evaluate,sidecar


def summary_from_areas(areas):
    tp,fp,fn,tn=[float(areas[k]) for k in AREA_KEYS]
    ratio=lambda n,d:n/d if d else None
    return {**dict(zip(AREA_KEYS,(tp,fp,fn,tn))), 'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),
            'f1':ratio(2*tp,2*tp+fp+fn),'iou':ratio(tp,tp+fp+fn),'false_positive_rate':ratio(fp,fp+tn),
            'reference_water_ha':tp+fn,'reference_nonwater_ha':fp+tn,'evaluated_ha':tp+fp+fn+tn}


def aggregate(records):
    groups={}
    for record in records:groups.setdefault(record['region'],[]).append(record)
    result={}
    for region,rows in groups.items():
        areas={k:sum((r.get('metrics') or {}).get(k,0) for r in rows) for k in AREA_KEYS}
        total=sum(r['reference_support_ha'] for r in rows);classified=sum(r['classified_reference_ha'] for r in rows)
        result[region]={**summary_from_areas(areas),'scenes':len(rows),'reference_support_ha':total,
                        'classified_reference_ha':classified,'reference_coverage':classified/total if total else None}
    pooled=summary_from_areas({k:sum(v[k] for v in result.values()) for k in AREA_KEYS})
    values=[r['f1'] for r in result.values() if r['f1'] is not None]
    return {'regions':result,'pooled':pooled,'macro_region_f1':float(np.mean(values)) if values else None,
            'minimum_region_f1':min(values) if values else None}


def prediction(spec,model,cache):
    o,truth,evaluate,sidecar=load_scene(spec)
    cache=Path(cache)/spec['id'];cache.mkdir(parents=True,exist_ok=True)
    receipt=cache/'prediction.json';array=cache/'prediction.npz'
    fingerprint=identity({'scene':spec['sha256'],'model':model.fingerprint})
    if receipt.exists():
        record=read_json(receipt)
        if record['fingerprint']!=fingerprint or sha256(array)!=record['sha256']:raise ValueError('Prediction cache integrity mismatch')
        with np.load(array,allow_pickle=False) as z:labels,scores=z['labels'].copy(),z['scores'].copy()
    else:
        print('PREDICT',spec['split'],model.metadata['model_type'],spec['id'],flush=True)
        labels,scores,info=model.predict(o)
        np.savez_compressed(array,labels=labels,scores=scores)
        record={'fingerprint':fingerprint,'sha256':sha256(array),'model_fingerprint':model.fingerprint,
                'scene_sha256':spec['sha256'],'info':info,'created_at':utcnow()}
        write_json(receipt,record)
    return o,truth,evaluate,labels,scores,record


def score(spec,o,truth,evaluate,labels,scores,threshold,*,boundary=False):
    support=(truth<=1)&o.valid&(o.weights>0)&(~evaluate if boundary else evaluate)
    common=support&(labels<=1)&np.isfinite(scores)
    predicted=scores>=threshold
    metrics=confusion_metrics(truth[common]==1,predicted[common],o.weights[common].astype(float)*.04) if common.any() else None
    total=float((o.weights[support]*.04).sum(dtype=float));known=float((o.weights[common]*.04).sum(dtype=float))
    return {'id':spec['id'],'region':spec['region'],'acquired_at':spec['acquired_at'],'split':spec['split'],
            'metrics':metrics,'reference_support_ha':total,'classified_reference_ha':known,
            'withheld_reference_ha':total-known,'reference_coverage':known/total if total else None,
            'water_reference_cells':int((support&(truth==1)).sum()),'nonwater_reference_cells':int((support&(truth==0)).sum())}


def passes(summary):
    # Areas below four hectares contain fewer than 100 independent grid cells;
    # record the rate but do not pretend that such a sample establishes specificity.
    checks=[]
    for region,row in summary['regions'].items():
        checks.append((row['reference_coverage'] or 0)>=.98)
        if row['reference_water_ha']>0:checks.append((row['recall'] or 0)>=.90)
        if row['precision'] is not None:checks.append(row['precision']>=.90)
        if row['reference_nonwater_ha']>=4:checks.append(row['false_positive_rate']<=.04)
    return bool(checks) and all(checks)


def validate_and_freeze(manifest_path,output):
    manifest_path,output=Path(manifest_path),Path(output);manifest=read_json(manifest_path)
    if (output/'freeze.json').exists():raise FileExistsError('Validation decision is already frozen')
    output.mkdir(parents=True,exist_ok=True)
    plan={'manifest_sha256':sha256(manifest_path),'thresholds':[i/10 for i in range(1,10)],
          'families':['hgb','unet','opera'],'test_opened':False,
          'selection_rule':'Prefer passing gate, then maximum minimum-region F1, then macro-region F1; stable family and threshold tie break',
          'gate':{'region_precision':.9,'region_recall':.9,'region_fpr':.04,'minimum_reference_coverage':.98,
                  'fpr_minimum_reference_land_ha':4},
          'evaluation_support':'40m-eroded spectral+SCL consensus interiors; boundaries separate weak diagnostic',
          'production_approval':False,'expert_reference_available':False}
    write_json(output/'selection-plan.json',plan)
    specs=[s for s in manifest['scenes'] if s['split']=='validation'];reports=[];family_best={}
    for family in plan['families']:
        model=load_model(ROOT/'models/round3'/family)
        prepared=[(s,*prediction(s,model,output/'validation-predictions'/family)) for s in specs]
        candidates=[]
        for threshold in ([.5] if family=='opera' else plan['thresholds']):
            rows=[score(s,o,t,e,l,p,threshold) for s,o,t,e,l,p,_ in prepared]
            report={'family':family,'threshold':threshold,**aggregate(rows),'scenes':rows,'test_used':False}
            report['passes_validation_gate']=passes(report);candidates.append(report);reports.append(report)
        rank=lambda r:(r['passes_validation_gate'],r['minimum_region_f1'] or 0,r['macro_region_f1'] or 0,-r['threshold'])
        family_best[family]=max(candidates,key=rank)
        print('VALIDATION',family,family_best[family]['threshold'],family_best[family]['minimum_region_f1'],flush=True)
    selected=max(family_best.values(),key=lambda r:(r['passes_validation_gate'],r['minimum_region_f1'] or 0,r['macro_region_f1'] or 0,r['family']))
    write_json(output/'validation-report.json',{'reports':reports,'family_best':family_best,'selected_family':selected['family'],
               'test_used':False,'expert_reference_available':False})
    snapshots={}
    for family,best in family_best.items():
        source=ROOT/'models/round3'/family;destination=output/'frozen'/family;destination.mkdir(parents=True,exist_ok=True)
        metadata=read_json(source/'model.json');metadata['threshold']=best['threshold']
        if family!='opera':metadata['decision_policy']={'kind':'threshold','threshold':best['threshold']}
        metadata['model_id']+='-validation-frozen';metadata['validation_selection']={'report_sha256':sha256(output/'validation-report.json'),
                'threshold':best['threshold'],'test_used':False,'production_ready':False}
        for name in ('weights.pt','estimator.joblib'):
            if (source/name).exists():shutil.copyfile(source/name,destination/name)
        write_json(destination/'model.json',metadata)
        snapshots[family]={'directory':str(destination.resolve()),'model_sha256':sha256(destination/'model.json'),
                           'threshold':best['threshold']}
    target=ROOT/'models/round3/selected'
    if target.exists():raise FileExistsError('Selected model directory already exists')
    shutil.copytree(output/'frozen'/selected['family'],target)
    freeze={'created_at':utcnow(),'manifest_sha256':sha256(manifest_path),'validation_report_sha256':sha256(output/'validation-report.json'),
            'selection_plan_sha256':sha256(output/'selection-plan.json'),'families':snapshots,'selected_family':selected['family'],
            'selected_directory':str(target.resolve()),'selected_model_sha256':sha256(target/'model.json'),
            'test_scene_ids':[s['id'] for s in manifest['scenes'] if s['split']=='test'],
            'test_opened_by_fitting_or_selection':False,'expert_reference_available':False,
            'implementation_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in
                [Path(__file__),ROOT/'src/nisar_water/neural.py',ROOT/'src/nisar_water/model.py',ROOT/'src/nisar_water/features.py',
                 ROOT/'src/nisar_water/inputs.py',ROOT/'src/nisar_water/opera_adapter.py',ROOT/'src/nisar_water/opera_model.py']}}
    write_json(output/'freeze.json',freeze);return freeze


def evaluate_test(manifest_path,output):
    output=Path(output);freeze=read_json(output/'freeze.json')
    if sha256(manifest_path)!=freeze['manifest_sha256']:raise ValueError('Manifest changed after freeze')
    for path,expected in freeze['implementation_hashes'].items():
        if sha256(ROOT/path)!=expected:raise ValueError('Implementation changed after freeze: '+path)
    manifest=read_json(manifest_path);specs=[s for s in manifest['scenes'] if s['split']=='test']
    report={'created_at':utcnow(),'label_kind':'weak','split':'unseen_geographic_test','freeze_sha256':sha256(output/'freeze.json'),
            'selected_family':freeze['selected_family'],'expert_reference_available':False,'families':{}}
    for family,snapshot in freeze['families'].items():
        path=Path(snapshot['directory'])
        if sha256(path/'model.json')!=snapshot['model_sha256']:raise ValueError('Frozen model changed')
        model=load_model(path);rows=[];boundaries=[]
        for spec in specs:
            o,t,e,l,p,_=prediction(spec,model,output/'test-predictions'/family)
            rows.append(score(spec,o,t,e,l,p,snapshot['threshold']))
            boundaries.append(score(spec,o,t,e,l,p,snapshot['threshold'],boundary=True))
        report['families'][family]={**aggregate(rows),'scenes':rows,
                'weak_boundary_diagnostic':{**aggregate(boundaries),'scenes':boundaries,'is_expert_shoreline_accuracy':False},
                'threshold':snapshot['threshold'],'changed_after_test':False}
        write_json(output/'test-report.json',report)
    selected=report['families'][freeze['selected_family']]
    report['selected_passes_weak_reference_gate']=passes(selected)
    report['readiness']='research_prototype_requires_independent_expert_and_failure_mode_validation'
    write_json(output/'test-report.json',report);return report

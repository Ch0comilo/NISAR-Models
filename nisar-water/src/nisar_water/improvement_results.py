"""Honest summaries and real-image comparisons across frozen improvement rounds."""
from pathlib import Path
import numpy as np
from .common import read_json,write_json,sha256,utcnow
from .round4_evaluation import load_scene
from .round3_results import native_composite
from .review import aligned_rgb
ROOT=Path(__file__).resolve().parents[2]


def table(experiment):
    report=read_json(ROOT/f'outputs/round{experiment}/test-report.json');rows=[]
    for family,result in report['families'].items():
        for region,row in result['regions'].items():
            rows.append({'round':experiment,'family':family,'region':region,
                **{k:row[k] for k in ('precision','recall','false_positive_rate','reference_water_ha','reference_land_ha','reference_coverage')},
                'low_history_water_recall':row['historical_water_diagnostic']['low_historical_occurrence_water']['recovered_fraction_of_full_reference'],
                'low_history_water_ha':row['historical_water_diagnostic']['low_historical_occurrence_water']['reference_water_ha']})
    return rows


def build_assessment(experiment):
    output=ROOT/f'outputs/round{experiment}';report=read_json(output/'test-report.json');freeze=read_json(output/'freeze.json')
    validation=read_json(output/'validation-report.json');selected=report['families'][freeze['selected_family']]
    unchanged=all(sha256(ROOT/p)==h for p,h in freeze['implementation_hashes'].items())
    unchanged&=sha256(Path(freeze['selected_directory'])/'model.json')==freeze['selected_model_sha256']
    for snap in freeze['families'].values():
        for name,digest in snap['artifacts'].items():unchanged&=sha256(Path(snap['directory'])/name)==digest
    failures=[{'region':name,'failed_checks':[k for k,v in checks.items() if not v]} for name,checks in selected['gate_checks'].items() if not all(checks.values())]
    assessment={'created_at':utcnow(),'state':'geographic_pilot_passed_weak_reference_gates_requires_expert_review' if selected['passes_test_gate'] else 'not_ready_for_general_use',
        'experiment_round':experiment,'selected_family':freeze['selected_family'],'model_sha256':freeze['selected_model_sha256'],
        'production_ready':False,'models_changed_after_test':not bool(unchanged),'expert_reviews_completed':0,
        'new_test_images':len(selected['scenes']),'new_test_regions':len(selected['regions']),
        'weak_reference_gate_passed':selected['passes_test_gate'],'blocking_findings':failures,
        'selected_pooled_weak_metrics':selected['pooled'],'historical_water_diagnostic':selected['historical_water_diagnostic'],
        'scope':'Current supported NISAR HH/HV L-GCOV 20m bounded UTM crops with genuine30m terrain and optional soft JRC2021 context',
        'limitations':['Optical consensus is weak reference, not independent expert shoreline truth',
            'Historical water and terrain can be stale where reservoirs or drainage have changed',
            'A soft historical feature can still induce strong learned dependence; its emerging-water errors must be measured',
            'Low historical occurrence alone does not establish a construction date or cause',
            'No worldwide, narrow-river, flooded-vegetation or human-impact accuracy established'],
        'human_review_materials':'review/earthengine/README.md; versioned import-review command',
        'context_fallback':'Missing physical context is explicit254/unknown; precompute it or configure bounded context fetching for a new AOI'}
    write_json(output/'assessment.json',assessment)
    write_json(output/'report.json',{**report,'validation_family_best':validation['family_best'],'region_table':table(experiment),
        'model_unchanged':bool(unchanged),'assessment_state':assessment['state']})
    return assessment


def plot_atlas(experiment,*,one_per_region=False):
    import matplotlib.pyplot as plt
    output=ROOT/f'outputs/round{experiment}';report=read_json(output/'test-report.json')
    specs=[s for s in read_json(ROOT/f'data/round{experiment}-prepared/manifest.json')['scenes'] if s['split']=='test']
    if one_per_region:
        seen=set();specs=[s for s in specs if not(s['region'] in seen or seen.add(s['region']))]
    baseline=next(f for f,v in report['families'].items() if v['is_fixed_baseline'])
    families=[baseline,'radar','terrain','terrain_prior'];titles=['Previous frozen model','Expanded radar-only','Radar + terrain','Radar + terrain + prior']
    fig,axes=plt.subplots(len(specs),5,figsize=(17,3.1*len(specs)),squeeze=False,layout='constrained')
    for row,spec in enumerate(specs):
        o,_,_,side=load_scene(spec);axes[row,0].imshow(aligned_rgb(o,side));axes[row,0].set_title(spec['region']+' '+spec['acquired_at'][:10]+'\nDated optical',fontsize=8)
        for column,(family,title) in enumerate(zip(families,titles),start=1):
            with np.load(output/'test-predictions'/family/spec['id']/'prediction.npz',allow_pickle=False) as z:labels=z['labels']
            axes[row,column].imshow(native_composite(o,labels));axes[row,column].set_title(title,fontsize=8)
        for ax in axes[row]:ax.axis('off')
    fig.suptitle(f'Round {experiment}: frozen models on new locations — blue predicted water, cyan boundary\nAll decisions shown; optical reference is not independent expert truth',fontsize=12)
    return fig


def plot_error(experiment,region):
    import matplotlib.pyplot as plt
    output=ROOT/f'outputs/round{experiment}';freeze=read_json(output/'freeze.json')
    spec=next(s for s in read_json(ROOT/f'data/round{experiment}-prepared/manifest.json')['scenes'] if s['split']=='test' and s['region']==region)
    o,t,e,side=load_scene(spec)
    with np.load(output/'test-predictions'/freeze['selected_family']/spec['id']/'prediction.npz',allow_pickle=False) as z:labels=z['labels']
    errors=np.full((*t.shape,3),240,np.uint8);errors[e&(t==1)&(labels==1)]=[35,153,245]
    errors[e&(t==0)&(labels==1)]=[217,76,61];errors[e&(t==1)&(labels==0)]=[244,183,55]
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for ax,data,title in zip(axes,[aligned_rgb(o,side),native_composite(o,labels),errors],
        ['Dated optical','Frozen selected model','Blue correct water; red false water; gold missed water']):
        ax.imshow(data);ax.set_title(title,fontsize=9);ax.axis('off')
    fig.suptitle(f"Round {experiment}: {region} {spec['acquired_at'][:10]} | unknown/cloud/boundary pixels unscored",fontsize=11)
    return fig

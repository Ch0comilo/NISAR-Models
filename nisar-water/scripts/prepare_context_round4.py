"""Precompute genuine terrain/prior context; no predictions or model fitting."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json,sha256,utcnow
from nisar_water.round4_evaluation import load_scene
from nisar_water.terrain_context import load_context

if __name__=='__main__':
    if (ROOT/'models/round4/plan.json').exists():raise ValueError('Preserve fitted inputs; context preparation already closed')
    directory=ROOT/'data/round4-prepared';source=directory/'scenes-manifest.json';manifest=read_json(source)
    cache=ROOT/'.cache/terrain-context';records=[];scenes=[]
    clarification=ROOT/'data/round4/selection-clarification-before-fit.json'
    if not clarification.exists():write_json(clarification,{'created_at':utcnow(),'fits_started':False,'new_test_predictions_inspected':False,
        'gate':'Require98% reference coverage per supported class as well as total; water recovery uses full water reference, so abstention cannot hide missed water',
        'ranking':'Worst-region then macro mean full-reference correct water/land fraction for supported classes; unknown is unrecovered, never labelled land',
        'tie_break':'Canonical family order radar/terrain/terrain_prior then highest threshold to prefer conservative decisions when validation ties exactly'})
    for spec in manifest['scenes']:
        # This is deterministic geospatial preprocessing of every partition.
        # New test values are not displayed, scored, or used by fitting/selection.
        o,_,_,_=load_scene(spec)
        path=directory/'context-hand-v2'/(spec['id']+'.npz');meta_path=path.with_suffix('.json')
        if not path.exists():
            total=sum(read_json(p).get('body_bytes',0) for p in (cache/'regions').glob('*/transfer.json')) if (cache/'regions').exists() else 0
            if total>=3*1024**3:raise ValueError('Round4 public ancillary transfer ceiling reached')
            print('CONTEXT',spec['split'],spec['id'],flush=True)
            context=load_context(o,cache_dir=cache,include_prior=True)
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,features=context['features'],available=context['available'],names=np.array(context['names']),
                prior_available=context['prior_available'],prior_occurrence_available=context['prior_occurrence_available'],
                transform=np.array(list(o.transform)[:6]),transform6=np.array(list(o.transform)[:6]),epsg=np.array(o.crs.to_epsg()),
                scene_sha256=np.array(spec['sha256']))
            metadata={**context['metadata'],'scene_sha256':spec['sha256'],'scene_id':spec['id'],
                      'target_transform':list(o.transform)[:6],'target_epsg':o.crs.to_epsg()}
            write_json(meta_path,metadata)
        with np.load(path,allow_pickle=False) as z:
            if str(z['scene_sha256'].item())!=spec['sha256']:raise ValueError('Cached context scene binding changed')
            support=o.valid&(o.weights>0);available=z['available'];fraction=float(available[support].mean()) if support.any() else None
            prior_fraction=float(z['prior_occurrence_available'][support].mean()) if support.any() else None
        scenes.append({**spec,'context_path':str(path.resolve()),'context_sha256':sha256(path),
                       'context_metadata_path':str(meta_path.resolve()),'context_metadata_sha256':sha256(meta_path)})
        records.append({'id':spec['id'],'region':spec['region'],'split':spec['split'],
                        'physical_observed_coverage':fraction,'historical_occurrence_observed_coverage':prior_fraction})
        write_json(directory/'context-preparation-progress.json',{'prepared':len(scenes),'total':len(manifest['scenes']),'records':records})
    final={**manifest,'context_prepared_at':utcnow(),'context_features':'Physical COP GLO30+HAND2021; optional soft historical JRC2021',
           'scenes':scenes,'context_source_manifest_sha256':sha256(source),'selection_clarification_sha256':sha256(clarification),
           'deterministic_preprocessing_includes_test':True,'test_predictions_or_image_inspection':False}
    write_json(directory/'manifest.json',final);print('Context manifest complete',len(scenes),flush=True)

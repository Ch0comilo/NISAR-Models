"""Close deterministic preparation and record development-only fit release."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json,sha256,utcnow
from nisar_water.context_model import load_context_sidecar
if __name__=='__main__':
    manifest_path=ROOT/'data/round4-prepared/manifest.json';manifest=read_json(manifest_path)
    qa=ROOT/'evidence/round4-development-qa-review.json';assert read_json(qa)['review_status']=='assistant_visual_qa_completed'
    assert not (ROOT/'models/round4').exists()
    regions={};records=[]
    for spec in manifest['scenes']:
        regions.setdefault(spec['region'],set()).add(spec['split'])
        assert sha256(spec['path'])==spec['sha256']
        assert sha256(spec['context_path'])==spec['context_sha256']
        assert sha256(spec['context_metadata_path'])==spec['context_metadata_sha256']
        if spec['split']=='test':continue
        with np.load(spec['path'],allow_pickle=False) as z:
            labels=z['labels'];valid=z['valid'];weights=z['weights'];aoi=z['aoi_weights'];transform=z['transform'];epsg=int(z['epsg'])
        context=load_context_sidecar(spec,valid.shape,transform,epsg,scene_sha256=spec['sha256'])
        observed=valid&(aoi>0)
        fraction=float(context['available'][observed].mean()) if observed.any() else None
        if fraction is None or fraction<.98:raise ValueError('Insufficient development context support: '+spec['id'])
        records.append({'id':spec['id'],'split':spec['split'],'region':spec['region'],'context_observed_fraction':fraction,
            'water_cells':int(((labels==1)&valid&(weights>0)).sum()),'land_cells':int(((labels==0)&valid&(weights>0)).sum())})
    assert all(len(v)==1 for v in regions.values())
    old=read_json(ROOT/'outputs/round3/freeze.json')
    assert sha256(ROOT/'models/round3/selected/model.json')==old['selected_model_sha256']
    release={'created_at':utcnow(),'manifest_sha256':sha256(manifest_path),'qa_review_sha256':sha256(qa),
        'precollection_plan_sha256':sha256(ROOT/'data/round4/plan.json'),
        'selection_clarification_sha256':sha256(ROOT/'data/round4/selection-clarification-before-fit.json'),
        'planned_separation_audit_sha256':sha256(ROOT/'data/round4/planned-separation-audit.json'),
        'scenes_by_split':{s:sum(r['split']==s for r in manifest['scenes']) for s in ('train','validation','test')},
        'development_support':records,'label_kind':'weak','expert_reviews_completed':0,
        'fits_authorized':{'radar':1,'terrain':1,'terrain_prior':1},'validation_pixels_read_by_fit':False,
        'new_test_images_inspected_or_predicted':False,'deterministic_test_preprocessing_and_hash_verification':True,
        'historical_round3_model_unchanged':True}
    write_json(ROOT/'evidence/round4-training-release.json',release);print(release['scenes_by_split'])

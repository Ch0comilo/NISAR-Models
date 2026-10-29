from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json,sha256,utcnow
from nisar_water.registry import load_model
from nisar_water.round5_evaluation import check_test_release,prediction,score_scene,aggregate,passes
if __name__=='__main__':
    output=ROOT/'outputs/round5';manifest=ROOT/'data/round5-prepared/manifest.json';check_test_release(manifest,output)
    frozen=read_json(output/'agreement-review-freeze.json');release=read_json(output/'test-release.json')
    if release['agreement_policy_freeze_sha256']!=sha256(output/'agreement-review-freeze.json'):raise ValueError('Review declaration changed after test release')
    model=load_model(frozen['model_directory'])
    if model.fingerprint!=frozen['model_sha256']:raise ValueError('Review policy changed')
    rows=[]
    for spec in read_json(manifest)['scenes']:
        if spec['split']!='test':continue
        o,t,e,l,p,occ,_,_=prediction(spec,model,output/'test-predictions/agreement_review')
        rows.append(score_scene(spec,o,t,e,l,p,.5,occ))
    report={'created_at':utcnow(),**aggregate(rows),'scenes':rows,'is_primary_model':False,'production_ready':False,
        'policy_chosen_before_test':True,'agreement_freeze_sha256':sha256(output/'agreement-review-freeze.json'),
        'interpretation':'Conditional metrics on agreed cells must be read with full-reference coverage and withheld water/land; unknown is not land'}
    report['passes_full_weak_gate']=passes(report);write_json(output/'agreement-test-report.json',report)
    print('Review test coverage',report['pooled']['reference_coverage'],'full gate',report['passes_full_weak_gate'])

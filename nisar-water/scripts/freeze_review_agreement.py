from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.agreement_model import create_review_model
from nisar_water.common import read_json,write_json,sha256,utcnow
from nisar_water.registry import load_model
from nisar_water.round5_evaluation import prediction,score_scene,aggregate,passes
if __name__=='__main__':
    output=ROOT/'outputs/round5';freeze=read_json(output/'freeze.json')
    if (output/'test-release.json').exists():raise ValueError('Declare the review policy before test release')
    model_dir=ROOT/'models/agreement-review'
    create_review_model(model_dir,output/'frozen'/freeze['selected_family'],ROOT/'outputs/round4/frozen/terrain_prior')
    model=load_model(model_dir);rows=[]
    for spec in read_json(ROOT/'data/round5-prepared/manifest.json')['scenes']:
        if spec['split']!='validation':continue
        o,t,e,l,p,occ,_,_=prediction(spec,model,output/'validation-predictions/agreement_review')
        rows.append(score_scene(spec,o,t,e,l,p,.5,occ))
    report={**aggregate(rows),'scenes':rows,'role':'Review aid, excluded from primary family selection','test_used':False}
    report['passes_weak_gate']=passes(report);write_json(output/'agreement-validation-report.json',report)
    write_json(output/'agreement-review-freeze.json',{'created_at':utcnow(),'model_directory':str(model_dir.resolve()),
        'model_sha256':model.fingerprint,'validation_report_sha256':sha256(output/'agreement-validation-report.json'),
        'primary_freeze_sha256':sha256(output/'freeze.json'),'test_used':False,'no_hyperparameters_tuned':True,
        'policy':'Fixed agreement of the validation-selected round5 classifier and previously frozen round4 prior model; disagreement254',
        'full_reference_coverage_reported':True,'production_ready':False})
    print('Review aid frozen before test; validation coverage',report['pooled']['reference_coverage'])

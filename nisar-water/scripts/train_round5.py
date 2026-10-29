from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json,sha256
from nisar_water.context_model import train_context_models
if __name__=='__main__':
    manifest=ROOT/'data/round5-prepared/manifest.json';release=read_json(ROOT/'evidence/round5-training-release.json')
    if release['manifest_sha256']!=sha256(manifest):raise ValueError('Exact reviewed training release required')
    if release['assistant_visual_qa_completed'] is not True:raise ValueError('Review development reference preparation first')
    output=ROOT/'models/round5';train_context_models(manifest,output)
    # The common fitting implementation and physical feature contract are
    # unchanged. Give this new dataset experiment distinct model identities.
    for family in ('radar','terrain','terrain_prior'):
        path=output/family/'model.json';meta=read_json(path);meta['model_id']=meta['model_id'].replace('round4','round5')
        meta['experiment_round']=5;write_json(path,meta)

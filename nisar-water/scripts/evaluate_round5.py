from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.round5_evaluation import validate_and_freeze,evaluate_test
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['validation','test']);a=p.parse_args()
    manifest=ROOT/'data/round5-prepared/manifest.json';output=ROOT/'outputs/round5'
    if a.operation=='validation':
        r=validate_and_freeze(manifest,output,ROOT/'models/round5',ROOT/'models/round4/selected');print('Frozen round5 selected',r['selected_family'])
    else:r=evaluate_test(manifest,output);print('Round5 tests finished',r['readiness'])

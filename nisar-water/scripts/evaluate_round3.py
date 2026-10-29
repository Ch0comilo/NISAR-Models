from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.round3_evaluation import validate_and_freeze,evaluate_test
from nisar_water.opera_model import create_model
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['validation','test'])
    args=parser.parse_args();manifest=ROOT/'data/round3-prepared/manifest.json';output=ROOT/'outputs/round3'
    if args.operation=='validation':
        if not (ROOT/'models/round3/opera/model.json').exists():create_model(ROOT/'models/round3/opera',ROOT)
        freeze=validate_and_freeze(manifest,output);print('Frozen selected family:',freeze['selected_family'])
    else:
        report=evaluate_test(manifest,output);print('Test completed:',report['readiness'])

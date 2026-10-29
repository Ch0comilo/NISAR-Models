"""Evaluate an additional fixed cohort without changing the validation decision."""
from pathlib import Path
import argparse
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json,sha256,utcnow,safe_id
from nisar_water.registry import load_model
from nisar_water.round3_evaluation import prediction,score,aggregate,passes

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path);parser.add_argument('--name',required=True)
    args=parser.parse_args();name=safe_id(args.name);out=ROOT/'outputs/round3';freeze=read_json(out/'freeze.json')
    manifest=read_json(args.manifest)
    report={'created_at':utcnow(),'cohort':name,'manifest_sha256':sha256(args.manifest),'freeze_sha256':sha256(out/'freeze.json'),
            'used_for_training_or_selection':False,'model_changed':False,'label_kind':'weak',
            'expert_reference_available':False,'families':{}}
    for family,snapshot in freeze['families'].items():
        path=Path(snapshot['directory'])
        if sha256(path/'model.json')!=snapshot['model_sha256']:raise ValueError('Frozen model changed')
        model=load_model(path);records=[];boundaries=[]
        for spec in manifest['scenes']:
            o,t,e,l,p,_=prediction(spec,model,out/(name+'-predictions')/family)
            records.append(score(spec,o,t,e,l,p,snapshot['threshold']))
            boundaries.append(score(spec,o,t,e,l,p,snapshot['threshold'],boundary=True))
        report['families'][family]={**aggregate(records),'scenes':records,'threshold':snapshot['threshold'],
            'passes_weak_reference_gate':passes(aggregate(records)),
            'weak_boundary_diagnostic':{**aggregate(boundaries),'scenes':boundaries,'is_expert_shoreline_accuracy':False}}
        write_json(out/(name+'-report.json'),report)
    print('Completed frozen cohort',name,flush=True)

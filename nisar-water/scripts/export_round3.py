"""Publish selected-model real observations, keeping viewing geometries separate."""
from pathlib import Path
import argparse
import sys
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json,sha256
from nisar_water.registry import load_model
from nisar_water.round3_evaluation import load_scene
from nisar_water.pipeline import run_series

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path);parser.add_argument('--cohort',required=True)
    args=parser.parse_args();freeze=read_json(ROOT/'outputs/round3/freeze.json')
    model=load_model(ROOT/'models/round3/selected')
    if model.fingerprint!=freeze['selected_model_sha256']:raise ValueError('Selected model changed')
    specs=read_json(args.manifest)['scenes']
    if args.cohort=='test':specs=[s for s in specs if s['split']=='test']
    groups=defaultdict(list)
    for spec in specs:
        source=read_json(Path(spec['path']).with_suffix('.json'))['source_observation']
        key=f"{spec['region']}-{source['track']}-{source['pass_direction']}-{source['frame']}"
        groups[key].append(spec)
    index=[]
    for key,items in groups.items():
        destination=ROOT/'outputs/round3/series'/args.cohort/key
        if (destination/'series.json').exists():series=read_json(destination/'series.json')
        else:
            print('EXPORT',key,len(items),flush=True)
            series=run_series((load_scene(s)[0] for s in items),model,destination,series_id=key)
        index.append({'key':key,'cohort':args.cohort,'path':str((destination/'series.json').resolve()),'observations':len(series['observations'])})
    write_json(ROOT/'outputs/round3'/('series-index-'+args.cohort+'.json'),{'series':index,'model_sha256':model.fingerprint})

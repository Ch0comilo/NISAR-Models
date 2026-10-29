"""Export the frozen selected family on all new test observations."""
from pathlib import Path
from collections import defaultdict
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json,sha256
from nisar_water.round4_evaluation import load_scene
from nisar_water.registry import load_model
from nisar_water.pipeline import run_series
if __name__=='__main__':
    output=ROOT/'outputs/round4';freeze=read_json(output/'freeze.json');model=load_model(ROOT/'models/round4/selected')
    if model.fingerprint!=freeze['selected_model_sha256']:raise ValueError('Frozen selected checkpoint changed')
    specs=[s for s in read_json(ROOT/'data/round4-prepared/manifest.json')['scenes'] if s['split']=='test'];groups=defaultdict(list)
    for spec in specs:
        meta=read_json(Path(spec['path']).with_suffix('.json'))['source_observation']
        key=f"{spec['region']}-{meta['track']}-{meta['pass_direction']}-{meta['frame']}";groups[key].append(spec)
    index=[]
    for key,scenes in groups.items():
        destination=output/'series'/key
        if (destination/'series.json').exists():series=read_json(destination/'series.json')
        else:
            print('EXPORT',key,len(scenes),flush=True)
            series=run_series((load_scene(s)[0] for s in scenes),model,destination,series_id=key)
        index.append({'id':key,'path':str((destination/'series.json').resolve()),'observations':len(series['observations'])})
    write_json(output/'series-index.json',{'series':index,'selected_model_sha256':model.fingerprint})

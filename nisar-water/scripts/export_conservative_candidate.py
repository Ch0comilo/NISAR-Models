from pathlib import Path
from collections import defaultdict
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json
from nisar_water.round5_evaluation import load_scene
from nisar_water.registry import load_model
from nisar_water.pipeline import run_series
if __name__=='__main__':
    output=ROOT/'outputs/round5';model=load_model(output/'frozen/terrain_prior');groups=defaultdict(list)
    for spec in read_json(ROOT/'data/round5-prepared/manifest.json')['scenes']:
        if spec['split']!='test':continue
        meta=read_json(Path(spec['path']).with_suffix('.json'))['source_observation']
        groups[f"{spec['region']}-{meta['track']}-{meta['pass_direction']}-{meta['frame']}"].append(spec)
    for key,specs in groups.items():
        destination=output/'candidate-series/terrain_prior'/key
        if not (destination/'series.json').exists():
            print('EXPORT frozen candidate',key,flush=True)
            run_series((load_scene(s)[0] for s in specs),model,destination,series_id='terrain-prior-'+key)

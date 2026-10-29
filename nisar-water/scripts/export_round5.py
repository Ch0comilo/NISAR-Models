from pathlib import Path
from collections import defaultdict
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json
from nisar_water.round5_evaluation import load_scene
from nisar_water.registry import load_model
from nisar_water.pipeline import run_series
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['primary','review'],default='primary');a=p.parse_args()
    output=ROOT/'outputs/round5';directory=ROOT/('models/round5/selected' if a.mode=='primary' else 'models/agreement-review')
    model=load_model(directory);all_specs=read_json(ROOT/'data/round5-prepared/manifest.json')['scenes'];groups=defaultdict(list)
    for spec in all_specs:
        if spec['split']!='test' and spec['region']!='gerd-shore':continue
        cohort='unseen-test' if spec['split']=='test' else 'exposed-training-regression'
        meta=read_json(Path(spec['path']).with_suffix('.json'))['source_observation']
        key=f"{cohort}/{spec['region']}-{meta['track']}-{meta['pass_direction']}-{meta['frame']}";groups[key].append(spec)
    index=[]
    for key,specs in groups.items():
        destination=output/('series' if a.mode=='primary' else 'review-series')/key
        if (destination/'series.json').exists():series=read_json(destination/'series.json')
        else:
            print('EXPORT round5',a.mode,key,len(specs),flush=True)
            series=run_series((load_scene(s)[0] for s in specs),model,destination,series_id=a.mode+'-'+key.replace('/','-'))
        index.append({'key':key,'path':str((destination/'series.json').resolve()),'observations':len(series['observations'])})
    write_json(output/('series-index-'+a.mode+'.json'),{'series':index,'model_sha256':model.fingerprint})

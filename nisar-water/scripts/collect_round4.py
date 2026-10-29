from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import write_json
from nisar_water.round4_data import create_plan,prepare_scenes
from nisar_water.round3_data import collect_site
if __name__=='__main__':
    directory=ROOT/'data/round4';plan=create_plan(directory);results=[]
    with ProcessPoolExecutor(max_workers=2) as executor:
        jobs={executor.submit(collect_site,site,directory,ROOT.parent/'nisar-landcover-phase1'):site['id'] for site in plan['sites']}
        for future in as_completed(jobs):
            try:results.append(future.result())
            except Exception as error:results.append({'site':{'id':jobs[future]},'state':'failed','message':str(error)[:500]})
            write_json(directory/'collection-summary.json',{'results':results,'test_predictions_inspected':False})
    manifest=prepare_scenes(directory,ROOT/'data/round4-prepared');print('Prepared',len(manifest['scenes']),'scenes')

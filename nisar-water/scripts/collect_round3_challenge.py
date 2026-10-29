"""Prospective hard-negative challenge; no predictions select dates or labels."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import write_json,read_json,utcnow,sha256
from nisar_water.round3_data import collect_site
from nisar_water.round3_scenes import aligned_reference,save_scene
from nisar_water.inputs import load_catalog_observation

if __name__=='__main__':
    root=ROOT/'data/round3-challenge';root.mkdir(parents=True,exist_ok=True)
    definitions=[('salar-uyuni',[-67.50,-20.25,-67.45,-20.20]),
                 ('denver-airport',[-104.69,39.84,-104.64,39.89]),
                 ('andes-shadow',[-70.29,-33.70,-70.24,-33.65])]
    sites=[]
    for name,(w,s,e,n) in definitions:
        sites.append({'id':name,'split':'challenge','start':'2026-06-01','end':'2026-10-27',
                      'minimum_reference_water_cells':0,'maximum_dates':1,
                      'aoi':{'type':'Polygon','coordinates':[[[w,s],[e,s],[e,n],[w,n],[w,s]]]}})
    plan=root/'plan.json'
    if not plan.exists():write_json(plan,{'created_at':utcnow(),'sites':sites,'test_role':'Prospective hard-negative stress set',
        'used_for_training_or_selection':False,'selection':'Fixed geography, chronological candidates, reference availability, clear fraction>=0.3; no water minimum',
        'expected_classes_are_not_truth':'Use optical consensus, not place name, to assign each known cell'})
    else:sites=read_json(plan)['sites']
    results=[]
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures={executor.submit(collect_site,s,root,ROOT.parent/'nisar-landcover-phase1'):s['id'] for s in sites}
        for future in as_completed(futures):
            try:results.append(future.result())
            except Exception as error:results.append({'state':'failed','site':{'id':futures[future]},'message':str(error)[:500]})
            write_json(root/'collection-summary.json',{'results':results})
    scenes=[]
    for result in results:
        for selected in result.get('selected',[]):
            o=load_catalog_observation(selected['observation']);labels,weights,evaluate,rgb,provenance=aligned_reference(o,selected['reference_dir'])
            oid=selected['region']+'-'+o.metadata['acquired_at'].replace(':','-')
            spec={'id':oid,'region':selected['region'],'group':selected['region'],'split':'challenge','acquired_at':o.metadata['acquired_at']}
            scenes.append(save_scene(o,labels,weights,evaluate,root/'scenes'/(oid+'.npz'),spec,provenance,rgb))
    write_json(root/'manifest.json',{'created_at':utcnow(),'scenes':scenes,'data_origin':'observed','label_kind':'weak',
                 'plan_sha256':sha256(plan),'used_for_training_or_selection':False})

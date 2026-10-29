"""Additional prospective shoreline cohort; all model choices already frozen."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json,utcnow,sha256,safe_id
from nisar_water.round3_data import collect_site
from nisar_water.round3_scenes import aligned_reference,save_scene
from nisar_water.inputs import load_catalog_observation

if __name__=='__main__':
    root=ROOT/'data/round3-shorelines';root.mkdir(parents=True,exist_ok=True)
    freeze=ROOT/'outputs/round3/freeze.json'
    if not freeze.exists():raise ValueError('Freeze family and thresholds first')
    plan=root/'plan.json'
    if not plan.exists():
        definitions=[('powell-shore',[-111.53,36.96,-111.45,37.04]),
                     ('malawi-shore',[34.88,-14.12,34.96,-14.04]),
                     ('jamuna-river',[89.66,24.37,89.74,24.45])]
        sites=[]
        for name,(w,s,e,n) in definitions:
            sites.append({'id':name,'split':'prospective_shoreline_test','start':'2026-06-01','end':'2026-10-27',
                'minimum_reference_water_cells':100,'minimum_reference_land_cells':100,'maximum_dates':2,
                'aoi':{'type':'Polygon','coordinates':[[[w,s],[e,s],[e,n],[w,n],[w,s]]]}})
        write_json(plan,{'created_at':utcnow(),'sites':sites,'freeze_sha256':sha256(freeze),'used_for_training_or_selection':False,
            'reason':'Initial held-out lake crops mostly contain open water and cannot establish specificity or shoreline behavior',
            'selection':'Fixed geography; date spread; >=100 clear SCL water and land source cells; optical availability only, never model predictions',
            'protocol':'Additional prospective cohort after frozen selection; report separately from original geographic test'})
    sites=read_json(plan)['sites'];results=[]
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
            oid=safe_id(selected['region']+'-'+o.metadata['acquired_at'])
            spec={'id':oid,'region':selected['region'],'group':selected['region'],'split':'prospective_shoreline_test','acquired_at':o.metadata['acquired_at']}
            scenes.append(save_scene(o,labels,weights,evaluate,root/'scenes'/(oid+'.npz'),spec,provenance,rgb))
    write_json(root/'manifest.json',{'created_at':utcnow(),'scenes':scenes,'data_origin':'observed','label_kind':'weak',
        'plan_sha256':sha256(plan),'freeze_sha256':sha256(freeze),'used_for_training_or_selection':False})

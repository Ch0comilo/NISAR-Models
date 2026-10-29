"""Correct two development AOIs using reference availability, before any model fit."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from nisar_water.common import read_json,write_json,utcnow
from nisar_water.round3_data import collect_site

if __name__=='__main__':
    root=Path('data/round3').resolve()
    sites=[]
    for name,(w,s,e,n) in [('nasser-east',(32.87,23.94,32.92,23.99)),('marion-north',(-80.48,33.48,-80.43,33.53))]:
        sites.append({'id':name,'split':'train','aoi':{'type':'Polygon','coordinates':[[[w,s],[e,s],[e,n],[w,n],[w,s]]]},
                      'start':'2026-06-01','end':'2026-10-27'})
    plan=root/'supplement-plan.json'
    if not plan.exists():write_json(plan,{'created_at':utcnow(),'sites':sites,'reason':'Original development windows had no optical SCL water support; adjust geography before any model predictions/fitting. Original failed attempts remain preserved.','model_scores_used':False,'test_allocation_unchanged':True})
    with ProcessPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(collect_site,sites,[root]*2,[Path('../nisar-landcover-phase1').resolve()]*2))
    write_json(root/'supplement-collection.json',{'results':results})


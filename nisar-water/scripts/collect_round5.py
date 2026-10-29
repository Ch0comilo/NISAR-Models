"""Prospective recent-reservoir follow-up after the exposed round4 failures."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json,sha256,utcnow,safe_id
from nisar_water.round4_data import separation_audit
from nisar_water.round3_data import collect_site
from nisar_water.inputs import load_catalog_observation
from nisar_water.round3_scenes import aligned_reference,save_scene

DEFINITIONS=[
 ('ilisu-shore','validation','mixed',[41.39,37.68,41.47,37.76]),
 ('madrid-airport','validation','land',[-3.59,40.44,-3.51,40.52]),
 ('omodeo-shore','validation','mixed',[8.91,40.08,8.99,40.16]),
 ('yusufeli-shore','test','mixed',[41.51,40.79,41.59,40.87]),
 ('nam-theun1-shore','test','mixed',[104.18,18.34,104.26,18.42]),
 ('hidroituango-shore','test','mixed',[-75.77,7.01,-75.69,7.09]),
 ('alice-springs-airport','test','land',[133.86,-23.85,133.94,-23.77]),
]

if __name__=='__main__':
    directory=ROOT/'data/round5';directory.mkdir(parents=True,exist_ok=True);plan_path=directory/'plan.json'
    if not plan_path.exists():
        sites=[]
        for name,split,kind,(w,s,e,n) in DEFINITIONS:
            sites.append({'id':name,'split':split,'kind':kind,'aoi':{'type':'Polygon','coordinates':[[[w,s],[e,s],[e,n],[w,n],[w,s]]]},
                'start':'2026-06-01','end':'2026-10-27','maximum_dates':2,'minimum_reference_land_cells':100,
                'minimum_reference_water_cells':100 if kind=='mixed' else 0})
        prior_sites=[{**s,'split':'train'} for s in read_json(ROOT/'data/round4/plan.json')['sites']]
        write_json(directory/'planned-separation-audit.json',separation_audit(prior_sites+sites))
        write_json(plan_path,{'created_at':utcnow(),'sites':sites,'previous_round4_manifest_sha256':sha256(ROOT/'data/round4-prepared/manifest.json'),
            'selection_reason':'Round4 reduced false water but failed current water with low historical occurrence. All71 exposed scenes become TRAIN; fresh recent-reservoir and dry-land validation/test remain separate.',
            'families':['radar','terrain','terrain_prior'],'accepted_fits_per_family':1,'parameters':'Same fixed HGB settings and matched samples as round4',
            'thresholds':[i/10 for i in range(1,10)],'selection_protocol':'Same class-coverage-guarded balanced validation rule; fresh recent-reservoir validation prevents selecting solely on persistent lakes',
            'no_test_predictions_before_freeze':True,'new_optical_labels':'Same exactly-once calibrated SCL/spectral consensus; still weak, not independent expert labels',
            'prior_rule':'Soft features only, no hard historical water mask; report low historical occurrence current-water recall separately',
            'provenance_sources':{'Yusufeli_impoundment_started_2022':'https://dsi.gov.tr/Haber/Detay/10078',
                'Nam_Theun1_project':'https://www.nt1pc.com/about-us',
                'Hidroituango_filling_in_2018_distinct_from_2022_generation':'https://www.epm.com.co/content/dam/epm/institucional/documentos/todos/skava-informe-causa-raiz-fisica.pdf'},
            'site_interpretation':'Names identify study regions; low JRC occurrence is not automatically a reservoir construction date or a causal label'})
    sites=read_json(plan_path)['sites'];results=[]
    with ProcessPoolExecutor(max_workers=2) as executor:
        jobs={executor.submit(collect_site,s,directory,ROOT.parent/'nisar-landcover-phase1'):s['id'] for s in sites}
        for future in as_completed(jobs):
            try:results.append(future.result())
            except Exception as error:results.append({'site':{'id':jobs[future]},'state':'failed','message':str(error)[:500]})
            write_json(directory/'collection-summary.json',{'results':results,'new_test_predictions_inspected':False})
    target=ROOT/'data/round5-prepared';target.mkdir(parents=True,exist_ok=True)
    scenes=[{**s,'previous_round4_split':s['split'],'split':'train','role':'previously_exposed_through_round4'} for s in
        read_json(ROOT/'data/round4-prepared/manifest.json')['scenes']]
    for result in results:
        for selected in result.get('selected',[]):
            o=load_catalog_observation(selected['observation']);labels,weights,evaluate,rgb,provenance=aligned_reference(o,selected['reference_dir'])
            oid=safe_id(selected['region']+'-'+o.metadata['acquired_at']);spec={'id':oid,'region':selected['region'],
                 'group':selected['region'],'split':selected['split'],'acquired_at':o.metadata['acquired_at']}
            scenes.append(save_scene(o,labels,weights,evaluate,target/'scenes'/(oid+'.npz'),spec,provenance,rgb))
    write_json(target/'scenes-manifest.json',{'created_at':utcnow(),'data_origin':'observed','label_kind':'weak','scenes':scenes,
        'plan_sha256':sha256(plan_path),'test_used_for_selection':False,'expert_reviews_completed':0})
    print('Prepared round5 scenes',len(scenes),flush=True)

"""Predeclared hard-negative and physical-context experiment; prior tests exposed."""
from pathlib import Path
import numpy as np
from pyproj import Geod
from .common import read_json,write_json,utcnow,sha256,safe_id
from .inputs import load_catalog_observation
from .round3_scenes import aligned_reference,save_scene

ROOT=Path(__file__).resolve().parents[2]
SITES=[
 ('mesa-airport','train','land',[-111.69,33.28,-111.63,33.34]),
 ('black-rock-playa','train','land',[-119.09,40.75,-119.03,40.81]),
 ('imperial-fields','train','land',[-115.57,32.94,-115.51,33.00]),
 ('mississippi-vicksburg','train','mixed',[-90.96,32.31,-90.88,32.39]),
 ('tucson-airport','validation','land',[-110.98,32.09,-110.92,32.15]),
 ('eufaula-shore','validation','mixed',[-95.64,35.28,-95.56,35.36]),
 ('titicaca-shore','validation','mixed',[-69.17,-16.22,-69.09,-16.14]),
 ('lubbock-airport','test','land',[-101.84,33.63,-101.78,33.69]),
 ('bonneville-playa','test','land',[-113.93,40.74,-113.85,40.82]),
 ('alamosa-fields','test','land',[-105.93,37.42,-105.85,37.50]),
 ('cumberland-shore','test','mixed',[-85.08,36.89,-85.00,36.97]),
 ('glenbawn-shore','test','mixed',[150.94,-32.14,151.02,-32.06]),
 ('gerd-shore','test','mixed',[35.15,11.10,35.23,11.18]),
]


def exposed_scenes():
    originals=read_json(ROOT/'data/round3-prepared/manifest.json')['scenes'][:]
    for dataset in ('round3-challenge','round3-shorelines'):originals+=read_json(ROOT/'data'/dataset/'manifest.json')['scenes']
    return [{**s,'split':'train','previous_split':s['split'],'role':'previously_exposed_training'} for s in originals]


def separation_audit(sites):
    geod=Geod(ellps='WGS84');regions={}
    def add(region,split,aoi):
        ring=np.asarray(aoi['coordinates'][0]);lon=float((ring[:,0].min()+ring[:,0].max())/2);lat=float((ring[:,1].min()+ring[:,1].max())/2)
        radius=max(abs(geod.inv(lon,lat,float(x),float(y))[2]) for x,y in ring)+20
        regions[region]={'split':split,'centre':[lon,lat],'radius_m':radius}
    for scene in exposed_scenes():
        meta=read_json(Path(scene['path']).with_suffix('.json'))['source_observation'];add(scene['region'],'train',meta['aoi'])
    for site in sites:add(site['id'],site['split'],site['aoi'])
    comparisons=[]
    for i,(name,a) in enumerate(regions.items()):
        for other,b in list(regions.items())[i+1:]:
            if a['split']==b['split']:continue
            distance=abs(geod.inv(*a['centre'],*b['centre'])[2])-a['radius_m']-b['radius_m']
            comparisons.append({'first':name,'second':other,'conservative_separation_m':distance})
    closest=min(comparisons,key=lambda r:r['conservative_separation_m'])
    if closest['conservative_separation_m']<100_000:raise ValueError('New geographic split too close: '+str(closest))
    return {'created_at':utcnow(),'regions':regions,'closest_cross_split_pair':closest,
            'minimum_cross_split_separation_m':closest['conservative_separation_m'],'new_test_pixels_inspected':False,
            'method':'Geodesic bounding-centre distance minus enclosing vertex radii+20m; lower bounds >=100km'}


def create_plan(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);path=directory/'plan.json'
    if path.exists():return read_json(path)
    sites=[]
    for name,split,kind,(w,s,e,n) in SITES:
        sites.append({'id':name,'split':split,'kind':kind,'aoi':{'type':'Polygon','coordinates':[[[w,s],[e,s],[e,n],[w,n],[w,s]]]},
            'start':'2026-06-01','end':'2026-10-27','maximum_dates':2,'minimum_reference_land_cells':100,
            'minimum_reference_water_cells':100 if kind=='mixed' else 0})
    audit=separation_audit(sites);write_json(directory/'planned-separation-audit.json',audit)
    plan={'created_at':utcnow(),'sites':sites,'prior_exposed_training_scenes':46,
          'families':['radar','terrain','terrain_prior'],'fits_per_family':1,
          'physical_context':'COP GLO30 slope+relief and HAND2021; no absolute geographic coordinates/elevation feature',
          'soft_prior_context':'JRC2021 occurrence+distance-to-water+prior availability; learned input only, never a hard water mask',
          'selection':'Fixed geographies/date-spread/reference availability+clear SCL land/water support; no model predictions',
          'test_rule':'No new test model predictions or imagery inspection before validation family/threshold freeze',
          'labels':'Same exactly-once calibrated SCL/spectral consensus policy as round3; independent expert review pending',
          'history':'All46 exposed round3 scenes are TRAIN in this new experiment; frozen round3 artifacts remain intact',
          'radar_transfer_cap_bytes_per_case':128*1024**2,'halo_metres':320,
          'training_support':'Identical sampled training cells with required terrain availability for fair feature ablations',
          'thresholds':[i/10 for i in range(1,10)],
          'validation_gate':{'precision_when_water_ha_at_least_4':.9,'recall_when_water_ha_at_least_4':.9,
                             'land_fpr_when_land_ha_at_least_4':.04,'minimum_reference_coverage':.98},
          'rank':'Passing gate, then worst-region balanced accuracy, then macro-region balanced accuracy; deterministic ties',
          'reporting':'All scene/region class support and withholding; current water on low JRC occurrence as emerging-water diagnostic, not attributed construction/flood truth',
          'production_readiness':'Independent expert shoreline accuracy and global generalization remain unestablished'}
    write_json(path,plan);return plan


def prepare_scenes(data_root,output):
    data_root,output=Path(data_root),Path(output);output.mkdir(parents=True,exist_ok=True)
    scenes=exposed_scenes();audit=[]
    for path in sorted(data_root.glob('*/collection.json')):
        for selected in read_json(path).get('selected',[]):
            region,split=selected['region'],selected['split'];o=load_catalog_observation(selected['observation'])
            print('PREPARE',region,split,flush=True)
            labels,weights,evaluate,rgb,provenance=aligned_reference(o,selected['reference_dir'])
            oid=safe_id(region+'-'+o.metadata['acquired_at']);spec={'id':oid,'split':split,'group':region,'region':region,'acquired_at':o.metadata['acquired_at']}
            saved=save_scene(o,labels,weights,evaluate,output/'scenes'/(oid+'.npz'),spec,provenance,rgb)
            scenes.append(saved)
            if split!='test':audit.append({'region':region,'split':split,'water_cells':int((labels==1).sum()),'land_cells':int((labels==0).sum()),'interior_cells':int(evaluate.sum())})
    result={'created_at':utcnow(),'data_origin':'observed','label_kind':'weak','scenes':scenes,
            'plan_sha256':sha256(data_root/'plan.json'),'test_used_for_fitting_or_selection':False,
            'exposed_old_test_policy':'All prior round3 test and validation scenes now TRAIN; new regions supply validation/test',
            'expert_reference_completed':False}
    write_json(output/'scenes-manifest.json',result);write_json(output/'preparation-audit.json',{'records':audit})
    return result

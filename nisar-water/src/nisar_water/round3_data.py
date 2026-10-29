"""Geographically reserved collection and spectrally screened weak supervision.

No predictions select an image or supply a label. New test regions are excluded
from training manifests and their scores are not opened until model freeze.
"""
from pathlib import Path
import json
import numpy as np
import requests
import rasterio
from rasterio.warp import reproject,Resampling
from scipy.ndimage import binary_erosion
from .common import read_json,write_json,sha256,utcnow
from .online import discover,acquire
from .optical import fetch_reference,crop_public_cog

SITES=[
 ("lanier","train",[-83.99,34.20,-83.94,34.25]),
 ("marion","train",[-80.48,33.39,-80.43,33.44]),
 ("nasser","train",[32.79,23.96,32.84,24.01]),
 ("victoria","train",[32.58,-.07,32.63,-.02]),
 ("kariba","train",[28.71,-16.58,28.76,-16.53]),
 ("garda","train",[10.69,45.60,10.74,45.65]),
 ("biwa","train",[135.96,35.12,136.01,35.17]),
 ("winnipeg","train",[-96.82,50.62,-96.77,50.67]),
 ("hartwell","train",[-82.89,34.43,-82.84,34.48]),
 ("st-clair","train",[-82.75,42.43,-82.70,42.48]),
 ("travis","validation",[-97.98,30.39,-97.93,30.44]),
 ("champlain","validation",[-73.36,44.39,-73.31,44.44]),
 ("constance","validation",[9.37,47.58,9.42,47.63]),
 ("tahoe","test",[-120.04,39.05,-119.99,39.10]),
 ("pontchartrain","test",[-90.17,30.24,-90.12,30.29]),
 ("itaipu","test",[-54.58,-25.43,-54.53,-25.38]),
 ("naivasha","test",[36.32,-.79,36.37,-.74]),
]


def create_plan(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/"plan.json").exists():return read_json(root/"plan.json")
    sites=[{"id":name,"split":split,"aoi":{"type":"Polygon","coordinates":[[[w,s],[e,s],[e,n],[w,n],[w,s]]]},
            "start":"2026-06-01","end":"2026-10-27"} for name,split,(w,s,e,n) in SITES]
    plan={"created_at":utcnow(),"sites":sites,"max_dates_per_region":2,"max_reference_candidates":8,
          "selection":"Date/coverage/reference availability only, no radar model scores; each water body stays in one split",
          "halo_metres":320,"radar_range_cap_bytes_per_case":128*1024**2,
          "label_policy":{"water":"SCL6 & NDWI>0 & MNDWI>0.1","land":"SCL4/5 & (NDWI<0 or MNDWI<0)",
                          "interior_weight":1.,"boundary_weight":.3,"unknown_code":255,"reference_max_hours":24},
          "readiness":{"minimum_region_precision":.9,"minimum_region_recall":.9,"maximum_nonwater_false_positive_rate":.04,
                       "scope":"New test region/date weak-reference pilot; global and expert shoreline accuracy require further evidence"},
          "test_rule":"No test model predictions or model scores inspected before family/threshold freeze; all test imagery and labels withheld from fitting",
          "review_rule":"Automated SCL/spectral screening plus assistant visual QA is not independent human/field validation"}
    write_json(root/"plan.json",plan);return plan


def fetch_spectral(reference_dir,aoi):
    reference_dir=Path(reference_dir)
    path=reference_dir/"spectral.json"
    ref=read_json(reference_dir/"reference.json")
    if ref["state"]!="ready":return {"state":"unavailable"}
    item=next(i for i in read_json(reference_dir/"search.json")["items"] if i["id"]==ref["source_id"])
    if path.exists():
        saved=read_json(path)
        if saved.get('state')=='ready' and saved.get('calibration_policy')!='earthsearch-boa-once-v2':
            previous=path.with_name('spectral-rejected-offset-v1.json')
            if not previous.exists():write_json(previous,saved)
            for key,asset in saved['assets'].items():
                declared=item['assets'][key].get('raster:bands',[{}])[0]
                scale,offset,basis=optical_calibration(item,declared)
                asset.update(scale=scale,offset=offset,declared_offset=declared.get('offset'),calibration_basis=basis)
            saved['calibration_policy']='earthsearch-boa-once-v2';write_json(path,saved)
        return saved
    ledger={"body_bytes":0,"cap_bytes":64*1024**2,"requests":[]}
    result={"state":"ready","source_id":ref["source_id"],"assets":{},"calibration_policy":"earthsearch-boa-once-v2"}
    session=requests.Session();session.trust_env=False
    try:
        for key in ("green","nir","swir16"):
            asset=item["assets"][key]
            calibration=asset.get("raster:bands",[{}])[0]
            scale,offset,basis=optical_calibration(item,calibration)
            data,t,crs=crop_public_cog(asset["href"],aoi,ledger,session)
            output=reference_dir/(key+".tif")
            with rasterio.open(output,"w",driver="GTiff",count=1,height=data.shape[1],width=data.shape[2],
                               dtype=data.dtype,crs=crs,transform=t,compress="DEFLATE") as dst:dst.write(data[:1])
            result["assets"][key]={"path":str(output.resolve()),"sha256":sha256(output),"source_url":asset["href"],
                                   "scale":scale,"offset":offset,"declared_offset":calibration.get('offset'),
                                   "calibration_basis":basis,"nodata":calibration.get("nodata",0)}
    except Exception as e:
        result.update(state="failed",error=type(e).__name__,message=str(e)[:300])
    finally:
        session.close();write_json(reference_dir/"spectral-transfer.json",ledger)
    write_json(path,result);return result


def optical_calibration(item,declared):
    """Earth Search COGs may already have the BOA offset removed despite asset metadata.

    Provider issue Element84/earth-search#71 and the item-level flag must be
    considered. A True applied flag means do not subtract the offset twice.
    Ambiguous modern items are withheld instead of guessing calibration.
    """
    if 'scale' not in declared:raise ValueError('Reflectance scale unavailable')
    scale=float(declared['scale']);applied=item.get('properties',{}).get('earthsearch:boa_offset_applied')
    if applied is True:return scale,0.,'item_BOA_offset_already_applied; asset_offset_not_reapplied'
    if applied is False and 'offset' in declared:return scale,float(declared['offset']),'item_BOA_offset_not_applied; declared_offset_applied_once'
    try:baseline=float(item.get('properties',{}).get('s2:processing_baseline'))
    except (TypeError,ValueError):baseline=None
    if baseline is not None and baseline<4:return scale,0.,'pre04_baseline_has_no_BOA_add_offset'
    raise ValueError('Ambiguous modern Sentinel-2 BOA offset; reference withheld')



def candidate_order(candidates):
    valid=[c for c in candidates if c.get("compatible") and c.get("coverage_fraction",0)>=.8]
    valid=sorted(valid,key=lambda c:c["acquired_at"])
    seen=set();unique=[]
    for c in valid:
        stamp=c["acquired_at"]
        if stamp not in seen:unique.append(c);seen.add(stamp)
    if not unique:return []
    # Spread across the record; prefer geometrical/date diversity without predictions.
    positions=np.linspace(0,len(unique)-1,min(8,len(unique))).round().astype(int)
    return [unique[i] for i in positions]


def collect_site(site,root,reference_root):
    root=Path(root);folder=root/site["id"];folder.mkdir(parents=True,exist_ok=True)
    if (folder/"collection.json").exists():return read_json(folder/"collection.json")
    print("COLLECT",site["id"],site["split"],flush=True)
    discovery_path=folder/"discovery.json"
    if discovery_path.exists():found=read_json(discovery_path)
    else:
        found=discover(site["aoi"],site["start"],site["end"],folder,reference_root);write_json(discovery_path,found)
    selected=[];attempts=[]
    for candidate in candidate_order(found.get("candidates",[])):
        cid=candidate["concept_id"]+"-"+str(candidate.get("revision_id",1))
        ref_dir=folder/"references"/cid
        stub={"id":cid,"acquired_at":candidate["acquired_at"],"aoi":site["aoi"]}
        ref=fetch_reference(stub,ref_dir)
        diagnostic={"candidate_id":candidate["id"],"datetime":candidate["acquired_at"],"reference_state":ref["state"]}
        if ref["state"]!="ready":attempts.append(diagnostic);continue
        with rasterio.open(ref["local_assets"]["scl"]["path"]) as src:
            scl=src.read(1)
        clear=float(np.isin(scl,[4,5,6]).mean());water=int((scl==6).sum());land=int(np.isin(scl,[4,5]).sum())
        diagnostic.update(clear_fraction=clear,water_cells=water,land_cells=land)
        if clear<.3 or water<site.get('minimum_reference_water_cells',20) or land<site.get('minimum_reference_land_cells',0):
            diagnostic["selection_state"]="insufficient_clear_water_reference";attempts.append(diagnostic);continue
        spectral=fetch_spectral(ref_dir,site["aoi"])
        if spectral["state"]!="ready":
            diagnostic["selection_state"]="spectral_unavailable";attempts.append(diagnostic);continue
        receipt_path=folder/("acquisition-"+cid+".json")
        if receipt_path.exists():acquired=read_json(receipt_path)
        else:
            acquired=acquire(candidate,site["aoi"],folder,reference_root,halo_metres=320);write_json(receipt_path,acquired)
        diagnostic["acquisition_state"]=acquired["state"];attempts.append(diagnostic)
        if acquired["state"] not in ("acquired","cached"):continue
        selected.append({"observation":acquired["observation"],"reference_dir":str(ref_dir.resolve()),
                         "receipt":str(receipt_path.resolve()),"split":site["split"],"region":site["id"]})
        if len(selected)>=site.get('maximum_dates',2):break
    result={"site":site,"state":"acquired" if selected else "unavailable","selected":selected,"attempts":attempts}
    write_json(folder/"collection.json",result)
    print("COLLECTED",site["id"],len(selected),"images",flush=True)
    return result


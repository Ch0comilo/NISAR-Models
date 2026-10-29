"""Bounded physical terrain context and a separately identified historical prior.

No radar labels, model scores, credentials or future observations enter these
features. Source windows are genuine public COP DEM GLO30, HAND2021 and optional
JRC2021 COG crops. Native nominal 30 m resolution is retained in provenance;
the returned 20 m grid does not create finer independent terrain measurements.

``cache_dir=None`` permits existing caches/covering OPERA windows only. Passing
an explicit cache directory opts into bounded public downloads when necessary.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter, uniform_filter

from .common import identity, read_json, sha256, utcnow, write_json

TERRAIN_VERSION = "copdem-glo30-hand2021-physical-to20m-v2"
PRIOR_VERSION = "jrc2021-occurrence-soft-water50-distance1000m-v1"
TERRAIN_NAMES = ("dem_slope_deg", "dem_relief31_m", "hand_log1p_m", "hand_mean7_m", "hand_min7_m")
PRIOR_NAMES = ("jrc2021_occurrence_pct", "jrc2021_water50_distance_capped1000_m")
MAX_TRANSFER_BYTES = 128 * 1024**2
MAX_TARGET_PIXELS = 4_000_000
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / ".cache/terrain-context"
SOURCE_DESCRIPTIONS = {
    "dem": {"product": "Copernicus DEM GLO-30 DSM", "native_nominal_resolution_m": 30, "units": "metres; source vertical datum retained"},
    "hand": {"product": "GLO-30 HAND v1/2021", "native_nominal_resolution_m": 30, "units": "metres above nearest drainage"},
    "reference_water": {"product": "JRC Global Surface Water occurrence v1.4/2021", "native_nominal_resolution_m": 30, "units": "percent occurrence 0..100; 255 unavailable"},
}


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _grid(observation):
    h, w = observation.valid.shape
    t = observation.transform
    if h*w > MAX_TARGET_PIXELS or h < 1 or w < 1:
        raise ValueError("Terrain context needs a nonempty bounded target of at most four million cells")
    if t.b or t.d or not np.isclose(t.a, 20) or not np.isclose(t.e, -20):
        raise ValueError("Terrain context requires a north-up 20 m target grid")
    if not observation.crs.is_projected:
        raise ValueError("Terrain derivatives require a projected metre grid")
    descriptor = {"shape": [h,w], "transform": [float(v) for v in list(t)[:6]], "epsg": observation.crs.to_epsg()}
    return descriptor, tuple(array_bounds(h,w,t))


def _expanded(bounds, padding):
    left,bottom,right,top = bounds
    return left-padding,bottom-padding,right+padding,top+padding


def _covers(path, crs, bounds):
    try:
        with rasterio.open(path) as source:
            needed = transform_bounds(crs, source.crs, *bounds, densify_pts=21)
            actual = source.bounds
            tolerance = max(abs(source.transform.a),abs(source.transform.e))*1e-4
            return (actual.left <= needed[0]+tolerance and actual.bottom <= needed[1]+tolerance
                    and actual.right >= needed[2]-tolerance and actual.top >= needed[3]-tolerance)
    except (OSError, ValueError, rasterio.errors.RasterioError):
        return False


def _source_record(path, source_urls, kind, *, reuse_origin, extra=None):
    path = Path(path).resolve()
    with rasterio.open(path) as src:
        nodata = src.nodata
        native = {"crs": str(src.crs), "transform": [float(v) for v in list(src.transform)[:6]],
                  "shape": list(src.shape), "nodata": "nan" if nodata is not None and not np.isfinite(nodata) else nodata}
    return {**SOURCE_DESCRIPTIONS[kind], "source_urls": list(source_urls), "cropped_window_path": str(path),
            "cropped_window_sha256": sha256(path), "hash_scope": "cropped_window; full remote raster checksum unavailable",
            "native_window": native, "reuse_origin": reuse_origin, **(extra or {})}


def _candidate_records(cache):
    candidates = {kind:[] for kind in SOURCE_DESCRIPTIONS}
    for receipt in cache.glob("sources/*/source.json"):
        try:
            record = read_json(receipt)
            kind = record.get("kind")
            if kind in candidates and record.get("cropped_window_path"):
                candidates[kind].append(record)
        except (OSError, ValueError, KeyError):
            continue
    # Only public ancillary manifests are inspected; no model outputs or labels.
    paths = list((ROOT/".cache/opera-inference").glob("*/request.json"))
    paths += list((ROOT/"outputs/opera-round3").glob("*/request.json"))
    for receipt in paths:
        try:
            request = read_json(receipt)
            for kind in candidates:
                asset = request.get("ancillary",{}).get(kind)
                if asset and asset.get("path") and asset.get("sha256"):
                    candidates[kind].append({"cropped_window_path":asset["path"],
                         "cropped_window_sha256":asset["sha256"], "source_urls":asset.get("source_urls",[]),
                         "reuse_origin":"existing_OPERA_public_window", "source_manifest_path":str(receipt),
                         "source_manifest_sha256":sha256(receipt)})
        except (OSError, ValueError, KeyError):
            continue
    return candidates


def _existing_source(records, kind, crs, bounds):
    for record in records:
        path = Path(record["cropped_window_path"])
        if not path.is_file() or not _covers(path,crs,bounds):
            continue
        if sha256(path) != record["cropped_window_sha256"]:
            continue
        extra = {k:record[k] for k in ("source_manifest_path","source_manifest_sha256") if k in record}
        return _source_record(path,record.get("source_urls",[]),kind,
                              reuse_origin=record.get("reuse_origin","terrain_source_cache"),extra=extra)
    return None


def _read_clean(source, kind, window=None):
    values = source.read(1, window=window, masked=True).astype(np.float32).filled(np.nan)
    good = np.isfinite(values)
    if kind == "hand":
        good &= values >= 0
    elif kind == "reference_water":
        good &= (values >= 0) & (values <= 100)
    values[~good] = np.nan
    return values


def _download_source(kind, bounds, crs, cache, ledger, session):
    from .opera_adapter import ancillary_sources, PublicRangeFile
    geographic = transform_bounds(crs,"EPSG:4326",*bounds,densify_pts=21)
    urls = ancillary_sources(geographic)[kind]
    key = identity({"kind":kind,"bounds_wgs84":list(geographic),"urls":urls})[:28]
    folder = cache/"sources"/(kind+"-"+key)
    folder.mkdir(parents=True,exist_ok=True)
    receipt, path = folder/"source.json", folder/"window.tif"
    if receipt.exists():
        record = read_json(receipt)
        if path.is_file() and sha256(path)==record.get("cropped_window_sha256"):
            return {**record,"reuse_origin":"terrain_source_cache"}
        raise ValueError("Terrain cropped source cache integrity failed")
    memory_files, readers, errors = [], [], []
    source_info = []
    try:
        for url in urls:
            blocks = {}
            def opener(value,mode="rb"):
                if value!=url: raise FileNotFoundError(value)
                return PublicRangeFile(url,blocks,ledger,session)
            try:
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                    with rasterio.open(url,opener=opener) as src:
                        needed = transform_bounds(crs,src.crs,*bounds,densify_pts=21)
                        raw = from_bounds(*needed, transform=src.transform)
                        c0,r0 = max(0,math.floor(raw.col_off)-3),max(0,math.floor(raw.row_off)-3)
                        c1,r1 = min(src.width,math.ceil(raw.col_off+raw.width)+3),min(src.height,math.ceil(raw.row_off+raw.height)+3)
                        if c1<=c0 or r1<=r0: continue
                        if (c1-c0)*(r1-r0)>8_000_000:
                            raise ValueError("Ancillary source crop exceeds eight million native cells")
                        window = Window(c0,r0,c1-c0,r1-r0)
                        values = _read_clean(src,kind,window)
                        affine = src.window_transform(window)
                        source_info.append({"url":url,"crs":str(src.crs),"native_transform":list(src.transform)[:6],
                                            "native_dtype":str(src.dtypes[0]),"window":[c0,r0,c1-c0,r1-r0]})
                        memory=MemoryFile();memory_files.append(memory)
                        with memory.open(driver="GTiff",height=values.shape[0],width=values.shape[1],count=1,
                                         dtype="float32",crs=src.crs,transform=affine,nodata=np.nan) as dst:
                            dst.write(values,1)
                        readers.append(memory.open())
            except Exception as error:
                errors.append({"url":url,"error_type":type(error).__name__,"message":str(error)[:300]})
                if ledger["body_bytes"]>=ledger["cap_bytes"]: break
        if not readers:
            raise ValueError("No usable public ancillary windows: "+json.dumps(errors)[:1000])
        merged, affine = merge(readers,nodata=np.nan)
        temporary = folder/("window-"+uuid.uuid4().hex+".part.tif")
        with rasterio.open(temporary,"w",driver="GTiff",width=merged.shape[2],height=merged.shape[1],count=1,
                           dtype="float32",crs=readers[0].crs,transform=affine,nodata=np.nan,compress="DEFLATE") as dst:
            dst.write(merged)
        temporary.replace(path)
        record = _source_record(path,urls,kind,reuse_origin="new_bounded_public_window",
                                extra={"kind":kind,"native_sources":source_info,"tile_errors":errors,
                                       "extent_fully_covers_requested_bounds":_covers(path,crs,bounds),
                                       "created_at":utcnow()})
        write_json(receipt,record)
        return record
    finally:
        for reader in readers:reader.close()
        for memory in memory_files:memory.close()


def _warp(record,kind,shape,affine,crs):
    result = np.full(shape,np.nan,np.float32)
    support = np.zeros(shape,np.float32)
    if record is None:
        return result,np.zeros(shape,bool)
    with rasterio.open(record["cropped_window_path"]) as source:
        raw = _read_clean(source,kind)
        valid = np.isfinite(raw).astype(np.float32)
        kwargs={"src_transform":source.transform,"src_crs":source.crs,"dst_transform":affine,"dst_crs":crs}
        reproject(raw,result,src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.bilinear,**kwargs)
        # Zeros in the mask are data: unknown neighbors may not be ignored.
        reproject(valid,support,dst_nodata=0,resampling=Resampling.bilinear,**kwargs)
    good=(support>=1-1e-6)&np.isfinite(result)
    result[~good]=np.nan
    return result,good


def physical_features(dem30,dem30_valid,dem30_transform,hand20,hand20_valid,dem20,dem20_valid,target_transform,crs):
    """Derive physical planes; every neighborhood requires complete known support."""
    dem30_valid=np.asarray(dem30_valid,bool)&np.isfinite(dem30)
    valid_slope=minimum_filter(dem30_valid.astype(np.uint8),size=3,mode="constant",cval=0)==1
    safe=np.where(dem30_valid,dem30,0).astype(np.float64)
    dy,dx=np.gradient(safe,abs(dem30_transform.e),abs(dem30_transform.a))
    slope30=np.rad2deg(np.arctan(np.hypot(dx,dy))).astype(np.float32)
    slope30[~valid_slope]=np.nan
    shape=dem20.shape
    slope=np.full(shape,np.nan,np.float32);support=np.zeros(shape,np.float32)
    kwargs={"src_transform":dem30_transform,"src_crs":crs,"dst_transform":target_transform,"dst_crs":crs}
    reproject(slope30,slope,src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.bilinear,**kwargs)
    reproject(valid_slope.astype(np.float32),support,dst_nodata=0,resampling=Resampling.bilinear,**kwargs)
    slope[(support<1-1e-6)|~np.isfinite(slope)]=np.nan
    valid_relief=minimum_filter(dem20_valid.astype(np.uint8),size=31,mode="constant",cval=0)==1
    maximum=maximum_filter(np.where(dem20_valid,dem20,-np.inf),size=31,mode="constant",cval=-np.inf)
    minimum=minimum_filter(np.where(dem20_valid,dem20,np.inf),size=31,mode="constant",cval=np.inf)
    relief=(maximum-minimum).astype(np.float32);relief[~valid_relief]=np.nan
    hand20_valid=np.asarray(hand20_valid,bool)&np.isfinite(hand20)&(hand20>=0)
    valid_hand7=minimum_filter(hand20_valid.astype(np.uint8),size=7,mode="constant",cval=0)==1
    loghand=np.full(shape,np.nan,np.float32)
    np.log1p(hand20,out=loghand,where=hand20_valid)
    handmean=uniform_filter(np.where(hand20_valid,hand20,0).astype(np.float64),size=7,mode="constant",cval=0).astype(np.float32)
    # A mean of nonnegative heights cannot be negative mathematically. Running
    # filter subtraction can leave tiny negative roundoff near zero; enforce
    # that physical invariant before the explicit validity/range checks.
    np.maximum(handmean,0,out=handmean)
    handmin=minimum_filter(np.where(hand20_valid,hand20,np.inf),size=7,mode="constant",cval=np.inf).astype(np.float32)
    handmean[~valid_hand7]=np.nan;handmin[~valid_hand7]=np.nan
    planes=np.stack([slope,relief,loghand,handmean,handmin]).astype(np.float32)
    return planes,np.isfinite(planes).all(0)


def prior_features(occurrence,valid,spacing=20.,cap=1000.):
    """Occurrence and conservative capped distance; unknown areas never become land."""
    valid=np.asarray(valid,bool)&np.isfinite(occurrence)&(occurrence>=0)&(occurrence<=100)
    water=valid&(occurrence>=50)
    # Explicit outside unknown/water-free padding prevents scipy's all-ones
    # boundary convention from inventing a nearest water pixel outside the crop.
    padded_water=np.pad(water,1,constant_values=False)
    if water.any():
        distance=distance_transform_edt(~padded_water,sampling=spacing)[1:-1,1:-1]
    else:
        distance=np.full(valid.shape,np.inf,np.float64)
    known_radius=distance_transform_edt(np.pad(valid,1,constant_values=False),sampling=spacing)[1:-1,1:-1]
    capped=np.minimum(distance,cap)
    distance_valid=valid&(known_radius>capped+spacing*.5)
    occurrence_plane=np.where(valid,occurrence,np.nan).astype(np.float32)
    distance_plane=np.where(distance_valid,capped,np.nan).astype(np.float32)
    return np.stack([occurrence_plane,distance_plane]),distance_valid,valid


def _load_cached(path,manifest,descriptor,include_prior):
    record=read_json(manifest)
    if record.get("profile")!=TERRAIN_VERSION or record.get("grid")!=descriptor or record.get("include_prior")!=include_prior:
        raise ValueError("Terrain feature cache contract mismatch")
    if sha256(path)!=record["features_sha256"]:
        raise ValueError("Terrain feature cache integrity failed")
    with np.load(path,allow_pickle=False) as archive:
        result={key:archive[key].copy() for key in ("features","available","prior_available","prior_occurrence_available")}
        result["names"]=archive["names"].tolist()
    result["metadata"]={**record["metadata"],"cache_hit":True,"body_bytes":0,
                        "feature_cache_path":str(path),"feature_cache_sha256":record["features_sha256"]}
    return result


def load_context(observation,cache_dir=None,*,include_prior=False):
    """Return aligned physical features and optional soft prior with provenance.

    Missing required terrain returns NaNs and ``available=False``. Optional JRC
    occurrence/distance availability is separate, and never masks the radar or
    rewrites labels. Explicit ``cache_dir`` authorizes public bounded fetching;
    otherwise this function uses existing local public windows/caches only.
    """
    descriptor,bounds=_grid(observation)
    cache=Path(cache_dir).resolve() if cache_dir is not None else DEFAULT_CACHE
    allow_network=cache_dir is not None
    key=identity({"profile":TERRAIN_VERSION,"prior":PRIOR_VERSION if include_prior else None,"grid":descriptor})
    folder=cache/"features"/key
    archive_path,manifest=folder/"features.npz",folder/"manifest.json"
    if manifest.exists():return _load_cached(archive_path,manifest,descriptor,include_prior)
    aoi=observation.metadata.get("aoi")
    region_key=identity({"epsg":descriptor["epsg"],"aoi":aoi if aoi is not None else list(bounds)})[:24]
    region=cache/"regions"/region_key
    with _lock(region/"lock"):
        if manifest.exists():return _load_cached(archive_path,manifest,descriptor,include_prior)
        ledger_path=region/"transfer.json"
        ledger=read_json(ledger_path) if ledger_path.exists() else {"body_bytes":0,"cap_bytes":MAX_TRANSFER_BYTES,"requests":[]}
        if ledger.get("cap_bytes")!=MAX_TRANSFER_BYTES or ledger["body_bytes"]>MAX_TRANSFER_BYTES:
            raise ValueError("Unexpected terrain region transfer ledger")
        before=ledger["body_bytes"]
        candidates=_candidate_records(cache)
        sources,errors={},{}
        import requests
        session=requests.Session();session.trust_env=False
        try:
            for kind in (("dem","hand","reference_water") if include_prior else ("dem","hand")):
                needed=_expanded(bounds,1060 if kind=="reference_water" else 360)
                source=_existing_source(candidates[kind],kind,observation.crs,needed)
                if source is None and allow_network:
                    try:source=_download_source(kind,needed,observation.crs,cache,ledger,session)
                    except Exception as error:errors[kind]={"error_type":type(error).__name__,"message":str(error)[:1000]}
                elif source is None:
                    errors[kind]={"error_type":"cache_only_missing_source","message":"No covering local public ancillary crop; network was not authorized by an explicit cache_dir"}
                sources[kind]=source
        finally:
            session.close();write_json(ledger_path,ledger)
        # Derivative processing uses a real 30 m metre grid with enough context.
        expanded=_expanded(bounds,340)
        t30=from_origin(math.floor(expanded[0]/30)*30,math.ceil(expanded[3]/30)*30,30,30)
        shape30=(math.ceil((t30.f-expanded[1])/30),math.ceil((expanded[2]-t30.c)/30))
        dem30,dem30_valid=_warp(sources.get("dem"),"dem",shape30,t30,observation.crs)
        physical_halo=15
        t20=observation.transform@rasterio.Affine.translation(-physical_halo,-physical_halo)
        padded_shape=(observation.valid.shape[0]+2*physical_halo,observation.valid.shape[1]+2*physical_halo)
        dem20,dem20_valid=_warp(sources.get("dem"),"dem",padded_shape,t20,observation.crs)
        hand20,hand20_valid=_warp(sources.get("hand"),"hand",padded_shape,t20,observation.crs)
        padded_planes,padded_available=physical_features(dem30,dem30_valid,t30,hand20,hand20_valid,dem20,dem20_valid,t20,observation.crs)
        physical_crop=(slice(physical_halo,-physical_halo),slice(physical_halo,-physical_halo))
        planes=padded_planes[:,physical_crop[0],physical_crop[1]]
        available=padded_available[physical_crop]
        prior_available=np.zeros(observation.valid.shape,bool);prior_occurrence_available=prior_available.copy()
        names=list(TERRAIN_NAMES)
        if include_prior:
            target=observation.transform
            halo=52
            tp=target@rasterio.Affine.translation(-halo,-halo)
            prior_shape=(observation.valid.shape[0]+2*halo,observation.valid.shape[1]+2*halo)
            occurrence,occ_valid=_warp(sources.get("reference_water"),"reference_water",prior_shape,tp,observation.crs)
            pp,pa,po=prior_features(occurrence,occ_valid)
            crop=(slice(halo,-halo),slice(halo,-halo))
            planes=np.concatenate([planes,pp[:,crop[0],crop[1]]],axis=0)
            prior_available,prior_occurrence_available=pa[crop],po[crop]
            names+=list(PRIOR_NAMES)
        weights=np.asarray(observation.weights,dtype=float)
        observed=np.asarray(observation.valid,bool)&(weights>0)
        metadata={"profile":TERRAIN_VERSION,"prior_profile":PRIOR_VERSION if include_prior else None,
            "source_hashes":{kind:source["cropped_window_sha256"] for kind,source in sources.items() if source is not None},
            "sources":{kind:source for kind,source in sources.items() if source is not None},"source_errors":errors,
            "hash_scope":"cropped_window; full remote raster checksum unavailable",
            "native_nominal_resolution_m":30,"target_pixel_spacing_m":20,"target_epsg":descriptor["epsg"],
            "target_transform6":descriptor["transform"],"target_shape":descriptor["shape"],
            "feature_units":{"dem_slope_deg":"degrees","dem_relief31_m":"metres",
                "hand_log1p_m":"dimensionless log(1+HAND_metres/1_metre)","hand_mean7_m":"metres","hand_min7_m":"metres",
                "jrc2021_occurrence_pct":"percent","jrc2021_water50_distance_capped1000_m":"metres, right-censored at1000"},
            "methods":{"slope":"central derivative on 30m UTM grid, complete3x3 source support, bilinear20m view",
                "relief":"max-min on31x31 target cells, complete known DEM support",
                "hand":"negative/nodata unavailable; continuous bilinear20m view; complete7x7 support",
                "prior":"optional soft learned features only; occurrence>=50 seeds distance; cap1000m; unknown closer support invalidates distance"},
            "body_bytes":ledger["body_bytes"]-before,"region_total_body_bytes":ledger["body_bytes"],"transfer_cap_bytes":MAX_TRANSFER_BYTES,
            "transfer_ledger_path":str(ledger_path),"transfer_ledger_sha256":sha256(ledger_path),
            "network_permitted":allow_network,"cache_hit":False,
            "physical_available_cells":int(available.sum()),"physical_unknown_cells":int((~available).sum()),
            "observed_context_withheld_ha":float((weights[observed&~available]*.04).sum()),
            "prior_occurrence_available_cells":int(prior_occurrence_available.sum()),"prior_available_cells":int(prior_available.sum()),
            "limitations":["20m posting is an interpolated view of nominal30m terrain, not independent20m measurements",
                "Copernicus DSM includes surface objects; terrain derivatives are not a complete radar shadow/layover mask",
                "HAND and static historical occurrence do not establish current water presence or human cause"]}
        result={"features":planes.astype(np.float32),"available":available,"names":names,"metadata":metadata,
                "prior_available":prior_available,"prior_occurrence_available":prior_occurrence_available}
        # Cache complete acquisitions only. Cache-only misses and failed sources
        # may be repaired by a later explicitly authorized acquisition.
        required=["dem","hand"]+(["reference_water"] if include_prior else [])
        if all(sources.get(kind) is not None for kind in required):
            folder.mkdir(parents=True,exist_ok=True)
            temporary=folder/("features-"+uuid.uuid4().hex+".npz.part")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle,features=result["features"],available=available,names=np.array(names),
                    prior_available=prior_available,prior_occurrence_available=prior_occurrence_available,
                    transform6=np.array(descriptor["transform"]),epsg=np.int32(descriptor["epsg"]),metadata_json=json.dumps(metadata,allow_nan=False))
            temporary.replace(archive_path)
            digest=sha256(archive_path)
            write_json(manifest,{"profile":TERRAIN_VERSION,"grid":descriptor,"include_prior":include_prior,
                                  "features_sha256":digest,"metadata":metadata})
            result["metadata"]={**metadata,"feature_cache_path":str(archive_path),"feature_cache_sha256":digest}
        return result


"""Independent audit of the exposed v2 inputs, not the new round3 holdout.

The only data roots opened are the old phase1 catalog, the old spatial-v2
development archive, and new-images-20261028. Optional remote reads inspect only
the already exposed Mead/Okeechobee source metadata through the protected client.
No model fitting, threshold selection, credential serialization or new holdout
access occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nisar_water.common import read_json, write_json, sha256, utcnow
from nisar_water.features import FEATURE_NAMES, tiles
from nisar_water.inputs import load_catalog_observation


def strict_aggregate(power, valid, spacing):
    """Independent reference calculation for native-to-20m linear power."""
    power, valid = np.asarray(power, np.float32), np.asarray(valid, bool)
    good = valid & np.isfinite(power).all(0) & (power > 0).all(0)
    if spacing == 20:
        return power.copy(), good
    if spacing != 10:
        raise ValueError("Audit expects native 10/20 metre pixels")
    h, w = good.shape
    h, w = h // 2, w // 2
    result = np.full((2, h, w), np.nan, np.float32)
    accepted = np.zeros((h, w), bool)
    # Independent slice implementation avoids sharing the production reshape.
    blocks = [power[:, dr:2*h:2, dc:2*w:2] for dr in (0, 1) for dc in (0, 1)]
    accepted = np.logical_and.reduce([good[dr:2*h:2, dc:2*w:2] for dr in (0, 1) for dc in (0, 1)])
    result[:] = np.mean(np.stack(blocks).astype(np.float64), axis=0).astype(np.float32)
    result[:, ~accepted] = np.nan
    return result, accepted


def crop_interior_mask(shape, radius=15):
    """Cells whose full 31x31 neighborhood lies inside the cached source crop."""
    h, w = shape
    result = np.zeros(shape, bool)
    if h > radius * 2 and w > radius * 2:
        result[radius:h-radius, radius:w-radius] = True
    return result


def feature_rows(observation, take):
    values = np.full((*take.shape, len(FEATURE_NAMES)), np.nan, np.float32)
    eligible = np.zeros(take.shape, bool)
    for sl, block, good in tiles(observation.power, observation.valid):
        values[sl] = block
        eligible[sl] = good
    return values[take & eligible], eligible


def quantify(values):
    if not len(values):
        return None
    q = np.percentile(values, [1, 10, 50, 90, 99], axis=0)
    return {name: dict(zip(("p01", "p10", "p50", "p90", "p99"), [float(x) for x in q[:, i]]))
            for i, name in enumerate(FEATURE_NAMES)}


def inspect_extra_metadata(client=None, record=None, *, h5=None, max_bytes=16*1024**2):
    """Return bounded GCOV flags and LUT descriptors; never perform correction.

    Pass an already open h5py File through ``h5=`` to reuse a reader. Otherwise
    pass the existing protected range client and exact public source record.
    Noise values and incidence cubes are deliberately not materialized. An
    incidence channel needs actual terrain-height-aware interpolation; absence
    or a zero-height placeholder is never represented as a measured angle.
    """
    import h5py

    def collect(source):
        base = "/science/LSAR/GCOV/metadata"
        parameters = base+"/processingInformation/parameters"
        flags = ("noiseCorrectionApplied", "radiometricTerrainCorrectionApplied",
                 "shadowMaskingApplied", "postProcessingFilteringApplied",
                 "preprocessingMultilookingApplied", "validSamplesSubSwathMaskingApplied",
                 "polarimetricSymmetrizationApplied", "polarizationOrientationCorrectionApplied",
                 "faradayRotationApplied")
        result = {"schema":"nisar-gcov-extra-metadata-v1", "state":"inspected",
                  "processing_flags":{}, "processing_flag_source_values":{},
                  "datasets":{}, "small_scalar_metadata":{},
                  "incidence_angle_status":"unavailable",
                  "noise_lut_status":"unavailable", "radiometric_arithmetic_applied":False}
        def scalar(path):
            if path not in source or not isinstance(source[path],h5py.Dataset) or source[path].shape != ():
                return None
            value = source[path][()]
            if isinstance(value,bytes): value=value.decode("utf-8",errors="replace")
            elif isinstance(value,np.generic): value=value.item()
            if isinstance(value,float) and not np.isfinite(value): return None
            return value if isinstance(value,(str,int,float,bool)) else None
        for flag in flags:
            value=scalar(parameters+"/"+flag)
            result["processing_flag_source_values"][flag]=value
            if isinstance(value,str) and value.lower() in {"true","false"}:
                value=value.lower()=="true"
            result["processing_flags"][flag]=value if isinstance(value,bool) else None
        scalar_paths=[parameters+"/geocoding/isSourceDataUpsampled",
                      parameters+"/preprocessing/frequencyA/numberOfAzimuthLooks",
                      parameters+"/preprocessing/frequencyA/numberOfRangeLooks"]
        for path in scalar_paths:
            result["small_scalar_metadata"][path]=scalar(path)
        roots=[base+"/radarGrid",base+"/calibrationInformation/frequencyA/noiseEquivalentBackscatter"]
        for root in roots:
            for name in ("incidenceAngle","heightAboveEllipsoid","xCoordinates","yCoordinates","projection","HH","HV"):
                path=root+"/"+name
                if path not in source or not isinstance(source[path],h5py.Dataset): continue
                dataset=source[path]
                result["datasets"][path]={"shape":list(dataset.shape),"dtype":str(dataset.dtype),
                    "attributes":{str(k):str(v) for k,v in dataset.attrs.items() if k in {"description","units","_FillValue","long_name","valid_min","valid_max"}}}
                if name=="projection":result["small_scalar_metadata"][path]=scalar(path)
        if base+"/radarGrid/incidenceAngle" in result["datasets"]:
            result["incidence_angle_status"]="cube_available_not_interpolated_at_terrain_height"
        noise=base+"/calibrationInformation/frequencyA/noiseEquivalentBackscatter/"
        if noise+"HH" in result["datasets"] and noise+"HV" in result["datasets"]:
            result["noise_lut_status"]="HH_HV_descriptors_available_values_not_applied"
        result["limitations"]=["No terrain-height interpolation or incidence angle feature was created",
                                "Noise LUT descriptions/radiometric convention must be verified before subtraction or SNR arithmetic",
                                "Native GCOV subswath validity is not a complete geometry shadow/layover screen"]
        return result

    if h5 is not None:
        return collect(h5)
    if client is None or record is None:
        raise ValueError("Provide an open HDF5 file, or the existing protected client and source record")
    sys.path.insert(0,str(ROOT.parent/"nisar-models/src"))
    from nisar_models.acquisition import HTTPSRangeReader
    with HTTPSRangeReader(client,record["https_urls"][0],max_bytes=max_bytes) as reader, h5py.File(reader,"r") as source:
        result=collect(source)
        result.update(source_granule=record.get("granule_id"),range_bytes=reader.bytes_transferred)
        return result


def audit_entry(entry):
    paths = entry["assets"]
    with rasterio.open(paths["HHHH"]["reference"]) as hh, rasterio.open(paths["HVHV"]["reference"]) as hv, rasterio.open(paths["mask"]["reference"]) as mask:
        raw = np.stack([hh.read(1), hv.read(1)]).astype(np.float32)
        decoded = np.isin(mask.read(1), entry.get("mask_valid_values", [1]))
        expected, good = strict_aggregate(raw, decoded, abs(hh.transform.a))
        source_tags = {"HHHH": hh.tags(), "HVHV": hv.tags()}
        spacing = abs(hh.transform.a)
    observation = load_catalog_observation(entry)
    assert np.array_equal(good, observation.valid), "Validity changed relative to independent aggregation"
    delta = float(np.max(np.abs(expected[:, good] - observation.power[:, good]))) if good.any() else None
    assert delta is None or delta <= 2e-6 * max(1, float(np.max(expected[:, good])))
    mask_agreement = None
    if "native_quality_mask" in paths:
        with rasterio.open(paths["native_quality_mask"]["reference"]) as source:
            native = source.read(1)
        expected_native = (native > 0) & (native < 255) & np.isfinite(raw).all(0) & (raw > 0).all(0)
        mask_agreement = bool(np.array_equal(expected_native, decoded))
        assert mask_agreement, "Derived validity does not match native mask and finite positive powers"
    _, supported = feature_rows(observation, np.zeros_like(observation.valid))
    meta = entry.get("source_metadata", {})
    layers = meta.get("scientific_layers", {})
    for channel in ("HHHH", "HVHV"):
        if layers:
            assert layers[channel]["dataset"].endswith("/"+channel), "Stored HDF5 channel identity differs"
        assert source_tags[channel].get("polarization", channel) == channel
        assert source_tags[channel].get("units", "gamma0_power") == "gamma0_power"
    source = meta.get("source", {})
    ident = meta.get("hdf5_identification", {})
    check = {"id": entry["id"], "granule_id": entry["granule_id"], "native_spacing_m": spacing,
             "native_shape": list(raw.shape[1:]), "analysis_shape": list(observation.valid.shape),
             "mode": source.get("mode", entry["granule_id"].split("_")[8]),
             "processing_version": entry.get("processing_version"), "product_specification_version": ident.get("productSpecificationVersion"),
             "crid": entry.get("crid"), "track": entry.get("track"), "pass_direction": entry.get("pass_direction"),
             "frequency": entry["frequency"], "channels": entry["polarizations"], "units": entry["units"],
             "rtc_applied": entry["rtc_applied"], "independent_aggregation_max_abs_error": delta,
             "native_mask_agreement": mask_agreement, "channel_tags": source_tags,
             "source_hdf5_layers": {k: {"dataset": v.get("dataset"), "attributes": {a: v.get("attributes", {}).get(a) for a in ("units", "long_name", "valid_min", "_FillValue")}} for k,v in layers.items()},
             "source_halo_policy": entry.get("acquisition_identity", {}).get("preprocessing"),
             "available_geometry_or_noise_fields_in_retained_metadata": []}
    def visit(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                path = prefix+"/"+str(key)
                if any(part in key.lower() for part in ("incidence", "noise", "radiometric", "nlooks")):
                    check["available_geometry_or_noise_fields_in_retained_metadata"].append(path)
                visit(value, path)
    visit(meta)
    return observation, check


def inspect_exposed_remote_metadata(entries, output):
    """Shape/attribute and small scalar inspection, never full geometry cubes."""
    sys.path.insert(0, str(ROOT.parent/"nisar-models/src"))
    from nisar_models.acquisition import HTTPSRangeReader
    from nisar_models.earthdata_access import EarthdataHTTP, TransferLedger
    import h5py
    ledger = TransferLedger(output.with_name("round3-input-audit-remote-ledger.json"), limit=64*1024**2,
                            storage_root=output.parent, storage_limit=12*1024**3)
    client = EarthdataHTTP(ledger, timeout=30, retries=1)
    result = []
    for entry in entries:
        urls = entry.get("source_metadata", {}).get("source", {}).get("https_urls", [])
        if not urls:
            urls = [value for value in entry.get("provenance", []) if value.startswith("https://nisar.asf.earthdatacloud.nasa.gov/")]
        if not urls:
            result.append({"id":entry["id"], "state":"source_url_unavailable"})
            continue
        url = urls[0]
        row = {"id":entry["id"], "source_url":url, "datasets":{}, "metadata_scalar_values":{}}
        try:
            with HTTPSRangeReader(client, url, max_bytes=16*1024**2) as reader, h5py.File(reader,"r") as h5:
                roots = ["/science/LSAR/GCOV/metadata/radarGrid",
                         "/science/LSAR/GCOV/metadata/calibrationInformation/frequencyA/noiseEquivalentBackscatter",
                         "/science/LSAR/GCOV/metadata/processingInformation/parameters"]
                for root in roots:
                    if root not in h5:
                        row["datasets"][root] = {"present":False}
                        continue
                    def collect(name, item):
                        if not isinstance(item, h5py.Dataset): return
                        path = root+"/"+name
                        row["datasets"][path] = {"shape":list(item.shape),"dtype":str(item.dtype),
                                                 "attributes":{str(k):str(v) for k,v in item.attrs.items() if k in {"description","units","_FillValue","long_name"}}}
                        if item.shape == () and item.dtype.kind in {"b","i","u","f","S","U","O"}:
                            value = item[()]
                            if isinstance(value, bytes):value=value.decode("utf-8", errors="replace")
                            elif isinstance(value, np.generic):value=value.item()
                            if isinstance(value,float) and not np.isfinite(value):value=None
                            if isinstance(value,str) and len(value)>512:
                                row["metadata_scalar_values"][path]={"text_length":len(value),"sha256":hashlib.sha256(value.encode()).hexdigest(),"text_not_serialized":True}
                            elif isinstance(value,(str,int,float,bool,type(None))):row["metadata_scalar_values"][path]=value
                    h5[root].visititems(collect)
                row["range_bytes"] = reader.bytes_transferred
                row["state"] = "inspected"
        except Exception as error:
            row.update(state="unavailable", error_type=type(error).__name__)
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"evidence/round3-input-audit.json")
    parser.add_argument("--remote-metadata",action="store_true")
    args = parser.parse_args()
    old_root = ROOT/"data/new-images-20261028"
    old_report = read_json(ROOT/"outputs/new-images-20261028/report.json")
    catalog = {e["id"]:e for e in read_json(ROOT.parent/"nisar-landcover-phase1/data/catalog.json")["observations"]}
    with np.load(ROOT/"models/spatial-v2/development/train.npz",allow_pickle=False) as z:
        train_water = z["context"][z["y"] == 1]
        positives = {str(oid):int(((z["observation_ids"]==oid)&(z["y"]==1)).sum()) for oid in np.unique(z["observation_ids"]) if ((z["observation_ids"]==oid)&(z["y"]==1)).any()}
    entries = []
    for path in sorted(old_root.glob("*/acquisition-*.json")):
        acquired = read_json(path)
        if acquired.get("state") in {"acquired","cached"}: entries.append((path.parent.name, acquired["observation"]))
    report = {"created_at":utcnow(), "scope":"Exposed 20261028 batch and old development only; new round3 holdout not accessed",
              "training_water_positive_counts":positives,"training_water_feature_quantiles":quantify(train_water),
              "source_checks":[],"reference_water_diagnostics":[],"new_round3_holdout_opened":False,"models_fitted":False}
    for oid in positives:
        _, check = audit_entry(catalog[oid]);check["role"]="old_positive_training_observation";report["source_checks"].append(check)
    train_low, train_high = np.percentile(train_water,[1,99],axis=0)
    for case, entry in entries:
        observation, check = audit_entry(entry)
        check.update(case=case,role="exposed_regression_observation")
        report["source_checks"].append(check)
        row = next(r for r in old_report["rows"] if r.get("observation_id")==entry["id"])
        folder = ROOT/"outputs/new-images-20261028/measurements"/case/entry["id"]
        reference_path = folder/"reference-support.npz"
        if not reference_path.exists(): continue
        with np.load(reference_path) as z:water=(z["classes"]==6)&z["evaluate"]
        if not water.any(): continue
        features, supported = feature_rows(observation, water)
        values = quantify(features)
        interior = crop_interior_mask(observation.valid.shape)
        with np.load(folder/"spatial_v2.npz") as z:
            labels, score = z["labels"], z["scores"]
        actual_water = water & supported
        interior_water = actual_water & interior
        missing_context = actual_water & ~interior
        record = {"case":case,"observation_id":entry["id"],"reference_water_cells":int(actual_water.sum()),
                  "feature_quantiles":values,"reference_water_near_source_crop_edge_fraction":float(missing_context.sum()/actual_water.sum()),
                  "interior_reference_water_cells":int(interior_water.sum()),
                  "interior_water_recall":float((labels[interior_water]==1).mean()) if interior_water.any() else None,
                  "all_reference_water_recall":float((labels[actual_water]==1).mean()),
                  "reference_water_score_p10_p50_p90_p99":np.percentile(score[actual_water],[10,50,90,99]).tolist(),
                  "feature_outside_training_water_p01_p99_fraction":{name:float(((features[:,i]<train_low[i])|(features[:,i]>train_high[i])).mean()) for i,name in enumerate(FEATURE_NAMES)},
                  "source_context_radius_m":300,"recorded_online_halo_m":80}
        report["reference_water_diagnostics"].append(record)
    metadata_entries = [catalog[oid] for oid in positives]
    for name in ("lake-mead", "lake-okeechobee"):
        scored = {row["observation_id"] for row in old_report["rows"] if row["case"] == name and row["models"]["spatial_v2"]["metrics"]}
        metadata_entries.append(next(e for case,e in entries if case == name and e["id"] in scored))
    report["remote_metadata"] = inspect_exposed_remote_metadata(metadata_entries,args.output) if args.remote_metadata else []
    report["conclusion"] = "No demonstrated channel, units, double-log or power-aggregation defect. Source-crop halo is insufficient for full 31-cell context near edges; interior failures and narrow training-water distributions remain. Missing retained geometry/noise metadata prevents causal attribution."
    write_json(args.output,report)
    for row in report["reference_water_diagnostics"]:
        print(row["case"],"recall",round(row["all_reference_water_recall"],4),"interior recall",round(row["interior_water_recall"],4),"edge fraction",round(row["reference_water_near_source_crop_edge_fraction"],4))
    print("Audit saved:",args.output)


if __name__ == "__main__":main()


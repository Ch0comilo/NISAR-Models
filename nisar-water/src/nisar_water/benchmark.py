"""Frozen-model measurements on new images; no fitting or test-based selection."""
from pathlib import Path
import csv
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from .common import read_json, write_json, sha256, utcnow
from .features import tiles
from .inputs import load_catalog_observation
from .model import WaterModel
from .pipeline import run_series


def confusion_metrics(truth, prediction, areas):
    truth, prediction, areas = np.asarray(truth, bool), np.asarray(prediction, bool), np.asarray(areas, float)
    tp = float(areas[truth & prediction].sum())
    fp = float(areas[~truth & prediction].sum())
    fn = float(areas[truth & ~prediction].sum())
    tn = float(areas[~truth & ~prediction].sum())
    return {"precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "iou": tp/(tp+fp+fn) if tp+fp+fn else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "false_positive_rate": fp/(fp+tn) if fp+tn else None,
            "true_water_ha":tp,"false_water_ha":fp,"missed_water_ha":fn,"true_nonwater_ha":tn,
            "reference_water_ha":tp+fn,"reference_nonwater_ha":tn+fp,"evaluated_ha":tp+fp+fn+tn}


def reference_support(observation, reference):
    raw = np.zeros(observation.valid.shape, np.uint8)
    with rasterio.open(reference["local_assets"]["scl"]["path"]) as source:
        reproject(source.read(1),raw,src_transform=source.transform,src_crs=source.crs,
                  dst_transform=observation.transform,dst_crs=observation.crs,resampling=Resampling.nearest)
    classes = np.zeros_like(raw)
    for code in (4,5,6):
        classes[binary_erosion(raw==code,iterations=2) & (observation.weights>=1-1e-6)] = code
    supported = np.zeros_like(observation.valid)
    for sl,_,valid in tiles(observation.power,observation.valid):
        supported[sl]=valid
    evaluate=(classes>0) & observation.valid & supported & ~observation.exclusions
    return raw,classes,evaluate


def geographic_distance(observation, experiment):
    rr,cc=np.where(observation.valid & (observation.weights>0))
    xy=np.column_stack(rasterio.transform.xy(observation.transform,rr,cc))
    distances={}
    for split in ("train","validation"):
        with np.load(Path(experiment)/"development"/(split+".npz"),allow_pickle=False) as z:
            if observation.metadata["id"] in z["observation_ids"]:
                raise ValueError("New observation occurs in development")
            minimum=float("inf")
            for epsg in np.unique(z["epsg"]):
                points=z["xy"][z["epsg"]==epsg]
                tx,ty=Transformer.from_crs(int(epsg),observation.crs,always_xy=True).transform(points[:,0],points[:,1])
                projected=np.column_stack((tx,ty))
                projected=projected[np.isfinite(projected).all(1)]
                if len(projected):
                    minimum=min(minimum,float(cKDTree(projected).query(xy,k=1)[0].min()))
            if not np.isfinite(minimum) or minimum<140:
                raise ValueError("New-image geographic separation failed")
            distances[split]=minimum
    return distances


def evaluate_batch(data_root, output, experiment, baseline):
    data_root,output,experiment,baseline=map(Path,(data_root,output,experiment,baseline))
    output.mkdir(parents=True,exist_ok=True)
    plan=read_json(data_root/"plan.json")
    freeze=read_json(experiment/"freeze.json")
    new=WaterModel(experiment/"selected")
    old=WaterModel(baseline)
    if new.fingerprint!=plan["model_sha256"] or new.fingerprint!=freeze["model_sha256"]:
        raise ValueError("Frozen model identity changed")
    rows=[]
    for case in plan["cases"]:
        folder=data_root/case["id"]
        entries=[]
        for receipt in sorted(folder.glob("acquisition-*.json")):
            acquired=read_json(receipt)
            if acquired.get("state") in ("acquired","cached"):
                entries.append(acquired["observation"])
        if not entries:
            rows.append({"case":case["id"],"state":"no_acquired_images"})
            continue
        for entry in entries:
            print("Benchmark",case["id"],entry["acquired_at"],flush=True)
            image_dir=output/"measurements"/case["id"]/entry["id"]
            image_dir.mkdir(parents=True,exist_ok=True)
            if (image_dir/"metrics.json").exists():
                rows.append(read_json(image_dir/"metrics.json"))
                continue
            o=load_catalog_observation(entry)
            distance=geographic_distance(o,experiment)
            ref=read_json(folder/"references"/entry["id"]/"reference.json")
            raw=classes=evaluate=None
            if ref["state"]=="ready":
                raw,classes,evaluate=reference_support(o,ref)
                np.savez_compressed(image_dir/"reference-support.npz",raw_scl=raw,classes=classes,evaluate=evaluate,
                                    transform=np.array(list(o.transform)[:6]))
            write_json(image_dir/"support-freeze.json",{"frozen_at":utcnow(),"model_sha256":new.fingerprint,
                       "reference":ref,"minimum_distance_to_development_cells_m":distance,
                       "reference_support_sha256":sha256(image_dir/"reference-support.npz") if evaluate is not None else None,
                       "predictions_used_for_reference_selection":False})
            record={"case":case["id"],"title":case["title"],"observation_id":entry["id"],
                    "datetime":entry["acquired_at"],"state":"scored" if evaluate is not None and evaluate.any() else "reference_unavailable",
                    "reference_state":ref["state"],"reference_difference_hours":ref.get("date_difference_hours"),
                    "minimum_distance_to_development_cells_m":distance,"models":{},
                    "coverage":{"requested_ha":o.metadata["requested_aoi_area_ha"],
                                "observed_ha":float((o.weights[o.valid]*.04).sum(dtype=np.float64))}}
            record["reference_reason"] = (ref.get("reason") if ref["state"]!="ready" else
                                          "no_retained_clear_reference_cells" if evaluate is not None and not evaluate.any() else None)
            for name,model in (("previous_v1",old),("spatial_v2",new)):
                labels,scores,info=model.predict(o)
                np.savez_compressed(image_dir/(name+".npz"),labels=labels,scores=scores)
                if evaluate is not None and evaluate.any():
                    if not (labels[evaluate]<=1).all():
                        raise ValueError("Eligible reference cells were withheld")
                    areas=o.weights[evaluate].astype(np.float64)*.04
                    measure=confusion_metrics(classes[evaluate]==6,labels[evaluate]==1,areas)
                    measure["water_cells"]=int((classes[evaluate]==6).sum())
                    measure["nonwater_cells"]=int((classes[evaluate]!=6).sum())
                else:
                    measure=None
                record["models"][name]={"metrics":measure,"model_sha256":model.fingerprint,"decision_policy":info["decision_policy"]}
            write_json(image_dir/"metrics.json",record)
            rows.append(record)
            print("v2:",record["models"]["spatial_v2"]["metrics"],flush=True)
        destination=output/"series"/case["id"]
        if not (destination/"series.json").exists():
            run_series((load_catalog_observation(e) for e in entries),new,destination,series_id=case["id"])
        write_json(output/"report.json",{"created_at":utcnow(),"rows":rows,"model_sha256":new.fingerprint,
                   "test_used_for_model_selection":False,"label_kind":"dated_machine_optical_weak_labels"})
    aggregated={}
    for name in ("previous_v1","spatial_v2"):
        valid=[r["models"][name]["metrics"] for r in rows if r.get("models",{}).get(name,{}).get("metrics")]
        tp,fp,fn,tn=[sum(m[k] for m in valid) for k in ("true_water_ha","false_water_ha","missed_water_ha","true_nonwater_ha")]
        aggregated[name]={"precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,
                          "iou":tp/(tp+fp+fn) if tp+fp+fn else None,"true_water_ha":tp,"false_water_ha":fp,
                          "missed_water_ha":fn,"true_nonwater_ha":tn,"scored_images":len(valid)}
    report={"created_at":utcnow(),"rows":rows,"aggregate":aggregated,"model_sha256":new.fingerprint,
            "model_unchanged":sha256(experiment/"selected/model.json")==freeze["model_sha256"],
            "test_used_for_model_selection":False,"label_kind":"dated_machine_optical_weak_labels",
            "limitations":["Four new regions and up to two dates each; spatial/temporal cells correlated",
                           "Aggregated scores are area-weighted across retained support, not an estimate of worldwide accuracy",
                           "SCL weak interiors do not validate shorelines; cloud/unknown/boundary cells are withheld"]}
    write_json(output/"report.json",report)
    measured=[r["models"]["spatial_v2"]["metrics"] for r in rows
              if r.get("models",{}).get("spatial_v2",{}).get("metrics")]
    failures=any((m["recall"] is not None and m["recall"]<.7)
                 or (m["precision"] is not None and m["precision"]<.5)
                 or (m["false_positive_rate"] is not None and m["false_positive_rate"]>.04) for m in measured)
    write_json(output/"assessment.json",{
        "state":"not_ready_for_general_use" if failures else "global_validation_not_established",
        "model_sha256":new.fingerprint,"new_radar_images":sum(bool(r.get("models")) for r in rows),
        "new_regions":len({r["case"] for r in rows}),"quantitatively_scored_images":len(measured),
        "source_report_sha256":sha256(output/"report.json"),"model_changed_or_refitted":False,
        "label_kind":"dated_machine_optical_weak_labels",
        "reason":"At least one scored image fails the declared operating targets; inspect per-location results and unavailable references."
                  if failures else "Available weak-reference evidence does not establish global accuracy."})
    flat=[]
    for row in rows:
        for name,value in row.get("models",{}).items():
            flat.append({"case":row["case"],"datetime":row["datetime"],"model":name,"reference_state":row["reference_state"],
                         **(value["metrics"] or {})})
    if flat:
        fields=list(dict.fromkeys(k for item in flat for k in item))
        with (output/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    return report


"""Single frozen-model evaluation on previously reserved Red Rock geography."""
from datetime import datetime
from pathlib import Path
import hashlib
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from shapely.geometry import shape

from .common import read_json, write_json, sha256, utcnow
from .development import TEST_ID
from .features import tiles
from .inputs import load_catalog_observation
from .model import WaterModel
from .pipeline import run_series
from .training import metrics


def prepare_holdout(source, experiment, output):
    source, experiment, output = Path(source), Path(experiment), Path(output)
    if output.exists():
        raise FileExistsError("Heldout output already exists; preserve the one-time evaluation")
    output.mkdir(parents=True)
    freeze = read_json(experiment / "freeze.json")
    model_path = experiment / "selected/model.json"
    if sha256(model_path) != freeze["model_sha256"] or sha256(experiment / "selected/estimator.joblib") != freeze["estimator_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    if sha256(experiment / "comparison.json") != freeze["selection_sha256"]:
        raise ValueError("Frozen development selection changed")
    allocation_path = source / "evidence/new-mixed-test-allocation-v1.json"
    allocation = read_json(allocation_path)
    if allocation["partition"] != "test" or allocation["model_scores_used"] is not False:
        raise ValueError("Holdout was not allocated independently")
    catalog = read_json(source / "data/catalog.json")["observations"]
    entry = next(e for e in catalog if e["id"] == TEST_ID)
    if not shape(entry["aoi"]).equals(shape(allocation["aoi"])):
        raise ValueError("Test AOI differs from historical allocation")
    o = load_catalog_observation(entry)
    ref_dir = source / "data/labels/water-correction-v4/red-rock-reference"
    reference = read_json(ref_dir / "reference.json")
    mismatch = abs((datetime.fromisoformat(reference["reference_date"].replace("Z", "+00:00")) -
                    datetime.fromisoformat(entry["acquired_at"].replace("Z", "+00:00"))).total_seconds())/3600
    if mismatch > 24:
        raise ValueError("Water reference exceeds the 24 hour date limit")
    scl = np.zeros(o.valid.shape, np.uint8)
    with rasterio.open(ref_dir / "scl.tif") as src:
        reproject(src.read(1), scl, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=o.transform, dst_crs=o.crs, resampling=Resampling.nearest, dst_nodata=0)
    source_class = np.zeros(o.valid.shape, np.uint8)
    for code in (4, 5, 6):
        source_class[binary_erosion(scl == code, iterations=2) & (o.weights >= 1-1e-6)] = code
    supported = np.zeros(o.valid.shape, bool)
    for sl, _, eligible in tiles(o.power, o.valid):
        supported[sl] = eligible
    evaluated = (source_class > 0) & o.valid & supported & ~o.exclusions
    if not (source_class[evaluated] == 6).any() or not np.isin(source_class[evaluated], [4, 5]).any():
        raise ValueError("Holdout lacks retained water and explicit nonwater")
    rr, cc = np.where(evaluated)
    xy = np.column_stack(rasterio.transform.xy(o.transform, rr, cc))
    minimums = {}
    for split in ("train", "validation"):
        with np.load(experiment / "development" / (split+".npz"), allow_pickle=False) as z:
            if TEST_ID in z["observation_ids"]:
                raise ValueError("Holdout acquisition leaked into development")
            minimum = float("inf")
            for epsg in np.unique(z["epsg"]):
                points = z["xy"][z["epsg"] == epsg]
                tx, ty = Transformer.from_crs(int(epsg), o.crs, always_xy=True).transform(points[:, 0], points[:, 1])
                minimum = min(minimum, float(cKDTree(np.column_stack((tx, ty))).query(xy, k=1)[0].min()))
            if minimum < 140:
                raise ValueError("Geographic train/validation-to-test buffer violation")
            minimums[split] = minimum
    groups = np.array([f"{int(x//600)}:{int(y//600)}" for x, y in xy])
    truth = (source_class[evaluated] == 6).astype(np.uint8)
    areas = o.weights[evaluated].astype(np.float64)*.04
    np.savez_compressed(output / "reference-support.npz", raw_scl=scl, source_class=source_class,
                        evaluated=evaluated, y=truth, areas_ha=areas, groups=groups,
                        transform=np.array(list(o.transform)[:6]), xy=xy)
    area = o.weights.astype(np.float64)*.04
    receipt = {"support_frozen_at": utcnow(), "selection_frozen_at": freeze["frozen_at"],
               "model_sha256": freeze["model_sha256"], "checkpoint_sha256": freeze["estimator_sha256"],
               "allocation_sha256": sha256(allocation_path), "allocation_created_at": allocation["created_at"],
               "observation_id": TEST_ID, "granule_id": entry["granule_id"], "acquired_at": entry["acquired_at"],
               "reference": reference, "reference_difference_hours": mismatch,
               "scl_sha256": sha256(ref_dir / "scl.tif"), "reference_support_sha256": sha256(output / "reference-support.npz"),
               "minimum_distance_to_development_cells_m": minimums,
               "water_cells": int(truth.sum()), "nonwater_cells": int((truth == 0).sum()),
               "geographic_600m_blocks": len(set(groups)),
               "coverage": {"requestedHa": o.metadata["requested_aoi_area_ha"],
                            "observedHa": float(area[o.valid].sum()),
                            "featureEligibleHa": float(area[o.valid & supported].sum()),
                            "evaluatedHa": float(areas.sum()), "referenceWaterHa": float(areas[truth == 1].sum())},
               "policy": "SCL4/5 nonwater, SCL6 water; 40 m interiors, full 20 m AOI cells, radar feature support; clouds/shadows/unknown never negatives",
               "label_kind": "machine_weak", "human_reviewed": False,
               "test_predictions_inspected_before_support_freeze": False,
               "source_assets": entry["assets"]}
    write_json(output / "support-freeze.json", receipt)
    return o, evaluated, truth, areas, groups, receipt


def evaluate_unseen(source, experiment, baseline, output):
    source, experiment, baseline, output = map(Path, (source, experiment, baseline, output))
    o, evaluated, truth, areas, groups, receipt = prepare_holdout(source, experiment, output)
    report = {"kind": "previously_unseen_geography", "label_kind": "machine_weak",
              "observation_id": TEST_ID, "support": receipt, "models": {},
              "test_used_for_selection_or_training": False,
              "scope": "one new geographic area and one acquisition; dated machine optical references; not global/field/shoreline validation"}
    saved = {}
    for name, directory in (("previous_v1", baseline), ("improved_v2", experiment / "selected")):
        model = WaterModel(directory)
        labels, scores, info = model.predict(o)
        if not (labels[evaluated] <= 1).all():
            raise ValueError("Model withheld an eligible evaluation cell")
        pred = labels[evaluated]
        m = metrics(np.where(truth == 1, 1, 2), pred, .5, areas)
        blocks = {str(group): metrics(np.where(truth[take] == 1, 1, 2), pred[take], .5, areas[take])
                  for group in np.unique(groups) for take in [groups == group]}
        report["models"][name] = {"model_sha256": model.fingerprint, "decision_policy": info["decision_policy"],
                                  "metrics": m, "blocks": blocks,
                                  "passes_overall_targets_here": m["precision"] >= .5 and (m["recall"] or 0) >= .7 and (m["false_positive_rate"] or 0) <= .04}
        np.savez_compressed(output / (name+"-prediction.npz"), labels=labels, scores=scores)
        run_series([o], model, output / name, series_id="red-rock-"+name)
        saved[name] = labels
        print(name, m, flush=True)
    report["model_frozen_before_test"] = True
    report["model_sha256_after_evaluation"] = sha256(experiment / "selected/model.json")
    if report["model_sha256_after_evaluation"] != receipt["model_sha256"]:
        raise ValueError("Selected model changed during evaluation")
    write_json(output / "unseen-evaluation.json", report)
    return report


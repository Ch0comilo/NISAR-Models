"""Reproducible fixed-budget comparison using existing geographic development labels.

Only train and validation arrays are accessed. Old exposed tests and the reserved
Lake Red Rock geography remain untouched. Scores are weak-reference agreement.
"""
from __future__ import annotations
import time
from pathlib import Path
import joblib
import numpy as np
from rasterio.transform import rowcol
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .common import read_json, write_json, sha256, utcnow
from .features import FEATURE_VERSION, FEATURE_NAMES, tiles
from .inputs import load_catalog_observation

POLICY = {"minimum_precision": .5, "minimum_recall": .7,
          "maximum_false_positive_rate_each_observation": .04,
          "thresholds": [round(float(x), 2) for x in np.arange(.05, 1, .05)] + [.97, .99],
          "choice": "maximum F1 among feasible thresholds; otherwise diagnostic maximum F1, marked experimental",
          "fitting": "two fixed HGB fits (local 11 features, multiscale 41 features), 250 iterations, 15 leaves; no test arrays loaded"}


def metrics(y, score, threshold, areas):
    pred = score >= threshold
    truth = y == 1
    tp, fp = float(areas[truth & pred].sum()), float(areas[~truth & pred].sum())
    fn, tn = float(areas[truth & ~pred].sum()), float(areas[~truth & ~pred].sum())
    p, r = tp/(tp+fp) if tp+fp else 0., tp/(tp+fn) if tp+fn else None
    return {"precision": p, "recall": r, "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "iou": tp/(tp+fp+fn) if tp+fp+fn else None,
            "false_positive_rate": fp/(fp+tn) if fp+tn else None,
            "true_water_ha": tp, "false_water_ha": fp, "missed_water_ha": fn,
            "nonwater_ha": tn+fp, "water_ha": tp+fn}


def train_from_existing(source_root, output_dir):
    source_root, output_dir = Path(source_root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Training output already exists; use a new version")
    output_dir.mkdir(parents=True)
    source = source_root / "data/labels/water-correction-v9/samples.npz"
    catalog_path = source_root / "data/catalog.json"
    source_manifest = read_json(source.with_name("dataset.json"))
    if sha256(source) != source_manifest["samples_sha256"]:
        raise ValueError("Existing label samples failed their dataset manifest hash")
    write_json(output_dir / "plan.json", {"created_at": utcnow(), "policy": POLICY,
               "samples_sha256": sha256(source), "catalog_sha256": sha256(catalog_path),
               "labels": "existing dated Dynamic World/Sentinel-2 SCL weak labels; no manual/field truth",
               "test_policy": "no test arrays or independent reserved geography accessed"})
    catalog = {o["id"]: o for o in read_json(catalog_path)["observations"]}
    parts = {}
    with np.load(source, allow_pickle=False) as z:
        for split in ("train", "validation"):
            parts[split] = {key: z[split+"__"+key] for key in
                            ("X", "y", "areas_ha", "observation_ids", "xy", "epsg", "groups")}
            parts[split]["context"] = np.full((len(parts[split]["y"]), len(FEATURE_NAMES)), np.nan, np.float32)
    if set(parts["train"]["groups"]) & set(parts["validation"]["groups"]):
        raise ValueError("Train/validation geographic groups overlap")
    from scipy.spatial import cKDTree
    separation = {}
    for epsg in set(parts["train"]["epsg"]) & set(parts["validation"]["epsg"]):
        train_xy = parts["train"]["xy"][parts["train"]["epsg"] == epsg]
        val_xy = parts["validation"]["xy"][parts["validation"]["epsg"] == epsg]
        distance = float(cKDTree(train_xy).query(val_xy, k=1)[0].min())
        if distance < 140-1e-6:
            raise ValueError("Train/validation retained-cell distance is less than 140 m")
        separation[str(epsg)] = distance
    write_json(output_dir / "geographic-audit.json", {"minimum_retained_cell_distance_m_by_epsg": separation,
                                                     "test_arrays_accessed": False})
    wanted = sorted(set(np.concatenate([p["observation_ids"] for p in parts.values()])))
    for oid in wanted:
        print("Extracting spatial context:", oid, flush=True)
        observation = load_catalog_observation(catalog[oid], catalog_path.parent)
        sample_positions = {}
        for split, p in parts.items():
            index = np.flatnonzero(p["observation_ids"] == oid)
            if index.size == 0:
                continue
            if not np.all(p["epsg"][index] == observation.crs.to_epsg()):
                raise ValueError("Sample CRS mismatch")
            rr, cc = rowcol(observation.transform, *p["xy"][index].T)
            rr, cc = np.asarray(rr), np.asarray(cc)
            if (rr < 0).any() or (cc < 0).any() or (rr >= observation.valid.shape[0]).any() or (cc >= observation.valid.shape[1]).any():
                raise ValueError("Training cell falls outside its source")
            sample_positions[split] = (index, rr, cc)
        for sl, features, eligible in tiles(observation.power, observation.valid):
            for split, (index, rr, cc) in sample_positions.items():
                inside = (rr >= sl[0].start) & (rr < sl[0].stop) & (cc >= sl[1].start) & (cc < sl[1].stop)
                if inside.any():
                    parts[split]["context"][index[inside]] = features[rr[inside]-sl[0].start, cc[inside]-sl[1].start]
    for split, p in parts.items():
        # This prevents a silent half-pixel/channel/unit mismatch with historic labels.
        if not np.isfinite(p["context"]).all():
            raise ValueError(f"Unsupported training cells in {split}; no silent exclusion allowed")
        if not np.allclose(p["context"][:, :3], p["X"][:, :3], atol=3e-4, rtol=1e-5):
            delta = np.max(np.abs(p["context"][:, :3] - p["X"][:, :3]))
            raise ValueError(f"Historic/context input parity failed ({delta})")
    train, val = parts["train"], parts["validation"]
    all_results, models = {}, {}
    for name, feature in (("local-hgb", "X"), ("context-hgb", "context")):
        print("Fitting", name, flush=True)
        estimator = HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=15, max_depth=None,
                      learning_rate=.07, min_samples_leaf=40, l2_regularization=5.,
                      early_stopping=False, random_state=20261028)
        start = time.perf_counter()
        with threadpool_limits(limits=4):
            estimator.fit(train[feature], (train["y"] == 1).astype(np.uint8))
            scores = estimator.predict_proba(val[feature])[:, 1]
        reports = []
        for threshold in POLICY["thresholds"]:
            overall = metrics(val["y"], scores, threshold, val["areas_ha"])
            regions = {str(oid): metrics(val["y"][m], scores[m], threshold, val["areas_ha"][m])
                       for oid in np.unique(val["observation_ids"])
                       for m in [val["observation_ids"] == oid]}
            feasible = (overall["precision"] >= POLICY["minimum_precision"]
                        and (overall["recall"] or 0) >= POLICY["minimum_recall"]
                        and all((r["false_positive_rate"] or 0) <= POLICY["maximum_false_positive_rate_each_observation"] for r in regions.values()))
            reports.append({"threshold": threshold, "feasible": feasible, "overall": overall, "regions": regions})
        feasible = [r for r in reports if r["feasible"]]
        chosen = max(feasible or reports, key=lambda r: (r["overall"]["f1"], r["overall"]["recall"] or 0))
        all_results[name] = {"seconds": time.perf_counter()-start, "selected": chosen,
                             "acceptance_passed": bool(feasible), "threshold_reports": reports}
        models[name] = estimator
        print(name, chosen["threshold"], chosen["overall"], "accepted:", bool(feasible), flush=True)
    # Keep the new spatial profile fixed, even if local HGB wins; report the comparison.
    selected = all_results["context-hgb"]
    model_path = output_dir / "estimator.joblib"
    joblib.dump(models["context-hgb"], model_path, compress=3)
    metadata = {"model_id": "nisar-water-context-hgb-v1", "feature_version": FEATURE_VERSION,
                "features": FEATURE_NAMES, "threshold": selected["selected"]["threshold"],
                "estimator_sha256": sha256(model_path), "created_at": utcnow(),
                "evaluation_status": "local_weak_validation_passed" if selected["acceptance_passed"] else "experimental_acceptance_failed",
                "acceptance_passed": selected["acceptance_passed"],
                "source_samples_sha256": sha256(source),
                "training_counts": {"water": int((train["y"] == 1).sum()), "nonwater": int((train["y"] != 1).sum())},
                "validation_counts": {"water": int((val["y"] == 1).sum()), "nonwater": int((val["y"] != 1).sum())},
                "validation": selected["selected"],
                "scope": "development geographic blocks in known regions; machine weak labels; no unseen-region, independent test or shoreline accuracy claim"}
    write_json(output_dir / "model.json", metadata)
    write_json(output_dir / "comparison.json", {"policy": POLICY, "models": all_results,
                                                "test_evaluated": False, "global_validated": False})
    return metadata


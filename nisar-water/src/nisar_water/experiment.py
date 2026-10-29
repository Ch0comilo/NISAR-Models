"""One additional fit and a declared spatial decision comparison, using development only."""
from pathlib import Path
import shutil
import time
import joblib
import numpy as np
from rasterio.transform import rowcol
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .common import read_json, write_json, sha256, utcnow, identity
from .development import load_development, TEST_ID
from .features import FEATURE_NAMES, FEATURE_VERSION
from .inputs import load_catalog_observation
from .model import WaterModel
from .spatial import water_decision
from .training import metrics


def run_experiment(source, baseline, output):
    source, baseline, output = Path(source).resolve(), Path(baseline).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Experiment output exists; preserve it and choose a new path")
    output.mkdir(parents=True)
    policies = [{"kind": "threshold", "threshold": round(float(t), 2)} for t in np.arange(.05, 1, .05)]
    policies += [{"kind": "region_growing", "seed_threshold": seed, "grow_threshold": grow,
                  "minimum_seed_pixels": minimum}
                 for seed in (.5, .7, .9) for grow in (.03, .1, .2) for minimum in (1, 4)]
    parameters = {"max_iter": 400, "max_leaf_nodes": 31, "learning_rate": .05,
                  "min_samples_leaf": 40, "l2_regularization": 10.,
                  "early_stopping": False, "random_state": 20261028}
    plan = {"created_at": utcnow(), "test_id": TEST_ID, "reserved_test_pixels_opened": False,
            "baseline_model_sha256": sha256(baseline / "model.json"),
            "additional_fits": 1, "fit_parameters": parameters, "water_training_weight": 2.,
            "decision_policies": policies, "selection": "Maximum development F1 among feasible candidates; otherwise maximum F1 diagnostic",
            "acceptance": {"overall_precision_min": .5, "overall_recall_min": .7, "nonwater_fpr_max_each_validation_observation": .04},
            "heldout_policy": "Freeze selected model and policy before reading Red Rock pixels/reference; evaluate once without test-based refitting"}
    write_json(output / "plan.json", plan)
    parts, catalog = load_development(source, output / "development")
    train, val = parts["train"], parts["validation"]
    old_model = WaterModel(baseline)
    estimator = HistGradientBoostingClassifier(**parameters)
    start = time.perf_counter()
    with threadpool_limits(limits=4):
        estimator.fit(train["context"], (train["y"] == 1).astype(np.uint8),
                      sample_weight=np.where(train["y"] == 1, 2., 1.))
    print("Additional fit seconds:", time.perf_counter()-start, flush=True)
    candidates = {"context-v1": old_model.estimator, "context-v2-weighted": estimator}
    reports = []
    for name, fitted in candidates.items():
        temporary = output / (name+"-inference")
        temporary.mkdir()
        joblib.dump(fitted, temporary / "estimator.joblib", compress=3)
        meta = {**old_model.metadata, "model_id": name,
                "estimator_sha256": sha256(temporary / "estimator.joblib"), "decision_policy": {"kind": "threshold", "threshold": .7}}
        write_json(temporary / "model.json", meta)
        model = WaterModel(temporary)
        predictions = [np.empty(len(val["y"]), np.uint8) for _ in policies]
        for oid in np.unique(val["observation_ids"]):
            print("Development spatial decisions:", name, oid, flush=True)
            o = load_catalog_observation(catalog[oid])
            labels, scores, _ = model.predict(o)
            eligible = labels <= 1
            indices = np.flatnonzero(val["observation_ids"] == oid)
            rr, cc = map(np.asarray, rowcol(o.transform, *val["xy"][indices].T))
            if not eligible[rr, cc].all():
                raise ValueError("A validation cell was withheld; no silent sample exclusions")
            for i, policy in enumerate(policies):
                water = water_decision(scores, eligible, policy)
                predictions[i][indices] = water[rr, cc]
        for policy, pred in zip(policies, predictions):
            overall = metrics(val["y"], pred, .5, val["areas_ha"])
            regions = {str(oid): metrics(val["y"][m], pred[m], .5, val["areas_ha"][m])
                       for oid in np.unique(val["observation_ids"]) for m in [val["observation_ids"] == oid]}
            feasible = (overall["precision"] >= .5 and (overall["recall"] or 0) >= .7
                        and all((r["false_positive_rate"] or 0) <= .04 for r in regions.values()))
            reports.append({"classifier": name, "policy": policy, "feasible": feasible,
                            "overall": overall, "regions": regions})
    feasible = [r for r in reports if r["feasible"]]
    selected = max(feasible or reports, key=lambda r: (r["overall"]["f1"], r["overall"]["recall"] or 0))
    final = output / "selected"
    final.mkdir()
    origin = output / (selected["classifier"]+"-inference")
    shutil.copyfile(origin / "estimator.joblib", final / "estimator.joblib")
    threshold = selected["policy"].get("threshold", selected["policy"].get("seed_threshold"))
    metadata = {**old_model.metadata, "model_id": "nisar-water-spatial-v2", "created_at": utcnow(),
                "estimator_sha256": sha256(final / "estimator.joblib"), "threshold": threshold,
                "decision_policy": selected["policy"], "classifier_origin": selected["classifier"],
                "acceptance_passed": selected["feasible"],
                "evaluation_status": "local_weak_validation_passed" if selected["feasible"] else "experimental_acceptance_failed",
                "validation": selected, "experiment_plan_sha256": sha256(output / "plan.json")}
    write_json(final / "model.json", metadata)
    write_json(output / "comparison.json", {"selected": selected, "candidates": reports,
               "test_pixels_accessed": False, "new_test_evaluated": False})
    freeze = {"frozen_at": utcnow(), "model_sha256": sha256(final / "model.json"),
              "estimator_sha256": sha256(final / "estimator.joblib"), "selection_sha256": sha256(output / "comparison.json"),
              "plan_sha256": sha256(output / "plan.json"), "source_hashes": {p.name: sha256(p) for p in
              [Path(__file__), Path(__file__).with_name("spatial.py"), Path(__file__).with_name("model.py"), Path(__file__).with_name("features.py")]},
              "test_id": TEST_ID, "test_pixels_accessed_before_freeze": False,
              "development_counts": {k: len(p["y"]) for k, p in parts.items()}}
    write_json(output / "freeze.json", freeze)
    print("FROZEN SELECTED:", selected, flush=True)
    return final


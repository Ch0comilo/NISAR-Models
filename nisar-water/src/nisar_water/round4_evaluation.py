"""Round-four validation-only selection and explicitly released geographic tests.

The manifest alone assigns partitions. Full weak-reference interiors remain the
same for every model; unsupported predictions remain withheld and are reported.
Historical-water strata are diagnostics, never eligibility gates for radar-only
or terrain-only models.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np
from affine import Affine

from .common import read_json, write_json, sha256, utcnow, identity
from .inputs import Observation
from .benchmark import confusion_metrics

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("radar", "terrain", "terrain_prior")
BASELINE = "old_round3_hgb"
THRESHOLDS = tuple(i / 10 for i in range(1, 10))
MIN_CLASS_HA = 4.0
AREA_KEYS = ("true_water_ha", "false_water_ha", "missed_water_ha", "true_nonwater_ha")
CONTEXT_KEYS = ("context_path", "context_sha256", "context_metadata_path", "context_metadata_sha256")


def class_supported(area):
    """Include exactly 100 full 20 m cells despite floating-point summation."""
    return area >= MIN_CLASS_HA - 1e-9


def load_model(directory):
    """Keep the new evaluator independent of changes to the shared registry."""
    directory = Path(directory)
    if read_json(directory / "model.json").get("model_type") == "context_hgb":
        from .context_model import ContextWaterModel
        return ContextWaterModel(directory)
    from .registry import load_model as registry_load_model
    return registry_load_model(directory)


def load_scene(spec):
    path = Path(spec["path"])
    if sha256(path) != spec["sha256"]:
        raise ValueError("Scene hash mismatch: " + spec["id"])
    sidecar = read_json(path.with_suffix(".json"))
    metadata = dict(sidecar["source_observation"])
    metadata["source_observation_id"] = metadata.get("id")
    metadata.update(id=spec["id"], scene_sha256=spec["sha256"], data_origin="observed", is_demo=False)
    for key in CONTEXT_KEYS:
        if key in spec:
            metadata[key] = spec[key]
    with np.load(path, allow_pickle=False) as z:
        observation = Observation(z["power"], z["valid"], Affine(*z["transform"]), int(z["epsg"]), metadata, z["aoi_weights"])
        labels, evaluate = z["labels"].copy(), z["evaluate"].astype(bool)
    if labels.shape != observation.valid.shape or evaluate.shape != labels.shape:
        raise ValueError("Scene reference shapes differ")
    if not np.isin(labels, [0, 1, 255]).all():
        raise ValueError("Unsupported reference label codes")
    return observation, labels, evaluate, sidecar


def historical_occurrence(spec, observation):
    """Read bound context only; missing prior does not withhold classification."""
    path = spec.get("context_path")
    if not path:
        return np.full(observation.valid.shape, np.nan, np.float32), "context_not_supplied"
    if not spec.get("context_sha256") or sha256(path) != spec["context_sha256"]:
        raise ValueError("Context sidecar hash mismatch")
    metadata_path = spec.get("context_metadata_path")
    if metadata_path and sha256(metadata_path) != spec.get("context_metadata_sha256"):
        raise ValueError("Context metadata hash mismatch")
    with np.load(path, allow_pickle=False) as z:
        names = [str(x) for x in z["names"].tolist()]
        if "jrc2021_occurrence_pct" not in names:
            return np.full(observation.valid.shape, np.nan, np.float32), "occurrence_not_supplied"
        if "scene_sha256" in z and str(z["scene_sha256"].item()) != spec["sha256"]:
            raise ValueError("Context is bound to a different scene")
        if "transform6" in z and not np.allclose(z["transform6"], list(observation.transform)[:6], rtol=0, atol=1e-8):
            raise ValueError("Context transform mismatch")
        if "epsg" in z and int(z["epsg"]) != observation.crs.to_epsg():
            raise ValueError("Context CRS mismatch")
        plane = np.asarray(z["features"][names.index("jrc2021_occurrence_pct")], np.float32).copy()
        available = z["prior_occurrence_available"].astype(bool) if "prior_occurrence_available" in z else np.isfinite(plane)
    if plane.shape != observation.valid.shape or available.shape != plane.shape:
        raise ValueError("Historical occurrence shape mismatch")
    plane[~available | ~np.isfinite(plane) | (plane < 0) | (plane > 100)] = np.nan
    return plane, "available"


def prediction(spec, model, cache):
    observation, truth, evaluate, sidecar = load_scene(spec)
    cache = Path(cache) / spec["id"]
    cache.mkdir(parents=True, exist_ok=True)
    receipt, array = cache / "prediction.json", cache / "prediction.npz"
    fingerprint = identity({"scene": spec["sha256"], "context": spec.get("context_sha256"),
                            "context_metadata": spec.get("context_metadata_sha256"), "model": model.fingerprint})
    if receipt.exists():
        record = read_json(receipt)
        if record["fingerprint"] != fingerprint or sha256(array) != record["sha256"]:
            raise ValueError("Prediction cache integrity mismatch")
        with np.load(array, allow_pickle=False) as z:
            labels, scores = z["labels"].copy(), z["scores"].copy()
    else:
        print("ROUND4 PREDICT", spec["split"], model.model_id, spec["id"], flush=True)
        labels, scores, info = model.predict(observation)
        labels, scores = np.asarray(labels), np.asarray(scores)
        if labels.shape != observation.valid.shape or scores.shape != labels.shape:
            raise ValueError("Prediction grid differs from input")
        if not np.isin(labels, [0, 1, 254, 255]).all():
            raise ValueError("Prediction has unsupported label codes")
        if not np.isfinite(scores[labels <= 1]).all():
            raise ValueError("Classified pixels have unavailable scores")
        np.savez_compressed(array, labels=labels.astype(np.uint8), scores=scores.astype(np.float32))
        record = {"fingerprint": fingerprint, "sha256": sha256(array), "model_fingerprint": model.fingerprint,
                  "scene_sha256": spec["sha256"], "context_sha256": spec.get("context_sha256"),
                  "info": info, "created_at": utcnow()}
        write_json(receipt, record)
    occurrence, occurrence_state = historical_occurrence(spec, observation)
    return observation, truth, evaluate, labels, scores, occurrence, occurrence_state, record


def _water_stratum(reference, classified, truth, predicted, area, mask):
    target = reference & (truth == 1) & mask
    covered = target & classified
    total, known = float(area[target].sum()), float(area[covered].sum())
    recovered = float(area[covered & predicted].sum())
    missed = float(area[covered & ~predicted].sum())
    return {"reference_water_ha": total, "classified_water_ha": known,
            "withheld_water_ha": total-known, "recovered_water_ha": recovered, "missed_water_ha": missed,
            "reference_coverage": known/total if total else None,
            "recall_on_classified": recovered/known if known else None,
            "recovered_fraction_of_full_reference": recovered/total if total else None}


def score_scene(spec, observation, truth, evaluate, labels, scores, threshold, occurrence=None, *, boundary=False):
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Threshold must lie strictly between zero and one")
    reference = (truth <= 1) & observation.valid & (observation.weights > 0) & (~evaluate if boundary else evaluate)
    classified = (labels <= 1) & np.isfinite(scores)
    common = reference & classified
    predicted = scores >= threshold
    area = observation.weights.astype(np.float64)*.04
    metrics = confusion_metrics(truth[common] == 1, predicted[common], area[common]) if common.any() else None
    water = float(area[reference & (truth == 1)].sum())
    land = float(area[reference & (truth == 0)].sum())
    known_water = float(area[common & (truth == 1)].sum())
    known_land = float(area[common & (truth == 0)].sum())
    total, known = water+land, known_water+known_land
    if occurrence is None:
        occurrence = np.full(truth.shape, np.nan, np.float32)
    occurrence_known = np.isfinite(occurrence) & (occurrence >= 0) & (occurrence <= 100)
    strata = {"low_historical_occurrence_water": _water_stratum(reference, classified, truth, predicted, area, occurrence_known & (occurrence <= 10)),
              "historical_water": _water_stratum(reference, classified, truth, predicted, area, occurrence_known & (occurrence >= 50)),
              "intermediate_occurrence_water": _water_stratum(reference, classified, truth, predicted, area, occurrence_known & (occurrence > 10) & (occurrence < 50)),
              "prior_unknown_water": _water_stratum(reference, classified, truth, predicted, area, ~occurrence_known)}
    return {"id": spec["id"], "region": spec["region"], "acquired_at": spec["acquired_at"], "split": spec["split"],
            "metrics": metrics, "reference_support_ha": total, "reference_water_ha": water, "reference_land_ha": land,
            "classified_reference_ha": known, "classified_water_reference_ha": known_water,
            "classified_land_reference_ha": known_land, "withheld_reference_ha": total-known,
            "withheld_water_reference_ha": water-known_water, "withheld_land_reference_ha": land-known_land,
            "reference_coverage": known/total if total else None,
            "water_reference_coverage": known_water/water if water else None,
            "land_reference_coverage": known_land/land if land else None,
            "water_reference_cells": int((reference & (truth == 1)).sum()),
            "land_reference_cells": int((reference & (truth == 0)).sum()), "historical_water_diagnostic": strata,
            "diagnostic_caution": "Low historical occurrence is not proof of newly formed water; prior never changes scoring eligibility"}


def _areas_summary(areas):
    tp, fp, fn, tn = [float(areas[k]) for k in AREA_KEYS]
    ratio = lambda n, d: n/d if d else None
    return {**dict(zip(AREA_KEYS, (tp, fp, fn, tn))), "precision": ratio(tp, tp+fp), "recall": ratio(tp, tp+fn),
            "specificity": ratio(tn, tn+fp), "false_positive_rate": ratio(fp, fp+tn),
            "f1": ratio(2*tp, 2*tp+fp+fn), "iou": ratio(tp, tp+fp+fn)}


def _aggregate_strata(rows):
    result = {}
    for name in ("low_historical_occurrence_water", "historical_water", "intermediate_occurrence_water", "prior_unknown_water"):
        keys = ("reference_water_ha", "classified_water_ha", "withheld_water_ha", "recovered_water_ha", "missed_water_ha")
        values = {key: sum(r["historical_water_diagnostic"][name][key] for r in rows) for key in keys}
        total, known, recovered = values["reference_water_ha"], values["classified_water_ha"], values["recovered_water_ha"]
        result[name] = {**values, "reference_coverage": known/total if total else None,
                        "recall_on_classified": recovered/known if known else None,
                        "recovered_fraction_of_full_reference": recovered/total if total else None}
    return result


def aggregate(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["region"], []).append(row)
    regions = {}
    for region, scenes in grouped.items():
        areas = {key: sum((r["metrics"] or {}).get(key, 0) for r in scenes) for key in AREA_KEYS}
        result = _areas_summary(areas)
        keys = ("reference_support_ha", "reference_water_ha", "reference_land_ha", "classified_reference_ha",
                "classified_water_reference_ha", "classified_land_reference_ha", "withheld_reference_ha",
                "withheld_water_reference_ha", "withheld_land_reference_ha")
        result.update({key: sum(r[key] for r in scenes) for key in keys})
        total, water, land = result["reference_support_ha"], result["reference_water_ha"], result["reference_land_ha"]
        result["reference_coverage"] = result["classified_reference_ha"]/total if total else None
        result["water_reference_coverage"] = result["classified_water_reference_ha"]/water if water else None
        result["land_reference_coverage"] = result["classified_land_reference_ha"]/land if land else None
        result["recovered_fraction_of_full_water_reference"] = result["true_water_ha"]/water if water else None
        result['correct_fraction_of_full_land_reference'] = result['true_nonwater_ha']/land if land else None
        supported = []
        if class_supported(water):
            supported.append(result['recovered_fraction_of_full_water_reference'] or 0.)
        if class_supported(land):
            supported.append(result['correct_fraction_of_full_land_reference'] or 0.)
        result["balanced_accuracy_supported_classes"] = float(np.mean(supported)) if supported else None
        result["scenes"] = len(scenes)
        result["historical_water_diagnostic"] = _aggregate_strata(scenes)
        regions[region] = result
    values = [r["balanced_accuracy_supported_classes"] for r in regions.values() if r["balanced_accuracy_supported_classes"] is not None]
    pooled = _areas_summary({key: sum(r[key] for r in regions.values()) for key in AREA_KEYS})
    pooled.update({key: sum(r[key] for r in regions.values()) for key in
                   ("reference_support_ha", "reference_water_ha", "reference_land_ha", "classified_reference_ha", "withheld_reference_ha")})
    total = pooled["reference_support_ha"]
    pooled["reference_coverage"] = pooled["classified_reference_ha"]/total if total else None
    return {"regions": regions, "pooled": pooled,
            "minimum_region_balanced_accuracy": min(values) if values else None,
            "macro_region_balanced_accuracy": float(np.mean(values)) if values else None,
            "historical_water_diagnostic": _aggregate_strata(rows) if rows else {}}


def gate_checks(summary):
    checks = {}
    for region, row in summary["regions"].items():
        values = {"reference_coverage": (row["reference_coverage"] or 0) >= .98}
        if class_supported(row["reference_water_ha"]):
            values["full_reference_water_recovery"] = (row['recovered_fraction_of_full_water_reference'] or 0) >= .90
            values['water_reference_coverage'] = (row['water_reference_coverage'] or 0) >= .98
            values["water_precision"] = (row["precision"] or 0) >= .90
        if class_supported(row["reference_land_ha"]):
            values["land_false_positive_rate"] = row["false_positive_rate"] is not None and row["false_positive_rate"] <= .04
            values['land_reference_coverage'] = (row['land_reference_coverage'] or 0) >= .98
        checks[region] = values
    return checks


def passes(summary):
    checks = gate_checks(summary)
    return bool(checks) and all(all(values.values()) for values in checks.values())


def _rank(report, family_order=FAMILIES):
    worst = report["minimum_region_balanced_accuracy"]
    macro = report["macro_region_balanced_accuracy"]
    return (bool(report["passes_validation_gate"]), -1. if worst is None else worst,
            -1. if macro is None else macro, -family_order.index(report["family"]), report["threshold"])


def validate_and_freeze(manifest_path, output, model_root=None, baseline=None):
    manifest_path, output = Path(manifest_path), Path(output)
    model_root = Path(model_root or ROOT/"models/round4")
    baseline = Path(baseline or ROOT/"outputs/round3/frozen/hgb")
    if (output/"freeze.json").exists():
        raise FileExistsError("Round4 validation is already frozen")
    manifest = read_json(manifest_path)
    specs = [s for s in manifest["scenes"] if s["split"] == "validation"]
    if not specs:
        raise ValueError("No validation scenes supplied")
    output.mkdir(parents=True, exist_ok=True)
    plan = {"created_at": utcnow(), "manifest_sha256": sha256(manifest_path), "families": list(FAMILIES),
            "thresholds": list(THRESHOLDS), "baseline": str(baseline.resolve()), "baseline_reselected": False,
            "gate": {"minimum_class_support_ha": MIN_CLASS_HA, "precision_min_water_regions": .9,
                     "recall_min_water_regions": .9, "nonwater_false_positive_rate_max": .04, "reference_coverage_min": .98,
                     'per_class_reference_coverage_min':.98,'recall_gate_denominator':'full water reference; withheld remains unrecovered'},
            "selection_rule": "Passing gate, then worst-region supported-class balanced accuracy, then macro balanced accuracy",
            "tie_break": "Canonical family order radar/terrain/terrain_prior, then highest threshold",
            "score_support": "Identical full weak interior reference; unknown prediction remains withheld and coverage is reported",
            "historical_water_diagnostic": "Current optical water on valid JRC2021 occurrence<=10 versus>=50; independent of distance validity and classification eligibility",
            "test_opened": False, "expert_reference_available": False, "production_approval": False}
    write_json(output/"selection-plan.json", plan)
    reports, best = [], {}
    for family in FAMILIES:
        model = load_model(model_root/family)
        prepared = [(s, *prediction(s, model, output/"validation-predictions"/family)) for s in specs]
        candidates = []
        for threshold in THRESHOLDS:
            rows = [score_scene(s, o, t, e, l, p, threshold, occurrence) for s, o, t, e, l, p, occurrence, _, _ in prepared]
            report = {"family": family, "threshold": threshold, **aggregate(rows), "scenes": rows, "test_used": False}
            report["passes_validation_gate"] = passes(report)
            report["gate_checks"] = gate_checks(report)
            candidates.append(report)
            reports.append(report)
        best[family] = max(candidates, key=_rank)
        print("ROUND4 VALIDATION", family, best[family]["threshold"], best[family]["minimum_region_balanced_accuracy"], flush=True)
    baseline_model = load_model(baseline)
    baseline_threshold = float(baseline_model.metadata["threshold"])
    baseline_rows = []
    for spec in specs:
        o, t, e, l, p, occurrence, _, _ = prediction(spec, baseline_model, output/"validation-predictions"/BASELINE)
        baseline_rows.append(score_scene(spec, o, t, e, l, p, baseline_threshold, occurrence))
    baseline_report = {"family": BASELINE, "threshold": baseline_threshold, **aggregate(baseline_rows),
                       "scenes": baseline_rows, "test_used": False, "eligible_for_new_family_selection": False}
    baseline_report["passes_validation_gate"] = passes(baseline_report)
    selected = max(best.values(), key=_rank)
    write_json(output/"validation-report.json", {"reports": reports, "family_best": best, "baseline": baseline_report,
               "selected_family": selected["family"], "test_used": False, "expert_reference_available": False})
    snapshots = {}
    sources = {family: model_root/family for family in FAMILIES}
    sources[BASELINE] = baseline
    for family, source in sources.items():
        destination = output/"frozen"/family
        if destination.exists():
            raise FileExistsError("Frozen model directory already exists: "+family)
        destination.mkdir(parents=True)
        metadata = read_json(source/"model.json")
        threshold = baseline_threshold if family == BASELINE else best[family]["threshold"]
        if family != BASELINE:
            metadata["threshold"] = threshold
            metadata["decision_policy"] = {"kind": "threshold", "threshold": threshold}
            metadata["model_id"] += "-round4-validation-frozen"
            metadata["validation_selection"] = {"report_sha256": sha256(output/"validation-report.json"),
                                                "threshold": threshold, "test_used": False, "production_ready": False}
        for name in ("estimator.joblib", "weights.pt"):
            if (source/name).is_file():
                shutil.copyfile(source/name, destination/name)
        write_json(destination/"model.json", metadata)
        snapshots[family] = {"directory": str(destination.resolve()), "model_sha256": sha256(destination/"model.json"),
                             "threshold": threshold, "is_fixed_baseline": family == BASELINE,
                             "artifacts": {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}}
    target = model_root/"selected"
    if target.exists():
        raise FileExistsError("Selected model directory already exists")
    shutil.copytree(output/"frozen"/selected["family"], target)
    code = [Path(__file__), ROOT/"src/nisar_water/context_model.py", ROOT/"src/nisar_water/terrain_context.py",
            ROOT/"src/nisar_water/model.py", ROOT/"src/nisar_water/features.py", ROOT/"src/nisar_water/inputs.py"]
    freeze = {"created_at": utcnow(), "manifest_sha256": sha256(manifest_path),
              "validation_report_sha256": sha256(output/"validation-report.json"),
              "selection_plan_sha256": sha256(output/"selection-plan.json"), "families": snapshots,
              "selected_family": selected["family"], "selected_directory": str(target.resolve()),
              "selected_model_sha256": sha256(target/"model.json"),
              "test_scene_ids": [s["id"] for s in manifest["scenes"] if s["split"] == "test"],
              "test_opened_by_fitting_or_selection": False, "expert_reference_available": False,
              "implementation_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in code if p.is_file()}}
    write_json(output/"freeze.json", freeze)
    return freeze


def check_test_release(manifest_path, output, release_path=None):
    output = Path(output)
    freeze = read_json(output/"freeze.json")
    if sha256(manifest_path) != freeze["manifest_sha256"]:
        raise ValueError("Manifest changed after validation freeze")
    release_path = Path(release_path or output/"test-release.json")
    if not release_path.is_file():
        raise ValueError("New test pixels remain closed until root writes the frozen-test release receipt")
    release = read_json(release_path)
    if release.get("authorized") is not True or release.get("freeze_sha256") != sha256(output/"freeze.json"):
        raise ValueError("Test release is absent or does not match this freeze")
    if release.get("manifest_sha256") != freeze["manifest_sha256"]:
        raise ValueError("Test release manifest differs from the frozen manifest")
    for path, expected in freeze["implementation_hashes"].items():
        if sha256(ROOT/path) != expected:
            raise ValueError("Implementation changed after freeze: "+path)
    for snapshot in freeze["families"].values():
        directory = Path(snapshot["directory"])
        for name, expected in snapshot["artifacts"].items():
            if sha256(directory/name) != expected:
                raise ValueError("Frozen model artifact changed")
    return freeze, release_path


def evaluate_test(manifest_path, output, release_path=None):
    output = Path(output)
    freeze, release_path = check_test_release(manifest_path, output, release_path)
    manifest = read_json(manifest_path)
    specs = [s for s in manifest["scenes"] if s["split"] == "test"]
    if [s["id"] for s in specs] != freeze["test_scene_ids"]:
        raise ValueError("Reserved test scene list changed")
    report = {"created_at": utcnow(), "label_kind": "weak", "split": "unseen_geographic_test",
              "freeze_sha256": sha256(output/"freeze.json"), "test_release_sha256": sha256(release_path),
              "selected_family": freeze["selected_family"], "expert_reference_available": False,
              "families": {}, "test_used_for_selection": False}
    for family, snapshot in freeze["families"].items():
        model = load_model(snapshot["directory"])
        rows, boundaries = [], []
        for spec in specs:
            o, t, e, l, p, occurrence, occurrence_state, _ = prediction(spec, model, output/"test-predictions"/family)
            row = score_scene(spec, o, t, e, l, p, snapshot["threshold"], occurrence)
            row["historical_occurrence_state"] = occurrence_state
            rows.append(row)
            boundaries.append(score_scene(spec, o, t, e, l, p, snapshot["threshold"], occurrence, boundary=True))
        report["families"][family] = {**aggregate(rows), "scenes": rows, "threshold": snapshot["threshold"],
             "changed_after_test": False, "is_fixed_baseline": snapshot["is_fixed_baseline"],
             "weak_boundary_diagnostic": {**aggregate(boundaries), "scenes": boundaries, "is_expert_shoreline_accuracy": False}}
        report["families"][family]["passes_test_gate"] = passes(report["families"][family])
        report["families"][family]["gate_checks"] = gate_checks(report["families"][family])
        write_json(output/"test-report.json", report)
    report["selected_passes_weak_reference_gate"] = passes(report["families"][freeze["selected_family"]])
    report["readiness"] = "research_prototype_requires_independent_expert_and_failure_mode_validation"
    write_json(output/"test-report.json", report)
    return report


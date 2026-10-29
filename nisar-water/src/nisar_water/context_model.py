"""Matched-sample HGB ablations with physical terrain and optional historical water.

The historical water layers are learned covariates, never a hard water/nonwater
mask. Missing required physical context is explicitly withheld at inference.
Only training scene pixels are opened by the fitting entry point.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .common import read_json, write_json, sha256, utcnow
from .features import FEATURE_NAMES, FEATURE_VERSION, tiles

SCHEMA = "nisar-water-context-hgb-v1"
CONTEXT_VERSION = "nisar-radar-terrain-soft-jrc2021-v2"
TERRAIN_NAMES = ("dem_slope_deg", "dem_relief31_m", "hand_log1p_m", "hand_mean7_m", "hand_min7_m")
PRIOR_NAMES = ("jrc2021_occurrence_pct", "jrc2021_water50_distance_capped1000_m")
ABLATIONS = ("radar", "terrain", "terrain_prior")
PARAMETERS = {"max_iter": 300, "max_leaf_nodes": 31, "learning_rate": .05,
              "min_samples_leaf": 40, "l2_regularization": 10.,
              "early_stopping": False, "random_state": 20261028}


def feature_names(ablation):
    if ablation not in ABLATIONS:
        raise ValueError("Unsupported context ablation")
    names = list(FEATURE_NAMES)
    if ablation != "radar":
        names += list(TERRAIN_NAMES)
    if ablation == "terrain_prior":
        names += list(PRIOR_NAMES)+["prior_available"]
    return names


def _path(value, base):
    path = Path(value)
    return path if path.is_absolute() else Path(base)/path


def _verify_asset(path, expected, description):
    if not expected or len(str(expected)) != 64 or sha256(path) != expected:
        raise ValueError(description+" hash mismatch")


def _verify_source_windows(metadata, *, required_physical=True, require_prior=False):
    """Hashes identify downloaded cropped windows, not full remote global products."""
    sources = metadata.get("sources", {})
    hashes = metadata.get("source_hashes", {})
    receipts = {}
    for key in ("dem", "hand", "reference_water"):
        source = sources.get(key)
        required = (key in ("dem", "hand") and required_physical) or (key == "reference_water" and require_prior)
        if not source:
            if required:
                raise ValueError("Context metadata lacks required cropped source "+key)
            continue
        path = source.get("cropped_window_path")
        expected = source.get("cropped_window_sha256") or hashes.get(key)
        if not path and not expected and not required:
            continue  # Explicitly missing optional/unavailable context, not a zero value.
        if not path or not expected:
            raise ValueError("Context source lacks cropped path/hash "+key)
        _verify_asset(path, expected, "Context cropped source "+key)
        if hashes.get(key) and hashes[key] != expected:
            raise ValueError("Context source hash declarations disagree")
        receipts[key] = {"path": str(Path(path).resolve()), "sha256": expected,
                         "hash_scope": "downloaded_or_reused_cropped_window; full_remote_raster_checksum_unavailable"}
    return receipts


def context_arrays(context, expected_shape):
    """Validate feature semantics while preserving optional prior NaNs."""
    values = np.asarray(context["features"], np.float32)
    names = [str(name) for name in context["names"]]
    available = np.asarray(context["available"], bool)
    if values.ndim != 3 or values.shape[1:] != expected_shape or len(names) != values.shape[0] or available.shape != expected_shape:
        raise ValueError("Context features/availability do not match the radar grid")
    if len(set(names)) != len(names) or any(name not in names for name in TERRAIN_NAMES):
        raise ValueError("Required terrain feature names are missing or duplicated")
    terrain = np.stack([values[names.index(name)] for name in TERRAIN_NAMES], axis=-1)
    available = available & np.isfinite(terrain).all(-1)
    available &= (terrain[..., 0] >= 0) & (terrain[..., 0] <= 90) & (terrain[..., 1:] >= 0).all(-1)
    prior = np.full((*expected_shape, len(PRIOR_NAMES)), np.nan, np.float32)
    for i, name in enumerate(PRIOR_NAMES):
        if name in names:
            prior[..., i] = values[names.index(name)]
    for i, maximum in enumerate((100., 1000.)):
        bad = ~np.isfinite(prior[..., i]) | (prior[..., i] < 0) | (prior[..., i] > maximum)
        prior[..., i][bad] = np.nan
    # A valid occurrence value is retained even if distance is unknown.
    prior_available = np.asarray(context.get("prior_available", np.isfinite(prior).all(-1)), bool)
    if prior_available.shape != expected_shape:
        raise ValueError("Prior availability does not match the grid")
    prior_available &= np.isfinite(prior).all(-1)
    occurrence_available = np.isfinite(prior[..., 0])
    return {"terrain": terrain, "available": available, "prior": prior,
            "prior_available": prior_available, "prior_occurrence_available": occurrence_available,
            "metadata": context.get("metadata", {})}


def load_context_sidecar(binding, shape, transform, epsg, *, base=".", scene_sha256=None):
    """Bind the derived context to the exact prepared image and its native grid."""
    for key in ("context_path", "context_sha256", "context_metadata_path", "context_metadata_sha256"):
        if not binding.get(key):
            raise ValueError("Manifest lacks "+key)
    path = _path(binding["context_path"], base)
    metadata_path = _path(binding["context_metadata_path"], base)
    _verify_asset(path, binding["context_sha256"], "Context sidecar")
    _verify_asset(metadata_path, binding["context_metadata_sha256"], "Context metadata")
    metadata = read_json(metadata_path)
    with np.load(path, allow_pickle=False) as archive:
        if not {"features", "available", "names", "transform", "epsg", "scene_sha256"}.issubset(archive.files):
            raise ValueError("Context sidecar lacks physical grid/scene binding")
        if str(archive["scene_sha256"].item()) != (scene_sha256 or binding.get("scene_sha256")):
            raise ValueError("Context sidecar is bound to a different prepared scene")
        if not np.allclose(archive["transform"], np.asarray(transform), atol=1e-7, rtol=0) or int(archive["epsg"]) != int(epsg):
            raise ValueError("Context sidecar and radar grid differ")
        context = {"features": archive["features"].copy(), "available": archive["available"].copy(),
                   "names": archive["names"].tolist(), "metadata": metadata}
        if "prior_available" in archive:
            context["prior_available"] = archive["prior_available"].copy()
    if metadata.get("target_transform") is not None and not np.allclose(metadata["target_transform"], transform, atol=1e-7, rtol=0):
        raise ValueError("Context metadata target transform differs")
    if metadata.get("target_epsg") is not None and int(metadata["target_epsg"]) != int(epsg):
        raise ValueError("Context metadata target EPSG differs")
    checked = context_arrays(context, shape)
    checked["source_receipts"] = _verify_source_windows(metadata,
        required_physical=bool(checked["available"].any()),
        require_prior=bool(checked["prior_occurrence_available"].any()))
    checked["context_sha256"] = binding["context_sha256"]
    checked["context_metadata_sha256"] = binding["context_metadata_sha256"]
    return checked


def _append_context(radar_features, context, slices, ablation):
    if ablation == "radar":
        return radar_features
    arrays = [radar_features, context["terrain"][slices]]
    if ablation == "terrain_prior":
        arrays += [context["prior"][slices], context["prior_available"][slices][..., None].astype(np.float32)]
    return np.concatenate(arrays, axis=-1)


def _manifest_checks(manifest):
    partitions, ids = defaultdict(set), set()
    for spec in manifest.get("scenes", []):
        for key in ("id", "path", "sha256", "split", "group", "region"):
            if not spec.get(key):
                raise ValueError("Scene manifest lacks "+key)
        if spec["id"] in ids:
            raise ValueError("Duplicate scene ID")
        ids.add(spec["id"])
        if spec["split"] not in ("train", "validation", "test"):
            raise ValueError("Invalid split")
        partitions[("group", spec["group"])].add(spec["split"])
        partitions[("region", spec["region"])].add(spec["split"])
    if any(len(splits) > 1 for splits in partitions.values()):
        raise ValueError("A group or entire region occurs in multiple partitions")


def train_context_models(manifest_path, output_dir, *, seed=20261028, per_class_scene_cap=2048,
                         allow_synthetic_fixture=False):
    """Three named fixed fits on identical physically supported training samples."""
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    manifest = read_json(manifest_path)
    if output_dir.exists():
        raise FileExistsError("Preserve old fits; choose a new output directory")
    if manifest.get("data_origin") != "observed" and not (allow_synthetic_fixture and manifest.get("data_origin") == "synthetic_fixture"):
        raise ValueError("Observed training origin required; fixture override is for software tests only")
    if manifest.get("label_kind") not in ("weak", "mixed", "reviewed") or per_class_scene_cap < 1:
        raise ValueError("Declare valid label provenance and a positive sample budget")
    _manifest_checks(manifest)
    output_dir.mkdir(parents=True)
    parameters = {**PARAMETERS, "random_state": seed}
    plan = {"created_at": utcnow(), "manifest_sha256": sha256(manifest_path), "parameters": parameters,
            "families": list(ABLATIONS), "features": {a: feature_names(a) for a in ABLATIONS},
            "per_class_scene_cap": per_class_scene_cap, "seed": seed,
            "sampling": "Common radar+required-physical-context-supported labelled cells across all3 fits; at most2048 per available class per scene by default",
            "weighting": "Retain confidence weights; normalize their total equally by training region",
            "validation_pixels_read": False, "test_paths_opened": False,
            "selection": "No family/threshold/epoch selection here; integrator uses a separate frozen validation protocol",
            "historical_prior": "Soft learned covariates only; missing prior remainsNaN plus an availability indicator",
            "implementation_sha256": sha256(__file__), "radar_features_sha256": sha256(Path(__file__).with_name("features.py"))}
    write_json(output_dir/"plan.json", plan)
    rng = np.random.default_rng(seed)
    parts, audits = [], []
    for spec in manifest["scenes"]:
        if spec["split"] != "train":
            continue  # No validation/test pixel files or sidecars are opened.
        path = _path(spec["path"], manifest_path.parent)
        _verify_asset(path, spec["sha256"], "Prepared training scene")
        with np.load(path, allow_pickle=False) as archive:
            power = archive["power"].astype(np.float32)
            valid = archive["valid"].astype(bool)
            labels = archive["labels"].copy()
            weights = archive["weights"].astype(np.float32) if "weights" in archive else np.ones(valid.shape, np.float32)
            transform, epsg = archive["transform"].copy(), int(archive["epsg"])
        if power.shape != (2, *valid.shape) or labels.shape != valid.shape or weights.shape != valid.shape:
            raise ValueError("Prepared scene arrays differ")
        if not np.isin(labels, [0, 1, 255]).all() or not np.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
            raise ValueError("Invalid label or confidence weights")
        valid &= np.isfinite(power).all(0) & (power > 0).all(0)
        context = load_context_sidecar(spec, valid.shape, transform, epsg, base=manifest_path.parent, scene_sha256=spec["sha256"])
        supported = np.zeros_like(valid)
        for sl, _, block_support in tiles(power, valid):
            supported[sl] = block_support
        labelled = valid & supported & (labels != 255) & (weights > 0)
        eligible = labelled & context["available"]
        selected = []
        for code in (0, 1):
            indices = np.flatnonzero(eligible & (labels == code))
            if len(indices) > per_class_scene_cap:
                indices = rng.choice(indices, per_class_scene_cap, replace=False)
            selected.extend(indices.tolist())
        audit = {"id": spec["id"], "region": spec["region"], "scene_sha256": spec["sha256"],
                 "context_sha256": spec["context_sha256"], "context_metadata_sha256": spec["context_metadata_sha256"],
                 "radar_labelled_cells": int(labelled.sum()), "common_eligible_cells": int(eligible.sum()),
                 "dropped_required_context_cells": int((labelled & ~context["available"]).sum()),
                 "sampled_cells": len(selected), "source_receipts": context["source_receipts"],
                 "prior_occurrence_available_cells": int((eligible & context["prior_occurrence_available"]).sum()),
                 "prior_complete_available_cells": int((eligible & context["prior_available"]).sum())}
        audits.append(audit)
        if not selected:
            print("Context fit skipped unlabelled/unavailable scene", spec["id"], flush=True)
            continue
        selected = np.asarray(selected)
        rr, cc = np.unravel_index(selected, valid.shape)
        radar = np.full((len(selected), len(FEATURE_NAMES)), np.nan, np.float32)
        for sl, features, _ in tiles(power, valid):
            inside = (rr >= sl[0].start) & (rr < sl[0].stop) & (cc >= sl[1].start) & (cc < sl[1].stop)
            radar[inside] = features[rr[inside]-sl[0].start, cc[inside]-sl[1].start]
        terrain = context["terrain"][rr, cc]
        if not np.isfinite(radar).all() or not np.isfinite(terrain).all():
            raise ValueError("A sampled common-support feature is not finite")
        parts.append({"radar": radar, "terrain": terrain, "prior": context["prior"][rr, cc],
                      "prior_available": context["prior_available"][rr, cc, None].astype(np.float32),
                      "y": labels[rr, cc].astype(np.uint8), "weights": weights[rr, cc].astype(np.float64),
                      "region": spec["region"], "id": spec["id"]})
        print("Context matched samples", spec["id"], len(selected), flush=True)
    if not parts:
        raise ValueError("No retained training samples")
    y = np.concatenate([p["y"] for p in parts])
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both explicit water and nonwater training classes are required")
    regional_weight = defaultdict(float)
    for part in parts:
        regional_weight[part["region"]] += float(part["weights"].sum())
    sample_weight = np.concatenate([p["weights"]*len(y)/(len(regional_weight)*regional_weight[p["region"]]) for p in parts])
    audit = {"manifest_sha256": sha256(manifest_path), "scenes": audits, "total_sampled_cells": len(y),
             "training_counts": {"water": int((y == 1).sum()), "nonwater": int((y == 0).sum())},
             "common_samples_identical_across_families": True, "regional_confidence_weight_before_normalization": dict(regional_weight),
             "validation_pixels_read": False, "test_paths_opened": False}
    write_json(output_dir/"training-audit.json", audit)
    outputs = {}
    for ablation in ABLATIONS:
        directory = output_dir/ablation; directory.mkdir()
        columns = [np.concatenate([p["radar"] for p in parts])]
        if ablation != "radar":
            columns.append(np.concatenate([p["terrain"] for p in parts]))
        if ablation == "terrain_prior":
            columns += [np.concatenate([p["prior"] for p in parts]), np.concatenate([p["prior_available"] for p in parts])]
        X = np.concatenate(columns, axis=1)
        model = HistGradientBoostingClassifier(**parameters)
        started = time.perf_counter()
        with threadpool_limits(limits=4):
            model.fit(X, y, sample_weight=sample_weight)
        joblib.dump(model, directory/"estimator.joblib", compress=3)
        metadata = {"schema_version": SCHEMA, "model_type": "context_hgb", "ablation": ablation,
                    "model_id": "nisar-water-"+ablation.replace("_", "-")+"-round4-v1",
                    "feature_version": FEATURE_VERSION, "context_feature_version": CONTEXT_VERSION,
                    "features": feature_names(ablation), "threshold": .5, "decision_policy": {"kind": "threshold", "threshold": .5},
                    "estimator_sha256": sha256(directory/"estimator.joblib"), "created_at": utcnow(),
                    "evaluation_status": "synthetic_fixture" if manifest["data_origin"] == "synthetic_fixture" else "experimental_round4_validation_pending",
                    "training_origin": manifest["data_origin"], "label_kind": manifest["label_kind"],
                    "manifest_sha256": sha256(manifest_path), "training_audit_sha256": sha256(output_dir/"training-audit.json"),
                    "parameters": parameters, "fit_seconds": time.perf_counter()-started,
                    "training_counts": audit["training_counts"], "training_regions": sorted(regional_weight),
                    "required_context": [] if ablation == "radar" else list(TERRAIN_NAMES),
                    "optional_context": list(PRIOR_NAMES) if ablation == "terrain_prior" else [],
                    "validation_pixels_read": False, "test_evaluated": False,
                    "source_hash_scope": "cropped source windows and derived sidecars, not full remote raster checksums",
                    "implementation_sha256": sha256(__file__), "software": {"numpy": np.__version__}}
        write_json(directory/"model.json", metadata)
        outputs[ablation] = str(directory.resolve())
        print("Context HGB saved", ablation, X.shape, flush=True)
    write_json(output_dir/"families.json", {"families": outputs, "manifest_sha256": sha256(manifest_path), "test_opened": False})
    return outputs


class ContextWaterModel:
    def __init__(self, directory, *, context_cache_dir=None):
        self.directory = Path(directory)
        self.metadata = read_json(self.directory/"model.json")
        if self.metadata.get("schema_version") != SCHEMA or self.metadata.get("model_type") != "context_hgb":
            raise ValueError("Unsupported context model schema")
        self.ablation = self.metadata["ablation"]
        if self.metadata.get("feature_version") != FEATURE_VERSION or self.metadata.get("context_feature_version") != CONTEXT_VERSION or self.metadata.get("features") != feature_names(self.ablation):
            raise ValueError("Context model physical feature contract differs")
        path = self.directory/"estimator.joblib"
        _verify_asset(path, self.metadata["estimator_sha256"], "Context checkpoint")
        self.estimator = joblib.load(path)  # Locally fitted trusted artifact only.
        if self.estimator.n_features_in_ != len(feature_names(self.ablation)):
            raise ValueError("Fitted feature count differs from metadata")
        self.model_id, self.threshold = self.metadata["model_id"], float(self.metadata["threshold"])
        self.fingerprint = sha256(self.directory/"model.json")
        self.context_cache_dir = context_cache_dir

    def _context(self, observation):
        binding = observation.metadata
        if binding.get("context_path"):
            return load_context_sidecar(binding, observation.valid.shape, list(observation.transform)[:6], observation.crs.to_epsg(),
                                        scene_sha256=binding.get("scene_sha256"))
        try:
            from .terrain_context import load_context
        except ImportError:
            return {"terrain": np.full((*observation.valid.shape, len(TERRAIN_NAMES)), np.nan, np.float32),
                    "prior": np.full((*observation.valid.shape, len(PRIOR_NAMES)), np.nan, np.float32),
                    "available": np.zeros(observation.valid.shape, bool),
                    "prior_available": np.zeros(observation.valid.shape, bool),
                    "prior_occurrence_available": np.zeros(observation.valid.shape, bool),
                    "metadata": {"state": "context_runtime_unavailable"}}
        # None is a read-only cache lookup. Explicit cache_dir opts into the
        # terrain adapter's bounded acquisition only when configured by a caller.
        result = load_context(observation, cache_dir=self.context_cache_dir, include_prior=self.ablation == "terrain_prior")
        return context_arrays(result, observation.valid.shape)

    def predict(self, observation, *, threshold=None, tile_size=256):
        threshold = self.threshold if threshold is None else float(threshold)
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Threshold must be finite and strictly between zero and one")
        context = None if self.ablation == "radar" else self._context(observation)
        score = np.full(observation.valid.shape, np.nan, np.float32)
        base_support = np.zeros_like(observation.valid)
        support = np.zeros_like(observation.valid)
        water_column = int(np.flatnonzero(self.estimator.classes_ == 1)[0])
        for sl, radar, base_eligible in tiles(observation.power, observation.valid, tile_size=tile_size):
            base_support[sl] = base_eligible
            eligible = base_eligible if context is None else base_eligible & context["available"][sl]
            support[sl] = eligible
            if eligible.any():
                features = _append_context(radar, context, sl, self.ablation)
                values = self.estimator.predict_proba(features[eligible])[:, water_column]
                if not np.isfinite(values).all():
                    raise ValueError("Context HGB produced nonfinite scores")
                block = score[sl]; block[eligible] = values
        observed = observation.valid & (observation.weights > 0)
        base_eligible = observed & base_support & ~observation.exclusions
        classified = observed & support & ~observation.exclusions
        labels = np.full(observation.valid.shape, 255, np.uint8)
        labels[observed] = 254
        labels[classified] = (score[classified] >= threshold).astype(np.uint8)
        score[~classified] = np.nan
        area = observation.weights.astype(np.float64)*.04
        denominator = float(area[base_eligible].sum())
        available = float(area[classified].sum())
        prior_area = (float(area[base_eligible & context["prior_available"]].sum()) if self.ablation == "terrain_prior" else None)
        occurrence_area = (float(area[base_eligible & context["prior_occurrence_available"]].sum()) if self.ablation == "terrain_prior" else None)
        return labels, score, {"threshold": threshold, "threshold_override": threshold != self.threshold,
            "decision_policy": {"kind": "threshold", "threshold": threshold},
            "evaluation_status": self.metadata["evaluation_status"],
            "score_semantics": "uncalibrated context HGB score, not measured probability of water",
            "ablation": self.ablation, "base_feature_supported_ha": denominator, "classified_ha": available,
            "required_context_withheld_ha": max(0., denominator-available),
            "required_context_available_fraction": available/denominator if denominator else None,
            "prior_available_fraction": prior_area/denominator if prior_area is not None and denominator else None,
            "prior_occurrence_available_fraction": occurrence_area/denominator if occurrence_area is not None and denominator else None,
            "context_metadata": context.get("metadata", {}) if context is not None else {"required": False}}


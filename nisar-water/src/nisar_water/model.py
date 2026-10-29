"""Local fitted model loading and bounded inference. No model downloads."""
from pathlib import Path
import joblib
import numpy as np

from .common import read_json, sha256
from .features import FEATURE_VERSION, FEATURE_NAMES, tiles
from .spatial import water_decision


class WaterModel:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = read_json(self.directory / "model.json")
        if self.metadata["feature_version"] != FEATURE_VERSION or self.metadata["features"] != FEATURE_NAMES:
            raise ValueError("Checkpoint feature contract differs from installed code")
        weights = self.directory / "estimator.joblib"
        if sha256(weights) != self.metadata["estimator_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        # Only load the locally fitted trusted artifact; joblib is executable serialization.
        self.estimator = joblib.load(weights)
        self.threshold = float(self.metadata["threshold"])
        self.model_id = self.metadata["model_id"]
        self.fingerprint = sha256(self.directory / "model.json")

    def predict(self, observation, *, threshold=None, tile_size=256):
        policy = self.metadata.get("decision_policy", {"kind": "threshold", "threshold": self.threshold})
        if threshold is not None and policy["kind"] != "threshold":
            raise ValueError("Region-growing checkpoints use a frozen seed/growth policy; threshold overrides are unsupported")
        threshold = self.threshold if threshold is None else float(threshold)
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Threshold must be finite and strictly between zero and one")
        score = np.full(observation.valid.shape, np.nan, np.float32)
        supported = np.zeros(observation.valid.shape, bool)
        water_column = int(np.flatnonzero(self.estimator.classes_ == 1)[0])
        for sl, features, eligible in tiles(observation.power, observation.valid, tile_size):
            supported[sl] = eligible
            if eligible.any():
                values = self.estimator.predict_proba(features[eligible])[:, water_column]
                block = score[sl]
                block[eligible] = values
        observed = observation.valid & (observation.weights > 0)
        classified = observed & supported & ~observation.exclusions
        labels = np.full(observed.shape, 255, np.uint8)
        labels[observed] = 254
        if policy["kind"] == "threshold":
            policy = {"kind": "threshold", "threshold": threshold}
        water = water_decision(score, classified, policy)
        labels[classified] = water[classified].astype(np.uint8)
        score[~observed | observation.exclusions] = np.nan
        return labels, score, {"threshold": threshold,
                             "threshold_override": threshold != self.threshold,
                             "evaluation_status": self.metadata["evaluation_status"],
                             "decision_policy": policy,
                             "score_semantics": "uncalibrated model score, not measured probability of water"}


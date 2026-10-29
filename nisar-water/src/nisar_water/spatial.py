"""Whole-crop hysteresis decisions; no labels, future dates or historical masks."""
import numpy as np
from scipy.ndimage import binary_propagation, label


def water_decision(score, eligible, policy):
    score, eligible = np.asarray(score), np.asarray(eligible, bool)
    if score.shape != eligible.shape or score.ndim != 2:
        raise ValueError("Scores and eligible mask must share a 2D grid")
    valid = eligible & np.isfinite(score)
    if policy["kind"] == "threshold":
        threshold = float(policy["threshold"])
        if not 0 < threshold < 1:
            raise ValueError("Invalid decision threshold")
        return valid & (score >= threshold)
    if policy["kind"] != "region_growing":
        raise ValueError("Unsupported spatial decision policy")
    seed, grow = float(policy["seed_threshold"]), float(policy["grow_threshold"])
    minimum = int(policy["minimum_seed_pixels"])
    if not 0 < grow <= seed < 1 or minimum < 1:
        raise ValueError("Require 0 < grow <= seed < 1 and positive seed support")
    seeds = valid & (score >= seed)
    if minimum > 1:
        components, _ = label(seeds)  # Four-connected, including during growth.
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        seeds = sizes[components] >= minimum
    return binary_propagation(seeds, mask=valid & (score >= grow))


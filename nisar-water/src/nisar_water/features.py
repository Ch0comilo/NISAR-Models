"""Fixed physical features with 60, 140, 300 and 620 m spatial context."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter, sobel

WINDOWS = (3, 7, 15, 31)
FEATURE_VERSION = "nisar-l-hhhv-multiscale-20m-v1"
FEATURE_NAMES = ["HH_db", "HV_db", "HH_minus_HV_db"]
for size in WINDOWS:
    FEATURE_NAMES += [f"support_{size}"]
    for channel in ("HH", "HV"):
        FEATURE_NAMES += [f"{channel}_mean_power_{size}_db", f"{channel}_std_db_{size}",
                          f"{channel}_contrast_{size}_db", f"{channel}_cv_power_{size}"]
FEATURE_NAMES += ["HH_gradient", "HV_gradient"]
HALO = 15


def _sum(a, size):
    return uniform_filter(np.asarray(a, np.float64), size=size, mode="constant", cval=0) * size**2


def compute(power, valid):
    valid = np.asarray(valid, bool) & np.isfinite(power).all(0) & (power > 0).all(0)
    db = 10*np.log10(np.where(valid[None], power, 1))
    feats = [db[0], db[1], db[0]-db[1]]
    eligible = valid.copy()
    for size in WINDOWS:
        count = _sum(valid, size)
        feats.append((count / size**2).astype(np.float32))
        if size <= 7:
            eligible &= count >= np.ceil(.8 * size**2) - 1e-6
        for channel in range(2):
            p = np.where(valid, power[channel], 0).astype(np.float64)
            d = np.where(valid, db[channel], 0).astype(np.float64)
            denominator = np.maximum(count, 1)
            mean_power = _sum(p, size) / denominator
            mean_db = _sum(d, size) / denominator
            mean_power_db = 10 * np.log10(np.maximum(mean_power, 1e-20))
            std_db = np.sqrt(np.maximum(_sum(d*d, size) / denominator - mean_db**2, 0))
            cv = np.sqrt(np.maximum(_sum(p*p, size) / denominator - mean_power**2, 0)) / np.maximum(mean_power, 1e-20)
            feats.extend([mean_power_db, std_db, db[channel]-mean_power_db, cv])
    for channel in range(2):
        feats.append(np.hypot(sobel(db[channel], axis=0, mode="nearest"),
                              sobel(db[channel], axis=1, mode="nearest"))/8)
    features = np.stack(feats, axis=-1).astype(np.float32)
    features[~eligible] = np.nan
    return features, eligible


def tiles(power, valid, tile_size=256):
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    height, width = valid.shape
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            endrow, endcol = min(row+tile_size, height), min(col+tile_size, width)
            r0, c0 = max(0, row-HALO), max(0, col-HALO)
            r1, c1 = min(height, endrow+HALO), min(width, endcol+HALO)
            f, support = compute(power[:, r0:r1, c0:c1], valid[r0:r1, c0:c1])
            rs, cs = slice(row-r0, endrow-r0), slice(col-c0, endcol-c0)
            yield (slice(row, endrow), slice(col, endcol)), f[rs, cs], support[rs, cs]


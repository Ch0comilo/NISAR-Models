"""Reusable development inputs, with no access to the reserved test arrays."""
from pathlib import Path
import numpy as np
from rasterio.transform import rowcol
from scipy.spatial import cKDTree

from .common import read_json, write_json, sha256
from .inputs import load_catalog_observation
from .features import FEATURE_NAMES, tiles

TEST_ID = "explore-20261019T112151Z-497081d067e5"


def load_development(source, cache):
    source, cache = Path(source), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    sample_path = source / "data/labels/water-correction-v9/samples.npz"
    manifest = read_json(sample_path.with_name("dataset.json"))
    if sha256(sample_path) != manifest["samples_sha256"]:
        raise ValueError("Development sample hash mismatch")
    catalog = {e["id"]: e for e in read_json(source / "data/catalog.json")["observations"]}
    parts = {}
    with np.load(sample_path, allow_pickle=False) as z:
        for split in ("train", "validation"):
            parts[split] = {k: z[split+"__"+k] for k in ("X", "y", "areas_ha", "xy", "epsg", "groups", "observation_ids")}
    if any(TEST_ID in p["observation_ids"] for p in parts.values()):
        raise ValueError("Reserved Red Rock observation entered development")
    if set(parts["train"]["groups"]) & set(parts["validation"]["groups"]):
        raise ValueError("Train/validation geographic overlap")
    distances = {}
    for epsg in set(parts["train"]["epsg"]) & set(parts["validation"]["epsg"]):
        train = parts["train"]["xy"][parts["train"]["epsg"] == epsg]
        val = parts["validation"]["xy"][parts["validation"]["epsg"] == epsg]
        distance = float(cKDTree(train).query(val, k=1)[0].min())
        if distance < 140-1e-6:
            raise ValueError("Geographic separation below 140 m")
        distances[str(epsg)] = distance
    for p in parts.values():
        p["context"] = np.full((len(p["y"]), len(FEATURE_NAMES)), np.nan, np.float32)
    wanted = sorted(set(np.concatenate([p["observation_ids"] for p in parts.values()])))
    for oid in wanted:
        print("Development features:", oid, flush=True)
        o = load_catalog_observation(catalog[oid])
        for sl, values, supported in tiles(o.power, o.valid):
            for p in parts.values():
                ids = np.flatnonzero(p["observation_ids"] == oid)
                if not len(ids):
                    continue
                rr, cc = map(np.asarray, rowcol(o.transform, *p["xy"][ids].T))
                inside = (rr >= sl[0].start) & (rr < sl[0].stop) & (cc >= sl[1].start) & (cc < sl[1].stop)
                p["context"][ids[inside]] = values[rr[inside]-sl[0].start, cc[inside]-sl[1].start]
    for split, p in parts.items():
        if not np.isfinite(p["context"]).all() or not np.allclose(p["context"][:, :3], p["X"][:, :3], atol=3e-4, rtol=1e-5):
            raise ValueError("Feature availability/physical input parity failed")
        np.savez_compressed(cache / (split+".npz"), **p)
    write_json(cache / "audit.json", {"source_samples_sha256": sha256(sample_path),
               "development_observations": wanted, "minimum_retained_cell_distance_m": distances,
               "reserved_id_absent": True, "test_arrays_accessed": False})
    return parts, catalog


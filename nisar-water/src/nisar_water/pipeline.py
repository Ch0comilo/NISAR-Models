"""Batch inference and changes measured only on shared classified support."""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from rasterio.warp import reproject
from rasterio.enums import Resampling

from .common import write_json, identity, safe_id, sha256
from .exports import export_observation, write_raster


def _aligned_labels(labels, observation, reference):
    aligned = np.full(reference.valid.shape, 255, np.uint8)
    # Nearest preserves codes. No change measurement is made at unknown pixels.
    reproject(labels, aligned, src_transform=observation.transform, src_crs=observation.crs,
              dst_transform=reference.transform, dst_crs=reference.crs,
              src_nodata=255, dst_nodata=255, resampling=Resampling.nearest)
    return aligned


def compare(before, after, weights=None):
    """Codes: stable land=0, stable water=1, gain=2, loss=3, unavailable=255."""
    if before.shape != after.shape:
        raise ValueError("Change inputs must be aligned")
    common = (before <= 1) & (after <= 1)
    if weights is None:
        weights = np.ones(before.shape, np.float32)
    common &= weights > 0
    change = np.full(before.shape, 255, np.uint8)
    change[common] = 0
    change[common & (before == 1) & (after == 1)] = 1
    change[common & (before == 0) & (after == 1)] = 2
    change[common & (before == 1) & (after == 0)] = 3
    area = weights.astype(np.float64)*.04
    support = float(area[common].sum())
    if support == 0:
        return change, {"state": "no_common_support", "commonClassifiedHa": 0.0,
                        "gainHa": None, "lossHa": None, "netChangeHa": None,
                        "beforeWaterHa": None, "afterWaterHa": None}
    gain, loss = float(area[change == 2].sum()), float(area[change == 3].sum())
    return change, {"state": "comparable", "commonClassifiedHa": support,
                    "gainHa": gain, "lossHa": loss, "netChangeHa": gain-loss,
                    "beforeWaterHa": float(area[common & (before == 1)].sum()),
                    "afterWaterHa": float(area[common & (after == 1)].sum())}


def run_series(observations, model, output, *, series_id="water", threshold=None):
    """Consume an iterable one acquisition at a time; no temporal smoothing/future data."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    series_id = safe_id(series_id)
    manifest_path = output / "series.json"
    if manifest_path.exists():
        raise FileExistsError("Output already contains a series; choose a fresh directory")
    results, aligned, observations_meta, animations, weights = [], [], [], [], []
    reference = None
    geometry = None
    seen = set()
    for observation in observations:
        meta = observation.metadata
        key = tuple(meta.get(k) for k in ("track", "pass_direction", "frame", "frequency"))
        if any(x is None for x in key):
            raise ValueError("Each observation must identify track, direction, frame and frequency")
        if geometry is None:
            geometry = key
        elif key != geometry:
            raise ValueError("Time series must use one track/direction/frame/frequency; run separate series")
        oid = safe_id(meta["id"])
        if oid in seen:
            raise ValueError("Duplicate observation identifier")
        seen.add(oid)
        labels, score, info = model.predict(observation, threshold=threshold)
        # Identity includes actual pixel inputs, model and all scientific settings.
        pixel_inputs = {"power": __import__("hashlib").sha256(observation.power.tobytes()).hexdigest(),
                                 "valid": __import__("hashlib").sha256(observation.valid.tobytes()).hexdigest(),
                                 "weights": __import__("hashlib").sha256(observation.weights.tobytes()).hexdigest(),
                                 "exclusions": __import__("hashlib").sha256(observation.exclusions.tobytes()).hexdigest(),
                                 "transform": list(observation.transform)[:6], "crs": str(observation.crs),
                                 "model": model.fingerprint, "threshold": info["threshold"]}
        from .context_provenance import scientific_context
        context=scientific_context(info)
        if context is not None:pixel_inputs['physical_context']=context['input_sha256']
        pixel_digest=identity(pixel_inputs)
        destination = output / "observations" / f"{oid}-{pixel_digest[:12]}"
        if destination.exists():
            raise FileExistsError(f"Observation output exists: {destination}")
        result = export_observation(observation, labels, score, info, model, destination, asset_root=output)
        results.append(result)
        observations_meta.append(meta)
        if reference is None:
            reference = observation
        aligned.append(_aligned_labels(labels, observation, reference))
        aligned_weights = np.zeros(reference.valid.shape, np.float32)
        reproject(observation.weights, aligned_weights, src_transform=observation.transform, src_crs=observation.crs,
                  dst_transform=reference.transform, dst_crs=reference.crs, dst_nodata=0, resampling=Resampling.nearest)
        weights.append(aligned_weights)
        animations.append(destination / "composite.png")
    if reference is None:
        raise ValueError("No observations supplied")
    order = sorted(range(len(results)), key=lambda i: results[i]["datetime"])
    results, aligned, weights, animations = [[collection[i] for i in order]
                                            for collection in (results, aligned, weights, animations)]
    dates = [r["datetime"] for r in results]
    if len(set(dates)) != len(dates):
        raise ValueError("Multiple revisions/crops at the same acquisition time; select one revision per time series")
    transitions = []
    for i in range(1, len(results)):
        pair_weights = np.minimum(weights[i-1], weights[i])
        change, metrics = compare(aligned[i-1], aligned[i], pair_weights)
        name = f"change-{i-1:03d}-{i:03d}.tif"
        write_raster(output / name, change, reference.transform, reference.crs, nodata=255)
        transitions.append({"beforeId": results[i-1]["id"], "afterId": results[i]["id"],
                            "before": dates[i-1], "after": dates[i], **metrics,
                            "changeUrl": "/assets/"+name})
    # A fixed intersection prevents apparent shrinking caused by changing coverage.
    common_all = np.logical_and.reduce([a <= 1 for a in aligned])
    common_weights = np.minimum.reduce(weights)
    common_all &= common_weights > 0
    area_all = float((common_weights[common_all]*.04).sum(dtype=np.float64))
    timeline = []
    for result, label in zip(results, aligned):
        timeline.append({"id": result["id"], "datetime": result["datetime"],
                         **result["statistics"], "commonAllDatesHa": area_all,
                         "waterOnCommonAllDatesHa": float((common_weights[common_all & (label == 1)]*.04).sum(dtype=np.float64)) if area_all else None})
    counts = np.sum([a <= 1 for a in aligned], axis=0, dtype=np.uint16)
    water_counts = np.sum([a == 1 for a in aligned], axis=0, dtype=np.uint16)
    occurrence = np.full(reference.valid.shape, np.nan, np.float32)
    np.divide(water_counts, counts, out=occurrence, where=counts > 0)
    write_raster(output / "water-occurrence.tif", occurrence, reference.transform, reference.crs, nodata=np.nan)
    write_raster(output / "classified-observation-count.tif", counts, reference.transform, reference.crs)
    with (output / "timeline.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(timeline[0]))
        writer.writeheader()
        writer.writerows(timeline)
    animation_url = None
    same_view = all(r["bounds"] == results[0]["bounds"] for r in results)
    if len(animations) > 1 and same_view:
        frames = []
        for path, result in zip(animations, results):
            with Image.open(path) as image:
                frame = image.convert("RGB")
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, frame.width, 25), fill=(15, 20, 30))
            draw.text((8, 7), result["datetime"] + " | experimental water extent", fill="white")
            frames.append(frame)
        frames[0].save(output / "water-evolution.gif", save_all=True, append_images=frames[1:],
                       duration=1100, loop=0, disposal=2)
        animation_url = "/assets/water-evolution.gif"
    series = {"schemaVersion": "nisar-water-series-1", "id": series_id,
              "kind": "water_extent_time_series", "isDemo": any(r["isDemo"] for r in results),
              "modelId": model.model_id, "evaluationStatus": model.metadata["evaluation_status"],
              "track": geometry[0], "pass": geometry[1], "frame": geometry[2], "frequency": geometry[3],
              "observations": results, "timeline": timeline, "transitions": transitions,
              "animationUrl": animation_url,
              "occurrenceUrl": "/assets/water-occurrence.tif",
              "observationCountUrl": "/assets/classified-observation-count.tif",
              "changeCodes": {"0": "stable_land", "1": "stable_water", "2": "water_gain", "3": "water_loss", "255": "not_comparable"},
              "comparisonPolicy": "consecutive exact acquisitions; nearest categorical reprojection to first grid; common classified support and minimum fractional AOI coverage only",
              "temporalSmoothing": False, "humanImpactAttribution": "not_established"}
    write_json(manifest_path, series)
    return series


def predict_dataset(dataset, model_dir, output_dir, *, series_id="notebook-water",context_cache_dir=None):
    from .inputs import from_dataset
    from .registry import load_model
    model=load_model(model_dir) if context_cache_dir is None else load_model(model_dir,context_cache_dir=context_cache_dir)
    return run_series(from_dataset(dataset), model, output_dir, series_id=series_id)


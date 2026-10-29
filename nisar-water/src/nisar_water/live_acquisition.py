"""Standalone public CMR discovery and authenticated bounded GCOV acquisition."""
from datetime import date
from pathlib import Path
import math

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import transform as project, unary_union

from .common import identity, read_json, sha256, write_json
from .earthdata import EarthdataError, EarthdataHTTP, TransferLedger
from .remote_gcov import AcquisitionError, _summarize_granule, _write_cog, extract_gcov_window, parse_product_id

COLLECTION_ID = "C2854338529-ASF"
COLLECTION_NAME = "NISAR_L2_GCOV_PROVISIONAL_V1"
CMR = "https://cmr.earthdata.nasa.gov/search/"


def validate_aoi(aoi, halo=320):
    if not 80 <= int(halo) <= 1000:
        raise ValueError("Context halo must be 80 to 1000 metres")
    if aoi.get("type") == "Feature":
        aoi = aoi["geometry"]
    geometry = shape(aoi)
    if geometry.geom_type not in ("Polygon", "MultiPolygon") or not geometry.is_valid or geometry.is_empty:
        raise ValueError("Use a nonempty valid WGS84 Polygon or MultiPolygon")
    west, south, east, north = geometry.bounds
    if not all(math.isfinite(v) for v in geometry.bounds) or not (-180 < west < east < 180 and -80 <= south < north <= 84):
        raise ValueError("Use an ordered WGS84 AOI outside polar/antimeridian regions")
    zone = min(60, max(1, int((geometry.centroid.x + 180) // 6) + 1))
    crs = CRS.from_epsg((32600 if geometry.centroid.y >= 0 else 32700) + zone)
    forward = Transformer.from_crs(4326, crs, always_xy=True).transform
    backward = Transformer.from_crs(crs, 4326, always_xy=True).transform
    projected = project(forward, geometry)
    crop = project(backward, box(*projected.bounds).buffer(halo, join_style=2)).bounds
    if projected.area > 400e6 or crop[2] - crop[0] > .2 or crop[3] - crop[1] > .2:
        raise ValueError("Use an AOI at most 400 km2 and 0.2 degrees per side including context")
    pixels = math.ceil((projected.bounds[2] - projected.bounds[0] + 2 * halo) / 10 + 2) * math.ceil((projected.bounds[3] - projected.bounds[1] + 2 * halo) / 10 + 2)
    if pixels > 4_000_000:
        raise ValueError("AOI plus context exceeds four million native pixels")
    return {"aoi": mapping(geometry), "bbox": list(geometry.bounds), "crop_bbox": list(crop)}


def _footprint(record):
    geometry = (record.get("spatial_extent") or {}).get("HorizontalSpatialDomain", {}).get("Geometry", {})
    shapes = []
    for polygon in geometry.get("GPolygons", []):
        points = polygon.get("Boundary", {}).get("Points", [])
        if len(points) >= 3:
            item = Polygon([(p["Longitude"], p["Latitude"]) for p in points])
            if item.is_valid and not item.is_empty:
                shapes.append(item)
    for rectangle in geometry.get("BoundingRectangles", []):
        shapes.append(box(rectangle["WestBoundingCoordinate"], rectangle["SouthBoundingCoordinate"],
                          rectangle["EastBoundingCoordinate"], rectangle["NorthBoundingCoordinate"]))
    return unary_union(shapes) if shapes else None


def candidate_from_record(record, aoi):
    attributes = {a["Name"]: a.get("Values", []) for a in record.get("additional_attributes", [])}
    collection = record.get("collection", {})
    reasons = []
    if collection.get("ShortName") != COLLECTION_NAME or str(collection.get("Version")) != "1":
        reasons.append("Unsupported collection/version")
    if record.get("processing_type") != "PR" or not record.get("polarization_mode", "").startswith("DH"):
        reasons.append("Unsupported processing/polarization profile")
    if set(attributes.get("FREQUENCY_A_POLARIZATION", [])) != {"HH", "HV"}:
        reasons.append("Frequency-A HH/HV is required")
    if not record.get("https_urls"):
        reasons.append("No scientific HTTPS asset")
    footprint = _footprint(record)
    geometry = shape(aoi)
    coverage = min(1., footprint.intersection(geometry).area / geometry.area) if footprint is not None else 0.
    if coverage <= 0:
        reasons.append("No intersecting metadata footprint")
    return {"id": f"{record.get('concept_id')}:{record.get('revision_id')}",
            "granule_id": record["granule_id"], "acquired_at": record["start"],
            "revision_id": record.get("revision_id"), "compatible": not reasons,
            "compatibility_notes": reasons, "coverage_fraction": coverage, "record": record}


def discover(aoi, start, end, work_dir, reference_root=None):
    validated = validate_aoi(aoi)
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or (last - first).days > 366:
        raise ValueError("Use an ordered interval no longer than 366 days")
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    client = EarthdataHTTP(TransferLedger(root / "catalog-transfer.json", limit=32 * 1024**2))
    verified = client.public_json(CMR + "collections.umm_json", {"concept_id": COLLECTION_ID, "page_size": 1})
    items = verified.get("items", [])
    if len(items) != 1 or items[0]["umm"].get("ShortName") != COLLECTION_NAME or str(items[0]["umm"].get("Version")) != "1":
        raise EarthdataError("UNSUPPORTED_COLLECTION", "The supported CMR collection changed; no radar was fetched.")
    query = {"collection_concept_id": COLLECTION_ID, "bounding_box": ",".join(map(str, validated["bbox"])),
             "temporal": f"{start}T00:00:00Z,{end}T23:59:59Z", "page_size": 100,
             "sort_key[]": ["-start_date", "producer_granule_id"]}
    candidates, skipped, count = [], 0, 0
    for page in (1, 2):
        result = client.public_json(CMR + "granules.umm_json", {**query, "page_num": page})
        records = result.get("items", [])
        count += len(records)
        for raw in records:
            try:
                candidates.append(candidate_from_record(_summarize_granule(raw), validated["aoi"]))
            except (ValueError, KeyError, TypeError):
                skipped += 1
        if len(records) < 100:
            break
    compatible = sum(c["compatible"] for c in candidates)
    return {"state": "ready" if compatible else "no_compatible_data", "candidates": candidates,
            "truncated": page == 2 and len(records) == 100, "record_count": count,
            "unsupported_records": skipped, "query": query,
            "message": f"{compatible} compatible observations; metadata footprint is preliminary.",
            "standalone": True}


def acquire(candidate, aoi, work_dir, reference_root=None, max_bytes=128 * 1024**2, *, halo_metres=320):
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    ledger = TransferLedger(root / "radar-transfer.json", limit=max_bytes, storage_root=root)
    try:
        halo = halo_metres if halo_metres is not None else 320
        validated = validate_aoi(aoi, halo)
        record = candidate["record"]
        checked = candidate_from_record(record, validated["aoi"])
        if not checked["compatible"] or any(candidate[k] != checked[k] for k in ("id", "granule_id", "acquired_at")):
            raise ValueError("Candidate identity/compatibility differs from its source record")
        fingerprint = identity({"record": record, "aoi": validated["aoi"], "crop_bbox": validated["crop_bbox"],
                                "recipe": f"native-gcov-power-halo{halo}-v1"})
        folder = root / "crops" / fingerprint
        saved = folder / "observation.json"
        if saved.is_file():
            observation = read_json(saved)
            for asset in observation["assets"].values():
                if sha256(asset["reference"]) != asset["sha256"]:
                    raise ValueError("Cached crop hash mismatch")
            return {"state": "cached", "observation": observation, "transfer_bytes": 0}
        client = EarthdataHTTP(ledger)
        window = extract_gcov_window(client, record, validated["crop_bbox"], max_bytes=max_bytes)
        fields = parse_product_id(record["granule_id"])
        observation = {"id": "nisar-" + fields["start"].replace("-", "").replace(":", "") + "-" + fingerprint[:12],
            "granule_id": record["granule_id"], "acquired_at": fields["start"],
            "sensor": "NISAR", "radar_band": "L", "product_type": "GCOV", "frequency": "A",
            "polarizations": ["HHHH", "HVHV"], "units": "gamma0_power", "rtc_applied": True,
            "processing_version": window["identification"]["productVersion"], "crid": fields["crid"],
            "track": fields["track"], "pass_direction": fields["pass_direction"], "frame": fields["frame"],
            "data_origin": "observed", "aoi": validated["aoi"], "mask_valid_values": [1],
            "source_metadata": {"cmr": record, "hdf5_identification": window["identification"],
                "source_window": window["source_window"], "layers": window["layers"],
                "native_pixel_spacing_m": window["pixel_spacing_m"], "remote_full_integrity_verified": False},
            "context_halo_metres": halo,
            "provenance": [record["https_urls"][0], "Bounded authenticated HTTPS range crop; no full download fallback",
                f"Current NISAR GCOV linear gamma0 power; native validity; {halo}m context; no optical input"]}
        assets = {}
        for channel in ("HHHH", "HVHV"):
            assets[channel] = _write_cog(folder / f"{channel}.tif", window["arrays"][channel], window,
                channel=channel, granule_id=record["granule_id"], units="gamma0_power", nodata=np.nan)
        assets["mask"] = _write_cog(folder / "valid-mask.tif", window["valid"].astype(np.uint8), window,
                channel="valid", granule_id=record["granule_id"], units="valid_mask", nodata=0)
        assets["native_quality_mask"] = _write_cog(folder / "native-gcov-mask.tif", window["raw_mask"], window,
                channel="native-mask", granule_id=record["granule_id"], units="native_mask", nodata=255)
        observation["assets"] = assets
        # Publish the observation receipt only after complete grids/hashes verify.
        from .inputs import load_catalog_observation
        load_catalog_observation(observation)
        write_json(saved, observation)
        return {"state": "acquired", "observation": observation, "transfer_bytes": ledger.total,
                "metrics": window["metrics"]}
    except (AcquisitionError, EarthdataError, ValueError, KeyError, OSError) as error:
        return {"state": "failed", "error_code": getattr(error, "code", type(error).__name__),
                "message": str(error) if isinstance(error, ValueError) else "Scientific crop could not be completed.",
                "transfer_bytes": ledger.total}

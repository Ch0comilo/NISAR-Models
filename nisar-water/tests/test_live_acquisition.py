"""Cache identity and failure accounting for the standalone worker."""
from pathlib import Path

import numpy as np
import pytest

from nisar_water import live_acquisition as live
from nisar_water.common import sha256
from nisar_water.remote_gcov import AcquisitionError


def inputs():
    name = "NISAR_L2_PR_GCOV_023_135_A_021_4005_DHDH_A_20260623T131753_20260623T131828_P05023_N_F_J_001"
    aoi = {"type": "Polygon", "coordinates": [[[-121.52, 39.50], [-121.48, 39.50],
            [-121.48, 39.54], [-121.52, 39.54], [-121.52, 39.50]]]}
    record = {"granule_id": name, "start": "2026-06-23T13:17:53Z", "concept_id": "G-fixture",
              "revision_id": 1, "processing_type": "PR", "polarization_mode": "DHDH",
              "https_urls": ["https://nisar.asf.earthdatacloud.nasa.gov/fixture/" + name + ".h5"],
              "collection": {"ShortName": live.COLLECTION_NAME, "Version": "1"},
              "additional_attributes": [{"Name": "FREQUENCY_A_POLARIZATION", "Values": ["HH", "HV"]}],
              "spatial_extent": {"HorizontalSpatialDomain": {"Geometry": {"BoundingRectangles": [{
                  "WestBoundingCoordinate": -122, "SouthBoundingCoordinate": 39,
                  "EastBoundingCoordinate": -121, "NorthBoundingCoordinate": 40}]}}}}
    return aoi, live.candidate_from_record(record, aoi)


def test_cached_crop_cannot_reuse_an_old_polygon_when_a_hole_changes(tmp_path, monkeypatch):
    aoi, candidate = inputs()
    calls = []
    def extract(client, record, bbox, **kwargs):
        calls.append(bbox)
        return {"arrays": {"HHHH": np.full((4, 4), .02, np.float32),
                           "HVHV": np.full((4, 4), .002, np.float32)},
                "valid": np.ones((4, 4), bool), "raw_mask": np.ones((4, 4), np.uint8),
                "identification": {"productVersion": "fixture"}, "source_window": [0, 0, 4, 4],
                "layers": {}, "pixel_spacing_m": [20, 20], "metrics": {"transferred_bytes": 0}}
    def cog(path, values, window, **kwargs):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(values.tobytes())
        return {"reference": str(path), "sha256": sha256(path), "band": 1}
    monkeypatch.setattr(live, "extract_gcov_window", extract)
    monkeypatch.setattr(live, "_write_cog", cog)
    monkeypatch.setattr("nisar_water.inputs.load_catalog_observation", lambda *a: None)
    first = live.acquire(candidate, aoi, tmp_path)
    repeated = live.acquire(candidate, aoi, tmp_path)
    hole = [[-121.51, 39.51], [-121.49, 39.51], [-121.49, 39.53], [-121.51, 39.53], [-121.51, 39.51]]
    changed = {"type": "Polygon", "coordinates": aoi["coordinates"] + [hole]}
    second = live.acquire(candidate, changed, tmp_path)
    assert first["state"] == second["state"] == "acquired"
    assert repeated["state"] == "cached" and len(calls) == 2
    assert first["observation"]["aoi"] != second["observation"]["aoi"]
    assert first["observation"]["id"] != second["observation"]["id"]


def test_failed_scientific_read_reports_consumed_bytes_to_the_outer_budget(tmp_path, monkeypatch):
    aoi, candidate = inputs()
    def fail(client, *args, **kwargs):
        client.ledger.record("nisar.asf.earthdatacloud.nasa.gov", 206, 1024, "scientific_range")
        raise AcquisitionError("FIXTURE_FAILURE", "Deliberate synthetic read failure")
    monkeypatch.setattr(live, "extract_gcov_window", fail)
    result = live.acquire(candidate, aoi, tmp_path)
    assert result["state"] == "failed" and result["transfer_bytes"] == 1024
    assert not list(tmp_path.rglob("observation.json"))


@pytest.mark.parametrize("halo", [0, -100, 79, 1001])
def test_context_bounds_rejected_before_network(halo):
    aoi, _ = inputs()
    with pytest.raises(ValueError, match="halo"):
        live.validate_aoi(aoi, halo)

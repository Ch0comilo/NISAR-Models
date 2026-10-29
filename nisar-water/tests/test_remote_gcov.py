"""Offline tests with actual HDF5 encoding and an injected byte-range client."""
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import rasterio
from pyproj import Transformer

from nisar_water.remote_gcov import (
    AcquisitionError, HTTPSRangeReader, extract_gcov_window, parse_product_id,
    _summarize_granule, _write_cog,
)


NAME = "NISAR_L2_PR_GCOV_023_135_A_021_4005_DHDH_A_20260623T131753_20260623T131828_P05023_N_F_J_001"
URL = "https://nisar.asf.earthdatacloud.nasa.gov/fixture/" + NAME + ".h5"


class ByteClient:
    """Serve only explicitly bounded slices; no HTTP or full-object read API."""
    def __init__(self, payload, cache_dir=None):
        self.payload = payload
        self.calls = []
        if cache_dir is not None:
            self.ledger = SimpleNamespace(storage_root=Path(cache_dir))

    def stat(self, url):
        assert url == URL
        return {"size": len(self.payload), "etag": "synthetic-hdf5-fixture"}

    def read_range(self, url, start, end):
        assert url == URL and 0 <= start < end <= len(self.payload)
        self.calls.append((start, end))
        return self.payload[start:end]


def hdf_fixture(tmp_path, *, missing=None, updates=None, spacing=20):
    path = tmp_path / "scientific-fixture.h5"
    shape = (48, 56)
    with h5py.File(path, "w") as h5:
        identity = h5.create_group("science/LSAR/identification")
        values = {"granuleId": NAME, "zeroDopplerStartTime": "2026-06-23T13:17:53Z",
                  "trackNumber": 135, "frameNumber": 21, "productVersion": "1.2.3",
                  "compositeReleaseId": "P05023", "productType": "GCOV",
                  "radarBand": "L", "missionId": "NISAR"}
        values.update(updates or {})
        for key, value in values.items():
            if key != missing:
                identity.create_dataset(key, data=value)
        grid = h5.create_group("science/LSAR/GCOV/grids/frequencyA")
        grid.create_dataset("xCoordinates", data=500010 + spacing*np.arange(shape[1]))
        grid.create_dataset("yCoordinates", data=4399990 - spacing*np.arange(shape[0]))
        grid.create_dataset("projection", data=32610)
        hh = np.full(shape, .02, np.float32)
        hv = np.full(shape, .002, np.float32)
        mask = np.ones(shape, np.uint8)
        mask[15, 16] = 0
        mask[16, 17] = 255
        hv[18, 20] = 0
        hh[20, 21] = np.nan
        for channel, array in (("HHHH", hh), ("HVHV", hv)):
            ds = grid.create_dataset(channel, data=array, chunks=(8, 8), compression="gzip")
            ds.attrs["units"] = "1"
        grid.create_dataset("mask", data=mask)
    inverse = Transformer.from_crs(32610, 4326, always_xy=True)
    west, south = inverse.transform(500200, 4399400)
    east, north = inverse.transform(500600, 4399800)
    return path.read_bytes(), [west, south, east, north]


def record():
    return {"granule_id": NAME, "https_urls": [URL]}


def test_extracts_hdf5_power_and_native_mask_without_log_or_fake_validity(tmp_path):
    payload, bbox = hdf_fixture(tmp_path)
    client = ByteClient(payload)
    result = extract_gcov_window(client, record(), bbox, max_bytes=2*1024**2)
    assert result["crs"] == "EPSG:32610" and result["pixel_spacing_m"] == [20, 20]
    assert result["identification"]["productVersion"] == "1.2.3"
    assert result["metrics"]["transferred_bytes"] == sum(b-a for a, b in client.calls)
    assert result["metrics"]["remote_size_bytes"] == len(payload)
    c0, r0, _, _ = result["source_window"]
    for source_row, source_col in ((15, 16), (16, 17), (18, 20), (20, 21)):
        row, col = source_row-r0, source_col-c0
        assert not result["valid"][row, col]
        assert all(np.isnan(result["arrays"][key][row, col]) for key in ("HHHH", "HVHV"))
    np.testing.assert_allclose(result["arrays"]["HHHH"][result["valid"]], .02)
    np.testing.assert_allclose(result["arrays"]["HVHV"][result["valid"]], .002)


@pytest.mark.parametrize("missing,updates,code", [
    ("productVersion", None, "SOURCE_METADATA_UNAVAILABLE"),
    ("granuleId", None, "SOURCE_METADATA_UNAVAILABLE"),
    (None, {"granuleId": NAME.replace("20260623", "20260624")}, "SOURCE_IDENTITY_MISMATCH"),
    (None, {"trackNumber": 134}, "SOURCE_IDENTITY_MISMATCH"),
    (None, {"productType": "GSLC"}, "SOURCE_IDENTITY_MISMATCH"),
    (None, {"zeroDopplerStartTime": "2026-06-24T13:17:53Z"}, "SOURCE_DATE_MISMATCH"),
])
def test_rejects_missing_and_mismatched_native_identity(tmp_path, missing, updates, code):
    payload, bbox = hdf_fixture(tmp_path, missing=missing, updates=updates)
    with pytest.raises(AcquisitionError) as error:
        extract_gcov_window(ByteClient(payload), record(), bbox, max_bytes=2*1024**2)
    assert error.value.code == code


def test_rejects_coarse_grid_and_wrong_url_before_classification(tmp_path):
    payload, bbox = hdf_fixture(tmp_path, spacing=40)
    with pytest.raises(AcquisitionError) as error:
        extract_gcov_window(ByteClient(payload), record(), bbox, max_bytes=2*1024**2)
    assert error.value.code == "UNSUPPORTED_GRID"
    client = ByteClient(payload)
    with pytest.raises(AcquisitionError) as error:
        extract_gcov_window(client, {"granule_id": NAME, "https_urls": [URL.replace(NAME, "wrong")]}, bbox)
    assert error.value.code == "SOURCE_IDENTITY_MISMATCH" and not client.calls


def test_reader_enforces_explicit_bounds_transfer_cap_and_exact_response_length():
    client = ByteClient(bytes(range(256))*64)
    with HTTPSRangeReader(client, URL, block_size=4096, max_bytes=4096) as reader:
        with pytest.raises(AcquisitionError, match="Unbounded"):
            reader.read()
        with pytest.raises(AcquisitionError, match="outside"):
            reader.seek(-1)
        assert len(reader.read(4096)) == 4096
        with pytest.raises(AcquisitionError, match="ceiling"):
            reader.read(1)
    class Truncated(ByteClient):
        def read_range(self, url, start, end):
            return super().read_range(url, start, end)[:-1]
    with HTTPSRangeReader(Truncated(client.payload), URL, block_size=4096, max_bytes=4096) as reader:
        with pytest.raises(AcquisitionError, match="unexpected byte count"):
            reader.read(1)


def test_cached_range_reuse_is_sha_verified_and_ram_cache_is_bounded(tmp_path):
    payload = bytes(range(256))*64
    client = ByteClient(payload, tmp_path)
    with HTTPSRangeReader(client, URL, block_size=4096, cache_blocks=1, max_bytes=8192) as reader:
        assert reader.read(1) == payload[:1]
        reader.seek(4096)
        reader.read(1)
        assert len(reader.cache) == 1
    reused = ByteClient(payload, tmp_path)
    with HTTPSRangeReader(reused, URL, block_size=4096, max_bytes=4096) as reader:
        assert reader.read(1) == payload[:1]
        assert not reused.calls and reader.bytes_transferred == 0
    block = next((tmp_path / "_range-cache").rglob("000000000000.bin"))
    block.write_bytes(b"x"*4096)
    with HTTPSRangeReader(ByteClient(payload, tmp_path), URL, block_size=4096, max_bytes=4096) as reader:
        with pytest.raises(AcquisitionError, match="SHA256"):
            reader.read(1)


def test_cog_output_is_a_derived_verified_power_asset(tmp_path):
    payload, bbox = hdf_fixture(tmp_path)
    window = extract_gcov_window(ByteClient(payload), record(), bbox, max_bytes=2*1024**2)
    asset = _write_cog(tmp_path / "HHHH.tif", window["arrays"]["HHHH"], window,
                       channel="HHHH", granule_id=NAME, units="gamma0_power", nodata=np.nan)
    assert len(asset["sha256"]) == 64
    with rasterio.open(asset["reference"]) as source:
        assert source.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        assert source.transform == window["transform"]
        assert source.tags()["units"] == "gamma0_power"
        np.testing.assert_array_equal(source.read(1), window["arrays"]["HHHH"])


def test_cmr_summary_retains_catalog_identity_and_no_sibling_runtime():
    item = {"meta": {"concept-id": "G123-ASF", "revision-id": 2}, "umm": {
        "GranuleUR": NAME, "RelatedUrls": [{"Type": "GET DATA", "URL": URL}],
        "TemporalExtent": {"RangeDateTime": {"BeginningDateTime": "2026-06-23T13:17:53Z"}},
        "CollectionReference": {"ShortName": "NISAR_L2_GCOV_PROVISIONAL_V1", "Version": "1"},
    }}
    summary = _summarize_granule(item)
    assert summary["revision_id"] == 2 and summary["https_urls"] == [URL]
    assert parse_product_id(NAME)["start"] == "2026-06-23T13:17:53Z"

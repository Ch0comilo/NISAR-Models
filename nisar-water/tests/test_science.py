from pathlib import Path
import json

import h5py
import numpy as np
import pytest
from affine import Affine
from fastapi.testclient import TestClient
from PIL import Image
import rasterio
import xarray as xr

from nisar_water.inputs import Observation, prepare, from_dataset, from_hdf5, parse_granule, aoi_weights
from nisar_water.features import compute, tiles
from nisar_water.pipeline import compare, run_series
from nisar_water.api import create_app
from nisar_water.common import write_json

NAME = "NISAR_L2_PR_GCOV_023_135_A_021_4005_DHDH_A_20260623T131753_20260623T131828_P05023_N_F_J_001.h5"


def obs(date="2026-06-23T13:17:53Z", shift=0):
    power = np.full((2, 40, 48), .02, np.float32)
    power[1] = .002
    meta = {**parse_granule(NAME), "id": "synthetic-"+date, "acquired_at": date,
            "frequency": "A", "data_origin": "synthetic_fixture"}
    return Observation(power, np.ones((40, 48), bool), Affine(20, 0, 627120+shift, 0, -20, 4381220), 32610, meta)


class FixtureModel:
    model_id = "synthetic-fixture-only"
    fingerprint = "test-fixture"
    metadata = {"evaluation_status": "synthetic_fixture"}

    def predict(self, observation, **kwargs):
        labels = np.zeros(observation.valid.shape, np.uint8)
        labels[5:30, 7:37] = 1
        labels[12:18, 15:24] = 0  # A real hole must survive vectorization.
        labels[~observation.valid] = 255
        labels[observation.exclusions & observation.valid] = 254
        return labels, np.where(labels == 1, .9, .1).astype(np.float32), {
            "threshold": .5, "threshold_override": False, "evaluation_status": "synthetic_fixture",
            "score_semantics": "fixture only"}


def test_aggregation_averages_power_before_log_and_preserves_missing():
    power = np.array([[[1, 100], [1, 100]]]*2, np.float32)
    o = prepare(power, np.ones((2, 2), bool), Affine(10, 0, 0, 0, -10, 20), 32610, {})
    assert o.power[0, 0, 0] == 50.5
    assert 10*np.log10(o.power[0, 0, 0]) != pytest.approx(10.)
    mask = np.array([[True, True], [False, True]])
    missing = prepare(power, mask, Affine(10, 0, 0, 0, -10, 20), 32610, {})
    assert not missing.valid.any()


def test_halo_tiling_has_no_seams_and_no_reflected_support():
    rng = np.random.default_rng(17)
    power = np.exp(rng.normal(-4, 1, (2, 72, 83))).astype(np.float32)
    valid = np.ones((72, 83), bool)
    valid[24:29, 39:48] = False
    whole, support = compute(power, valid)
    tiled = np.full_like(whole, np.nan)
    tiled_support = np.zeros_like(valid)
    for sl, values, good in tiles(power, valid, tile_size=17):
        tiled[sl], tiled_support[sl] = values, good
    np.testing.assert_array_equal(tiled_support, support)
    np.testing.assert_allclose(tiled, whole, atol=1e-5, equal_nan=True)
    assert not support[0].any()


def test_unobserved_water_is_not_reported_as_loss():
    before = np.array([[1, 1, 0, 254, 1]], np.uint8)
    after = np.array([[255, 0, 1, 1, 1]], np.uint8)
    change, report = compare(before, after)
    assert change.tolist() == [[255, 3, 2, 255, 1]]
    assert report["gainHa"] == pytest.approx(.04)
    assert report["lossHa"] == pytest.approx(.04)
    assert report["commonClassifiedHa"] == pytest.approx(.12)
    _, empty = compare(before, np.full_like(before, 255))
    assert empty["gainHa"] is None


def notebook_cube():
    return xr.Dataset({"HHHH": (("time", "y", "x"), np.full((1, 40, 48), .01, np.float32)),
                       "HVHV": (("time", "y", "x"), np.full((1, 40, 48), .001, np.float32))},
              coords={"time": np.array(["2026-06-23T13:17:53"], dtype="datetime64[s]"),
                      "nombre": ("time", [NAME]), "x": 627130+20*np.arange(48),
                      "y": 4381210-20*np.arange(40)},
              attrs={"epsg": 32610, "frecuencia": "A", "multilook": 2,
                     "descripcion": "gamma0 (RTC) en potencia lineal, NISAR L2 GCOV"})


def test_exact_notebook_xarray_contract_and_time_validation():
    ds = notebook_cube()
    loaded = list(from_dataset(ds))
    assert loaded[0].power.shape == (2, 40, 48)
    assert loaded[0].metadata["track"] == 135
    ds = ds.assign_coords(time=np.array(["2026-06-24T13:17:53"], dtype="datetime64[s]"))
    with pytest.raises(ValueError, match="time disagrees"):
        list(from_dataset(ds))


def test_notebook_requires_physical_units_and_both_polarizations():
    ds = notebook_cube()
    ds.attrs = {"epsg": 32610}
    with pytest.raises(ValueError, match="linear gamma0"):
        list(from_dataset(ds))
    with pytest.raises(ValueError, match="HHHH and HVHV"):
        list(from_dataset(notebook_cube().drop_vars("HVHV")))


def test_notebook_netcdf_roundtrip(tmp_path):
    path = tmp_path / "input.nc"
    notebook_cube().to_netcdf(path, engine="scipy")
    with xr.open_dataset(path, engine="scipy") as ds:
        result = list(from_dataset(ds))
    assert len(result) == 1
    assert result[0].metadata["acquired_at"] == "2026-06-23T13:17:53Z"


def test_reject_web_mercator_area_and_coarse_upsampling():
    with pytest.raises(ValueError, match="local UTM"):
        Observation(np.ones((2, 20, 20)), np.ones((20, 20), bool),
                    Affine(20, 0, 0, 0, -20, 1000), 3857, {})
    with pytest.raises(ValueError, match="no upsampling"):
        prepare(np.ones((2, 20, 20)), np.ones((20, 20), bool),
                Affine(40, 0, 0, 0, -40, 1000), 32610, {})


def test_real_checkpoint_is_tile_invariant_and_keeps_qa_unknown():
    from nisar_water.model import WaterModel
    checkpoint = Path(__file__).resolve().parents[1] / "models/context-hgb-v1"
    if not checkpoint.exists():
        pytest.skip("Local fitted checkpoint not present")
    o = obs()
    rng = np.random.default_rng(382)
    o.power[:] *= np.exp(rng.normal(0, .3, o.power.shape))
    o.valid[12:16, 15:22] = False
    o.exclusions[20:22, 8:15] = True
    model = WaterModel(checkpoint)
    labels1, score1, _ = model.predict(o, tile_size=13)
    labels2, score2, _ = model.predict(o, tile_size=256)
    np.testing.assert_array_equal(labels1, labels2)
    np.testing.assert_allclose(score1, score2, atol=1e-6, equal_nan=True)
    assert (labels1[~o.valid] == 255).all()
    assert (labels1[o.valid & o.exclusions] == 254).all()
    assert np.isnan(score1[~o.valid | o.exclusions]).all()


def test_local_hdf5_reads_bounded_window_and_qa(tmp_path):
    path = tmp_path / NAME
    with h5py.File(path, "w") as h:
        ident = h.create_group("science/LSAR/identification")
        ident["productType"], ident["radarBand"] = np.bytes_("GCOV"), np.bytes_("L")
        g = h.create_group("science/LSAR/GCOV/grids/frequencyA")
        g["xCoordinates"], g["yCoordinates"] = 627125+10*np.arange(100), 4381215-10*np.arange(80)
        g["projection"] = 32610
        g["HHHH"], g["HVHV"] = np.full((80, 100), .02), np.full((80, 100), .001)
        mask = np.ones((80, 100), np.uint8)
        mask[0, 0] = 255
        g["mask"] = mask
    o = from_hdf5(path, window=(0, 0, 48, 40))
    assert o.power.shape == (2, 20, 24)
    assert not o.valid[0, 0]
    with pytest.raises(ValueError, match="Provide"):
        from_hdf5(path)


def test_export_web_registration_vector_holes_api_and_no_overwrite(tmp_path):
    out = tmp_path / "results" / "river"
    first = obs()
    second = obs("2026-10-21T13:17:50Z")
    second.valid[:8] = False
    series = run_series([second, first], FixtureModel(), out)
    assert series["isDemo"] is True
    assert series["timeline"][0]["datetime"] < series["timeline"][1]["datetime"]
    assert series["transitions"][0]["lossHa"] == 0
    assert series["animationUrl"]
    result = series["observations"][0]
    water_path = out / result["assets"]["water"]["url"].removeprefix("/assets/")
    with Image.open(water_path) as png:
        assert png.mode == "RGBA"
        assert np.asarray(png)[0, 0, 3] == 0
    for vector in out.rglob("water.geojson"):
        gj = json.loads(vector.read_text())
        assert any(len(f["geometry"]["coordinates"]) > 1 for f in gj["features"])
    with TestClient(create_app(tmp_path / "results")) as client:
        listed = client.get("/water/series").json()
        assert len(listed) == 1
        response = client.get(listed[0]["url"])
        assert response.status_code == 200
        published = response.json()
        png_url = published["observations"][0]["assets"]["composite"]["url"]
        assert client.get(png_url).status_code == 200
        assert client.get("/water/series/missing").status_code == 404
    with pytest.raises(FileExistsError):
        run_series([first], FixtureModel(), out)


def test_no_cross_track_series(tmp_path):
    first, second = obs(), obs("2026-10-21T13:17:50Z")
    second.metadata["track"] = 42
    with pytest.raises(ValueError, match="one track"):
        run_series([first, second], FixtureModel(), tmp_path)


def test_empty_series_rejected(tmp_path):
    with pytest.raises(ValueError, match="No observations"):
        run_series([], FixtureModel(), tmp_path)


def test_direct_root_api_key_is_fetchable(tmp_path):
    run_series([obs()], FixtureModel(), tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        result = client.get("/water/series").json()[0]
        assert client.get(result["url"]).status_code == 200


def test_requested_aoi_outside_crop_is_counted_as_unobserved(tmp_path):
    from pyproj import Transformer
    from shapely.geometry import mapping, box
    from shapely.ops import transform as geom_transform
    reference = obs()
    west, south, east, north = rasterio.transform.array_bounds(*reference.valid.shape, reference.transform)
    larger = box(west-200, south-200, east+200, north+200)
    aoi = mapping(geom_transform(Transformer.from_crs(32610, 4326, always_xy=True).transform, larger))
    o = prepare(reference.power, reference.valid, reference.transform, reference.crs, reference.metadata, aoi=aoi)
    result = run_series([o], FixtureModel(), tmp_path)["observations"][0]
    assert result["statistics"]["requestedHa"] == pytest.approx(larger.area/10000)
    assert result["statistics"]["unobservedHa"] > 0
    assert result["statistics"]["observedHa"]+result["statistics"]["unobservedHa"] == pytest.approx(larger.area/10000)


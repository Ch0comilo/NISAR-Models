"""Synthetic CPU checks for matched context ablations; no science accuracy claim."""
from pathlib import Path
import copy
import numpy as np
import pytest
from affine import Affine
from nisar_water.common import write_json, read_json, sha256
from nisar_water.inputs import Observation
from nisar_water.context_model import (ContextWaterModel, train_context_models, context_arrays,
    load_context_sidecar, feature_names, TERRAIN_NAMES, PRIOR_NAMES)


def fixture_dataset(root):
    root.mkdir(parents=True, exist_ok=True)
    shape = (48, 56)
    labels = np.zeros(shape, np.uint8); labels[:, :28] = 1
    labels[:4] = 255; labels[-4:] = 255; labels[:, :4] = 255; labels[:, -4:] = 255
    power = np.zeros((2, *shape), np.float32)
    power[0] = np.where(np.indices(shape)[1] < 28, .002, .1)
    power[1] = np.where(np.indices(shape)[1] < 28, .0002, .03)
    valid = np.ones(shape, bool)
    transform = np.array([20, 0, 500000, 0, -20, 4400000], float)
    scene_path = root/"scene.npz"
    np.savez_compressed(scene_path, power=power, valid=valid, labels=labels,
                        weights=(labels != 255).astype(np.float32), transform=transform, epsg=np.array(32610))
    names = list(TERRAIN_NAMES)+list(PRIOR_NAMES)
    features = np.ones((7, *shape), np.float32)
    features[0] = 2; features[1] = 3; features[2] = 1; features[3] = 2; features[4] = 1
    features[5] = 100  # Constant historical water everywhere: cannot decide the class.
    features[6] = 0
    available = np.ones(shape, bool)
    available[8:15, 8:16] = False
    sidecar = root/"context.npz"
    np.savez_compressed(sidecar, features=features, available=available,
                        prior_available=np.ones(shape, bool), names=np.array(names),
                        transform=transform, epsg=np.array(32610), scene_sha256=np.array(sha256(scene_path)))
    sources = {}; hashes = {}
    for key in ("dem", "hand", "reference_water"):
        path = root/(key+"-synthetic-source.bin"); path.write_bytes((key+" explicit synthetic fixture").encode())
        hashes[key] = sha256(path)
        sources[key] = {"cropped_window_path": str(path), "cropped_window_sha256": hashes[key]}
    meta = write_json(root/"context.json", {"state": "synthetic_fixture", "target_transform": transform.tolist(),
                     "target_epsg": 32610, "source_hashes": hashes, "sources": sources,
                     "hash_scope": "synthetic source windows only"})
    spec = {"id": "fixture-train", "path": str(scene_path), "sha256": sha256(scene_path), "split": "train",
            "group": "train-group", "region": "train-region", "acquired_at": "2026-10-01T00:00:00Z",
            "context_path": str(sidecar), "context_sha256": sha256(sidecar),
            "context_metadata_path": str(meta), "context_metadata_sha256": sha256(meta)}
    val = {"id": "unopened-validation", "path": str(root/"missing-validation.npz"), "sha256": "1"*64,
           "split": "validation", "group": "val-group", "region": "val-region"}
    test = {"id": "unopened-test", "path": str(root/"missing-test.npz"), "sha256": "2"*64,
            "split": "test", "group": "test-group", "region": "test-region"}
    manifest = write_json(root/"manifest.json", {"data_origin": "synthetic_fixture", "label_kind": "weak", "scenes": [spec, val, test]})
    return manifest, spec, power, valid, transform, available


@pytest.fixture(scope="module")
def trained_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("context-fixture")
    manifest, spec, power, valid, transform, available = fixture_dataset(root/"data")
    output = root/"models"
    directories = train_context_models(manifest, output, per_class_scene_cap=128, allow_synthetic_fixture=True)
    return root, manifest, spec, power, valid, transform, available, output, directories


def test_matched_training_reads_no_validation_or_test_pixels(trained_fixture):
    root, manifest, spec, power, valid, transform, available, output, directories = trained_fixture
    audit = read_json(output/"training-audit.json")
    assert audit["common_samples_identical_across_families"] is True
    assert audit["test_paths_opened"] is False
    assert audit["validation_pixels_read"] is False
    assert audit["scenes"][0]["dropped_required_context_cells"] > 0
    assert set(directories) == {"radar", "terrain", "terrain_prior"}
    assert [len(feature_names(name)) for name in directories] == [41, 46, 49]
    counts = [read_json(Path(path)/"model.json")["training_counts"] for path in directories.values()]
    assert counts[0] == counts[1] == counts[2]


def test_missing_terrain_withholds_but_radar_remains_classified(trained_fixture):
    root, manifest, spec, power, valid, transform, available, output, directories = trained_fixture
    metadata = {"id": spec["id"], "data_origin": "synthetic_fixture", **spec, "scene_sha256": spec["sha256"]}
    observation = Observation(power, valid.copy(), Affine(*transform), 32610, metadata)
    radar = ContextWaterModel(directories["radar"])
    terrain = ContextWaterModel(directories["terrain"])
    a, sa, ia = radar.predict(observation, tile_size=16)
    b, sb, ib = terrain.predict(observation, tile_size=32)
    assert (a[~available] <= 1).all()
    assert (b[~available] == 254).all()
    assert np.isnan(sb[~available]).all()
    assert ib["required_context_withheld_ha"] > 0
    assert ia["required_context_withheld_ha"] == 0
    assert ib["required_context_available_fraction"] < 1


def test_historical_water_is_soft_and_tiles_have_no_seams(trained_fixture):
    root, manifest, spec, power, valid, transform, available, output, directories = trained_fixture
    observation = Observation(power, valid.copy(), Affine(*transform), 32610,
                              {**spec, "scene_sha256": spec["sha256"], "data_origin": "synthetic_fixture"})
    model = ContextWaterModel(directories["terrain_prior"])
    a, sa, _ = model.predict(observation, tile_size=16)
    b, sb, _ = model.predict(observation, tile_size=256)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(sa, sb, atol=1e-6, equal_nan=True)
    # Every source prior says water100%, but the learned result still contains land.
    assert (a == 0).any() and (a == 1).any()


def test_optional_prior_missing_stays_nan_without_withholding_physical_context():
    features = np.ones((7, 10, 12), np.float32)
    features[5] = 0
    features[6] = np.nan
    context = context_arrays({"features": features, "names": list(TERRAIN_NAMES)+list(PRIOR_NAMES),
                              "available": np.ones((10, 12), bool), "prior_available": np.zeros((10, 12), bool)}, (10, 12))
    assert context["available"].all()
    assert (context["prior"][..., 0] == 0).all()
    assert np.isnan(context["prior"][..., 1]).all()
    assert not context["prior_available"].any()


def test_context_binding_rejects_wrong_scene_and_grid(tmp_path):
    manifest, spec, power, valid, transform, available = fixture_dataset(tmp_path/"binding")
    with pytest.raises(ValueError, match="different prepared scene"):
        load_context_sidecar(spec, valid.shape, transform, 32610, scene_sha256="3"*64)
    changed = transform.copy(); changed[2] += 20
    with pytest.raises(ValueError, match="grid differ"):
        load_context_sidecar(spec, valid.shape, changed, 32610, scene_sha256=spec["sha256"])
    source = Path(read_json(spec["context_metadata_path"])["sources"]["hand"]["cropped_window_path"])
    source.write_bytes(b"tampered synthetic source")
    with pytest.raises(ValueError, match="cropped source hand hash mismatch"):
        load_context_sidecar(spec, valid.shape, transform, 32610, scene_sha256=spec["sha256"])


def test_group_or_region_leakage_rejected_before_any_scene_read(tmp_path):
    manifest, spec, *_ = fixture_dataset(tmp_path/"leakage")
    content = read_json(manifest)
    content["scenes"][1]["region"] = "train-region"
    write_json(manifest, content)
    with pytest.raises(ValueError, match="multiple partitions"):
        train_context_models(manifest, tmp_path/"should-not-fit", allow_synthetic_fixture=True)


"""Synthetic software checks only; no accuracy conclusions or GPU workloads."""
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

from affine import Affine
from nisar_water.common import write_json, sha256
from nisar_water.inputs import Observation
from nisar_water.neural import (SmallUNet, NeuralWaterModel, normalize_power, masked_bce_dice,
    fit_normalization, development_scenes, PatchDataset, classification_support,
    _save_weights, train_unet, SCHEMA, FEATURE_VERSION, CHANNELS)


NORM = {"mean_db": [-15., -25.], "std_db": [5., 5.], "clip_db": [-60., 10.]}


def fixture_scene(path, scene_id, split, group, *, labels=None):
    power = np.full((2, 40, 48), .01, np.float32)
    power[1] = .001
    valid = np.ones((40, 48), bool)
    if labels is None:
        labels = np.zeros((40, 48), np.uint8)
        labels[8:32, 16:36] = 1
    np.savez_compressed(path, power=power, valid=valid, labels=labels)
    return {"id": scene_id, "path": str(path), "split": split, "group": group, "region": group}


def fixture_checkpoint(directory):
    directory.mkdir()
    torch.manual_seed(410)
    model = SmallUNet(4)
    _save_weights(directory/"weights.pt", model)
    write_json(directory/"model.json", {"schema_version": SCHEMA, "model_type": "unet",
        "model_id": "synthetic_fixture_untrained", "feature_version": FEATURE_VERSION,
        "channel_order": CHANNELS, "base_channels": 4, "threshold": .5,
        "normalization": NORM, "weights_sha256": sha256(directory/"weights.pt"),
        "evaluation_status": "synthetic_fixture"})
    return directory


def test_fixed_normalization_converts_power_once_and_does_not_use_validation():
    p = np.stack([np.full((6, 6), .01), np.full((6, 6), .001)]).astype(np.float32)
    train = {"power": p, "valid": np.ones((6, 6), bool)}
    norm = fit_normalization([train])
    np.testing.assert_allclose(norm["mean_db"], [-20., -30.], atol=1e-5)
    np.testing.assert_allclose(normalize_power(p, train["valid"], norm), 0, atol=1e-5)
    assert norm["fitted_on"] == "train_only"
    assert norm["std_db"] == [2., 2.]
    p[:, 2, 2] = np.nan
    valid = train["valid"].copy(); valid[2, 2] = False
    assert (normalize_power(p, valid, norm)[:, 2, 2] == 0).all()


def test_masked_loss_ignores_unknown_pixels_and_their_gradients():
    logits = torch.tensor([[[[0., 80.], [0., -80.]]]], requires_grad=True)
    target = torch.tensor([[[[1., 0.], [0., 1.]]]])
    weights = torch.tensor([[[[1., 0.], [1., 0.]]]])
    loss = masked_bce_dice(logits, target, weights)
    loss.backward()
    assert torch.isfinite(loss)
    assert (logits.grad[..., 1] == 0).all()
    zero = masked_bce_dice(logits.detach().requires_grad_(), target, torch.zeros_like(weights))
    assert float(zero.detach()) == 0


def test_development_never_opens_test_paths_and_rejects_shared_regions(tmp_path):
    train = fixture_scene(tmp_path/"train.npz", "train", "train", "train-region")
    val = fixture_scene(tmp_path/"val.npz", "val", "validation", "val-region")
    test = {"id": "test", "path": str(tmp_path/"DOES-NOT-EXIST.npz"), "split": "test",
            "group": "test-region", "region": "test-region"}
    path = write_json(tmp_path/"manifest.json", {"data_origin": "synthetic_fixture", "label_kind": "weak", "scenes": [train, val, test]})
    _, parts = development_scenes(path, allow_synthetic_fixture=True)
    assert len(parts["train"]) == len(parts["validation"]) == 1
    with pytest.raises(ValueError, match="observed"):
        development_scenes(path)
    val["region"] = "train-region"
    write_json(path, {"data_origin": "synthetic_fixture", "label_kind": "weak", "scenes": [train, val]})
    with pytest.raises(ValueError, match="multiple partitions"):
        development_scenes(path, allow_synthetic_fixture=True)


def test_unknown_targets_do_not_become_negative_patch_labels():
    labels = np.full((40, 48), 255, np.uint8)
    labels[18, 22] = 1
    scene = {"id": "fixture", "group": "g", "power": np.ones((2, 40, 48), np.float32)*.01,
             "valid": np.ones((40, 48), bool), "labels": labels, "weights": np.ones((40, 48), np.float32)}
    dataset = PatchDataset([scene], NORM, patch_size=32, patches_per_scene=3)
    for index in range(len(dataset)):
        x, y, weight = dataset[index]
        assert int((weight > 0).sum()) == 1
        assert (y[weight > 0] == 1).all()
        assert torch.isfinite(x).all()


def test_geographic_group_sampling_is_balanced_despite_more_dates_in_one_group():
    labels = np.zeros((40, 48), np.uint8); labels[5:30, 5:35] = 1
    scenes = [{"id": str(i), "group": "a" if i < 3 else "b", "power": np.ones((2, 40, 48), np.float32)*.01,
               "valid": np.ones((40, 48), bool), "labels": labels, "weights": np.ones((40, 48), np.float32)}
              for i in range(4)]
    dataset = PatchDataset(scenes, NORM, patch_size=32, patches_per_scene=5)
    groups = [scenes[s[0]]["group"] for s in dataset.samples]
    assert groups.count("a") == groups.count("b") == 10


def test_tensor_checkpoint_and_tiled_inference_have_no_seams_or_qa_relabeling(tmp_path):
    directory = fixture_checkpoint(tmp_path/"model")
    model = NeuralWaterModel(directory, device="cpu")
    rng = np.random.default_rng(91)
    power = np.exp(rng.normal(-4, .6, (2, 168, 184))).astype(np.float32)
    valid = np.ones((168, 184), bool); valid[70:74, 70:75] = False
    exclusions = np.zeros_like(valid); exclusions[90:93, 90:93] = True
    o = Observation(power, valid, Affine(20, 0, 500000, 0, -20, 4400000), 32610,
                    {"data_origin": "synthetic_fixture"}, exclusions=exclusions)
    a, score_a, info = model.predict(o, tile_size=32)
    b, score_b, _ = model.predict(o, tile_size=256)
    np.testing.assert_allclose(score_a, score_b, atol=3e-6, rtol=1e-5, equal_nan=True)
    np.testing.assert_array_equal(a, b)
    assert (a[~valid] == 255).all()
    assert (a[exclusions] == 254).all()
    assert np.isnan(score_a[~classification_support(valid)]).all()
    assert np.isfinite(score_a[a <= 1]).all()
    assert info["evaluation_status"] == "synthetic_fixture"
    with (directory/"weights.pt").open("ab") as stream: stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        NeuralWaterModel(directory, device="cpu")


def test_one_cpu_fixture_training_epoch_preserves_provenance_and_validation_outputs(tmp_path):
    train = fixture_scene(tmp_path/"train.npz", "train", "train", "train-region")
    val = fixture_scene(tmp_path/"val.npz", "val", "validation", "val-region")
    path = write_json(tmp_path/"manifest.json", {"data_origin": "synthetic_fixture", "label_kind": "weak", "scenes": [train, val]})
    output = tmp_path/"fit"
    metadata = train_unet(path, output, epochs=1, patch_size=32, patches_per_scene=1,
                          batch_size=1, base_channels=4, device="cpu", allow_synthetic_fixture=True)
    assert metadata["evaluation_status"] == "synthetic_fixture"
    assert metadata["best_epoch"] == 1
    assert metadata["test_evaluated"] is False
    assert (output/"validation-thresholds.json").exists()
    assert (output/"validation-scores/val.npz").exists()
    assert NeuralWaterModel(output, device="cpu").metadata["training_origin"] == "synthetic_fixture"


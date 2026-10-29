"""Synthetic evaluator contracts; no reserved round4 scene pixels are read."""
from pathlib import Path
import numpy as np
import pytest
from affine import Affine

from nisar_water.common import read_json, sha256, write_json
from nisar_water.inputs import Observation
from nisar_water import round4_evaluation as ev
from nisar_water import round4_evaluation,round5_evaluation


@pytest.fixture(params=[round4_evaluation,round5_evaluation],autouse=True)
def evaluation_protocol(request,monkeypatch):
    import sys
    monkeypatch.setattr(sys.modules[__name__],'ev',request.param)


def observation(shape=(20, 20)):
    return Observation(np.ones((2, *shape), np.float32), np.ones(shape, bool),
                       Affine(20, 0, 500000, 0, -20, 4100000), 32611,
                       {"id": "synthetic-only", "data_origin": "synthetic", "is_demo": True})


def scene_row(truth=None, scores=None, labels=None, *, region="synthetic", occurrence=None):
    o = observation()
    truth = np.zeros(o.valid.shape, np.uint8) if truth is None else truth
    scores = np.zeros(o.valid.shape, np.float32) if scores is None else scores
    labels = np.zeros(o.valid.shape, np.uint8) if labels is None else labels
    spec = {"id": "synthetic-only", "region": region, "acquired_at": "2026-01-01", "split": "validation"}
    return ev.score_scene(spec, o, truth, np.ones(o.valid.shape, bool), labels, scores, .5, occurrence)


def test_all_land_precision_not_required_but_fpr_is():
    scores = np.zeros((20, 20), np.float32)
    scores.flat[:12] = 1
    summary = ev.aggregate([scene_row(scores=scores)])
    assert summary["regions"]["synthetic"]["precision"] == 0
    assert ev.passes(summary)
    assert "water_precision" not in ev.gate_checks(summary)["synthetic"]
    scores.flat[:17] = 1
    assert not ev.passes(ev.aggregate([scene_row(scores=scores)]))


def test_supported_water_needs_precision_and_recall_at_exact_4ha():
    truth = np.zeros((20, 20), np.uint8)
    truth.flat[:100] = 1
    scores = truth.astype(np.float32)
    assert ev.class_supported(float(np.full(100, .04).sum()))
    assert ev.passes(ev.aggregate([scene_row(truth, scores)]))
    scores.flat[:11] = 0
    checks = ev.gate_checks(ev.aggregate([scene_row(truth, scores)]))["synthetic"]
    assert not checks["full_reference_water_recovery"]
    assert checks["water_precision"]
    scores = truth.astype(np.float32)
    scores.flat[100:113] = 1
    checks = ev.gate_checks(ev.aggregate([scene_row(truth, scores)]))["synthetic"]
    assert checks["full_reference_water_recovery"]
    assert not checks["water_precision"]


def test_withheld_is_reported_and_does_not_become_land():
    truth = np.zeros((20, 20), np.uint8)
    truth[:10] = 1
    labels = np.zeros((20, 20), np.uint8)
    labels.flat[:12] = 254
    scores = truth.astype(np.float32)
    scores.flat[:12] = np.nan
    row = scene_row(truth, scores, labels)
    assert row["reference_water_ha"] == pytest.approx(8)
    assert row["withheld_water_reference_ha"] == pytest.approx(.48)
    assert row["reference_coverage"] == pytest.approx(.97)
    assert row["metrics"]["false_water_ha"] == 0
    assert row["metrics"]["missed_water_ha"] == 0
    assert not ev.passes(ev.aggregate([row]))


def test_small_class_does_not_control_gate_or_balanced_accuracy():
    truth = np.zeros((20, 20), np.uint8)
    truth.flat[:50] = 1  # Only 2 ha; no supported-water recall requirement.
    summary = ev.aggregate([scene_row(truth)])
    row = summary["regions"]["synthetic"]
    assert row["recall"] == 0
    assert row["balanced_accuracy_supported_classes"] == 1
    assert ev.passes(summary)
    water_only = np.ones((20, 20), np.uint8)
    row = ev.aggregate([scene_row(water_only, water_only.astype(np.float32))])["regions"]["synthetic"]
    assert row["balanced_accuracy_supported_classes"] == 1
    assert row["specificity"] is None


def test_rank_prioritizes_gate_then_worst_region_then_macro_and_stable_tie():
    base = {"family": "radar", "threshold": .3, "passes_validation_gate": True,
            "minimum_region_balanced_accuracy": .9, "macro_region_balanced_accuracy": .92}
    assert ev._rank(base) > ev._rank({**base, "passes_validation_gate": False, "minimum_region_balanced_accuracy": 1})
    assert ev._rank(base) > ev._rank({**base, "minimum_region_balanced_accuracy": .89, "macro_region_balanced_accuracy": 1})
    assert ev._rank(base) > ev._rank({**base, "macro_region_balanced_accuracy": .91})
    assert ev._rank(base) > ev._rank({**base, "family": "terrain"})
    assert ev._rank(base) < ev._rank({**base, "threshold": .4})


def bound_context(tmp_path, *, binding="scene-hash"):
    o = observation()
    features = np.zeros((7, *o.valid.shape), np.float32)
    features[5].flat[:100] = 5
    features[5].flat[100:200] = 80
    features[5].flat[200:300] = 255
    features[5].flat[300:] = np.nan
    names = np.array(["slope", "relief", "hand", "hand_mean", "hand_min", "jrc2021_occurrence_pct", "distance"])
    path = tmp_path/"context.npz"
    np.savez_compressed(path, features=features, names=names, available=np.ones(o.valid.shape, bool),
                        prior_available=np.zeros(o.valid.shape, bool),  # Distance missing everywhere.
                        prior_occurrence_available=np.ones(o.valid.shape, bool),
                        transform6=np.array(list(o.transform)[:6]), epsg=32611, scene_sha256=binding)
    return o, {"context_path": str(path), "context_sha256": sha256(path), "sha256": "scene-hash"}


def test_occurrence_diagnostic_independent_of_distance_and_never_gates_classification(tmp_path):
    o, spec = bound_context(tmp_path)
    occurrence, state = ev.historical_occurrence(spec, o)
    assert state == "available"
    assert np.isfinite(occurrence).sum() == 200
    truth = np.ones((20, 20), np.uint8)
    row = scene_row(truth, truth.astype(np.float32), occurrence=occurrence)
    strata = row["historical_water_diagnostic"]
    assert strata["low_historical_occurrence_water"]["reference_water_ha"] == pytest.approx(4)
    assert strata["historical_water"]["reference_water_ha"] == pytest.approx(4)
    assert strata["prior_unknown_water"]["reference_water_ha"] == pytest.approx(8)
    assert row["reference_coverage"] == 1


def test_wrong_context_scene_binding_is_rejected(tmp_path):
    o, spec = bound_context(tmp_path, binding="other-scene")
    with pytest.raises(ValueError, match="different scene"):
        ev.historical_occurrence(spec, o)


def minimal_freeze(tmp_path):
    manifest = write_json(tmp_path/"manifest.json", {"scenes": [{"id": "reserved-test", "split": "test"}]})
    output = tmp_path/"output"
    freeze = write_json(output/"freeze.json", {"manifest_sha256": sha256(manifest), "implementation_hashes": {}, "families": {}})
    return manifest, output, freeze


def test_release_guard_runs_before_test_pixel_loader(tmp_path, monkeypatch):
    manifest, output, freeze = minimal_freeze(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("Reserved pixels were read before release")
    monkeypatch.setattr(ev, "load_scene", forbidden)
    with pytest.raises(ValueError, match="remain closed"):
        ev.evaluate_test(manifest, output)
    write_json(output/"test-release.json", {"authorized": True, "freeze_sha256": "wrong", "manifest_sha256": sha256(manifest)})
    with pytest.raises(ValueError, match="does not match"):
        ev.evaluate_test(manifest, output)


def test_release_verifies_frozen_code_and_model_artifact(tmp_path, monkeypatch):
    manifest, output, freeze_path = minimal_freeze(tmp_path)
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    source = tmp_path/"implementation.py"
    source.write_text("original", encoding="utf8")
    model = tmp_path/"model"
    model.mkdir()
    artifact = model/"weights.bin"
    artifact.write_bytes(b"original")
    freeze = read_json(freeze_path)
    freeze["implementation_hashes"] = {"implementation.py": sha256(source)}
    freeze["families"] = {"radar": {"directory": str(model), "artifacts": {"weights.bin": sha256(artifact)}}}
    write_json(freeze_path, freeze)
    write_json(output/"test-release.json", {"authorized": True, "freeze_sha256": sha256(freeze_path), "manifest_sha256": sha256(manifest)})
    ev.check_test_release(manifest, output)
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        ev.check_test_release(manifest, output)
    artifact.write_bytes(b"original")
    source.write_text("changed", encoding="utf8")
    with pytest.raises(ValueError, match="Implementation changed"):
        ev.check_test_release(manifest, output)


def test_validation_freezes_without_opening_train_or_test_pixels(tmp_path, monkeypatch):
    o = observation()
    truth = np.zeros(o.valid.shape, np.uint8)
    truth[:10] = 1
    specs = [{"id": "synthetic-validation", "split": "validation", "region": "synthetic", "acquired_at": "2026-01-01"},
             {"id": "unopened-train", "split": "train"}, {"id": "unopened-test", "split": "test"}]
    manifest = write_json(tmp_path/"manifest.json", {"scenes": specs})
    model_root, baseline, output = tmp_path/"models", tmp_path/"baseline", tmp_path/"evaluation"
    for folder in [*(model_root/f for f in ev.FAMILIES), baseline]:
        write_json(folder/"model.json", {"model_id": folder.name, "model_type": "synthetic-test-double", "threshold": .5})
        (folder/"estimator.joblib").write_bytes(b"synthetic-model")
    class FakeModel:
        def __init__(self, folder):
            self.metadata = read_json(Path(folder)/"model.json")
            self.model_id = self.metadata["model_id"]
            self.fingerprint = sha256(Path(folder)/"model.json")
    monkeypatch.setattr(ev, "load_model", FakeModel)
    calls = []
    def only_validation(spec, model, cache):
        assert spec["split"] == "validation", "Reserved split was opened by selection"
        calls.append((spec["id"], model.model_id))
        return o, truth, np.ones(o.valid.shape, bool), truth, truth.astype(np.float32), np.full(o.valid.shape, np.nan), "not_supplied", {}
    monkeypatch.setattr(ev, "prediction", only_validation)
    frozen = ev.validate_and_freeze(manifest, output, model_root, baseline)
    assert len(calls) == 4
    assert frozen["selected_family"] == "radar"
    assert frozen["test_scene_ids"] == ["unopened-test"]
    assert frozen["families"][ev.BASELINE]["threshold"] == .5
    assert read_json(output/"validation-report.json")["baseline"]["eligible_for_new_family_selection"] is False
    assert read_json(model_root/"selected/model.json")["threshold"] == .9
    with pytest.raises(FileExistsError, match="already frozen"):
        ev.validate_and_freeze(manifest, output, model_root, baseline)


def test_rare_class_withholding_cannot_hide_in_high_total_coverage():
    o=observation((100,100));truth=np.zeros((100,100),np.uint8);truth.flat[:200]=1
    labels=np.zeros_like(truth);labels.flat[:20]=254
    scores=truth.astype(np.float32);scores.flat[:20]=np.nan
    spec={'id':'fixture','region':'r','split':'validation','acquired_at':'2026-01-01'}
    row=ev.score_scene(spec,o,truth,np.ones_like(truth,bool),labels,scores,.5,None)
    summary=ev.aggregate([row]);region=summary['regions']['r']
    assert region['reference_coverage']>.99 and region['recall']==1
    assert region['water_reference_coverage']==pytest.approx(.9)
    assert not ev.passes(summary)

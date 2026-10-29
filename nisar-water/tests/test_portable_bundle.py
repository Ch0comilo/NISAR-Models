from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nisar_water.api import create_app
from nisar_water.common import read_json, sha256, write_json
from nisar_water.registry import load_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("family", ["radar", "terrain", "terrain_prior"])
def test_bundled_checkpoints_are_bound_to_the_manifest(family):
    bundle = read_json(ROOT / "checkpoints/manifest.json")
    spec = bundle["models"][family]
    directory = ROOT / spec["directory"]
    for name, asset in spec["assets"].items():
        assert sha256(directory / name) == asset["sha256"]
    model = load_model(directory)
    assert model.threshold == spec["threshold"]
    assert model.estimator.n_features_in_ == spec["feature_count"]
    assert np.array_equal(model.estimator.classes_, [0, 1])


def test_explicit_assessment_keeps_new_inference_results_honest(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    assessment = tmp_path / "current.json"
    write_json(assessment, {"state": "not_ready_for_general_use", "production_ready": False})
    write_json(root / "assessment.json", {"state": "obsolete"})
    client = TestClient(create_app(root, assessment_path=assessment))
    assert client.get("/water/assessment").json()["production_ready"] is False
    assert client.get("/health").json()["state"] == "ready"
    assert client.get("/assets/../current.json").status_code == 404


def test_assessment_is_optional_but_explicit_missing_file_fails_at_startup(tmp_path):
    assert TestClient(create_app(tmp_path)).get("/water/assessment").status_code == 404
    with pytest.raises(FileNotFoundError):
        create_app(tmp_path, assessment_path=tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json")
    with pytest.raises(ValueError):
        create_app(tmp_path, assessment_path=invalid)

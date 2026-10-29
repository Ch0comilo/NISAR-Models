import numpy as np
import pytest
from pathlib import Path
from nisar_water.spatial import water_decision
from nisar_water.model import WaterModel
from test_science import obs


POLICY = {"kind": "region_growing", "seed_threshold": .9, "grow_threshold": .03, "minimum_seed_pixels": 4}


def test_grow_connected_weak_water_but_reject_isolated_dark_regions():
    scores = np.zeros((12, 20), np.float32)
    scores[3:5, 2:4] = .95
    scores[4:8, 4:8] = .1
    scores[1:6, 13:17] = .2
    water = water_decision(scores, np.ones_like(scores, bool), POLICY)
    assert water[6, 6]
    assert not water[3, 15]


def test_unknown_and_exclusion_masks_block_region_growth():
    scores = np.full((8, 12), .1, np.float32)
    scores[2:4, 1:3] = .95
    valid = np.ones_like(scores, bool)
    valid[:, 5] = False
    water = water_decision(scores, valid, POLICY)
    assert water[:, :5].any()
    assert not water[:, 5:].any()


def test_spatial_decision_needs_sufficient_seed_and_uses_four_connectivity():
    scores = np.zeros((8, 8), np.float32)
    scores[1, 1] = .99
    assert not water_decision(scores, np.ones_like(scores, bool), POLICY).any()
    scores[2:4, 2:4] = .99
    water = water_decision(scores, np.ones_like(scores, bool), POLICY)
    assert not water[1, 1]
    assert water[2:4, 2:4].all()


def test_frozen_spatial_model_is_independent_of_feature_tile_size():
    directory = Path(__file__).resolve().parents[1] / "models/spatial-v2/selected"
    if not directory.exists():
        pytest.skip("Fitted spatial checkpoint not present")
    model = WaterModel(directory)
    observation = obs()
    observation.power[:, 5:35, 3:40] *= .03
    a, score_a, info = model.predict(observation, tile_size=13)
    b, score_b, _ = model.predict(observation, tile_size=256)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(score_a, score_b, equal_nan=True, atol=1e-6)
    assert info["decision_policy"] == POLICY
    with pytest.raises(ValueError, match="frozen seed/growth"):
        model.predict(observation, threshold=.5)


def test_bad_growth_parameters_rejected():
    with pytest.raises(ValueError, match="grow <= seed"):
        water_decision(np.ones((5, 5)), np.ones((5, 5), bool), {**POLICY, "grow_threshold": .99})


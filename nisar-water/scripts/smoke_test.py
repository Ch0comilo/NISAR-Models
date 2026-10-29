"""Exercise the shipped model, exports and HTTP contract without downloads.

All input pixels are SYNTHETIC. This verifies software plumbing, not accuracy.
Requires the `test` extra for FastAPI's in-process HTTP client.
"""
import argparse
from pathlib import Path

import numpy as np
from affine import Affine
from fastapi.testclient import TestClient

from nisar_water.api import create_app
from nisar_water.common import sha256, write_json
from nisar_water.inputs import Observation
from nisar_water.pipeline import run_series
from nisar_water.registry import load_model

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh smoke-test output directory")
    rng = np.random.default_rng(20261029)
    rows, cols = np.indices((96, 96))
    observations = []
    for day, radius in ((1, 20), (13, 24)):
        dark = (rows - 48)**2 + (cols - 48)**2 < radius**2
        hh = np.where(dark, .006, .08) * rng.lognormal(0, .15, rows.shape)
        hv = np.where(dark, .001, .02) * rng.lognormal(0, .15, rows.shape)
        valid = np.ones(rows.shape, bool)
        valid[:4] = False
        metadata = {"id": f"synthetic-{day:02d}", "granule_id": "SYNTHETIC_FIXTURE_NOT_NISAR_DATA",
                    "acquired_at": f"2026-01-{day:02d}T12:00:00Z", "track": 1,
                    "pass_direction": "A", "frame": 1, "frequency": "A", "data_origin": "synthetic_fixture"}
        observations.append(Observation(np.stack([hh, hv]), valid,
                            Affine(20, 0, 500000, 0, -20, 4400000), 32610, metadata))
    model = load_model(ROOT / "checkpoints/round5/radar")
    series = run_series(observations, model, output / "series", series_id="synthetic-smoke")
    assert series["isDemo"] and len(series["observations"]) == 2
    assert len(series["transitions"]) == 1
    # Exercise exactly the API that the team's separate UI will consume.
    with TestClient(create_app(output, assessment_path=ROOT / "benchmarks/round5/assessment.json")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/water/assessment").json()["production_ready"] is False
        listing = client.get("/water/series").json()
        manifest = client.get(listing[0]["url"]).json()
        fetched = 0
        for observation in manifest["observations"]:
            assert observation["isDemo"]
            for asset in observation["assets"].values():
                response = client.get(asset["url"])
                assert response.status_code == 200
                import hashlib
                assert hashlib.sha256(response.content).hexdigest() == asset["sha256"]
                fetched += 1
        assert client.get("/openapi.json").status_code == 200
    write_json(output / "smoke-report.json", {"state": "passed", "input_origin": "synthetic_fixture",
               "accuracy_evaluated": False, "observations": 2, "http_assets_verified": fetched,
               "model_metadata_sha256": sha256(ROOT / "checkpoints/round5/radar/model.json")})
    print("Passed: synthetic two-date inference, exports, transition and HTTP asset hashes.")
    print("This is a software check; no real-world accuracy was measured.")


if __name__ == "__main__":
    main()

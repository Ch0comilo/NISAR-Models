"""Copy the compact, already-frozen research artifacts into the portable bundle.

Run in the original research workspace only. No fit, selection or metadata edit
is performed; a fresh clone already contains the resulting checkpoints.
"""
from pathlib import Path
import shutil

from nisar_water.common import read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    freeze_path = ROOT / "outputs/round5/freeze.json"
    freeze = read_json(freeze_path)
    models = {}
    for family in ("radar", "terrain", "terrain_prior"):
        source = ROOT / "outputs/round5/frozen" / family
        expected = freeze["families"][family]
        if sha256(source / "model.json") != expected["model_sha256"]:
            raise ValueError("Frozen model metadata differs: " + family)
        meta = read_json(source / "model.json")
        if sha256(source / "estimator.joblib") != meta["estimator_sha256"]:
            raise ValueError("Frozen estimator differs: " + family)
        destination = ROOT / "checkpoints/round5" / family
        destination.mkdir(parents=True, exist_ok=True)
        assets = {}
        for name in ("model.json", "estimator.joblib"):
            target = destination / name
            if target.exists() and sha256(target) != sha256(source / name):
                raise FileExistsError("Refusing to replace a different checkpoint: " + str(target))
            shutil.copyfile(source / name, target)
            assets[name] = {"sha256": sha256(target), "bytes": target.stat().st_size}
        models[family] = {
            "directory": destination.relative_to(ROOT).as_posix(),
            "model_id": meta["model_id"], "threshold": meta["threshold"],
            "feature_count": len(meta["features"]), "assets": assets,
            "role": "validation_selected_diagnostic" if family == "radar" else "frozen_comparison_candidate",
            "production_ready": False,
        }
    benchmark = ROOT / "benchmarks/round5"
    benchmark.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "outputs/round5/assessment.json", benchmark / "assessment.json")
    # Exact machine-readable assessment is portable and contains no local input paths.
    write_json(ROOT / "checkpoints/manifest.json", {
        "schema_version": "nisar-water-checkpoint-bundle-1",
        "scientific_status": "not_ready_for_general_use",
        "source_freeze_sha256": sha256(freeze_path),
        "source_manifest_sha256": freeze["manifest_sha256"],
        "assessment_path": "benchmarks/round5/assessment.json",
        "assessment_sha256": sha256(benchmark / "assessment.json"),
        "implementation_hashes": freeze["implementation_hashes"],
        "models": models,
        "selection_note": "No candidate passed all validation targets; comparisons were frozen before test use.",
    })
    images = ROOT / "docs/assets"
    images.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "evidence/round5-all-test-images.png", images / "round5-all-test-images.png")
    print("Prepared three verbatim frozen checkpoints and the current assessment.")


if __name__ == "__main__":
    main()

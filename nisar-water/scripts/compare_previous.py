"""Compare saved classifiers on identical validation support, without accessing test arrays."""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree

from nisar_water.common import read_json, write_json, sha256
from nisar_water.training import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve() / "src"))
    from nisar_landcover import ml
    samples = args.source / "data/labels/water-correction-v9/samples.npz"
    dataset_meta = read_json(samples.with_name("dataset.json"))
    if sha256(samples) != dataset_meta["samples_sha256"]:
        raise ValueError("Dataset integrity failure")
    with np.load(samples, allow_pickle=False) as z:
        x, y, areas = [z["validation__"+k] for k in ("X", "y", "areas_ha")]
        train_groups, val_groups = z["train__groups"], z["validation__groups"]
        if set(train_groups) & set(val_groups):
            raise ValueError("Geographic group overlap")
        distances = {}
        for epsg in set(z["train__epsg"]) & set(z["validation__epsg"]):
            t = z["train__xy"][z["train__epsg"] == epsg]
            v = z["validation__xy"][z["validation__epsg"] == epsg]
            distance = float(cKDTree(t).query(v, k=1)[0].min())
            if distance < 140-1e-6:
                raise ValueError("Geographic buffer violation")
            distances[str(epsg)] = distance
    reports = {}
    for name, folder in (("historical_baseline", "rf-final-weak-v3"),
                         ("previous_failed_water_correction", "water-correction-v9/binary-target-1")):
        directory = args.source / "models" / folder
        with np.load(directory / "forest.npz", allow_pickle=False) as z:
            forest = {k: z[k] for k in z.files}
        ml._validate_forest(forest, binary=name != "historical_baseline")
        column = int(np.flatnonzero(forest["classes"] == 1)[0])
        scores = ml._predict_forest(forest, x)[:, column]
        reports[name] = {"threshold": .3, **metrics(y, scores, .3, areas),
                         "checkpoint_sha256": sha256(directory / "forest.npz")}
    context = read_json(args.model / "model.json")
    comparison = read_json(args.model / "comparison.json")
    for name, result in comparison["models"].items():
        reports[name] = {"threshold": result["selected"]["threshold"], **result["selected"]["overall"]}
    report = {"scope": "Identical 118068 existing development validation cells; agreement with dated weak optical labels",
              "validation_samples_sha256": sha256(samples), "geographic_separation_m": distances,
              "classifiers": reports, "test_arrays_accessed": False, "independent_test_evaluated": False}
    write_json(args.output, report)
    for name, m in reports.items():
        print(name, {k: round(m[k], 4) for k in ("threshold", "precision", "recall", "false_water_ha", "missed_water_ha")})


if __name__ == "__main__":
    main()


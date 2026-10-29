"""Fit the declared radar/terrain/soft-prior HGB ablations on TRAIN scenes only."""
import argparse
from pathlib import Path
from nisar_water.context_model import train_context_models
from nisar_water.common import read_json,sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--per-class-scene-cap", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    release=read_json(root/'evidence/round4-training-release.json')
    if release['manifest_sha256']!=sha256(args.manifest):
        raise ValueError('A successful review release matching this exact manifest is required before fitting')
    train_context_models(args.manifest, args.output, seed=args.seed, per_class_scene_cap=args.per_class_scene_cap)


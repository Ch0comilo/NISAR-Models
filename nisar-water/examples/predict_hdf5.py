"""Minimal local scientific-input worker for later API job integration."""
import argparse
from pathlib import Path

from nisar_water.common import read_json
from nisar_water.inputs import from_hdf5
from nisar_water.pipeline import run_series
from nisar_water.registry import load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--aoi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/round5/terrain_prior"))
    parser.add_argument("--context-cache", type=Path)
    args = parser.parse_args(argv)
    model = load_model(args.model, context_cache_dir=args.context_cache)
    observation = from_hdf5(args.input, aoi=read_json(args.aoi))
    result = run_series([observation], model, args.output, series_id="my-water-observation")
    print("Saved", len(result["observations"]), "observation to", args.output.resolve())


if __name__ == "__main__":
    main()

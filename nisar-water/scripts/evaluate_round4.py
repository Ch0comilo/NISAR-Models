"""Select on round-four validation, then evaluate only an explicitly released test."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from nisar_water.round4_evaluation import validate_and_freeze, evaluate_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["validation", "test"])
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/round4-prepared/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/round4")
    parser.add_argument("--model-root", type=Path, default=ROOT/"models/round4")
    parser.add_argument("--baseline", type=Path, default=ROOT/"outputs/round3/frozen/hgb")
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.operation == "validation":
        result = validate_and_freeze(args.manifest, args.output, args.model_root, args.baseline)
        print("Frozen selected round4 family:", result["selected_family"])
    else:
        result = evaluate_test(args.manifest, args.output, args.release)
        print("Released round4 test completed:", result["readiness"])


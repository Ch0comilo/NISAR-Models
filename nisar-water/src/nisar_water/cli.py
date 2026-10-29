from __future__ import annotations
import argparse
from pathlib import Path
import sys
from .common import read_json, write_json, identity, safe_id


def main(argv=None):
    parser = argparse.ArgumentParser(description="NISAR open-water mapping without a UI")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fit the existing weak-label NISAR development split")
    train.add_argument("--source", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    improve = commands.add_parser("improve", help="One fixed additional fit and development-only spatial selection")
    improve.add_argument("--source", type=Path, required=True)
    improve.add_argument("--baseline", type=Path, required=True)
    improve.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate-unseen", help="Evaluate a frozen model on the reserved Red Rock observation once")
    evaluate.add_argument("--source", type=Path, required=True)
    evaluate.add_argument("--experiment", type=Path, required=True)
    evaluate.add_argument("--baseline", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    archive = commands.add_parser('aoi', help='Acquire and map every compatible NISAR observation for an AOI/date range')
    archive.add_argument('input',type=Path,help='WGS84 GeoJSON polygon or feature')
    archive.add_argument('--start',required=True);archive.add_argument('--end',required=True)
    archive.add_argument('--model',type=Path,required=True);archive.add_argument('--output',type=Path,required=True)
    archive.add_argument('--reference-root',type=Path,default=Path(__file__).resolve().parents[3]/'nisar-landcover-phase1')
    archive.add_argument('--transfer-cap-mib',type=int,default=1024)
    archive.add_argument('--limit',type=int,default=0,help='0 processes all compatible dates; a positive cap explicitly marks a partial run')
    archive.add_argument('--context-cache',type=Path,help='Explicitly enable bounded public terrain/prior fetching into this cache for new AOIs')
    review=commands.add_parser('import-review',help='Version human polygon labels without changing frozen experiments')
    review.add_argument('--packet',type=Path,required=True,help='Review packet source.json')
    review.add_argument('--annotations',type=Path,required=True,help='Reviewed WGS84 GeoJSON with class/reviewer/source attribution')
    review.add_argument('--output',type=Path,required=True)
    for name in ("catalog", "netcdf", "hdf5"):
        p = commands.add_parser(name)
        p.add_argument("input", type=Path)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--threshold", type=float)
        p.add_argument('--context-cache',type=Path,help='Explicitly enable bounded physical-context fetching; omitted uses existing cache only')
        if name == "catalog":
            p.add_argument("--observation", action="append", help="Repeat for exact IDs; omitted means every supplied catalog observation")
        if name == "hdf5":
            p.add_argument("--aoi", type=Path, required=True, help="WGS84 GeoJSON Polygon or Feature")
    serve = commands.add_parser("serve", help="Expose saved results through HTTP, without a UI")
    serve.add_argument("--root", type=Path, required=True)
    serve.add_argument("--port", type=int, default=8010)
    serve.add_argument("--host", default="127.0.0.1", help="Bind address; use 0.0.0.0 explicitly for a container")
    serve.add_argument("--assessment", type=Path, help="Explicit saved assessment for this result service")
    serve.add_argument("--origin", action="append", default=["http://localhost:5173", "http://127.0.0.1:5173"])
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            from .training import train_from_existing
            result = train_from_existing(args.source, args.output)
            print(result["evaluation_status"], args.output)
        elif args.command == "improve":
            from .experiment import run_experiment
            run_experiment(args.source, args.baseline, args.output)
        elif args.command == "evaluate-unseen":
            from .holdout import evaluate_unseen
            evaluate_unseen(args.source, args.experiment, args.baseline, args.output)
        elif args.command == "serve":
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(args.root, args.origin, assessment_path=args.assessment), host=args.host, port=args.port)
        elif args.command == 'aoi':
            if args.limit<0 or not 1<=args.transfer_cap_mib<=6144:raise ValueError('Use limit>=0 and transfer cap 1..6144 MiB')
            from .archive import process_aoi
            result=process_aoi(read_json(args.input),args.start,args.end,args.model,args.output,args.reference_root,
                               transfer_cap_bytes=args.transfer_cap_mib*1024**2,limit=args.limit,context_cache_dir=args.context_cache)
            print(result['state'],result['acquired_observations'],'observations',args.output.resolve())
        elif args.command == 'import-review':
            from .review_import import import_review
            result=import_review(args.packet,args.annotations,args.output)
            print('Imported',result['label_kind'],result['destination'])
        else:
            from .registry import load_model
            from .pipeline import run_series
            from .inputs import from_dataset, from_hdf5, load_catalog_observation, parse_granule
            model = load_model(args.model,context_cache_dir=args.context_cache)
            if args.command == "catalog":
                entries = read_json(args.input)["observations"]
                if args.observation:
                    unknown = set(args.observation)-{e["id"] for e in entries}
                    if unknown:
                        raise ValueError(f"Unknown observation IDs: {sorted(unknown)}")
                    entries = [e for e in entries if e["id"] in args.observation]
                groups = {}
                for entry in entries:
                    key = identity({k: entry.get(k) for k in ("track", "pass_direction", "frame", "frequency", "aoi")})[:16]
                    groups.setdefault(key, []).append(entry)
                report = []
                for key, entries in groups.items():
                    print("Processing series", key, "observations:", len(entries), flush=True)
                    series = run_series((load_catalog_observation(e, args.input.parent) for e in entries),
                                        model, args.output/key, series_id=key, threshold=args.threshold)
                    report.append({"id": key, "observations": len(series["observations"]), "path": str((args.output/key/"series.json").resolve())})
                if not groups:
                    raise ValueError("Input catalog has no selected observations")
                write_json(args.output/"batch.json", {"series": report})
            elif args.command == "netcdf":
                import xarray as xr
                with xr.open_dataset(args.input) as cube:
                    run_series(from_dataset(cube), model, args.output, threshold=args.threshold)
            else:
                paths = (sorted(p for p in args.input.glob("*GCOV*.h5") if "QA_STATS" not in p.name)
                         if args.input.is_dir() else [args.input])
                if not paths:
                    raise ValueError("No local GCOV HDF5 files found")
                groups = {}
                for path in paths:
                    meta = parse_granule(path.name)
                    group = f"{meta['track']}-{meta['pass_direction']}-{meta['frame']}"
                    groups.setdefault(group, []).append(path)
                for key, paths in groups.items():
                    destination = args.output if len(groups) == 1 else args.output/key
                    run_series((from_hdf5(path, aoi=read_json(args.aoi)) for path in paths), model,
                               destination, series_id=key, threshold=args.threshold)
            print("Saved", args.output.resolve())
        return 0
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


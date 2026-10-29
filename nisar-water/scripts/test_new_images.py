"""Acquire a declared set of new locations/dates before notebook creation."""
from pathlib import Path
import argparse
from nisar_water.common import write_json, read_json, sha256, utcnow
from nisar_water.online import discover, acquire, select_dates

CASES = [
    ("lake-mead", "Lake Mead: water and dry terrain", [-114.80, 36.07, -114.74, 36.13]),
    ("lake-texoma", "Lake Texoma: reservoir and agricultural surroundings", [-96.67, 33.84, -96.61, 33.90]),
    ("gatun-lake", "Gatun Lake: tropical water and forest", [-79.87, 9.14, -79.81, 9.20]),
    ("lake-okeechobee", "Lake Okeechobee: broad open water", [-80.82, 26.92, -80.76, 26.98]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/new-images-20261028"))
    parser.add_argument("--reference", type=Path, default=Path("../nisar-landcover-phase1"))
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output/"plan.json"
    if not plan_path.exists():
        cases = [{"id": key, "title": title, "aoi": {"type": "Polygon", "coordinates": [[[w,s],[e,s],[e,n],[w,n],[w,s]]]},
                  "start": "2026-10-01", "end": "2026-10-27"} for key,title,(w,s,e,n) in CASES]
        write_json(plan_path, {"created_at": utcnow(), "cases": cases, "model_sha256": sha256("models/spatial-v2/selected/model.json"),
                  "policy": "No fitting or threshold selection. Two dates per region, one geometry; coverage/date metadata select acquisitions before predictions.",
                  "maximum_images": 8, "maximum_range_transfer_per_image_bytes": 128*1024**2,
                  "optical_policy": "Nearest Sentinel-2 acquisition within 24h, selected by time, SCL4/5/6 interiors; unknown references remain unscored"})
    plan = read_json(plan_path)
    results = []
    for case in plan["cases"]:
        folder = args.output/case["id"]
        folder.mkdir(exist_ok=True)
        discovery_path = folder/"discovery.json"
        if discovery_path.exists():
            found = read_json(discovery_path)
        else:
            print("Discovering",case["id"],flush=True)
            found = discover(case["aoi"],case["start"],case["end"],folder,args.reference)
            write_json(discovery_path,found)
        selected = select_dates(found.get("candidates",[]))
        write_json(folder/"selection.json", {"selected": selected, "selected_at": utcnow(), "model_scores_used": False})
        print(case["id"],found["state"],"selected",len(selected),flush=True)
        acquisitions=[]
        if not args.discover_only:
            for i,candidate in enumerate(selected):
                receipt=folder/f"acquisition-{i}.json"
                if receipt.exists():
                    result=read_json(receipt)
                else:
                    print("Acquiring",case["id"],candidate["acquired_at"],flush=True)
                    result=acquire(candidate,case["aoi"],folder,args.reference)
                    write_json(receipt,result)
                print(case["id"],result["state"],result.get("coverage",{}),flush=True)
                acquisitions.append(result)
        results.append({"case":case,"discovery_state":found["state"],"selected":len(selected),"acquisitions":acquisitions})
        write_json(args.output/"batch.json", {"cases":results,"model_unchanged":sha256("models/spatial-v2/selected/model.json")==plan["model_sha256"]})


if __name__=="__main__":
    main()


"""Fetch independently dated references for all acquired images, before inference."""
from pathlib import Path
from nisar_water.common import read_json,write_json
from nisar_water.optical import fetch_reference


def main():
    root=Path("data/new-images-20261028")
    summary=[]
    for folder in sorted(root.iterdir()):
        if not folder.is_dir(): continue
        for path in sorted(folder.glob("acquisition-*.json")):
            acquired=read_json(path)
            if acquired.get("state") not in ("acquired","cached"): continue
            entry=acquired["observation"]
            print("Reference",folder.name,entry["acquired_at"],flush=True)
            ref=fetch_reference(entry,folder/"references"/entry["id"])
            print(ref["state"],ref.get("date_difference_hours"),ref.get("reason"),flush=True)
            summary.append(ref)
            write_json(root/"references.json",{"references":summary,"model_predictions_used":False})


if __name__=="__main__":main()


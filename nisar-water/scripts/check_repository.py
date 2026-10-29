"""Offline integrity and credential checks for the files Git would publish."""
from pathlib import Path
import json
import re
import subprocess

from nisar_water.common import read_json, sha256

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {".venv", ".venv-opera", ".cache", ".secrets", "data", "outputs", "models", "evidence", "review", "vendor"}
JWT = re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{40,}")
SIGNED = re.compile(r"[?&](?:X-Amz-Signature|access_token|token)=[^\s\"'<>]+", re.I)
PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")


def publication_files():
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=ROOT, check=True, capture_output=True)
    return sorted({ROOT / name.decode("utf-8") for name in result.stdout.split(b"\0") if name})


def check_bundle():
    bundle = read_json(ROOT / "checkpoints/manifest.json")
    if bundle["scientific_status"] != "not_ready_for_general_use":
        raise ValueError("Unexpected scientific readiness statement")
    for model in bundle["models"].values():
        directory = ROOT / model["directory"]
        for name, asset in model["assets"].items():
            if sha256(directory / name) != asset["sha256"]:
                raise ValueError("Checkpoint integrity failure: " + str(directory / name))
        meta = read_json(directory / "model.json")
        if meta["threshold"] != model["threshold"] or len(meta["features"]) != model["feature_count"]:
            raise ValueError("Checkpoint feature/threshold contract differs")
    for name, expected in bundle["implementation_hashes"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("Frozen scientific implementation changed: " + name)
    if sha256(ROOT / bundle["assessment_path"]) != bundle["assessment_sha256"]:
        raise ValueError("Current assessment changed")
    return bundle


def main():
    bundle = check_bundle()
    paths = publication_files()
    for path in paths:
        relative = path.relative_to(ROOT)
        if relative.parts[0] in FORBIDDEN or path.suffix in (".dpapi", ".pem", ".key") or path.name.startswith(".env"):
            raise ValueError("Private/generated file would be published: " + relative.as_posix())
        if path.stat().st_size >= 50 * 1024**2:
            raise ValueError("Unexpected large publication file: " + relative.as_posix())
        if path.suffix.lower() in (".py", ".md", ".json", ".toml", ".yml", ".yaml", ".ps1", ".ipynb", ".txt"):
            text = path.read_text(encoding="utf-8")
            if JWT.search(text) or SIGNED.search(text) or PRIVATE_KEY.search(text):
                # Print only a path, never the matching credential.
                raise ValueError("Credential-like content in " + relative.as_posix())
            if path.suffix == ".json":
                json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
            if path.name == "NISAR_Water_Try_Your_Data.ipynb":
                notebook = json.loads(text)
                if any(c.get("outputs") or c.get("execution_count") is not None
                       for c in notebook["cells"] if c["cell_type"] == "code"):
                    raise ValueError("Clear the public notebook outputs/execution counts before publication")
    print(json.dumps({"state": "passed", "publication_files": len(paths),
                      "publication_bytes": sum(p.stat().st_size for p in paths),
                      "checkpoint_families": list(bundle["models"]),
                      "scientific_status": bundle["scientific_status"]}, indent=2))


if __name__ == "__main__":
    main()

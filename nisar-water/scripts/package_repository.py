"""Create a source/checkpoint archive from Git's non-ignored publication files."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from check_repository import ROOT, main as check_repository, publication_files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "repository-ready.zip")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    check_repository()
    files = [path for path in publication_files() if path != output]
    if output.exists():
        raise FileExistsError("Use a new archive filename; existing archives are preserved")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path("nisar-water") / path.relative_to(ROOT))
    print("Packaged", len(files), "files into", output)


if __name__ == "__main__":
    main()

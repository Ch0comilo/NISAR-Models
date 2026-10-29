# GitHub handoff

The repository root on the development computer is:

```text
C:\Users\danie\Documents\NISAC 2\nisar-water
```

This is the folder to open in your editor and upload/push as a repository.
It contains source, tests, lockfile, architecture/new-data/API guides, a sample
AOI, a minimal Python worker, the current scientific assessment and three
portable frozen model directories under `checkpoints/round5/`.

The owner supplied the shared repository
[Ch0comilo/NISAR-Models](https://github.com/Ch0comilo/NISAR-Models), with this
package inside `nisar-water/`. Preserve that layout when cloning:

```sh
git clone https://github.com/Ch0comilo/NISAR-Models.git
cd NISAR-Models/nisar-water
```

For changes, review the candidate file list and run the checks before committing:

```sh
git status --short
uv run --no-sync python scripts/check_repository.py
git add .
git diff --cached --stat
git commit -m "Describe the concrete change"
```

The repository's root README links the public Earthdata-token notebook, and the
root `.github/workflows/ci.yml` runs checks from this package directory.
Choose the team's intended code/model
license before inviting open-source reuse; the current license notice grants
no additional reuse rights.

`scripts/package_repository.py` creates an optional `repository-ready.zip` from
the same non-ignored candidate files. It scans them first and excludes Git
metadata. The archive contains a single `nisar-water/` root and does not include
credentials, local environments, research data, caches or generated results.

```sh
uv run --no-sync python scripts/package_repository.py
```

Original `models/`, `data/`, `outputs/`, `evidence/`, `review/`, executed research
notebooks and vendored comparator source remain on this computer and are
excluded by `.gitignore`. They have not been deleted. Original checkpoint
metadata and scientific feature sources were not edited during packaging.
The portable copies preserve frozen threshold/hash identities.
`.gitattributes` disables line-ending conversion for SHA-bound metadata and
scientific sources so a real Git clone preserves those exact identities.

Fresh-clone software checks use synthetic input and include no satellite
downloads. The actual model result/failure summary is separate from those
software checks. Full historical retraining is not claimed reproducible without
the excluded training data and original research inputs.

For the team's later API implementation, start with `examples/predict_hdf5.py`
as a worker and `docs/api-integration.md` as the integration guide. The shipped
HTTP service reads completed results; proposed asynchronous inference job
endpoints are explicitly separate from the implemented routes.

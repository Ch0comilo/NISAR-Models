# GitHub handoff

The repository root on the development computer is:

```text
C:\Users\danie\Documents\NISAC 2\nisar-water
```

This is the folder to open in your editor and upload/push as a repository.
It contains source, tests, lockfile, architecture/new-data/API guides, a sample
AOI, a minimal Python worker, the current scientific assessment and three
portable frozen model directories under `checkpoints/round5/`.

A local Git repository has been initialized on branch `main`. No remote was
created, no files were committed and nothing was uploaded. Review the candidate
file list before your first commit:

```sh
git status --short
uv run --no-sync python scripts/check_repository.py
git add .
git diff --cached --stat
git commit -m "Add headless NISAR water inference and API contract"
```

Create the desired empty GitHub repository in your account, then connect it
using the exact remote URL supplied by GitHub and push `main`. A remote/name
has deliberately not been invented. Choose the team's intended code/model
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

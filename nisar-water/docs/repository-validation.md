# Repository delivery verification

Verified on 2026-10-29 using Windows and CPython 3.13.5.

- Created an isolated local Git repository from the non-ignored publication
  files, committed it only inside the ignored test workspace, and cloned it.
  The main project repository remains uncommitted and has no remote.
- Installed the clone with `uv sync --frozen --extra test --python 3.13`.
  No sibling project, original virtual environment or credentials were copied.
- Checkpoint, assessment and frozen scientific source hashes survived the Git
  round trip. The three compact model metadata/estimator pairs are unchanged.
- **79 tests passed, three skipped, no failures.** Skips are the optional Torch
  module and two historical local-checkpoint tests whose artifacts are not
  distributed. The original workspace additionally passed those two historical
  model tests, for 81 passing main-runtime tests.
- The synthetic two-date smoke check exercised shipped radar inference,
  exports, temporal transition, readiness and 18 HTTP asset hash checks. It
  measured no scientific accuracy.
- The clean clone also processed an existing, verified Nam Theun 1 NISAR crop
  through the conservative checkpoint and the CLI catalog adapter. Only the
  copied scientific input and existing public ancillary cache were supplied.
  One real observation passed COG layout, raster alignment, area partition,
  polygon/raster area equality, RGBA transparency, file hashes and HTTP asset
  checks. This reuses an already exposed observation for software verification;
  it is not another unseen test or a new accuracy claim.
- Active README/architecture/testing/API/handoff/model-card local links were
  checked. Python source and wheel builds completed. The lockfile check passed.
- Publication-file scanning passed for plaintext JWT-like credentials, private
  keys, signed-token URLs, accidental large files and excluded research folders.

The first clean-copy test attempt hit a pre-existing permission problem in the
system pytest temp directory. The final documented command uses project-local
`--basetemp=.pytest-work`, whose parent exists in a fresh clone and whose
contents are ignored. The full final clean-clone test run passed.

GitHub Actions is configured for Linux and Windows; it has not run on GitHub
because no remote was created or pushed. Docker CLI is installed, but its
Linux engine was unavailable; the optional container recipe was not executed.
Software verification does not change `not_ready_for_general_use`.

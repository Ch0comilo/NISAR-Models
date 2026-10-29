# Development

Install the locked Python 3.13 environment with `uv sync --frozen --extra test`.
Run `uv run --no-sync python -m pytest -q --basetemp=.pytest-work` and
`uv run --no-sync python scripts/check_repository.py` before proposing a change.
The included CI runs offline software/model checks on Windows and Linux; a
successful software check does not establish scientific model readiness.

Keep raw data, credentials and generated results out of Git. Only use the
shipped trusted checkpoint artifacts or your own trusted fits. Do not modify
an existing frozen profile or its scientific feature implementation in place;
create a versioned model/implementation and retain its source/input hashes.
The repository check deliberately rejects changes to bundled frozen sources.

Train, validation and test data must use separate geographic groups. Reserve
fresh locations and acquisitions before evaluation, freeze operating points
before opening test labels, preserve failures and report unknown coverage per
class. Test images that later enter fitting are no longer unseen. Human
annotations need reviewer/source/date attribution. Do not infer human causes
from model water change alone.

Historical fitting scripts depend on the original local manifests/source data
and sometimes earlier sibling projects. Full training data are not distributed;
the current fresh-clone workflow supports inference and software verification.
Document any new API job endpoints separately from the existing read-only
result service and preserve the versioned result schema.

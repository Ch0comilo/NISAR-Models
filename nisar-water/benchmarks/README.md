# Scientific assessment

`round5/assessment.json` is the unchanged current experiment assessment. It
reports `not_ready_for_general_use` and binds the validation-selected radar
profile by its metadata SHA256. Its pooled metrics describe that diagnostic
profile, not the conservative comparison. All three profiles remain unaccepted.
Profile-specific comparisons are in the portable model card and results guide.

The checkpoint bundle manifest records this assessment's SHA256. New inference
runs do not establish a new accuracy result. Serve this file explicitly with
`nisar-water serve --assessment benchmarks/round5/assessment.json` when viewing
outputs from these checkpoints. Do not reuse it to claim evaluation of a
different future model.

Raw imagery, large per-pixel references, local training manifests and all
original research results are excluded from Git. Consequently the complete
historical training/evaluation cannot be rerun from this minimal repository
alone. This repository includes reproducible inference, not a complete public
training dataset. See `docs/improved-classifier-results.md` for the experiment
record and remaining independent annotation work.

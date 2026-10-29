# Round4 evaluation contract

The three new families use the same manifest and full weak optical reference
interiors. Only `split=validation` scenes may be opened by selection. The old
round3 HGB remains a fixed baseline at its frozen threshold. Old sidecar split
names do not change the new manifest partition.

For each new family, selection tests thresholds 0.1 through 0.9. A region passes
when at least 98% of its full reference area and each supported class receive a classified prediction;
full-reference water recovery and classified precision are at least 90% if its reference water is at least
4 ha; and land false positive rate is at most 4% if its reference land is at
least 4 ha. Regions with no supported water class have no precision requirement.
Selection ranks passing candidates first, then worst-region balanced accuracy,
then mean region balanced accuracy. This ranking averages fractions correctly recovered out of the full reference for classes
with at least 4 ha support. Exact ties use radar, terrain, terrain_prior order,
then the highest threshold. Threshold choice never reads new test pixels. These refinements were recorded in
`data/round4/selection-clarification-before-fit.json` before any real round4 fit.

Confusion metrics use classified reference cells. Full water and land reference,
classified water and land area, withheld water and land area, total and class
coverage, and recovered water divided by full reference water are reported
separately. Codes254/255 and nonfinite scores remain withheld. They never become
land predictions. Weak boundary measurements are a separate diagnostic and do
not claim expert shoreline accuracy.

The evaluator reads an existing, hash-bound context sidecar for JRC2021
diagnostics. Current optical water is split into occurrence<=10%, >=50%,
intermediate, and unknown strata. Occurrence validity is independent of prior
distance availability. Historical context does not change reference support or
radar/terrain classification eligibility. Low historical occurrence is not
proof of newly formed water.

Run `scripts/evaluate_round4.py validation` after model artifacts and prepared
data are ready. This writes the full validation report, snapshots all four
families, freezes thresholds and implementation hashes, and copies the selected
new family to `models/round4/selected`. The evaluator refuses to overwrite a
freeze.

After reviewing validation, the coordinating agent must explicitly release
test evaluation with `outputs/round4/test-release.json`:

```json
{
  "authorized": true,
  "freeze_sha256": "SHA256 of outputs/round4/freeze.json",
  "manifest_sha256": "SHA256 of data/round4-prepared/manifest.json"
}
```

Then run `scripts/evaluate_round4.py test`. Before loading test pixels it checks
the receipt, manifest, code and frozen artifact hashes. Test evaluation scores
all frozen families without selecting or tuning on test results. Automated
weak-reference gates alone do not establish production readiness.

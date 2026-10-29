# Portable model card

**Status: research only; `not_ready_for_general_use`.** These are byte-for-byte
copies of the three round5 checkpoints frozen before the fresh test. The
bundle does not refit, reselect, modify thresholds or update frozen metadata.
[manifest.json](manifest.json) binds model files, source implementation and the
current [assessment](../benchmarks/round5/assessment.json) by SHA256.

| Directory | Features | Threshold | Role |
|---|---:|---:|---|
| `round5/radar` | 41 | 0.20 | Best diagnostic under validation ranking; joint gate failed |
| `round5/terrain` | 46 | 0.10 | Frozen physical-context comparison |
| `round5/terrain_prior` | 49 | 0.10 | Frozen conservative comparison |

Each profile contains `estimator.joblib` and `model.json`. The loader verifies
the estimator hash, feature order/version and supported schema. Do not load
untrusted joblib files: the format can execute Python code. Use the locked
scikit-learn 1.7.2 environment; cross-version pickle compatibility is not assumed.

Model: `sklearn.ensemble.HistGradientBoostingClassifier`, 300 iterations,
31 maximum leaves, learning rate 0.05, minimum 40 samples/leaf, L2=10,
early-stopping disabled, random seed 20261028. Each fit shares 203,678 sampled
training cells (95,393 water; 108,285 land), weighted equally by geographic
region's total confidence. Fitting uses TRAIN only. Round5 has 71 training,
six validation and five held-out test scenes; previously evaluated round4
scenes are exposed training in round5 and cannot be counted as unseen again.

Input: current NISAR L-band GCOV frequency-A HHHH/HVHV linear gamma0 power,
prepared at 20 m on a north-up UTM grid. The 41 radar features describe channel
power, contrast, texture, support and gradients over up to 31-cell windows.
The physical profiles add DEM slope/relief and three HAND features. The prior
profile also adds JRC2021 occurrence/distance and availability. No current
optical image, future acquisition or absolute-coordinate feature is supplied
at inference. Historical water is a soft learned input, not a hard mask.

The prior profile has pooled weak-reference precision 99.95%, recall 75.49%
and land FPR 0.000577% on the five fresh tests. Nam Theun 1 recall is 92.28%,
but Yusufeli recall is only 19.73%. The diagnostic radar profile has precision
27.18% and recall 74.14%. No family passed all validation/test targets. These
are selected regional, correlated observations with automated optical labels;
they do not establish global, expert shoreline, narrow-river or flood accuracy.

Scores are uncalibrated. Codes are 0 land, 1 water, 254 unknown/excluded and
255 unobserved/outside. Required missing terrain is unknown; optional missing
prior remains NaN plus availability. Interpret all extent/change estimates
alongside support and readiness. Human activity attribution is not established.

`evaluation_status` inside frozen files retains its historical pre-evaluation
label. Read the bundled assessment for current readiness; this packaging step
does not rewrite the scientific record to change that label.

Historical U-Net, OPERA and agreement-review artifacts are retained only in
the original workspace. They are not dependencies of these portable profiles.
The agreement model contains workspace-specific component bindings and is not
part of this distributable bundle.

See [architecture](../docs/architecture.md), [complete results](../docs/improved-classifier-results.md)
and [license notice](../LICENSE).

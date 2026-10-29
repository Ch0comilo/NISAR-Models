# nisar-water-spatial-v2

**Current readiness: not ready for general use.** A subsequent frozen-model test
on eight new radar acquisitions in four regions exposed major recall failures:
11.3% at Mead and 0% at Okeechobee, despite 98.6% at Texoma. Only three images
had usable dated optical reference support. See `docs/new-images-benchmark.md`
and `outputs/new-images-20261028/assessment.json`. The checkpoint was not changed;
the earlier local/Red Rock results below remain historical evidence.

Status: passed the declared development gate and one frozen, unseen-location
weak-reference evaluation. Global, field and shoreline accuracy remain unvalidated.

This profile retains the v1 NISAR-trained HH/HV spatial-context boosted classifier
and replaces independent per-pixel thresholding with four-connected region
growing. At least four adjacent raw-score ≥0.90 seed pixels can grow into
connected score ≥0.03 pixels. Unobserved, excluded and unsupported pixels block
growth. The selected policy was frozen before the reserved Red Rock pixels or
reference labels were opened. No optical inputs or test information enter
inference. Scores remain uncalibrated; 0.90 is a score threshold, not measured
90% confidence.

Training support remains 6,774 water and 77,232 nonwater weak-label cells from the
existing geographic development split. A larger, water-weighted 400-iteration
classifier was fitted as a bounded comparison and rejected on development.
The selected estimator is the original 250-iteration, 15-leaf HGB. Its spatial
policy is selected using 118,068 existing validation cells, with minimum 240 m
retained train/validation separation in the North American development areas.

Development precision/recall: 58.07% / 71.08%. This is a recall-oriented tradeoff,
with lower precision and F1 than the earlier conservative threshold. The declared
joint gate passed; the maximum per-observation nonwater false-positive rate was
1.372%, below the 4% cap.

Unseen geography: Lake Red Rock, Iowa, NISAR 2026-10-19 11:21:51 UTC, one
previously reserved acquisition and area, at least 71.90 km from development
cells. Same-day Sentinel-2 SCL reference, 5.84 hours later, screened to 40 m
homogeneous interiors. On 376.44 ha / 9,411 eligible cells, precision 98.64%,
recall 80.90%, F1 0.8889, IoU 0.8001, nonwater false-positive rate 0.5824%.
False-water area is 1.44 ha and missed-water area 24.68 ha on that labelled
support. Radar observes only 763.41 ha of the 3,573.50 ha requested AOI. Source
unknowns and missing coverage are not scored or filled with water predictions.

These are agreement measurements with machine-generated optical labels,
not independent field truth. One new region/date cannot establish global
generalization. Narrow channels, weak or wind-roughened water, vegetation,
boundary pixels and dry-land/terrain confusers remain limitations. The native
GCOV mask is not a complete geometry shadow/layover screen. Four-pixel seed
requirements can miss small water bodies. Incorrect confident seeds can grow
into connected nonwater. Water gain/loss does not establish a human cause.

The input/output, power aggregation, native QA and georeferencing contracts are
the same as v1. Region growing is global within each supplied crop: independent
crops can differ where they supply different spatial context or connectivity.
Feature tiling alone does not change the result. Tile-size invariance and growth
barriers are tested. Avoid overriding a single score threshold; this checkpoint
uses its frozen two-threshold decision policy.

Checkpoint: `models/spatial-v2/selected`.
Plan/selection/freeze: `models/spatial-v2/`.
Test report and reference support: `outputs/unseen-red-rock-v2/`.
Visual comparison: `evidence/unseen-red-rock-v2.png`.
Reproduction and detailed methodology: `docs/spatial-v2-results.md`.


# nisar-water-context-hgb-v1

**Status: experimental; joint water acceptance failed.**

This is a trained water/nonwater histogram gradient-boosted tree model, using
41 fixed spatial HH/HV features. It is fitted directly to NISAR development
examples, not a downloaded Sentinel-1 pretrained model and not a neural network.

Inputs are NISAR L-band GCOV frequency A ordered HHHH/HVHV linear gamma0 power,
with validity masks, a local UTM grid and 20 m analysis cells. Native 10 m inputs
are averaged in power before feature extraction; incomplete blocks are invalid.
The 3/7-cell neighborhoods require at least 80% valid source support. Larger
15/31-cell context uses available samples and supplies its support fraction as a
feature. Missing exterior is not reflected or filled with measured zero.

Training: 84,006 existing development cells, comprising 6,774 weak-reference water
and 77,232 nonwater samples. Dated Dynamic World/Sentinel-2 SCL reference interiors
and hard negatives were inherited from `water-correction-v9`. They are machine
weak labels, not human-reviewed boundaries. Existing geography is preserved;
train/validation minimum retained-cell separation is 240 m in both North American
UTM zones and 4,920 m in the Tapajós zone. This does not establish independence
from spatial or temporal autocorrelation.

Validation: 118,068 cells, with 1,058 water and 117,010 nonwater. The selected
threshold is 0.70. Precision is 0.8103, recall 0.5813, F1 0.6769 and IoU 0.5116,
against the weak references. False-water area on labelled validation support is
5.76 ha, and missed-water area is 17.72 ha. These areas are not full-scene error
estimates or confidence intervals. Raw tree scores are not calibrated physical
probabilities. Threshold selection used validation; an independent test was not
evaluated. Reserved Lake Red Rock labels and older exposed test arrays were not
accessed.

Acceptance requires overall precision ≥0.50 and recall ≥0.70, with nonwater
false-positive rate ≤0.04 in every validation observation. None of the predeclared
threshold candidates passed. Maximum validation F1 selected a diagnostic
operating point for an explicitly experimental pipeline. This checkpoint was not
promoted into either existing application's production configuration.

Appropriate use: development visualization, annotation prioritization, comparison
with alternative water methods and frontend integration. Generalization to new
regions, dates, water regimes and shorelines is unresolved. Wind-roughened water,
flooded vegetation, narrow rivers and mixed shore pixels can be missed. Smooth
dry surfaces, fields and radar shadows can cause false alarms. Native subswath
validity is not a full shadow/layover mask. Model water loss/gain does not establish
a flood, drought, dam-operation change or human cause.

Scientific outputs preserve water/nonwater, unsupported or excluded, and
unobserved states. Change reports use common classified support. Occurrence is
the fraction of supplied classified acquisitions; it is not day-weighted water
duration. Pixel spacing, raster-derived shoreline geometry and projected grid
areas do not establish true boundary accuracy or radar resolution.

Checkpoint and fit provenance: `models/context-hgb-v1/model.json`,
`models/context-hgb-v1/plan.json`, `models/context-hgb-v1/comparison.json`.
Identical-support comparison with previous models:
`evidence/same-support-comparison.json`.


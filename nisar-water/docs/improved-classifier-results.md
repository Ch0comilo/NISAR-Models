# Water classifier improvement: terrain, recent water and honest review

The pipeline is implemented and tested; **no model is ready for general use**.
Two further prospective experiments materially reduced false water and exposed
the need to distinguish persistent water from water missing in historical maps.
The executed handoff is `notebooks/NISAR_Water_Improved_Classifier.ipynb`.
Default execution uses cached actual data and makes no satellite downloads.

## Reproducible models and inputs

Round4 has53 train, six validation and12 fresh test scenes. All earlier46
exposed scenes became training, alongside new airport, playa, agricultural and
river examples. New validation/test geographies had a conservative minimum
cross-split separation of139.05km. Round5 has71 train, six validation and five
fresh test scenes. Its minimum planned cross-split separation is334.14km.
All round4 scenes are now exposed training in round5 and cannot be described
as fresh tests a second time. Hidroituango lacked acceptable nearby clear
optical references and is not assigned a scored accuracy result.

Three fixed HGB feature ablations share identical sampled training cells:

* Radar-only:41 HH/HV power/texture/context features.
* Radar+terrain:46 features, adding DEM slope, local relief and HAND descriptors.
* Radar+terrain+prior:49 features, adding JRC2021 occurrence/distance and prior
  availability. These are learned inputs, never a coded historical-water mask.

There is no absolute coordinate or absolute-elevation feature. Input radar
remains current-observation NISAR L-GCOV frequency-A HH/HV gamma0 power.
Native10m power is averaged to20m only with complete native validity. Terrain
measurements have nominal30m resolution;20m output posting does not create
finer independent terrain measurements. Required missing terrain is unknown,
while optional missing historical prior remains NaN plus its availability flag.

Round4 accepted fits used147,525 common samples (76,104 water/71,421 land).
Round5 used203,678 common samples; exact class counts are in its metadata.
Each accepted fit has300 iterations,31 leaves, learning rate0.05, min-leaf40,
L2=10, equal geographic-region confidence weights and a2,048/class/scene cap.
No new U-Net or other GPU fit was needed for these feature ablations. The
previous U-Net experiment remains preserved as history.

An initial unreleased round4 attempt ran before a failed release check was
handled. It was isolated in `models/round4-unreleased-first-attempt` before any
validation score was examined. Tiny negative uniform-filter means of genuine
nonnegative HAND data had incorrectly failed range checks. Physical context
was corrected to enforce nonnegative means, its version increased tov2, all
derived sidecars regenerated from unchanged source crops, and the training CLI
made to require a successful release matching the exact manifest. A regression
test reproduces cancellation at roughly10^-13 metres. The original feature
sidecars, manifest, source snapshots and three unreleased fits remain available;
they were not used in validation/test results. Three corrected accepted fits
per round followed the successful review release. This is not an unreported
hyperparameter search.

## Labels and selection

Supervision remains dated optical SCL/spectral consensus with exactly-once
calibration, explicit unknowns, weak boundary weighting and40m-eroded scoring
interiors. No independent expert labels were imported into these fits. Assistant
visual QA is not expert review or field truth.

Selection uses fresh validation only. A supported class needs at least4ha.
Targets are90% water precision/full-reference recovery, at most4% land FPR,
and at least98% total and per-class reference coverage. Withheld water is
unrecovered, never relabelled land. Ranking prefers passing gates, then worst
and macro full-reference balanced recovery, with fixed tie rules. Thresholds
0.1..0.9 were declared before fitting.

Round4 selected terrain+prior at0.30 on validation. Round5 introduced an extra
requirement: current water on JRC occurrence<=10% must also reach90% full-reference
recovery with98% coverage. No round5 family passed every validation target;
radar-only0.20 was retained as the best diagnostic choice, not an accepted
deployment. Terrain and terrain+prior comparison points are0.10. All candidate
weights, thresholds, input/source hashes and test-release receipts were frozen
before the new test predictions. Nothing was reselected after those results.

## What the new tests show

Round4's selected prior model suppressed airport/playa false water but missed
most current GERD water. On the same12-image test, radar-only recalled90.43%
of reference water but had31.48% land FPR; terrain+prior achieved99.95% precision
and0.00369% land FPR but only67.88% water recall. Its GERD recall was0.97%.
This motivated the separate round5 experiment with exposed recent-water
failures in TRAIN and new reservoir/dry-land holdouts.

The five fresh round5 images cover Yusufeli, Nam Theun1 and Alice Springs.
They contain442.40ha scored water and27,709.76ha land. These are correlated,
selected weak-reference observations, not a global accuracy sample.

| Frozen candidate on the same fresh support |Precision|Water recall|Land FPR|
|---|---:|---:|---:|
| Previous round4 terrain+prior |100%|5.39%|0%|
| Round5 radar-only, validation-selected diagnostic |27.18%|74.14%|3.17%|
| Round5 terrain |96.02%|72.17%|0.0478%|
| Round5 terrain+prior |99.95%|75.49%|0.000577%|

The round5 terrain+prior variant reached99.96% precision and92.28% water recall
at Nam Theun1, compared with4.33% recall for the previous variant. It predicted
no false water on the14,086.80ha Alice Springs land reference. Yusufeli remains
difficult: its recall is19.73%, and even the radar-only variant recalls51.17%.
The variant is retained for comparison, not promoted because it looks better
on these tests. To reproduce its frozen operating point, use
`outputs/round5/frozen/terrain_prior`, not the unfrozen-fit default0.50 threshold.

On350ha of current reference water with JRC occurrence<=10%, recall rose from
0.64% for the previous prior variant to69.26% for the new one. Low historical
occurrence is not itself proof of a recent construction date or a cause.
Official history can provide separate context: [DSI records Yusufeli filling
from November2022](https://dsi.gov.tr/Haber/Detay/10078), while
[the Nam Theun1 operator documents its project](https://www.nt1pc.com/about-us).
Hidroituango's2018 reservoir filling must not be confused with its2022
generation start; [EPM's root-cause report](https://www.epm.com.co/content/dam/epm/institucional/documentos/todos/skava-informe-causa-raiz-fisica.pdf)
describes the2018 filling. No automatic human-impact attribution is made.

## Review aid: useful unknowns, not hidden errors

Before the last test release, a fixed additional policy was declared using the
round5 validation-selected classifier and already frozen round4 prior model.
Only agreement is classified0/1. Disagreement or unavailable support becomes254,
displayed amber using the existing uncertainty layer and `review-composite`.
The score raster is deterministic binary agreement, not a probability.

This review aid is explicitly excluded from primary family selection and fails
the full acceptance gate. On the fresh test, it classified95.76% of the full
reference but recovered only18.40ha of442.40ha water;315.04ha of water was
withheld and other water was missed where both models agreed incorrectly.
The exact per-region and full-reference counts are in
`outputs/round5/agreement-test-report.json`. Conditional100% precision is not
a statement that it detected all water. Agreement is not independent
confirmation, and correlated errors remain possible. Use it to prioritize
human review, never to hide false negatives or claim global accuracy.

## Headless integration and all observations

The existing `pila`, HDF5 AOI, NetCDF and catalog inputs remain supported.
Outputs include WGS84 transparent water/border/uncertain PNGs, native UTM COGs,
GeoJSON, exact acquisition identity, areas, unknowns and same-geometry temporal
comparisons on common classified support. Physical models also export
`context-provenance.json`, and output identities bind the actually used
ancillary source hashes rather than cache-hit flags or arbitrary cache paths.

The latest read-only service uses `outputs/round5` on port8014. `/health` is
service availability; `/water/assessment` remains `not_ready_for_general_use`.
ModelId and series key distinguish diagnostic, candidate and review outputs.
The frontend has not been edited. Consult `docs/frontend-contract.md`.

```powershell
# Reproduce a fixed conservative candidate on every compatible query observation.
.\.venv\Scripts\python.exe -m nisar_water.cli aoi data\round5\hidroituango-aoi.geojson `
  --start 2026-10-10 --end 2026-10-27 `
  --model outputs\round5\frozen\terrain_prior --context-cache .cache\terrain-context `
  --output outputs\NEW_RUN --transfer-cap-mib 96
```

`--limit 0` is the default and means all compatible dates; a positive limit is
explicitly partial. Metadata truncation is handled by date subdivision.
Optical availability/model scores never filter an inference query. The radar
transfer budget and physical-context128MiB/region ceiling are separate. Omit
`--context-cache` to use existing local sources only; missing physical support
then remains unknown rather than fabricated land. The current profile supports
bounded UTM AOIs/current L-GCOV HH/HV, not arbitrary complete global frames or
other sensor profiles.

The separate Hidroituango cloud-period demonstration is unscored inference,
not an accuracy claim. GERD's two-date exported evolution is exposed training
regression, not a fresh temporal accuracy test or measured water level.

## Earth Engine and the team's independent review

The user's `nisac-512902` project was accessed with existing credentials.
Matching dated harmonized Sentinel-2 sources were queried for all13 original
and five new review packets. Two actual EE RGB crops were retrieved for the
old Denver/Powell packets; the new packets reuse cached local optical imagery.
No credential was displayed/copied into code, and no Drive export was started.
The [harmonized collection](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
uses DN*0.0001 without another BOA offset.

Prefer `review/earthengine-round5/README.md` and its five fresh-region packets
for an independent reference assessment. The original13 packets are now
training-region label-quality work. The existing Code Editor script supplies
dated optical imagery and case-bound drawing layers without model predictions.
The reviewer can prepare and manually start a GeoJSON export. A versioned
`import-review` command validates source identity, attribution and polygons;
unlabelled space remains unknown, and frozen experiments are never rewritten.
Expert declaration is not externally verified certification.

The next useful work is independent recent-reservoir/narrow-valley annotation,
correctly calibrated noise/incidence features and causal past-date dynamics.
More training or a stronger static prior alone does not establish a solution.
Farms, deforestation and wildfires still need separate labelled tasks.

A post-test diagnostic (`evidence/recent-water-post-test-diagnostic.json`) found
median HV backscatter around-17.0 and-14.5dB in Yusufeli's screened water pixels,
versus-21.9dB at Nam Theun1. This measures a difference at matched radar/weak-
reference pixels; it does not establish wind, noise, layover, RTC or registration
as the cause. Those hypotheses need native geometry/calibration inspection.
No noise LUT was blindly subtracted from gamma0 and no test-driven model change
was made after the diagnostic.

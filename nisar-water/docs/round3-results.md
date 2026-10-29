# Round 3: real training, frozen comparison and unseen failures

The headless pipeline works, but **none of the compared models is ready for
general use**. Broader training corrected many missed-water errors. Independent
dry-land and shoreline tests exposed serious remaining false-water errors.
The selected research checkpoint is `models/round3/selected` (HGB, threshold0.10).
It was selected on validation and was not replaced after seeing test outcomes.

## Open the executed notebook

`notebooks/NISAR_Water_Round3_Comparison.ipynb` has16 executed code cells,
42 saved outputs and no errors. Run `./launch-notebook.ps1` with the installed
NISAR Water kernel. Its default is cached actual data: no satellite requests.
It reruns inference, verifies cached prediction parity, constructs the original
lab-style `pila`, exports maps, compares dates, and calls the real HTTP API.
The older exploration notebook and all earlier checkpoints are preserved.

## Data and separation

The immutable fitting manifest is `data/round3-prepared/manifest.json`:
28 training scenes across16 geographic groups, five validation scenes across
three groups, and seven original test scenes across four groups. Entire water
bodies remain in one partition. A conservative geodesic AOI separation audit
found a minimum cross-partition lower bound of130.47km, including the original
Oroville, Iowa and Tapajós training controls. Fitting never opened test NPZ paths.

New weak supervision uses dated Sentinel-2 within24hours. Water requires SCL6,
NDWI>0 and MNDWI>0.1. Land requires SCL4/5 and NDWI<0 or MNDWI<0. Clouds, optical
shadows, disagreement and unlabelled pixels remain255/unknown. Strong interiors
have weight1 and weak consensus boundaries0.3. Scoring uses40m-eroded interiors;
boundary agreement is a separate weak diagnostic, not expert shoreline accuracy.

Original exposed Mead/Texoma/Okeechobee failures are explicitly training and
regression data in this round. They are not newly unseen tests. Development QA
was performed by the assistant, not an independent expert. There are zero
completed expert shoreline reviews. Thirteen blinded reference packets are in
`review/round3`; unannotated space remains unknown when polygons are imported.

The original seven test crops proved highly water-dominated:13,374.16ha of
scored water and just3.12ha of land. That reference cannot establish specificity.
Two prospective frozen-model hard-negative images and four prospective mixed
shoreline images were then collected and tested. These later cohorts were
registered separately, outside fitting and selection. All13 scored images across
eight unseen regions are retained. Uyuni had no usable clear optical support;
Jamuna's candidate references were cloudy or its radar crops had no native
coverage. All attempts remain in the corresponding collection receipts.

## Inputs and a calibration issue caught before fitting

Radar bands remain HH/HV gamma0 linear power until after aggregation. Native10m
data use2x2 power averaging with all four native samples required. Invalid source
data do not become land. The audit found no band-order or dB conversion defect
explaining the earlier model's poor recall. Broader water interiors have very
different context statistics from the old near-shore training examples.

New acquisitions use a320m context halo to support31x31 features at20m. Old80m
crops have a15-cell supervision inset. Historical receipts contain a legacy
hardcoded80m provenance sentence even when the acquisition identity says
`round3-gcov-power-halo320m-v1`; the actual crop bbox and preprocessing identity
are authoritative. The wrapper now records the effective halo and preserves
the reader's original sentence separately. Frozen historical inputs are intact.

During new label preparation, Earth Search optical COG metadata was found to
describe an offset that had already been applied to the pixels. The item-level
`earthsearch:boa_offset_applied` flag now prevents double subtraction. Rejected
preparations were preserved and never fitted. This is a new optical-label
preparation correction, not a retroactive explanation for the earlier radar
benchmark failures. See the provider's [offset issue](https://github.com/Element84/earth-search/issues/71).

Incidence metadata is a height-dependent cube, not a ready terrain incidence
image. The inspected noise-equivalent LUT is described in DN². Neither was
blindly applied to gamma0. Correct interpolation/calibration and terrain/shadow
handling remain scientific extensions to evaluate, not completed capabilities.

## Actual models and validation decision

One fixed HGB fit used43,264 water and39,884 land samples, at most2,048 per class
per scene, equal geographic-region total weights, and retained boundary weights.
The model uses41 dual-polarization multiscale features,300 iterations,31 leaves,
learning rate0.05, minimum leaf40 and L2=10. Early stopping was disabled.

One small U-Net was trained from scratch on actual NISAR with1,084,465 parameters,
20 CUDA epochs,128-cell patches, batch8, AdamW0.001 and masked BCE+0.5Dice.
Best epoch20 minimized macro validation-region loss. Normalization is fitted
only to training images; there is no per-image brightness normalization.
This is not a claimed pretrained Sentinel-1 fine tune. The existing Torch2.8
CUDA runtime was reused. Main-environment neural inference uses its CPU bridge;
the selected HGB needs no Torch installation.

Published comparison scores use the CPU bridge consistently for U-Net.
Default cuDNN TF32 inference differs from CPU by up to0.000800 in validation
scores and changes five near-threshold pixels across five scenes. A separate
full-float32 CUDA check (TF32 disabled) agrees with CPU within0.0000071 with
zero label differences at0.10. This numerical check neither refits the model nor
uses test images. Exact bit parity across numerical backends is not claimed.

OPERA uses the six unmodified official NI open-water stages at source commit
`d9ae8ed5178417dec22a6fc549c4368cfd51ebe9`, scientific defaults, real ancillary
data, and30m average linear-power input. This crop adapter omits full-frame
mosaicking/MGRS publication and disables inundated-vegetation processing.
It is not an official DAAC-validated product. See `docs/round3-opera.md`.

The declared threshold grid was0.1..0.9. Selection preferred a passing weak gate,
then maximum worst-region F1, then macro-region F1. Region targets were90%
precision/recall,4% land FPR when at least4ha of reference land exists, and98%
reference coverage. Rare land support was reported rather than treated as a
validated specificity estimate. HGB at0.10 won; U-Net's comparison operating
point is0.10; OPERA is binary. All three were frozen before the original test.

| Validation family | Worst region F1 | Gate |
|---|---:|---|
| HGB |0.962|Passed on available weak support|
| U-Net |0.948|Passed on available weak support|
| OPERA crop adapter |0.000|Failed: Constance water missed|

`outputs/round3/selection-plan.json`, `validation-report.json` and `freeze.json`
record the decision, hashes and chronology. No test-guided threshold, refit or
family replacement followed. Raw classifier scores are uncalibrated.

## Unseen measurements

All values below are agreement with retained weak interiors. Each cohort is
reported separately in `test-report.json`, `challenge-report.json` and
`shorelines-report.json`; per-scene counts, class support and withheld area are
included. All retained reference interiors were classified in these runs.

| Example, pooled dates within region | HGB | U-Net | OPERA crop adapter |
|---|---:|---:|---:|
| Malawi shoreline precision / recall |97.64% /99.98%|99.66% /100.00%|99.68% /99.92%|
| Powell shoreline precision / recall |12.94% /96.28%|13.18% /72.65%|49.89% /98.01%|
| Denver airport land false-positive rate |75.15%|65.31%|0.00%|
| Andes land false-positive rate |1.14%|0.00%|0.00%|

Across all13 images the area-weighted values are:

| Family |Precision|Recall|IoU|Land false-positive rate|
|---|---:|---:|---:|---:|
|HGB|72.15%|99.81%|72.05%|42.46%|
|U-Net|77.19%|98.77%|76.46%|32.15%|
|OPERA crop adapter|94.31%|74.70%|71.48%|4.97%|

These are descriptive pooled results over selected regions with correlated
pixels/dates, not worldwide accuracy. The learned models recover water well
but confuse smooth dry land with water. The OPERA adapter suppresses much of
that false water, but misses large open-water crops. All three struggle at
Powell. Changing network architecture alone did not solve the central problem.

On the old exposed regression support, HGB water recall rose from11.33% to95.22%
at Mead, from98.57% to99.90% at Texoma, and from0% to100% at Okeechobee.
Mead precision is98.80%, but its land FPR is4.15%. These are training-region
regression checks and must not be presented as independent validation.

## Working headless delivery

The selected checkpoint accepts the original `pila`, local HDF5 AOI windows,
NetCDF and crop catalogs. Outputs preserve the team's existing integration
contract: transparent WGS84 water/border PNGs, UTM COG mask/score, GeoJSON, exact
acquisition metadata, areas, unknowns and model identity. The read-only result
service runs at `http://127.0.0.1:8013`. Read `/water/assessment` alongside maps;
service `/health` being ready does not mean the classifier is scientifically ready.

The new `aoi` command discovers and processes every compatible observation in
a bounded date/AOI query, without optical/cloud or model-score date selection:

```powershell
.\.venv\Scripts\python.exe -m nisar_water.cli aoi data\round3\tahoe-aoi.geojson `
  --start 2026-06-04 --end 2026-06-17 --model models\round3\selected `
  --output outputs\my-tahoe-run --transfer-cap-mib 128
```

`--limit 0` (default) selects all compatible observations. A positive limit,
transfer exhaustion or acquisition failure is explicit in the batch report.
Truncated CMR searches are split by date rather than silently losing records.
Different tracks/passes/frames are separate series. Authentication uses the
existing current-user encrypted helper. No plaintext token is in the notebook.

The real demonstration in `outputs/round3/archive-demo` acquired and processed
all five compatible observations, grouped into three geometries, using83MiB of
radar range bodies. These additional dates are unscored inference examples.
They do not add accuracy claims. Across the13 evaluated images and five-date
archive demonstration,18 actual export sets passed raster/vector/area/hash/HTTP
checks. Common-support temporal comparisons exclude unknown/invalid cells and
do not use future-date smoothing. Extent alone does not establish a human cause.

## Verification and remaining work

36 tests passed in the main environment, and seven neural tests passed in the
existing Torch environment. The Torch module is skipped only in the main
environment without Torch. Tests cover physical conversion, validity, no test
reads during fitting, group sampling, tiled inference, masks, source-host
restrictions, seam-crossing ancillary enumeration, unknown annotations,
truncated discovery and explicit transfer exhaustion. The actual executed
notebook, HTTP and18-output checks are separate evidence from software fixtures.

The next model experiment needs independent labels for dry soil, airports,
shorelines and shadows; new negative training/validation geographies; physically
valid terrain/incidence context; and another untouched geographic test. Test
causal previous-date persistence without erasing real new water. Historical
water may help as a soft prior, but a hard historical mask would conceal new
inundation. Validate abstention/calibration on independent data. Do not claim
expert shore accuracy, global deployment readiness, wildfire classification,
deforestation classification or human-impact attribution from this experiment.

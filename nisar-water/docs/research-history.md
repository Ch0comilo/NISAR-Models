# Historical workspace README

This preserves the earlier experiment narrative and commands. Paths and live
service ports below refer to the original local research workspace; excluded
data, notebooks and results are not part of a minimal Git clone. Use the current
[repository README](../README.md) for installation and inference.

A headless water classifier and time-series pipeline for the NASA Space Apps
challenge. It accepts scientific NISAR data and produces blue water fills, cyan
boundaries, transparent globe overlays, vectors, scientific rasters and temporal
metrics. There is no application UI. The other projects are unchanged.

## Latest: terrain, recent reservoirs and Earth Engine review

Open **`notebooks/NISAR_Water_Improved_Classifier.ipynb`** with
`./launch-notebook.ps1`. It contains13 executed code cells,43 saved outputs and
no errors. The earlier notebooks remain preserved. See the
[full results and limitations](improved-classifier-results.md).

Two further experiments tested12 new images, then five fresh reservoir/airport
images. Round5 fits share203,678 real training samples and compare radar-only,
terrain, and terrain+historical-prior features. On the same five fresh images,
the new conservative prior variant improved water recall from5.39% to75.49%
relative to its previous version, with99.95% precision. Nam Theun1 recall is
92.28%; no false water was predicted on the Alice Springs land reference.
Yusufeli remains substantially underdetected. All scores are weak-reference
agreement, not independent field or global accuracy.

**Still not ready for general use.** No round5 candidate passed every validation
target; the validation-selected diagnostic family remains radar-only. All frozen
variants are retained in `outputs/round5/frozen`. No better-looking test variant
was silently promoted. A separately frozen review aid paints disagreement amber
and reports withheld water/land explicitly; its conditional precision must not
be mistaken for complete water detection.

The headless output/API contract remains compatible. The latest read-only result
service is on port8014. Current physical source hashes accompany its maps.
Use `--context-cache .cache/terrain-context` to explicitly enable bounded public
ancillary fetching for a new AOI. The Hidroituango cloud-period demonstration
processed both compatible radar observations without an optical filter; it is
unscored inference, not accuracy validation. No application UI was added.

The Earth Engine project `nisac-512902` is accessible through the existing
credential. Your team can start with
[the five fresh-region review packets](../review/earthengine-round5/README.md).
The Code Editor helper shows dated optical imagery without predictions and
prepares unstarted GeoJSON tasks for the reviewer. `import-review` versions
returned annotations while preserving frozen experiments. No expert-reviewed
labels have yet been imported into these fits.

## Historical round3: broader training and thirteen unseen comparisons

Open **`notebooks/NISAR_Water_Round3_Comparison.ipynb`** with
`./launch-notebook.ps1`. All16 code cells executed successfully, with42 saved
outputs. The original exploration notebook is preserved below as history.

I trained a broader HGB classifier and a1.08-million-parameter U-Net on actual
NISAR, compared them with pinned official OPERA NI stages through a crop adapter,
and froze the validation-selected HGB at threshold0.10 before the original test.
The fitting split has28 train, five validation and seven test scenes; additional
frozen-model stress tests bring the unseen evaluation to13 images/eight regions.

**Still not ready for general use.** Malawi shoreline precision/recall is
97.64%/99.98%, but HGB labels75.15% of clear airport reference land as water and
has12.94% precision at Powell. U-Net also fails there; OPERA suppresses much false
water but misses substantial water elsewhere. No model was reselected after
these results. Read [the results and limitations](round3-results.md).

Use `--model models/round3/selected` for the current **research** checkpoint.
The headless exports and read-only API on port8013 are verified; the UI must read
`/water/assessment` and display `not_ready_for_general_use`. No frontend changes
were made. See [the integration contract](frontend-contract.md).

The new `aoi` command processes every compatible observation in a bounded query
without requiring optical imagery. Its actual Tahoe demonstration processed
all five observations;18 real observation export sets passed hash/COG/vector/
area/HTTP checks. The notebook supports the original lab's `pila` structure,
fresh inference, temporal plots and actual HTTP calls. There are43 passing
software tests across the two existing Python runtimes. Thirteen unreviewed
reference packets are prepared in `review/round3` for independent annotation.

```powershell
.\.venv\Scripts\python.exe -m nisar_water.cli aoi data\round3\tahoe-aoi.geojson `
  --start 2026-06-04 --end 2026-06-17 --model models\round3\selected `
  --output outputs\my-tahoe-run --transfer-cap-mib 128
```

## Historical eight-image exploration notebook

Open **`notebooks/NISAR_Water_Exploration.ipynb`** using the installed
**NISAR Water (Python 3.13)** kernel. All 14 code cells have been executed, with
saved outputs and no errors. It follows the original lab notebook's API/data
workflow and includes all eight newly tested images, selectable locations,
scientific crop verification, a compatible `pila` cube, fresh selected-case
inference, colored maps, animations, time-series area, reference errors,
input-distribution diagnostics, exports and actual local API calls.

```powershell
cd 'C:\Users\danie\Documents\NISAC 2\nisar-water'
.\launch-notebook.ps1
# Restore the notebook environment if needed:
uv sync --frozen --extra test --extra notebook
```

Default mode uses cached actual inputs and performs no satellite downloads.
Set `MODE='online'` and choose AOI/dates to use the existing bounded reader and
protected Earthdata helper. The supplied credential was stored with current-user
Windows DPAPI; it is absent from notebook cells, outputs and source files. No
full multi-GB NISAR frames are downloaded. The new batch used 136 MiB of radar
ranges and 22.71 MiB of public optical-reference ranges.

**Historical v2 readiness: not ready for general use.** Before creating that notebook,
the frozen v2 was applied to two new dates each at Mead, Texoma, Gatun and
Okeechobee. Of eight real radar acquisitions, three have usable dated optical
reference support. On that support, water recall is **11.3% at Mead, 98.6% at
Texoma and 0% at Okeechobee**. Four dates lack an optical acquisition within
24 hours; Gatun's available reference is fully cloudy. These failures outweigh
any interpretation of the earlier single-region success as global validation.
The checkpoint was not changed. See `docs/new-images-benchmark.md` and
`outputs/new-images-20261028/assessment.json`.

The new result service is available at port **8012**; previous result services
were restored at 8010 and 8011. Its `/water/benchmark` and `/water/assessment`
endpoints expose the measured limitations. Twenty-three software tests passed,
all eight export sets were verified, and notebook delivery checks found no
plaintext JWT in the scanned source, notebook, receipts or outputs.

## Historical spatial v2: improvement and genuinely unseen evaluation

Use **`--model models/spatial-v2/selected`** to reproduce that water segmentation
profile. It retains the original fitted context classifier and adds a spatial
decision: four connected high-score seed pixels (score ≥0.90) can grow into
connected candidate water (score ≥0.03). Isolated low-score regions remain land.
The entire cropped observation is considered, so feature tiles do not terminate
connected water. No future date, optical reference or test label enters inference.

One additional, larger weighted classifier fit was compared, but its development
results did not pass the joint targets. The original classifier with the new
spatial decision was selected using **development data only**, and frozen before
the reserved test pixels/reference were opened. Development precision/recall are
58.07% / 71.08%, meeting the declared joint targets. This trades some precision
for recall; its development F1 is 0.639 versus 0.677 for the earlier conservative
threshold. It is not a uniform improvement on every development metric.

The one-time unseen test was **Lake Red Rock, Iowa, 2026-10-19**. Its acquisition
and geography were absent from fitting and selection. The nearest retained
training and validation cells are 72.69 km and 71.90 km away, respectively.
The dated Sentinel-2 reference is 5.84 hours after the radar acquisition.

| Same previously unseen reference support | Previous v1 | Spatial v2 |
|---|---:|---:|
| Precision | 99.64% | **98.64%** |
| Water recall | 42.85% | **80.90%** |
| F1 | 0.599 | **0.889** |
| Water IoU | 42.78% | **80.01%** |
| False-water area | 0.20 ha | **1.44 ha** |
| Missed-water area | 73.84 ha | **24.68 ha** |

These metrics cover **376.44 ha / 9,411 retained cells**, with 129.20 ha of
reference water. They measure agreement with conservative, dated SCL weak-label
interiors, not human-reviewed shoreline truth. Radar observes 763.41 ha of the
3,573.50 ha requested AOI; unobserved areas are not invented or scored. Cloud,
shadow, unknown, heterogeneous boundaries and unsupported cells are withheld.
This is one new region and one date, not evidence of global accuracy. The model
was not refitted or reselected after these results.

View `evidence/unseen-red-rock-v2.png` for the complete radar-covered strip,
dated optical context, old/new maps and remaining errors. Full native/web
outputs and the machine-readable report are in `outputs/unseen-red-rock-v2/`.
The freeze and experiment reports are `models/spatial-v2/freeze.json` and
`models/spatial-v2/comparison.json`. See `docs/spatial-v2-results.md` for full
provenance and reproduction. Nineteen software tests and both actual Red Rock
export/API checks passed.

## Historical v1 result

A real classifier was fitted to **6,774 water and 77,232 nonwater cells** from your
existing NISAR development data. It uses 41 HH/HV features, including physical
power averages, contrast, texture and spatial context at 60–620 m. Inference needs
only the current radar observation. No location coordinates, optical image,
historical water polygon or later observation supplies the prediction.

The following classifiers were compared on the **same 118,068 validation cells**
(1,058 water, 117,010 nonwater). These are comparisons against screened, dated
Dynamic World/Sentinel-2 weak labels in known regions. They are not independent
field accuracy, global validation or measured shoreline accuracy.

| Classifier | Threshold | Precision | Water recall | False-water area | Missed-water area |
|---|---:|---:|---:|---:|---:|
| Previously active RF | 0.30 | 2.35% | 79.30% | 1,392.24 ha | 8.76 ha |
| Previous failed RF correction v9 | 0.30 | 13.16% | 67.01% | 187.12 ha | 13.96 ha |
| New HGB, old local features | 0.80 | 42.20% | 40.64% | 23.56 ha | 25.12 ha |
| **New HGB with spatial context** | **0.70** | **81.03%** | **58.13%** | **5.76 ha** | **17.72 ha** |

This is a substantial reduction in false water, with an unresolved recall cost.
The selected spatial model still fails the declared joint requirement: overall
precision ≥50%, water recall ≥70%, and nonwater false-positive rate ≤4% in every
validation observation. **Its status is `experimental_acceptance_failed`.**
At threshold 0.10 recall rises to 70.13%, but precision falls to 46.09%.
No threshold in the frozen grid passed. Every classified validation cell enters
these metrics; uncertainty masks do not hide classifier errors.

Two fixed model fits were run, with no randomized pixel split or test tuning.
Train/validation geographic groups were checked for overlap and ≥140 m retained
cell separation. The previous exposed test and reserved Lake Red Rock test labels
were not read or scored. See `evidence/same-support-comparison.json`,
`models/context-hgb-v1/comparison.json` and `models/context-hgb-v1/plan.json`.

## Run locally

The environment and checkpoint are already available here. From PowerShell:

```powershell
cd 'C:\Users\danie\Documents\NISAC 2\nisar-water'
# Restore dependencies if needed:
uv sync --frozen --extra test

# Process every compatible observation in an existing crop catalog.
# A fresh output directory is required; previous results are preserved.
.\.venv\Scripts\nisar-water.exe catalog '..\nisar-landcover-phase1\data\catalog.json' `
  --model models\context-hgb-v1 --output outputs\my-batch

# Optional: repeat --observation EXACT_ID to restrict the run.
# The command groups different frames/AOIs into separate time series.
```

That catalog includes geography reserved for evaluation. To preserve it for a
future independent experiment, select development observations explicitly, as in
the verified pilot rather than processing that entire research catalog.

`outputs/pilot-v3/` contains seven actual observations: three two-date series
(Oroville reservoir, Iowa agriculture and Tapajós forest control), plus a Folsom
acquisition outside the fitting observations. Folsom tests input/inference/export
compatibility; its accuracy was not independently evaluated. The reservoir
series is `504e53bdc8433438`. Open its `water-evolution.gif`, individual
`composite.png` maps and `timeline.csv`. Changes there remain experimental model
estimates, including temporal model errors; they are not measured water levels.

## Use the notebook's exact data structure

Your notebook's `construir_pila(...)` returns an xarray Dataset named `pila`.
Request both channels and a 10 or 20 m input grid (`multilook=1` for native 20 m,
or `multilook=2` for native 10 m). Retain `epsg`, `frecuencia`, `descripcion`,
`aoi_wkt`, `time`, `nombre`, `x` and `y`. In a kernel with this package installed:

```python
from nisar_water.pipeline import predict_dataset

result = predict_dataset(
    pila,
    model_dir=r'C:\Users\danie\Documents\NISAC 2\nisar-water\models\context-hgb-v1',
    output_dir=r'C:\Users\danie\Documents\NISAC 2\nisar-water\outputs\notebook-run-01',
    series_id='my-river',
)
```

For an unchanged notebook environment, export `pila` through NetCDF instead:

```python
pila.to_netcdf('water-input.nc', engine='scipy')
```

```powershell
.\.venv\Scripts\nisar-water.exe netcdf 'C:\path\water-input.nc' `
  --model models\context-hgb-v1 --output outputs\notebook-run-02
```

The package does not open pickle caches. Use the notebook's existing trusted
cache loader to obtain `pila`, then pass it in memory or export NetCDF. Notebook
multilooking can average blocks with some missing samples; this inherited mask
policy is recorded, and cannot be reversed after aggregation. For strict native
QA, prefer the HDF5 or verified crop adapters.

## Local GCOV HDF5

```powershell
.\.venv\Scripts\nisar-water.exe hdf5 'C:\data\NISAR_L2_PR_GCOV_....h5' `
  --aoi 'C:\data\river-aoi.geojson' --model models\context-hgb-v1 `
  --output outputs\hdf5-run-01
```

The reader requests an AOI window with 320 m context, rejects oversized crops,
checks product/band identification and native validity, and never loads an entire
multi-GB frame. The input may also be a directory: every local GCOV HDF5 is
processed, with different tracks/directions/frames placed in separate series.
Select one processing revision per acquisition; duplicate acquisition times are
rejected rather than silently double-counted. Use the notebook's existing authenticated ASF streaming workflow
for online acquisition, then pass its `pila`. This package does not schedule new
satellite downloads or process the worldwide archive automatically.

The present fitted profile supports **NISAR L-band GCOV frequency A HH/HV**,
native 10/20 m north-up UTM grids, with 20 m analysis. It explicitly rejects
VV/VH substitution, frequency B, dB/amplitude inputs, unsupported products,
coarse-to-fine upsampling and mixed-track time series. Other NISAR acquisition
modes need a separately trained profile. Grid spacing is not radar resolution.

## Outputs and frontend service

Each observation exports:

- `sar.png`, `composite.png`: fixed −30 to +3 dB radar image, with/without water.
- `water.png`, `border.png`: transparent blue fill and cyan shoreline overlays.
- `uncertain.png`: insufficient texture support or explicitly excluded geometry.
- `water-mask.tif`: COG codes 0 nonwater, 1 water, 254 uncertain, 255 unobserved.
- `water-score.tif`: COG with the uncalibrated water score; NaN outside observation.
- `water.geojson`: WGS84 polygons, preserved holes, projected areas/perimeters.
- `aoi-fraction.tif`, `result.json`: fractional AOI coverage and provenance.

Web PNGs are actually reprojected to an EPSG:4326 grid and accompanied by
`bounds: [west,south,east,north]`. Native UTM rasters retain their exact transforms.
Source outside the AOI and nodata remain transparent.

Each series exports a manifest, CSV, water occurrence, classified observation
count and consecutive gain/loss COGs. Gains/losses use only shared classified
pixels, preventing missing radar coverage from masquerading as disappearing
water. The CSV also reports water on the intersection shared by **all dates**.
No observed water is carried forward into missing dates. Animations are generated
when all display bounds agree. Occurrence means fraction of classified supplied
acquisitions; it is not a time-weighted fraction of days or a permanent-water map.

```powershell
.\.venv\Scripts\nisar-water.exe serve --root outputs\pilot-v3 --port 8010
```

The API lists results at `GET /water/series`, serves one manifest at
`GET /water/series/{key}`, and provides its assets at `/assets/...`.
Default CORS supports local Vite on port 5173. It serves saved results; use the
CLI/Python worker for new inference. `docs/frontend-contract.md` includes strict
TypeScript types and a Cesium example for your team's UI.

## Reproduce and verify

```powershell
.\.venv\Scripts\nisar-water.exe train --source '..\nisar-landcover-phase1' `
  --output models\context-hgb-reproduction-01
.\.venv\Scripts\python.exe scripts\compare_previous.py --source '..\nisar-landcover-phase1' `
  --model models\context-hgb-v1 --output evidence\comparison-reproduction.json
.\.venv\Scripts\python.exe -m pytest -q --basetemp evidence\tests-NEW
.\.venv\Scripts\python.exe scripts\verify_outputs.py outputs\pilot-v3 `
  --report evidence\verification-reproduction.json
```

`uv.lock` pins the environment. Load only locally produced trusted joblib model
files. Fourteen software tests and seven actual-observation export/API checks passed.
The tests cover power aggregation, mask semantics, tile seams, temporal support,
notebook/HDF5 adapters, polygon holes, publication URLs and overwrite protection.
Actual-data checks additionally reconcile polygon/raster area, verify all asset
hashes and COG layout, and fetch all advertised assets through HTTP.

## Recommended next work

1. **Benchmark OPERA DSWx-NI on the same NISAR scenes.** Its maintained NISAR
   workflow explicitly includes preprocessing, fuzzy membership, region growing,
   ancillary masking and refinement. Its top-level README still says only
   Sentinel-1, while the repository contains a NISAR workflow; verify the chosen
   release and run configuration before deployment. It was reviewed, not installed
   or evaluated in this run. It is a stronger next comparator than another
   darkness threshold. Source: [official NISAR workflow](https://github.com/opera-adt/DSWX-SAR/blob/main/src/dswx_sar/dswx_ni.py).
2. **Collect independently reviewed shorelines and difficult negatives across
   regions/dates.** Include dark fields, smooth bare ground, mountain shadows,
   urban shadows, wind-roughened water, narrow rivers and reservoir margins.
   Homogeneous weak interiors leave the exact border mostly unvalidated.
3. **Compare a small spatial segmentation network after those labels exist.**
   Use an encoder/decoder with a dedicated NISAR HH/HV input adapter and fixed
   NISAR normalization. A Sentinel-1 pretrained encoder is a transfer-learning
   candidate, not a ready NISAR water model. TerraMind supports S1 modalities;
   its published card does not establish NISAR performance. Do not relabel HH/HV
   as VV/VH to satisfy an input shape. Source: [TerraMind model card](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny).
4. **Add geometry-aware exclusions and hydrological context.** The Python
   `Observation.exclusions` mask can withhold independently supplied shadow/layover
   pixels. Keep these unknown, and report excluded area. A DEM slope screen alone
   is not a complete radar geometry mask. Historical water can be a soft prior;
   it must not prohibit new water or floods.
5. **Add separate vegetation/fire specialists after water validation.** Reuse
   the observation/overlay/time-series contract and spatial test discipline.
   A water gain/loss pixel alone cannot attribute change to farming, dam operation,
   drought or human activity. Add dated land-use, weather and operational evidence
   before making those interpretations.

ASF documents that GCOV is RTC gamma0 **power** and explains why NISAR L-band and
Sentinel-1 C-band do not observe vegetation and soil identically:
[NISAR amplitude guide](https://nisar-docs.asf.alaska.edu/amplitude/).


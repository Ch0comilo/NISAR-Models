# Spatial v2 and the first unseen-location evaluation

The selected checkpoint is `models/spatial-v2/selected`. Use it with the existing
`catalog`, `netcdf`, `hdf5` and Python pipeline inputs. Its fitted tree weights are
identical to `context-hgb-v1`; its segmentation decision is new. A larger weighted
tree model was actually fitted and evaluated on development data, but rejected.

The chosen spatial policy starts with four-connected seed components of at least
four pixels, each with raw water score ≥0.90, then grows through four-connected
pixels with score ≥0.03. Unsupported, excluded and unobserved pixels block
growth. No hole filling or temporal carry-forward is applied. It can recover
weak water connected to a confident water body while rejecting unseeded dark
areas. It can also expand an incorrect seed; the development tradeoff below and
remaining test false alarms quantify that limitation on labelled support.

The pre-fit plan declared one additional HGB fit and 37 decision policies per
classifier: 19 simple thresholds, plus 18 combinations of seed threshold, growth
threshold and seed support. These were selected on the existing development
geographic split, with acceptance requiring overall precision ≥50%, recall ≥70%
and nonwater false-positive rate ≤4% in each validation observation. Among
feasible candidates, maximum F1 selected the final policy. The new larger tree
model had no feasible policy. Three spatial decisions using the old classifier
were feasible. The selected one achieved precision 58.07%, recall 71.08%,
false-positive rate 0.464%, F1 0.639 and IoU 0.470 on development. The maximum
per-observation nonwater false-positive rate was 1.372%. False-water and
missed-water areas were 21.72 ha and 12.24 ha. Compared with the previous default,
this recovers more water with more false alarms and a lower development F1.

## Evidence that the test was unseen

Lake Red Rock's allocation predates this experiment and states that geography
was chosen without model scores. Its exact observation ID is
`explore-20261019T112151Z-497081d067e5`, track 105 ascending, frame 22. Neither
that observation nor its retained physical cell locations occur in development.
Minimum retained-cell distances to training and validation are 72,689.67 m and
71,898.62 m. The audit transforms locations across UTM zones before checking;
it does not rely just on different observation names.

`models/spatial-v2/freeze.json` was written at 2026-10-28 18:18:46 UTC, before
`outputs/unseen-red-rock-v2/support-freeze.json` at 18:20:04 UTC. The second
receipt froze the optical support before any heldout prediction. The selected
checkpoint and model metadata hashes were checked before and after inference.
No model fit, threshold choice or policy selection followed the test result.

The NISAR acquisition is 2026-10-19 11:21:51 UTC. The reference is Sentinel-2
`S2A_15TVF_20261019_0_L2A`, acquired at 17:12:31 UTC, 5.8446 hours later.
SCL 6 supplies water; SCL 4/5 supplies explicit nonwater. Two iterations of
four-connected erosion on the 20 m reference grid retain homogeneous interiors
at least 40 m from class transitions. Cells must be fully inside the AOI and
have valid radar/texture support. Clouds, cloud shadows, snow, unknown and
unlabelled cells never become negatives. The reference is machine-generated
weak supervision, not human-reviewed ground truth.

The source AOI is 3,573.50 ha. Radar observes 763.41 ha, of which 737.46 ha has
feature support. Conservative evaluation retains 376.44 ha: 3,230 water cells
(129.20 ha) and 6,181 nonwater cells (247.24 ha), distributed over 31 occupied
600 m blocks. These cells and blocks are correlated; they are not 9,411
independent trials. Source collection/processing metadata, hashes, reference
source, support arrays and missing-area accounting are saved with the report.

## Frozen test results

| Metric on identical reference support | Earlier v1 | Spatial v2 |
|---|---:|---:|
| Precision | 0.9964 | 0.9864 |
| Water recall | 0.4285 | 0.8090 |
| F1 | 0.5993 | 0.8889 |
| Water IoU | 0.4278 | 0.8001 |
| Nonwater false-positive rate | 0.0809% | 0.5824% |
| Correctly recovered water | 55.36 ha | 104.52 ha |
| False-water area | 0.20 ha | 1.44 ha |
| Missed-water area | 73.84 ha | 24.68 ha |

Both methods classify every retained reference cell. No model-specific abstention
or test-selected threshold changes their scored support. The improved pipeline
passes the overall precision/recall/false-positive targets here. These results
do not validate true boundary accuracy, global performance or interpretation of
human drivers. Approximately 19% of reference water is still missed.

The error map uses blue for correctly detected reference water, orange for
missed reference water, magenta for false water and gray for correct nonwater.
Black includes unscored regions. Optical RGB is context only; its cloudy/unknown
pixels are masked. All panels show the complete radar-covered footprint, with
the same geographic view and no model-dependent cropping.

## Reproduce

From the `nisar-water` package directory:

```powershell
# A new development experiment; never overwrite frozen evidence.
.\.venv\Scripts\nisar-water.exe improve --source '..\nisar-landcover-phase1' `
  --baseline models\context-hgb-v1 --output models\spatial-v2-reproduction

# Reproduces a now-exposed test, not a new independent result.
.\.venv\Scripts\nisar-water.exe evaluate-unseen --source '..\nisar-landcover-phase1' `
  --experiment models\spatial-v2-reproduction --baseline models\context-hgb-v1 `
  --output outputs\red-rock-reproduction

# Apply the frozen v2 to all supplied compatible catalog observations:
.\.venv\Scripts\nisar-water.exe catalog 'C:\data\catalog.json' `
  --model models\spatial-v2\selected --output outputs\new-water-series

.\.venv\Scripts\python.exe -m pytest -q --basetemp evidence\tests-NEW
.\.venv\Scripts\python.exe scripts\verify_outputs.py outputs\unseen-red-rock-v2 `
  --report evidence\unseen-verify-reproduction.json

# Serve the two frozen test result sets for the team's UI:
.\.venv\Scripts\nisar-water.exe serve --root outputs\unseen-red-rock-v2 --port 8011
```

Red Rock is now an exposed regression location. A future change needs a new
reserved region/date to support another genuinely unseen evaluation. Keep this
test out of fitting unless a separate evaluation location has first been reserved.

Conceptual reference: OPERA's maintained NISAR pipeline also includes region
growing. This implementation uses the local learned water score; it is not the
OPERA algorithm or an evaluated OPERA product:
[official NISAR region-growing stage](https://github.com/opera-adt/DSWX-SAR/blob/main/src/dswx_sar/nisar/region_growing.py).


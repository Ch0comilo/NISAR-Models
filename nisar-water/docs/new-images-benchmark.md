# Eight new images before notebook creation

The frozen `nisar-water-spatial-v2` checkpoint was tested on eight newly acquired
NISAR GCOV HH/HV observations: two dates each at Lake Mead, Lake Texoma, Gatun
Lake and Lake Okeechobee. A declared plan fixed the regions, date window,
acquisition-selection rule, model hash and reference policy before predictions.
No fitting, score-threshold choice or decision-policy change used these images.

**The broader generalization test failed.** Strong performance at Texoma and the
previous Red Rock test does not establish reliability across water regimes.
The current external model assessment is `not_ready_for_general_use`. The frozen
model metadata preserves its earlier local validation history; the newer
assessment and benchmark must accompany user-facing maps.

| Location | Radar dates in October 2026 | Retained scored date | Precision | Water recall | Water IoU |
|---|---|---|---:|---:|---:|
| Mead | 6, 18 | 6 | 73.02% | 11.33% | 10.87% |
| Texoma | 9, 21 | 21 | 99.62% | 98.57% | 98.21% |
| Gatun | 5, 17 | none | unavailable | unavailable | unavailable |
| Okeechobee | 3, 15 | 15 | undefined: no predicted water | 0.00% | 0.00% |

All eight radar observations have actual scientific crops and prediction/export
outputs, including the poor cases. The smallest geographic separation from
retained training/validation cells is 695 km at Mead; the other locations are
about 932 km, 3,179 km and 2,037 km away. Physical locations, not just names,
were checked across UTM zones. Exact granules, processing identities, native
grids and derived-file SHA256s are retained in acquisition receipts.

## What was and was not scored

The public Sentinel-2 search selected the closest acquisition within 24 hours,
preferring tiles whose footprints cover at least 80% of the AOI when available.
This choice preceded all new-image model predictions. Each downloaded SCL and
visual window has a source URL, local hash and exact HTTP-range receipt.

Four radar dates have no optical acquisition within that window: Mead Oct 18,
Texoma Oct 9, Gatun Oct 17 and Okeechobee Oct 3. Gatun Oct 5 has a reference
4.90 hours later, but its crop contains only SCL8/9 cloud classes; it supplies
zero usable binary reference support. These five images are retained as
qualitative outputs with unavailable quantitative accuracy, not treated as
negative labels or zero-error cases.

The scored dates use SCL6 water and SCL4/5 explicit nonwater. Two four-connected
erosions on the 20 m reference grid preserve 40 m homogeneous interiors. Full
AOI cells and valid radar/texture support are required. Clouds, cloud shadows,
unknown classes and mixed boundaries never become negatives. The references
are machine-generated weak labels, not human-reviewed shoreline truth.

Scored support is 2,423.36 ha at Mead, 1,867.88 ha at Texoma and 3,869.04 ha at
Okeechobee. The Okeechobee sample contains only reference water; nonwater
false-positive rate is therefore unmeasurable, and precision is undefined
because the model predicted no positive cells. It would be misleading to call
that 100% precision or zero error.

Across the pooled retained support, v2 precision is 95.16%, water recall 23.03%
and IoU 22.76%, versus v1 recall 18.24%. These are area-weighted agreement
statistics on the available samples, not worldwide accuracy. The large
Okeechobee water-only support dominates the aggregate. The notebook therefore
shows per-location metrics before pooled values.

## Acquisition and credential handling

The user-supplied credential was passed through the existing encrypted Windows
DPAPI helper. It was not embedded in source, notebook cells, saved outputs,
public-reference requests or acquisition JSON. Protected Earthdata access was
verified by successful actual data range reads. The notebook's default cached
mode does not read credentials; live mode can use the protected helper or
hidden input. Credential-format scans of the delivered notebook and supporting
text artifacts found no plaintext JWT.

The eight radar crops transferred **142,606,336 bytes (136 MiB)**. Public optical
references used approximately **22.71 MiB**. Readers require exact 206 ranges
and refuse unbounded/full-frame fallbacks. Derived crop hashes are verified;
the full remote multi-GB HDF5 checksum is not claimed.

## Notebook and reproduction

The delivered file is `notebooks/NISAR_Water_Exploration.ipynb`. It was created
after the new-image benchmark and then executed with the installed `nisar-water`
kernel: **14 code cells, 32 saved outputs, zero execution errors**. It includes
all eight water maps, selected-case HH inputs and overlays, animations,
comparable-support timelines, weak-reference errors, training/test HH/HV
distribution diagnostics, scientific downloads and local HTTP API calls.
The default location is the difficult Mead case. It recomputes that selected
case with the frozen model. Cached and online modes are distinct and explicit.

```powershell
uv sync --frozen --extra test --extra notebook
.\.venv\Scripts\python.exe -m ipykernel install --user --name nisar-water `
  --display-name 'NISAR Water (Python 3.13)'
.\launch-notebook.ps1

# The following scripts preserve existing acquisition/results receipts:
.\.venv\Scripts\python.exe scripts\test_new_images.py
.\.venv\Scripts\python.exe scripts\fetch_new_references.py
.\.venv\Scripts\python.exe scripts\benchmark_new_images.py
.\.venv\Scripts\python.exe scripts\verify_notebook_delivery.py
.\.venv\Scripts\python.exe -m pytest -q --basetemp evidence\tests-NEW
```

Use `data/new-images-20261028/plan.json` for original selection rules,
`outputs/new-images-20261028/report.json` for measured results,
`metrics.csv` for tabular comparisons, and the `measurements/` directory for
reference masks, saved predictions and support-freeze receipts. The saved
result API runs on port 8012, with `/water/series`, `/water/benchmark` and
`/water/assessment`. Its images are computed results, not demonstration data.

The parameterized notebook can query a custom AOI and dates in live mode.
Changing the input does not establish new accuracy without an independent dated
reference. Different native analysis grids remain explicit; the scientific
time-series pipeline aligns categorical maps only for support-aware change
measurements. It does not silently construct a false coincident `pila` cube.

These eight acquisitions are now exposed regression data. A future modeling
round needs diverse reviewed water/shoreline/hard-negative supervision and a new
reserved evaluation geography. Backscatter distribution differences are visible
in the notebook, but wind, vegetation or geometry are not established as causes
by this experiment alone.


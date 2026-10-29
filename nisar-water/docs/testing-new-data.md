# Test new NISAR observations

Run these commands from the repository root after installing the locked
environment. Use a fresh output folder every time; completed results are not
overwritten. This guide tests inference. Measuring accuracy additionally needs
an independent, dated reference with known review/uncertainty provenance.

## Supported scientific input

The current profile requires NISAR L-band **GCOV, frequency A**, ordered
`HHHH` and `HVHV` covariance powers, representing HH/HV **linear gamma0 power**.
Native north-up UTM grids must have 10 m or 20 m square pixels. Native 10 m
data are averaged to 20 m in linear power with all four native samples valid.
No coarse-to-fine upsampling is performed. Grid posting is not radar resolution.

Raw data must already be a calibrated GCOV product. Do not square or log it
again. PNG/JPEG previews, dB values, amplitudes, other products, frequency B and
VV/VH substitutions are rejected or unsupported. Keep the exact NISAR filename;
acquisition geometry and UTC time are parsed from it and checked where native
identification fields are available.

## Local GCOV HDF5

Obtain the scientific `.h5` using your normal NASA Earthdata/ASF access. The
local inference command needs no Earthdata credential. Edit
`examples/aoi.geojson` to a small WGS84 polygon inside the granule. Coordinates
are `[longitude, latitude]` in degrees, with a closed polygon ring. The file is
a Polygon geometry or Feature, rather than a FeatureCollection.

```sh
uv run --frozen nisar-water hdf5 /path/to/actual-NISAR-GCOV-filename.h5 --aoi examples/aoi.geojson --model checkpoints/round5/terrain_prior --context-cache .cache/terrain-context --output outputs/new-observation-01
```

The placeholder filename must be replaced with the original exact product
filename. Quote paths with spaces. `hdf5` reads only the AOI plus a 320 m radar
context halo and rejects more than 16 million native crop pixels. Large study
areas should be divided into bounded jobs. The native GCOV mask rejects
invalid/no-observation cells; it is not an independently estimated geometry
shadow/layover mask.

For a sequence of observations, pass a directory of local `*GCOV*.h5` files.
Every supplied compatible file is processed. Select one processing revision
per acquisition; duplicate acquisition times are rejected. Different
track/pass/frame groups become separate series.

```sh
uv run --frozen nisar-water hdf5 /path/to/gcov-folder --aoi examples/aoi.geojson --model checkpoints/round5/terrain_prior --context-cache .cache/terrain-context --output outputs/new-sequence-01
```

To compare a radar-only profile, use `checkpoints/round5/radar` and omit
`--context-cache`. The frozen default thresholds are 0.20 radar and 0.10 for
both physical profiles. `--threshold` deliberately overrides the frozen point;
exports record that override. Do not tune it on your final test reference.

## Physical context on a new location

`--context-cache PATH` explicitly enables bounded public DEM/HAND and optional
JRC downloads into that cache. It does not download radar imagery or require
Earth Engine. The current adapter caps ancillary transfer at 128 MiB per
region and records source-window identities, grids and hashes. Repeated runs
can reuse covering sources. No cache option means existing-cache lookup only.

The equivalent Python setting is `context_cache_dir=...`, or configure
`NISAR_WATER_CONTEXT_CACHE` for context-HGB loading. The five DEM/HAND features
are required by physical profiles: unavailable support becomes code 254.
Historical prior is optional; missing values remain NaN with an availability
flag. Check `requiredContextWithheldHa`, context provenance and unknown area
before interpreting a map. A blank overlay can mean unsupported context, not
an absence of water.

## Python worker for API integration

`examples/predict_hdf5.py` is a minimal single-observation worker:

```sh
uv run --frozen python examples/predict_hdf5.py /path/to/actual-NISAR-GCOV-filename.h5 --aoi examples/aoi.geojson --model checkpoints/round5/terrain_prior --context-cache .cache/terrain-context --output outputs/python-worker-01
```

The underlying interface is:

```python
from nisar_water.common import read_json
from nisar_water.inputs import from_hdf5
from nisar_water.registry import load_model
from nisar_water.pipeline import run_series

model = load_model("checkpoints/round5/terrain_prior",
                   context_cache_dir=".cache/terrain-context")
observation = from_hdf5(input_path, aoi=read_json("examples/aoi.geojson"))
series = run_series([observation], model, "outputs/python-worker-02",
                    series_id="my-river")
```

A future job worker can invoke this path with server-managed inputs, a fixed
trusted checkpoint and a job-specific output directory. See the API guide.

## Original notebook `pila` / NetCDF

The original `construir_pila(...)` xarray Dataset must retain:

- `HHHH` and `HVHV` data variables with `time`, `y`, `x` dimensions.
- Regular ascending `x` and descending `y` pixel-center coordinates, at native
  10 m or 20 m spacing, and `attrs['epsg']` for the local UTM CRS.
- Exact `time` values and a per-time `nombre` coordinate with the NISAR granule
  name. Each timestamp must match the granule's UTC acquisition time.
- `attrs['frecuencia']='A'` and `attrs['units']='gamma0_power'`; the original
  Spanish `descripcion` documenting gamma0 lineal is also accepted.
- Optional `attrs['aoi_wkt']` as a WGS84 polygon. Keep the original `multilook`
  metadata to record inherited missing-sample behavior.

```python
from nisar_water.pipeline import predict_dataset

series = predict_dataset(
    pila,
    model_dir="checkpoints/round5/terrain_prior",
    output_dir="outputs/pila-01",
    context_cache_dir=".cache/terrain-context",
    series_id="my-river",
)
```

For interchange, use `pila.to_netcdf('water-input.nc', engine='scipy')` and:

```sh
uv run --frozen nisar-water netcdf water-input.nc --model checkpoints/round5/terrain_prior --context-cache .cache/terrain-context --output outputs/netcdf-01
```

The package does not open pickle caches. Obtain `pila` in your trusted notebook
environment first. A cube containing mixed acquisition geometries must be
separated into compatible series before calling `predict_dataset`.

## Existing scientific crop catalogs

`nisar-water catalog catalog.json --model ... --output ...` consumes a catalog
with `observations` entries. Each observation identifies the sensor, product,
frequency, ordered polarizations, power units, RTC status, acquisition time,
track/pass/frame and AOI. It supplies SHA256-bound, grid-aligned HHHH/HVHV and
native-validity mask raster assets. Relative asset paths resolve against the
catalog's directory. See `inputs.load_catalog_observation` for the full enforced
contract. Do not create a catalog around display PNGs. Repeat `--observation`
to select exact entries; otherwise all supplied entries are processed.

## Read the outputs and compare time

Inspect `composite.png`, `water.png`, `border.png`, `uncertain.png` and
`water-mask.tif` under each observation directory. Mask codes are 0 nonwater,
1 water, 254 observed/unknown and 255 invalid/outside. A water score is an
uncalibrated classifier value, not a verified confidence percentage.

`series.json` is the API-ready manifest. Use `timeline[].waterOnCommonAllDatesHa`
together with `commonAllDatesHa` for an evolution chart. Pairwise gains/losses
use common classified support; unknown cells are never treated as dry. Do not
compare raw total water areas across changing coverage or mix different radar
geometries. Extent change does not establish a human cause.

## Diagnose a failed run

| Symptom | Check |
|---|---|
| Filename/identification rejected | Exact GCOV product and original filename |
| Wrong bands/units/grid | Frequency A HHHH/HVHV power; local native 10/20 m UTM |
| AOI does not intersect | WGS84 longitude/latitude polygon and product footprint |
| Crop too large | Split the AOI into bounded jobs |
| All/large unknown area | Context availability, feature support and exclusions |
| Blank water with classified land | Actual model decision; inspect known failure modes |
| Output already exists | Choose a fresh output directory |
| Authentication/access error in `aoi` | Use your own valid Earthdata token and required agreements; see the public notebook |

For software verification only, run the synthetic smoke check from the README.
For accuracy, obtain independent reviewed polygons and score water recall,
precision, land FPR and per-class reference coverage. Keep new evaluation
geographies out of training/selection. Current measured limitations are in
[the results guide](improved-classifier-results.md).

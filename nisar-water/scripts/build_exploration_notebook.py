"""Build the API/data exploration notebook only after the new-image benchmark exists."""
from pathlib import Path
import nbformat as nbf
from nisar_water.common import read_json,sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    report=read_json(ROOT/"outputs/new-images-20261028/report.json")
    assert len(report["rows"])==8 and report["model_unchanged"]
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):cells.append(nbf.v4.new_code_cell(text))
    md("""# NISAR water exploration — API, observations and honest model checks

Explore real NISAR L2 GCOV data, highlight water, compare dates, inspect reference errors, and use the saved-result API.

**Read this first:** the frozen spatial-v2 model is **not ready for general use**. Before this notebook was created, it was tested on eight newly downloaded acquisitions in four locations. Texoma performed well, while Mead and Okeechobee exposed large missed-water errors. Every tested image is retained here. Three images have usable dated optical reference support; the others are unscored.

The workflow follows the original `NISAC.ipynb`: environment → authentication → AOI/dates → catalog → bounded reads → `pila` → maps → time series → exports. It adds the water checkpoint, independent reference checks and the local water API. It does not train or tune a model.

Use the **NISAR Water (Python 3.13)** kernel and run the cells in order. The default uses downloaded scientific crops, so you can inspect all saved results without a token or more downloads. Change `MODE` to `online` to query and read new AOI windows with the protected credential helper. Never paste a token into a cell.""")
    md("## 0. Workspace and runtime\nThe package directory is found from the notebook location or kernel environment. All dates below are exact UTC acquisition times.")
    code("""from pathlib import Path
import sys, json, os, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
%matplotlib inline

candidates = [Path.cwd(), *Path.cwd().parents, Path(sys.executable).resolve().parents[2]]
PROJECT_ROOT = next((p for p in candidates if (p/'models/spatial-v2/selected/model.json').exists()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError('Open this notebook inside the nisar-water workspace.')
from nisar_water.common import read_json, write_json, sha256
from nisar_water.exploration import entries_for_case, metrics_table, plot_series, plot_errors, to_notebook_cube, plot_water_features, plot_benchmark_overview
from nisar_water.inputs import load_catalog_observation, from_dataset
from nisar_water.model import WaterModel
from nisar_water.pipeline import run_series

DATA_ROOT = PROJECT_ROOT/'data/new-images-20261028'
RESULTS_ROOT = PROJECT_ROOT/'outputs/new-images-20261028'
MODEL_DIR = PROJECT_ROOT/'models/spatial-v2/selected'
PLAN = read_json(DATA_ROOT/'plan.json')
REPORT = read_json(RESULTS_ROOT/'report.json')
ASSESSMENT = read_json(RESULTS_ROOT/'assessment.json')
MODEL = WaterModel(MODEL_DIR)
assert MODEL.fingerprint == PLAN['model_sha256'] == REPORT['model_sha256']
display(pd.DataFrame([{'library':n,'version':importlib.metadata.version(n)} for n in ['nisar-water','numpy','rasterio','scikit-learn','xarray','nbclient']]))
print('Workspace:', PROJECT_ROOT)
print('Model assessment:', ASSESSMENT['state'])""")
    md("""## 1. Exploration parameters

Choose a location from `lake-mead`, `lake-texoma`, `gatun-lake`, or `lake-okeechobee`. Mead is the default so that a difficult case is visible first. `cached` reuses the exact tested inputs. `online` queries public CMR and reads small HH/HV windows from authenticated Earthdata assets; it does not download complete multi-GB frames.

The spatial model's score settings are frozen. Changing them after examining these test results would turn these locations into development data.""")
    code("""MODE = 'cached'                 # 'cached' or 'online'
CASE_ID = 'lake-mead'
START_DATE, END_DATE = '2026-10-01', '2026-10-27'
RECOMPUTE_SELECTED = True       # Execute the frozen model on the selected real inputs.
EXPORT_CUBE = False             # Optional NetCDF export of compatible 20 m grids.
API_BASE = 'http://127.0.0.1:8012'
REFERENCE_ROOT = PROJECT_ROOT.parent/'nisar-landcover-phase1'

assert MODE in {'cached','online'}
CASE = next(c for c in PLAN['cases'] if c['id'] == CASE_ID)
AOI = CASE['aoi']               # WGS84 Polygon; replace for a custom live AOI.
display(pd.DataFrame([{'location':c['id'],'description':c['title'],'start':c['start'],'end':c['end']} for c in PLAN['cases']]))
print('Selected:', CASE['title'], '| mode:', MODE)
print('Frozen spatial decision:', MODEL.metadata['decision_policy'])""")
    md("""## 2. Authentication without saved credentials

The supplied credential was configured through the existing current-user encrypted Earthdata helper. This notebook contains no token. Live mode first checks the protected store and asks through hidden input only if needed. The token is used only for authorized Earthdata requests; optical references use a separate public client. Cached mode does not access credentials.""")
    code("""def ensure_earthdata_credential():
    import getpass
    from nisar_water.online import reference_adapter
    reference_adapter(REFERENCE_ROOT)._reference()
    from nisar_models.earthdata_access import get_earthdata_token, store_token
    try:
        value = get_earthdata_token()
        ready = bool(value)
        del value
        return ready
    except Exception:
        value = getpass.getpass('Earthdata bearer token (hidden; never saved in notebook): ')
        if os.name == 'nt':
            store_token(value)
        else:
            os.environ['EARTHDATA_TOKEN'] = value
        del value
        return True

if MODE == 'online':
    print('Protected Earthdata credential configured:', ensure_earthdata_credential())
else:
    print('Cached mode: no credential access or new satellite download.')""")
    md("""## 3. Discover exact acquisitions through the catalog API

The saved queries show what was actually searched before testing. Live mode makes a new CMR query. Acquisition selection uses footprint coverage, exact geometry and dates, never model scores. Frame plans do not establish that a processed granule exists.""")
    code("""from datetime import datetime, timezone
from nisar_water.online import discover, acquire, select_dates

if MODE == 'online':
    LIVE_TAG = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    ACTIVE_DATA_ROOT = PROJECT_ROOT/'data/notebook-online'/LIVE_TAG
    LIVE_FOLDER = ACTIVE_DATA_ROOT/CASE_ID
    DISCOVERY = discover(AOI, START_DATE, END_DATE, LIVE_FOLDER, REFERENCE_ROOT)
    write_json(LIVE_FOLDER/'discovery.json', DISCOVERY)
else:
    ACTIVE_DATA_ROOT = DATA_ROOT
    DISCOVERY = read_json(DATA_ROOT/CASE_ID/'discovery.json')

candidate_rows = [{k:c.get(k) for k in ['id','acquired_at','granule_id','compatible','coverage_fraction']} for c in DISCOVERY.get('candidates',[])]
display(pd.DataFrame(candidate_rows))
print('Catalog state:', DISCOVERY['state'])
print('Query truncated:', DISCOVERY.get('truncated',False))""")
    md("""## 4. Read or reuse bounded scientific crops

At most two dates in one track/direction/frame are selected. Native masks reject invalid samples. Power is averaged before logarithms, and the analysis grid is 20 m. The cached test transferred 136 MiB of radar ranges across all eight images. Source-file hashes identify each derived crop; full remote HDF5 hashes are unavailable because only windows were read.""")
    code("""if MODE == 'online':
    selected = select_dates(DISCOVERY.get('candidates',[]), max_dates=2)
    if not selected:
        raise RuntimeError('No compatible acquisitions with >=80% footprint coverage; change AOI/dates.')
    for i,candidate in enumerate(selected):
        result = acquire(candidate, AOI, LIVE_FOLDER, REFERENCE_ROOT)
        write_json(LIVE_FOLDER/f'acquisition-{i}.json', result)
        print(candidate['acquired_at'], result['state'])

ENTRIES = entries_for_case(ACTIVE_DATA_ROOT, CASE_ID)
if not ENTRIES:
    raise RuntimeError('No actual scientific crops were acquired; no replacement data is generated.')
display(pd.DataFrame([{'id':e['id'],'datetime':e['acquired_at'],'track':e['track'],'pass':e['pass_direction'],'frame':e['frame'],'units':e['units'],'native_shape':(e['height'],e['width'])} for e in ENTRIES]))
PREPARED = [load_catalog_observation(e) for e in ENTRIES]
print('Input verification passed:', len(PREPARED), 'actual acquisitions')""")
    md("""## 5. Build the familiar `pila` input structure

When both acquisitions share an identical prepared grid, construct an xarray cube with `HHHH`, `HVHV`, `time`, `nombre`, `x`, `y`, `epsg` and linear-power metadata. This is the structure accepted from `construir_pila(...)` in the original lab notebook.

Different grids are kept explicit instead of being silently aligned. The water time-series pipeline can compare their classified maps on common support. A 20 m posting does not establish independent 20 m radar resolution.""")
    code("""try:
    pila = to_notebook_cube(ENTRIES)
    display(pila)
    assert len(list(from_dataset(pila))) == len(ENTRIES)
    if EXPORT_CUBE:
        export_path = PROJECT_ROOT/'outputs'/f'{CASE_ID}-water-input.nc'
        pila.to_netcdf(export_path, engine='scipy')
        print('Saved:', export_path)
except ValueError as error:
    pila = None
    print(error)

# To use your original notebook's in-memory pila directly:
# from nisar_water.pipeline import predict_dataset
# result = predict_dataset(pila, MODEL_DIR, output_dir, series_id='my-water-series')""")
    md("""## 6. Run the frozen water model

The selected classifier uses NISAR HH/HV power and spatial texture, followed by connected-region decisions. Confident seed components can grow through weaker connected scores. No optical reference, future date or test label enters this prediction. The score is uncalibrated; a seed threshold of 0.90 does not mean measured 90% certainty.""")
    code("""if RECOMPUTE_SELECTED or MODE == 'online':
    run_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    SERIES_ROOT = PROJECT_ROOT/'outputs/notebook-runs'/f'{CASE_ID}-{run_tag}'
    SERIES = run_series(PREPARED, MODEL, SERIES_ROOT, series_id=CASE_ID)
else:
    SERIES_ROOT = RESULTS_ROOT/'series'/CASE_ID
    SERIES = read_json(SERIES_ROOT/'series.json')
display(pd.DataFrame([{'datetime':o['datetime'],**o['statistics']} for o in SERIES['observations']]))
print('Saved result:', SERIES_ROOT/'series.json')
print('Model hash still frozen:', sha256(MODEL_DIR/'model.json') == REPORT['model_sha256'])""")
    md("## 7. Radar input and colored water across dates\nBlue is predicted open water, cyan is its raster-derived outline, and transparent/black areas have no observation. The HH display range stays fixed at −30 to +3 dB. These are model estimates, not verified changes in water level.")
    code("""figure = plot_series(SERIES_ROOT)
plt.show()
if SERIES.get('animationUrl'):
    display(Image(filename=str(SERIES_ROOT/SERIES['animationUrl'].removeprefix('/assets/'))))""")
    md("""## 8. Water evolution on comparable support

Use the intersection classified on **all dates** for the area curve. Consecutive gains/losses use pairwise common support. Missing coverage must not appear as disappearing water. Model errors can also change between dates, so a large water-area jump is not proof of a real environmental event.""")
    code("""timeline = pd.DataFrame(SERIES['timeline'])
display(timeline)
display(pd.DataFrame(SERIES['transitions']))
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(pd.to_datetime(timeline['datetime']),timeline['waterOnCommonAllDatesHa'],marker='o',label='Water on common classified support')
ax.set_ylabel('Predicted water (ha)'); ax.set_xlabel('UTC acquisition date'); ax.grid(alpha=.3); ax.legend()
fig.autofmt_xdate(); plt.show()""")
    md("""## 9. Results on all eight new radar images

The locations were declared before prediction, and the frozen model was not changed after this benchmark. **The broader generalization test failed.** On the retained reference support, spatial-v2 water recall is 11.3% for Mead, 98.6% for Texoma and 0% for Okeechobee. Gatun's available dated optical reference contains only cloud classes; four other dates have no optical acquisition within 24 hours.

These are machine optical labels, not field truth. SCL4/5 are explicit nonwater and SCL6 is water. Forty-metre homogeneous interiors, valid radar support and full AOI cells are retained; cloud/shadow/unknown/boundary pixels are not scored. A blank metric means unavailable or undefined, never zero error. Okeechobee's scored support contains only water, so its nonwater false-positive rate cannot be measured.""")
    code("""table = metrics_table(REPORT)
display(table)
display(pd.DataFrame(REPORT['aggregate']).T)
print('Model assessment:', ASSESSMENT['state'])
print('No fitting or threshold selection used these new images:', not REPORT['test_used_for_model_selection'])
overview_figure = plot_benchmark_overview(RESULTS_ROOT)
plt.show()""")
    md("## 10. Dated optical context and visible errors\nSelect one of the tested dates below. Orange marks missed reference water, magenta false water, blue correct water and gray correct nonwater. Black is unscored. Errors are measured only on the frozen reference support, not every displayed pixel.")
    code("""REFERENCE_SCENE_INDEX = 0
if MODE == 'cached':
    chosen_entry = ENTRIES[REFERENCE_SCENE_INDEX]
    error_figure, message = plot_errors(DATA_ROOT,RESULTS_ROOT,CASE_ID,chosen_entry['id'])
    print(chosen_entry['acquired_at'], '|', message)
    if error_figure is not None:
        plt.show()
else:
    print('This fresh live selection has no independent reference evaluation in the saved benchmark. No accuracy is inferred.')""")
    md("""## 11. Inspect the input shift

Compare HH/HV backscatter of reference-water cells with the training-water distribution. These plots can expose a domain shift; they do not establish whether wind, vegetation, viewing geometry or another physical cause produced it. The reference labels are used here for diagnosis only, after the frozen benchmark.""")
    code("""feature_figure, feature_summary = plot_water_features(DATA_ROOT,RESULTS_ROOT,PROJECT_ROOT/'models/spatial-v2')
display(feature_summary)
plt.show()""")
    md("""## 12. Explore the local result API

Start the service in a terminal if needed:

```powershell
.\\.venv\\Scripts\\python.exe -m nisar_water.cli serve --root outputs\\new-images-20261028 --port 8012
```

`GET /water/series` lists computed series; each series supplies WGS84 overlay bounds and asset URLs. `GET /water/benchmark` and `/water/assessment` expose the broader test and its current limitation. These are saved actual results, not on-demand global inference.""")
    code("""import requests
try:
    response = requests.get(API_BASE+'/health',timeout=5); response.raise_for_status()
    print('API:',response.json())
    listing = requests.get(API_BASE+'/water/series',timeout=5); listing.raise_for_status()
    display(pd.DataFrame(listing.json()))
    endpoint = next(item['url'] for item in listing.json() if item['id']==CASE_ID)
    response = requests.get(API_BASE+endpoint,timeout=10); response.raise_for_status()
    API_SERIES = response.json()
    print('Exact acquisition IDs:',[o['id'] for o in API_SERIES['observations']])
    asset = API_SERIES['observations'][0]['assets']['water']
    image_response = requests.get(API_BASE+asset['url'],timeout=10); image_response.raise_for_status()
    display(Image(data=image_response.content))
    assessment_response = requests.get(API_BASE+'/water/assessment',timeout=5)
    assessment_response.raise_for_status(); display(assessment_response.json())
except requests.RequestException as error:
    print('Result API unavailable; start it with the command above. Saved local results remain usable.')
    print(type(error).__name__)""")
    md("## 13. Scientific exports and frontend integration\nEach result includes water/score COGs, fractional AOI coverage, transparent fill and border PNGs, WGS84 polygons and provenance. Web PNGs are reprojected to EPSG:4326; native GeoTIFFs keep their UTM transforms. Keep unknown/excluded pixels distinct from land, and display the broader model assessment alongside any water map.")
    code("""selected_observation = SERIES['observations'][0]
display(pd.DataFrame([{'asset':key,**value} for key,value in selected_observation['assets'].items()]))
print('WGS84 overlay bounds [west,south,east,north]:',selected_observation['bounds'])
print('Native analysis CRS:',selected_observation['crs'])
print('Frontend contract:',PROJECT_ROOT/'docs/frontend-contract.md')
assert sha256(MODEL_DIR/'model.json') == REPORT['model_sha256']""")
    md("""## 14. What these results support

- The end-to-end workflow can discover, acquire, classify and export multiple real locations/dates.
- The present checkpoint is **not reliable across water regimes**. Strong Texoma and Red Rock results do not resolve its Mead/Okeechobee failures.
- Three scored images are not a worldwide accuracy estimate. Correlated pixels, optical label errors and omitted boundaries limit interpretation.
- The next modeling round needs diverse reviewed water/shoreline/hard-negative examples and a **new reserved test geography**. These eight acquisitions are now exposed regression data.
- Water extent alone does not establish human impact. Relate independently validated changes to dated land use, weather, hydrology or operational evidence before attributing causes.

Reproduction: `scripts/test_new_images.py`, `scripts/fetch_new_references.py`, `scripts/benchmark_new_images.py`. Full receipts and range budgets are in `data/new-images-20261028`; full measurements are in `outputs/new-images-20261028/report.json`. The Earthdata token is absent from this notebook and its saved outputs.""")
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"name":"nisar-water","display_name":"NISAR Water (Python 3.13)","language":"python"},
          "language_info":{"name":"python","version":"3.13"},"nisar_water":{"model_sha256":report["model_sha256"],
          "benchmark_report_sha256":sha256(ROOT/"outputs/new-images-20261028/report.json"),"contains_credentials":False,
          "radar_images_tested_before_creation":8,"new_regions":4}})
    destination=ROOT/"notebooks/NISAR_Water_Exploration.ipynb"
    destination.parent.mkdir(exist_ok=True)
    nbf.validate(nb);nbf.write(nb,destination)
    print(destination)


if __name__=="__main__":main()


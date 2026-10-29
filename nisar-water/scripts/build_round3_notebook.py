"""Create the comparison notebook after all frozen unseen cohorts were tested."""
from pathlib import Path
import nbformat as nbf
from nisar_water.common import read_json,sha256
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    assessment=read_json(ROOT/'outputs/round3/assessment.json')
    assert assessment['state']=='not_ready_for_general_use'
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):cells.append(nbf.v4.new_code_cell(text))
    md('''# NISAR water — trained models, unseen images, API and evolution

**Current result: a working headless research pipeline, not a model ready for general use.** A retrained tree classifier and a small U-Net were fitted to broader real NISAR data. They were compared with pinned official OPERA NI open-water algorithm stages through a crop adapter. Family and thresholds were frozen on three validation regions before the original test predictions.

Before this notebook was built, all three frozen models were tested on **13 images across eight unseen regions**: seven reserved lake images, two additional hard-negative images, and four additional shoreline images. The later cohorts remained outside fitting and selection and are reported separately. An airport and dry Powell terrain expose major false-water errors. This notebook retains the successes and failures.

Inputs are current-observation NISAR L-band frequency-A GCOV HH/HV linear power. Outputs include blue water fills, cyan borders, COG masks, GeoJSON, common-support time series and a read-only JSON API. There is no application UI. The default notebook run uses cached actual data and makes no satellite downloads.

Use the **NISAR Water (Python 3.13)** kernel. The historical `NISAR_Water_Exploration.ipynb` remains available for the earlier frozen-v2 benchmark.''')
    md('## 1. Runtime, artifacts and frozen identity')
    code('''from pathlib import Path
import sys, json, uuid, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
%matplotlib inline

PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'models/round3/selected/model.json').exists())
from nisar_water.common import read_json, sha256
from nisar_water.registry import load_model
from nisar_water.round3_evaluation import load_scene
from nisar_water.round3_results import cohort_specs, cohort_table, native_composite
from nisar_water.exploration import to_notebook_cube
from nisar_water.pipeline import predict_dataset

RESULTS = PROJECT_ROOT/'outputs/round3'
FREEZE = read_json(RESULTS/'freeze.json')
REPORT = read_json(RESULTS/'report.json')
ASSESSMENT = read_json(RESULTS/'assessment.json')
MODEL_DIR = PROJECT_ROOT/'models/round3/selected'
MODEL = load_model(MODEL_DIR)
assert MODEL.fingerprint == FREEZE['selected_model_sha256']
assert REPORT['model_unchanged']
assert all(sha256(PROJECT_ROOT/p)==h for p,h in FREEZE['implementation_hashes'].items())
print('Assessment:', ASSESSMENT['state'])
print('Selected by validation:', MODEL.model_id, 'threshold', MODEL.metadata['threshold'])
display(pd.DataFrame([{'package':n,'version':importlib.metadata.version(n)} for n in ['numpy','scikit-learn','rasterio','xarray','nbclient']]))''')
    md('''## 2. Select actual data; authentication stays outside the notebook

Choose a scene from `test`, `shorelines` or `challenge`. These choices inspect already evaluated data and do not change model settings. The encrypted Earthdata helper is needed only for new authenticated radar reads. Tokens never enter cells, environment dumps or notebook outputs.''')
    code('''COHORT = 'shorelines'
SPECS = cohort_specs(COHORT)
display(pd.DataFrame(SPECS)[['id','region','acquired_at','split']])
SCENE_ID = next(s['id'] for s in SPECS if s['region']=='malawi-shore')
SPEC = next(s for s in SPECS if s['id']==SCENE_ID)
OBSERVATION, TRUTH, EVALUATE, SIDECAR = load_scene(SPEC)
REFERENCE = SIDECAR['provenance']['reference']
print('Radar UTC:', SPEC['acquired_at'])
print('Optical UTC:', REFERENCE['reference_date'], '| signed difference hours:', round(REFERENCE['date_difference_hours'],2))
print('Analysis grid:', OBSERVATION.power.shape, OBSERVATION.crs, '| linear gamma0 power, HH then HV')
print('Current-user encrypted Earthdata helper available:', (PROJECT_ROOT.parent/'nisar-models/.secrets/earthdata-token.dpapi').is_file())
print('No credential read or live acquisition occurs in this cell.')''')
    md('''## 3. Broader data and geographic separation

The fitting manifest contains 40 scenes: **28 train, five validation, seven original test**. Entire water bodies stay in one partition. Training covers 16 geographic groups, including previously exposed Mead/Texoma/Okeechobee failures and older dry-land controls. Those exposed examples are now training/regression data, never called unseen tests.

New training labels combine dated Sentinel-2 SCL with NDWI/MNDWI consensus. Cloud, optical shadow, disagreements and unlabelled pixels remain unknown. Interior weights are 1 and weak boundary weights 0.3. Evaluation uses eroded interiors; weak boundary agreement is a separate diagnostic. **No expert shoreline labels have been completed.**''')
    code('''MANIFEST = read_json(PROJECT_ROOT/'data/round3-prepared/manifest.json')
frame = pd.DataFrame(MANIFEST['scenes'])
display(frame.groupby('split').agg(scenes=('id','size'),regions=('region','nunique')))
audit = read_json(PROJECT_ROOT/'evidence/round3-split-audit.json')
display(audit)
locations=[]
for scene in MANIFEST['scenes']:
    meta=read_json(Path(scene['path']).with_suffix('.json'))['source_observation']
    ring=np.array(meta['aoi']['coordinates'][0])
    locations.append({'region':scene['region'],'split':scene['split'],'lon':float(ring[:-1,0].mean()),'lat':float(ring[:-1,1].mean())})
points=pd.DataFrame(locations).drop_duplicates(['region','split'])
fig,ax=plt.subplots(figsize=(11,4))
for split,color in [('train','#467a50'),('validation','#ca8b25'),('test','#286bc0')]:
    p=points[points.split==split];ax.scatter(p.lon,p.lat,label=split,color=color,s=45)
ax.set(xlim=(-180,180),ylim=(-60,80),xlabel='Longitude',ylabel='Latitude',title='Whole-region partitions; no randomized pixel split')
ax.grid(alpha=.2);ax.legend();plt.show()''')
    md('''## 4. Input audit and calibration

The radar audit found no channel-order or power-conversion defect explaining the old failures. Native 10 m inputs are averaged in **linear power** to 20 m with all four source samples required. New acquisitions use 320 m context for 31-cell features. Old 80 m crops retain an explicit legacy provenance caveat and have a 15-cell supervision inset.

A separate new-label preparation error was caught before fitting: some Earth Search optical COGs already have the BOA offset applied despite the asset offset metadata. The provider item flag now controls exactly-once calibration. Rejected preparations are preserved and were not fitted. This does not explain away the earlier frozen-v2 radar failures.

Incidence and noise-equivalent metadata were inspected. The incidence cube has not been interpolated at terrain height. A noise LUT described in DN² is not blindly subtracted from gamma0. These physical extensions remain future evaluated work.''')
    code('''display(Image(filename=str(PROJECT_ROOT/'evidence/round3-development-label-qa-corrected.png'),width=1100))
display(read_json(PROJECT_ROOT/'evidence/round3-development-qa-review.json'))''')
    md('''## 5. Actual fits and validation-only choice

**HGB:** 41 HH/HV multiscale features, 43,264 water and 39,884 land samples, geography-balanced weights, one fixed 300-iteration fit.

**U-Net:** 1,084,465 parameters; trained from scratch on actual NISAR, not advertised as Sentinel-1 fine tuning. Twenty CUDA epochs; best epoch 20 by macro validation-region loss. Input normalization is fitted on training scenes only. Inference tiles discard 64-cell halos and use fixed BatchNorm statistics.

**OPERA:** pinned official NI open-water stages with unchanged scientific defaults, real DEM/HAND/WorldCover/JRC/GLAD ancillary inputs, 30 m power aggregation, and isolated GDAL runtime. This bounded crop adapter is **not an official DAAC product** and is not established as equivalent to full-frame processing.

Thresholds 0.1–0.9 and the declared validation rule selected HGB at **0.10**. U-Net's frozen comparison threshold is also 0.10; OPERA is deterministic binary. Raw scores are uncalibrated. All checkpoints and selection records are preserved.''')
    code('''VALIDATION = read_json(RESULTS/'validation-report.json')
best=[]
for family,r in VALIDATION['family_best'].items():
    best.append({'family':family,'threshold':r['threshold'],'minimum_region_F1':r['minimum_region_f1'],'macro_region_F1':r['macro_region_f1'],'weak_validation_gate':r['passes_validation_gate']})
display(pd.DataFrame(best))
display(pd.DataFrame([{'family':f,'region':region,**{k:v[k] for k in ['precision','recall','false_positive_rate','reference_nonwater_ha']}} for f,r in VALIDATION['family_best'].items() for region,v in r['regions'].items()]))
history=read_json(PROJECT_ROOT/'models/round3/unet/history.json')['epochs']
h=pd.DataFrame(history);fig,ax=plt.subplots(figsize=(9,3))
ax.plot(h.epoch,h.train_loss,label='train patch loss');ax.plot(h.epoch,h.validation_macro_region_loss,label='validation macro-region loss')
ax.set(xlabel='Epoch',ylabel='Loss',title='One fixed U-Net fit; no test-guided epochs');ax.legend();plt.show()
print('Freeze UTC:',FREEZE['created_at'],'| test opened by fitting/selection:',FREEZE['test_opened_by_fitting_or_selection'])''')
    md('''## 6. All unseen results, including insufficient reference support

The original seven lake images are overwhelmingly water-dominated: only 3.12 ha of scored land across the cohort. Their near-perfect learned-model water recall is useful but cannot establish land specificity. Additional prospective cohorts were therefore registered with the already frozen models: airport/mountains and mixed shorelines. No family or threshold was changed after these results.

The salt-flat reference remained unusable, and the Jamuna candidates had cloud or no native radar coverage; these attempts are retained in their collection receipts, not replaced by fabricated scores. Dates and pixels within a region are correlated. Area-weighted pooled metrics are not global accuracy estimates.''')
    code('''TABLE = pd.DataFrame(REPORT['region_table'])
display(TABLE.style.format({k:'{:.3f}' for k in ['precision','recall','false_positive_rate','iou','reference_coverage']},na_rep='unavailable'))
display(pd.DataFrame([{'family':f,**r['pooled']} for f,r in REPORT['pooled_across_cohorts'].items()]))''')
    md('### Original lake cohort: every tested image')
    code("display(Image(filename=str(PROJECT_ROOT/'evidence/round3-test-comparison.png'),width=1100))")
    md('### Additional mixed shorelines: every tested image')
    code("display(Image(filename=str(PROJECT_ROOT/'evidence/round3-shorelines-comparison.png'),width=1400))")
    md('### Hard negatives: every tested image')
    code("display(Image(filename=str(PROJECT_ROOT/'evidence/round3-challenge-comparison.png'),width=1400))")
    md('''## 7. Where the masks are wrong

Red marks false water on retained clear-land interiors; gold marks missed water. Gray/white boundary and cloud pixels are unscored. These are weak-reference diagnostics, not independently reviewed shoreline accuracy. The airport and Powell failures block general deployment despite the good Malawi result.''')
    code('''for region in ['malawi-shore','powell-shore','denver-airport']:
    display(Image(filename=str(PROJECT_ROOT/'evidence'/('round3-'+region+'-errors.png')),width=1300))
display(ASSESSMENT['blocking_findings'])''')
    md('''## 8. Fresh inference on the selected actual observation

This reruns the frozen selected model on the scientific power arrays and checks parity with its saved evaluated prediction. There are no optical, location, historical-water or future-date inference features in HGB/U-Net.''')
    code('''LABELS, SCORES, INFO = MODEL.predict(OBSERVATION)
with np.load(RESULTS/(COHORT+'-predictions')/'hgb'/SPEC['id']/'prediction.npz',allow_pickle=False) as cached:
    assert np.array_equal(LABELS,cached['labels'])
    np.testing.assert_allclose(SCORES,cached['scores'],atol=1e-7,equal_nan=True)
fig,ax=plt.subplots(figsize=(7,7));ax.imshow(native_composite(OBSERVATION,LABELS));ax.axis('off')
ax.set_title(SPEC['id']+' | blue open water, cyan border',fontsize=10);plt.show()
print('Water ha:',float((OBSERVATION.weights[LABELS==1]*.04).sum(dtype=float)))
print('Decision:',INFO['decision_policy'],'| scores:',INFO['score_semantics'])''')
    md('''## 9. The original lab `pila` structure → exported water and borders

The xarray adapter uses the original notebook conventions (`HHHH`, `HVHV`, `time`, `nombre`, `x`, `y`, `epsg`, `frecuencia`, linear gamma0 description and `aoi_wkt`). It processes every supplied compatible observation, requiring common acquisition geometry within a series. Different tracks/passes/frames belong in separate series.''')
    code('''dataset_root = {'test':'round3','challenge':'round3-challenge','shorelines':'round3-shorelines'}[COHORT]
collection = read_json(PROJECT_ROOT/'data'/dataset_root/SPEC['region']/'collection.json')
ENTRY = next(s['observation'] for s in collection['selected'] if s['observation']['acquired_at']==SPEC['acquired_at'])
pila = to_notebook_cube([ENTRY])
display(pila)
FRESH_OUTPUT = PROJECT_ROOT/'.cache/notebook-round3'/uuid.uuid4().hex
PILA_RESULT = predict_dataset(pila,MODEL_DIR,FRESH_OUTPUT,series_id='selected-notebook-observation')
display({k:PILA_RESULT['observations'][0][k] for k in ['id','datetime','bounds','modelId','decisionPolicy','statistics']})
print('Fresh exports:',FRESH_OUTPUT)
print('Assets:',list(PILA_RESULT['observations'][0]['assets']))''')
    md('''## 10. Evolution on compatible dates

Tahoe's two reserved acquisitions share track 34, ascending, frame 21. Compare water only on common classified support. These crops mostly cover stable open water; the plot does not establish water-level change or a human cause. Other test dates use different geometry and are kept separate.''')
    code('''TAHOE_ROOT = RESULTS/'series/test/tahoe-34-A-21'
SERIES = read_json(TAHOE_ROOT/'series.json')
timeline = pd.DataFrame(SERIES['timeline'])
display(timeline[['datetime','waterHa','commonAllDatesHa','waterOnCommonAllDatesHa']])
display(pd.DataFrame(SERIES['transitions']))
fig,ax=plt.subplots(figsize=(9,3));ax.plot(pd.to_datetime(timeline.datetime),timeline.waterOnCommonAllDatesHa,marker='o')
ax.set(xlabel='Exact acquisition UTC',ylabel='Estimated water ha',title='Fixed common support; model estimates, not measured water level')
ax.grid(alpha=.2);plt.show()
if SERIES['animationUrl']:
    display(Image(filename=str(TAHOE_ROOT/SERIES['animationUrl'].removeprefix('/assets/'))))''')
    md('''## 11. Actual read-only HTTP API for the team UI

Run `python -m nisar_water.cli serve --root outputs/round3 --port 8013` from this project if the service is not already running. `/health` reports service availability; `/water/assessment` reports scientific readiness. The frontend must respect the latter. The JSON/PNG/COG/GeoJSON contract is in `docs/frontend-contract.md`.''')
    code('''import requests
API_BASE = 'http://127.0.0.1:8013'
session=requests.Session();session.trust_env=False
health=session.get(API_BASE+'/health',timeout=15);health.raise_for_status()
assessment=session.get(API_BASE+'/water/assessment',timeout=15);assessment.raise_for_status()
items=session.get(API_BASE+'/water/series',timeout=15);items.raise_for_status()
display(health.json());print('Scientific readiness:',assessment.json()['state'])
display(pd.DataFrame(items.json()))
item=next(i for i in items.json() if 'malawi-shore' in i['key'])
response=session.get(API_BASE+item['url'],timeout=15);response.raise_for_status()
api_series=response.json();observation=api_series['observations'][0]
asset=observation['assets']['water']
image_response=session.get(API_BASE+asset['url'],timeout=15);image_response.raise_for_status()
assert __import__('hashlib').sha256(image_response.content).hexdigest()==asset['sha256']
display(Image(data=image_response.content,width=450))
display({k:observation[k] for k in ['datetime','bounds','displayCrs','modelId','statistics']})
print('Transparent water PNG hash and HTTP access verified.')''')
    md('''## 12. New online observations: every compatible date, no optical filter

Keep `RUN_ONLINE=False` for the reproducible cached run. Set it to `True` to query and acquire current NISAR observations with the protected helper. The AOI command recursively splits truncated metadata queries, selects processing revisions from metadata, preserves failed receipts, applies a total transfer budget and reports partial runs explicitly. It does not select dates using clouds, optical labels or classifier scores. It supports bounded AOIs in the current L-GCOV HH/HV collection, not arbitrary entire global frames or other sensor profiles.

The already executed `outputs/round3/archive-demo` is a separate real acquisition/inference demonstration. Its dates are additional unscored observations, not extra accuracy claims.''')
    code('''RUN_ONLINE = False
START_DATE = '2026-07-02'
END_DATE = '2026-07-14'
MAX_OBSERVATIONS = 0       # 0 means every compatible observation in the query
TOTAL_TRANSFER_MIB = 128
AOI = ENTRY['aoi']
if RUN_ONLINE:
    from nisar_water.archive import process_aoi
    online_output=PROJECT_ROOT/'outputs'/('notebook-online-'+uuid.uuid4().hex)
    acquisition=process_aoi(AOI,START_DATE,END_DATE,MODEL_DIR,online_output,
        PROJECT_ROOT.parent/'nisar-landcover-phase1',transfer_cap_bytes=TOTAL_TRANSFER_MIB*1024**2,limit=MAX_OBSERVATIONS)
    display(acquisition)
else:
    demo_path=RESULTS/'archive-demo/batch.json'
    if demo_path.exists():
        demo=read_json(demo_path)
        display({k:demo[k] for k in ['state','complete_for_compatible_query','candidate_observations','acquired_observations','radar_body_bytes','scope']})
    print('Cached mode: no satellite requests made.')''')
    md('''## 13. Review and the next model decision

`review/round3/index.json` contains 13 unreviewed packets with radar/dated optical imagery, georeferencing and empty annotation templates. Model predictions are absent from these packets. A reviewer should annotate `open_water`, `nonwater` and `exclude` polygons; unannotated pixels remain unknown. Automated spectral labels and assistant visual QA are not expert review.

The practical next improvement is a better discriminating data/physical-context experiment: independent dry-land and shoreline labels, new hard-negative training and validation geographies, terrain/incidence context and causal previous-date persistence, followed by another untouched test. Historical water can be tested as a soft prior but must not hard-mask new water. Noise correction needs verified calibration units. The current comparison does not justify replacing everything with a larger network.

Extent change by itself cannot attribute agriculture, deforestation, wildfire or human impact. Those need separate labelled tasks and contextual evidence. The API and export structure can carry future model layers, while keeping their scientific claims separate.''')
    code('''review=read_json(PROJECT_ROOT/'review/round3/index.json')
print('Review packets:',len(review['items']),'| completed independent expert reviews:',review['expert_reviews_completed'])
display(pd.DataFrame(review['items'])[['key','scene_id','status']])
assert sha256(MODEL_DIR/'model.json') == FREEZE['selected_model_sha256']
print('Selected checkpoint unchanged after evaluation and notebook execution.')''')
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'NISAR Water (Python 3.13)','language':'python','name':'nisar-water'},
        'language_info':{'name':'python','version':'3.13'},'nisar_water':{'freeze_sha256':sha256(ROOT/'outputs/round3/freeze.json'),
        'assessment_sha256':sha256(ROOT/'outputs/round3/assessment.json'),'test_images_before_notebook':13,'satellite_downloads_in_default_run':False}})
    path=ROOT/'notebooks/NISAR_Water_Round3_Comparison.ipynb';nbf.write(notebook,path);print(path)

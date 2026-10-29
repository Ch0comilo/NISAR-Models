"""Build the executed-improvement handoff only after fresh frozen tests exist."""
from pathlib import Path
import nbformat as nbf
from nisar_water.common import read_json,sha256
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    assert (ROOT/'outputs/round5/agreement-test-report.json').exists()
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# NISAR water: hard negatives, terrain, recent reservoirs and review

**Research status: no candidate meets every acceptance target.** This notebook follows two additional real-data experiments after round3. Round4 introduced hard-negative data and physical context; its selected conservative model suppressed false water but missed recent reservoir water. Round5 promoted exposed failures to training and reserved new recent-reservoir/dry-land geographies. All five fresh round5 test images, and all12 round4 test images, were evaluated before this notebook was built.

Current optical consensus is a weak reference, not expert shoreline truth. Area-weighted scores over selected correlated regions do not establish worldwide accuracy. Model choices and thresholds remain frozen. The diagnostic primary family is chosen by validation even when no family passes; a better-looking test variant is not silently promoted.

The model/backend remains headless. The notebook supplies scientific data exploration, original-lab `pila` compatibility, colored maps/borders, comparisons, temporal outputs, HTTP calls and Earth Engine annotation handoff. Default execution uses cached actual inputs and makes no satellite downloads.''')
    code('''from pathlib import Path
import sys, uuid, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
%matplotlib inline
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'outputs/round5/freeze.json').exists())
from nisar_water.common import read_json,sha256
from nisar_water.registry import load_model
from nisar_water.round5_evaluation import load_scene
from nisar_water.improvement_results import table
from nisar_water.round3_results import native_composite
R4=read_json(ROOT/'outputs/round4/test-report.json')
R5=read_json(ROOT/'outputs/round5/test-report.json')
F5=read_json(ROOT/'outputs/round5/freeze.json')
ASSESSMENT=read_json(ROOT/'outputs/round5/assessment.json')
for experiment in [4,5]:
    freeze=read_json(ROOT/f'outputs/round{experiment}/freeze.json')
    assert all(sha256(ROOT/p)==h for p,h in freeze['implementation_hashes'].items())
print('Scientific readiness:',ASSESSMENT['state'])
print('Validation-selected diagnostic family:',F5['selected_family'],'| accepted for production: False')''')
    md('''## Data, geography and fit provenance

All71 exposed round4 scenes are training in round5; they cannot be called fresh tests again. New whole regions supply six validation and five test scenes. Planned minimum conservative cross-split separation is334.14km in round5. Hidroituango lacked usable dated clear optical support and is retained as an unavailable scoring case; a separate radar-only acquisition demonstration is unscored.

Labels retain the same exactly-once calibrated SCL/NDWI/MNDWI consensus, unknown clouds/shadows/ambiguities and eroded interiors. No expert polygons were imported into these fits. Terrain is a20m view of nominal30m measurements, not finer independent terrain resolution.''')
    code('''splits=[]
for experiment in [4,5]:
    manifest=read_json(ROOT/f'data/round{experiment}-prepared/manifest.json')
    frame=pd.DataFrame(manifest['scenes'])
    for split,part in frame.groupby('split'):
        splits.append({'round':experiment,'split':split,'scenes':len(part),'regions':part.region.nunique()})
display(pd.DataFrame(splits))
display(read_json(ROOT/'data/round5/planned-separation-audit.json')['closest_cross_split_pair'])
models=[]
for family in ['radar','terrain','terrain_prior']:
    meta=read_json(ROOT/f'models/round5/{family}/model.json')
    models.append({'family':family,'features':len(meta['features']),**meta['training_counts'],'threshold_before_selection':meta['threshold']})
display(pd.DataFrame(models))
print('Same203,678 sampled cells, geography-balanced weights, fixed300 iterations per accepted round5 fit.')''')
    md('''## What physical context adds

The ablations share identical training samples:41 radar features;46 radar/terrain features;49 radar/terrain/prior features. Terrain adds slope, local relief and HAND descriptors, excluding absolute geographic coordinates/elevation as predictors. Prior features are historical JRC2021 occurrence/distance plus availability. They are learned covariates, never a coded hard mask; strong learned historical dependence can still suppress real new water.

Missing required terrain stays unknown. A near-zero numerical HAND-mean issue was caught by the release check: an unreleased initial fit was isolated, corrected feature caches were versionedv2, and all accepted fits followed the successful release. No validation/test score was used for that repair. Original failed artifacts remain reviewable.''')
    code('''from nisar_water.terrain_context import TERRAIN_NAMES,PRIOR_NAMES
display(pd.DataFrame({'physical_features':TERRAIN_NAMES}))
display(pd.DataFrame({'optional_history_features':PRIOR_NAMES}))
display(Image(filename=str(ROOT/'evidence/round4-development-label-qa.png'),width=1000))''')
    md('''## Round4: lower false water, but a major recent-water failure

Validation selected terrain+prior at0.30. On twelve fresh images, it was very conservative on land, but recovered less than1% of retained GERD water. Radar-only recovered most GERD water but confused extensive dry land elsewhere. This exposed the danger of stale terrain and historical-water context for the actual goal of monitoring evolving water bodies.''')
    code('''display(pd.DataFrame([{'family':f,**v['pooled']} for f,v in R4['families'].items()]))
display(pd.DataFrame(table(4)).round(4))
display(Image(filename=str(ROOT/'evidence/round4-overview.png'),width=1300))
display(Image(filename=str(ROOT/'evidence/round4-gerd-shore-errors.png'),width=1100))''')
    md('''## Round5: current water absent from old maps becomes a validation requirement

The new validation includes Ilısu recent-reservoir shoreline, persistent Omodeo shoreline and dry Madrid airport. A supported current-water stratum with JRC occurrence≤10% must recover90% of its full reference, alongside90% regional precision/recall,≤4% land FPR and98% total/per-class coverage. Unknown predictions cannot hide rare water.

All three families still failed at least one validation requirement. The frozen diagnostic choice is radar-only at0.20; terrain-only and terrain+prior comparison operating points are0.10. None was promoted to production. All five new test images remained outside fitting and selection.''')
    code('''validation=read_json(ROOT/'outputs/round5/validation-report.json')
display(pd.DataFrame([{'family':f,'threshold':v['threshold'],'passes_joint_gate':v['passes_validation_gate'],
    'minimum_region_balanced_recovery':v['minimum_region_balanced_accuracy']} for f,v in validation['family_best'].items()]))
display(Image(filename=str(ROOT/'evidence/round5-validation-label-qa.png'),width=1100))
print('Frozen before test:',F5['created_at'])''')
    md('''## Every fresh round5 test image and remaining errors

The conservative terrain+prior variant substantially improves current-water recovery over round4 on these fresh reservoirs, especially Nam Theun1. Yusufeli remains difficult, and radar-only still produces dry-airport false water. These are comparisons of frozen candidates, not test-based reselection. Consult reference class areas: this cohort is dominated by land.

Low historical occurrence is a diagnostic, not itself proof of a construction date or a human cause. Official project chronology can provide separate context; extent alone does not establish causality.''')
    code('''display(pd.DataFrame(table(5)).round(4))
display(pd.DataFrame([{'family':f,**v['pooled']} for f,v in R5['families'].items()]))
display(pd.DataFrame([{'family':f,**v['historical_water_diagnostic']['low_historical_occurrence_water']} for f,v in R5['families'].items()]))
display(Image(filename=str(ROOT/'evidence/round5-all-test-images.png'),width=1300))''')
    md('''## Fresh inference: inspect any frozen candidate

Changing the inspection choice below does not change selection or acceptance. Start with the validation-selected diagnostic family; choose `terrain` or `terrain_prior` to inspect another frozen candidate. Its saved test measurement uses the same frozen operating point. There are no contemporary optical inference features.''')
    code('''MODEL_VARIANT=F5['selected_family']
MODEL_DIR=ROOT/'outputs/round5/frozen'/MODEL_VARIANT
MODEL=load_model(MODEL_DIR)
manifest=read_json(ROOT/'data/round5-prepared/manifest.json')
SPEC=next(s for s in manifest['scenes'] if s['split']=='test' and s['region']=='nam-theun1-shore')
OBS,TRUTH,EVALUATE,SIDECAR=load_scene(SPEC)
LABELS,SCORES,INFO=MODEL.predict(OBS)
with np.load(ROOT/'outputs/round5/test-predictions'/MODEL_VARIANT/SPEC['id']/'prediction.npz',allow_pickle=False) as z:
    assert np.array_equal(LABELS,z['labels'])
    np.testing.assert_allclose(SCORES,z['scores'],atol=1e-7,equal_nan=True)
fig,ax=plt.subplots(figsize=(7,6));ax.imshow(native_composite(OBS,LABELS));ax.axis('off')
ax.set_title(MODEL_VARIANT+' | '+SPEC['id'],fontsize=10);plt.show()
display({'radar_utc':SPEC['acquired_at'],'optical_utc':SIDECAR['provenance']['reference']['reference_date'],
    'decision_policy':INFO['decision_policy'],'score_semantics':INFO['score_semantics']})''')
    md('''## Original lab `pila` → headless exports

Both channels remain ordered linear gamma0 power. The adapter preserves the lab conventions: time/nombre/x/y, epsg, frecuencia, aoi_wkt and power units. An explicit context-cache argument enables bounded ancillary acquisition for a new AOI; this cached example already has its sources. The read-only API is separate from inference.''')
    code('''from nisar_water.exploration import to_notebook_cube
from nisar_water.pipeline import predict_dataset
collection=read_json(ROOT/'data/round5'/SPEC['region']/'collection.json')
ENTRY=next(s['observation'] for s in collection['selected'] if s['observation']['acquired_at']==SPEC['acquired_at'])
pila=to_notebook_cube([ENTRY]);display(pila)
FRESH=ROOT/'.cache/improved-notebook'/uuid.uuid4().hex
fresh=predict_dataset(pila,MODEL_DIR,FRESH,context_cache_dir=ROOT/'.cache/terrain-context')
result=fresh['observations'][0]
display({k:result[k] for k in ['datetime','bounds','statistics','modelId','evaluationStatus']})
print('Assets:',list(result['assets']))
print('Fresh scientific/web outputs:',FRESH)''')
    md('''## Review-only disagreement layer

Before the last blind tests, a fixed agreement policy was declared using the round5 validation-selected classifier and previously frozen round4 context model. Agreement yields0/1; disagreement or unavailable support yields254/unknown, displayed amber. This policy is excluded from primary selection and is not independently verified confirmation.

Read conditional precision together with full-water recovery and withheld area. The policy can leave much true water unknown and can still miss water where both models agree incorrectly. It fails the full acceptance gate; it is an operator-review aid, not a way to advertise accuracy by excluding errors.''')
    code('''review=read_json(ROOT/'outputs/round5/agreement-test-report.json')
display(pd.DataFrame([review['pooled']]))
display(pd.DataFrame([{'region':name,**{k:r[k] for k in ['reference_coverage','water_reference_coverage',
    'recovered_fraction_of_full_water_reference','withheld_water_reference_ha','withheld_land_reference_ha']}} for name,r in review['regions'].items()]))
review_image=next((ROOT/'outputs/round5/review-series/unseen-test').glob('nam-theun1*/observations/*/review-composite.png'))
display(Image(filename=str(review_image),width=700))
print('Full weak gate passed:',review['passes_full_weak_gate'])''')
    md('''## Evolution example, clearly marked as exposed training regression

The two GERD observations share acquisition geometry. They demonstrate fixed-model, common-support temporal exports, not independent change-detection accuracy: these dates are now training data. Water-area changes are estimates, not measured water levels. Unknown support and wind/geometric model errors can affect interpretation; no future-date smoothing or automatic human-impact attribution is used.''')
    code('''path=next((ROOT/'outputs/round5/series/exposed-training-regression').glob('gerd*/series.json'))
series=read_json(path);timeline=pd.DataFrame(series['timeline'])
display(timeline[['datetime','waterHa','uncertainHa','commonAllDatesHa','waterOnCommonAllDatesHa']])
display(pd.DataFrame(series['transitions']))
fig,ax=plt.subplots(figsize=(8,3));ax.plot(pd.to_datetime(timeline.datetime),timeline.waterOnCommonAllDatesHa,marker='o')
ax.set(xlabel='Exact acquisition UTC',ylabel='Estimated water ha on fixed common support',title='Exposed training regression: temporal pipeline demonstration')
ax.grid(alpha=.2);plt.show()
if series['animationUrl']:display(Image(filename=str(path.parent/series['animationUrl'].removeprefix('/assets/'))))''')
    md('''## Actual HTTP API for the team UI

The latest result service is on port8014. `/health` is server status; `/water/assessment` is scientific readiness. Keep the latter visible beside maps. Series include diagnostic, conservative-candidate and review modes; modelId/key distinguishes them. Fetch water/border/uncertain PNGs or COG/GeoJSON assets against the service origin.''')
    code('''import requests
API='http://127.0.0.1:8014';session=requests.Session();session.trust_env=False
health=session.get(API+'/health',timeout=15);health.raise_for_status()
assessment=session.get(API+'/water/assessment',timeout=15);assessment.raise_for_status()
items=session.get(API+'/water/series',timeout=15);items.raise_for_status()
display(health.json());print('Scientific assessment:',assessment.json()['state'])
display(pd.DataFrame(items.json()))
item=next(i for i in items.json() if 'candidate-series/terrain_prior/nam-theun1' in i['key'])
r=session.get(API+item['url'],timeout=15);r.raise_for_status();ob=r.json()['observations'][0]
asset=ob['assets']['composite'];data=session.get(API+asset['url'],timeout=15);data.raise_for_status()
assert __import__('hashlib').sha256(data.content).hexdigest()==asset['sha256']
display(Image(data=data.content,width=650));display({'datetime':ob['datetime'],'bounds':ob['bounds'],'modelId':ob['modelId'],'statistics':ob['statistics']})
if 'context-provenance' in ob['assets']:
    provenance=session.get(API+ob['assets']['context-provenance']['url'],timeout=15);provenance.raise_for_status()
    display(provenance.json()['scientific_inputs'])''')
    md('''## Every compatible observation, including cloudy optical periods

`aoi` acquisition does not require an optical reference or choose dates by model score. Its bounded AOI/date query preserves failures, splits truncated metadata queries, groups geometry and reports partial runs/transfer use. `--context-cache .cache/terrain-context` explicitly enables public physical-context acquisition when needed. Radar and ancillary caps are separate.

The cached Hidroituango demonstration, when available, is unscored inference during a period without a usable close optical reference. It must not be presented as accuracy validation. Default notebook execution does not make new satellite requests.''')
    code('''demo=ROOT/'outputs/round5/hidroituango-cloud-demo/batch.json'
if demo.exists():
    batch=read_json(demo)
    display({k:batch[k] for k in ['state','complete_for_compatible_query','candidate_observations','acquired_observations','radar_body_bytes']})
else:print('Cloud-period acquisition demonstration has not completed; no accuracy claim.')
RUN_ONLINE=False
print('Online mode enabled:',RUN_ONLINE)
print('Example: python -m nisar_water.cli aoi data/round5/hidroituango-aoi.geojson --start 2026-10-10 --end 2026-10-27 --model models/agreement-review --context-cache .cache/terrain-context --output outputs/NEW_RUN --transfer-cap-mib 96')''')
    md('''## Earth Engine and team review handoff

Access to `nisac-512902` was verified using existing credentials, and matching dated EE optical sources were queried. No token is in this notebook. The original13 packets support label-quality/training work; the five fresh-region packets are preferable for an independent expert reference assessment. All currently fitted labels remain automated.

Use `review/earthengine-round5/review_round5.js` in the existing Code Editor. It shows no model predictions, supplies case-specific water/land/exclude drawing layers, and only prepares an export task when the reviewer requests it. No Drive export was started by the assistant. The versioned importer checks source hashes and attribution without rewriting frozen experiments.

Next scientific priorities are independent recent-reservoir/narrow-valley labels, terrain-staleness handling, calibrated noise/incidence features and causal temporal context. A larger network or stronger historical mask alone is not established as a solution. Separate labelled tasks remain necessary for farms, deforestation and wildfire.''')
    code('''display(read_json(ROOT/'evidence/round4-earthengine-access.json'))
review_index=read_json(ROOT/'review/round5/index.json')
display(pd.DataFrame(review_index['items'])[['key','scene_id','status']])
print('Expert review labels imported into these fits:0')
print('Guide:',ROOT/'review/earthengine-round5/README.md')
print('Import: python -m nisar_water.cli import-review --packet SOURCE.json --annotations REVIEWED.geojson --output review/imported')
assert sha256(ROOT/'models/round5/selected/model.json')==F5['selected_model_sha256']
print('No selected model or threshold changed after test.')''')
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'NISAR Water (Python3.13)','language':'python','name':'nisar-water'},
        'language_info':{'name':'python','version':'3.13'},'nisar_water':{'round4_test_images':12,'round5_test_images':5,
            'freeze_sha256':sha256(ROOT/'outputs/round5/freeze.json'),'default_satellite_downloads':False}})
    path=ROOT/'notebooks/NISAR_Water_Improved_Classifier.ipynb';nbf.write(notebook,path);print(path)

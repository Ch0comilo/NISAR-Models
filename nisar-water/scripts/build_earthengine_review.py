"""Use the team's EE project for blinded, dated optical review packets."""
from pathlib import Path
import json
import sys,argparse
from urllib.parse import urlparse,urljoin
import ee
import httplib2
import requests
import numpy as np
import rasterio
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json,sha256,utcnow
from nisar_water.round3_evaluation import load_scene
from nisar_water.round3_results import cohort_specs

def bounded_download(url,path,cap=32*1024**2):
    session=requests.Session();session.trust_env=False;length=0
    try:
        for _ in range(4):
            host=urlparse(url).hostname or ''
            if urlparse(url).scheme!='https' or not (host.endswith('.googleapis.com') or host=='storage.googleapis.com'):
                raise ValueError('Unapproved Earth Engine download host')
            response=session.get(url,stream=True,timeout=30,allow_redirects=False)
            if response.status_code in (301,302,303,307,308):
                url=urljoin(url,response.headers['Location']);response.close();continue
            response.raise_for_status()
            if int(response.headers.get('Content-Length','0'))>cap:raise ValueError('Review crop exceeds32MiB')
            temporary=path.with_suffix('.part')
            with temporary.open('wb') as handle:
                for chunk in response.iter_content(256*1024):
                    length+=len(chunk)
                    if length>cap:raise ValueError('Review crop exceeds32MiB')
                    handle.write(chunk)
            response.close();temporary.replace(path);return length
        raise ValueError('Too many Earth Engine download redirects')
    finally:session.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=int,choices=[3,5],default=3);args=parser.parse_args()
    ee.Initialize(project='nisac-512902',http_transport=httplib2.Http(timeout=30));ee.data.setDeadline(30000)
    folder=ROOT/('review/earthengine' if args.experiment==3 else 'review/earthengine-round5');folder.mkdir(parents=True,exist_ok=True)
    packets=ROOT/f'review/round{args.experiment}'
    cases=[];checks=[]
    specs=(sum([cohort_specs(name) for name in ('challenge','shorelines','test')],[]) if args.experiment==3 else
           [s for s in read_json(ROOT/'data/round5-prepared/manifest.json')['scenes'] if s['split']=='test'])
    for spec in specs:
        o,_,_,side=load_scene(spec);ref=side['provenance']['reference'];key='review-'+spec['sha256'][:12]
        tile=ref['source_id'].split('_')[1];time=ref['reference_date']
        case={'id':spec['id'],'key':key,'aoi':o.metadata['aoi'],'radar_utc':spec['acquired_at'],'optical_utc':time,
              'mgrs_tile':tile,'scene_sha256':spec['sha256'],'local_packet':str((packets/key).resolve())}
        cases.append(case);print('EE reference',spec['region'],spec['acquired_at'][:10],flush=True)
        collection=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(ee.Geometry(case['aoi'])).filterDate(
            ee.Date(time).advance(-10,'minute'),ee.Date(time).advance(10,'minute')).filter(ee.Filter.eq('MGRS_TILE',tile))
        count=collection.size().getInfo()
        check={'scene_id':spec['id'],'packet':key,'project':'nisac-512902','exact_reference_window_minutes':20,'matching_assets':count,
               'state':'available' if count else 'unavailable','expert_review_completed':False}
        if count:
            image=ee.Image(collection.sort('system:index',False).first())
            source=image.toDictionary(['system:id','system:index','system:time_start','PROCESSING_BASELINE','MGRS_TILE','SPACECRAFT_NAME']).getInfo()
            check['source']=source
            # Two challenging, already exposed TRAIN scenes exercise actual
            # authenticated EE image reads. No new validation/test labels change.
            if spec['region'] in ('denver-airport','powell-shore') and not any(c.get('downloaded') and c['scene_id'].split('-2026')[0]==spec['region'] for c in checks):
                packet=packets/key;path=packet/'earthengine-optical-rgb.tif'
                if not path.exists():
                    url=image.select(['B4','B3','B2']).toUint16().getDownloadURL({'crs':str(o.crs),
                        'crs_transform':list(o.transform)[:6],'dimensions':[o.valid.shape[1],o.valid.shape[0]],'format':'GEO_TIFF'})
                    # Signed download URLs are consumed in memory, never saved.
                    check['body_bytes']=bounded_download(url,path)
                else:check['body_bytes']=0
                with rasterio.open(path) as src:
                    assert src.shape==o.valid.shape and src.crs==o.crs and np.allclose(list(src.transform)[:6],list(o.transform)[:6])
                    values=src.read().astype(np.float32)*.0001
                rgb=np.moveaxis(np.uint8(np.clip(values/.3,0,1)**(1/1.2)*255),0,-1)
                Image.fromarray(rgb).save(packet/'earthengine-optical.png')
                check.update(downloaded=True,asset_path=str(path.resolve()),sha256=sha256(path),
                    calibration='EE S2_SR_HARMONIZED DN*0.0001; no additional BOA offset',model_predictions_included=False)
                write_json(packet/'earthengine-source.json',check)
        checks.append(check);write_json(ROOT/('evidence/round4-earthengine-review-check.json' if args.experiment==3 else 'evidence/round5-earthengine-review-check.json'),{'checked_at':utcnow(),'records':checks,'project':'nisac-512902'})
    write_json(folder/'cases.json',{'cases':cases,'no_model_predictions':True,'expert_reviews_completed':0})
    js='''// Paste into https://code.earthengine.google.com/?project=nisac-512902
// Blinded reference annotation; no model predictions or historical water shown.
// Save/download your annotations before switching case or rerunning this script.
var CASES = '''+json.dumps(cases,indent=2)+''';
var CASE_INDEX = 0;
var REVIEWER = 'ENTER_YOUR_NAME';
var INDEPENDENT_EXPERT = false; // Set true only for a real qualified independent review.
var selected = CASES[CASE_INDEX];
var aoi = ee.Geometry(selected.aoi);
Map.centerObject(aoi, 13);
Map.addLayer(ee.FeatureCollection([ee.Feature(aoi)]).style({color:'ffcc00',fillColor:'00000000'}),{},'Requested AOI');
print('Case, radar UTC, optical UTC:', selected.id, selected.radar_utc, selected.optical_utc);
var scenes = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(aoi)
  .filterDate(ee.Date(selected.optical_utc).advance(-10,'minute'),ee.Date(selected.optical_utc).advance(10,'minute'))
  .filter(ee.Filter.eq('MGRS_TILE', selected.mgrs_tile));
scenes.size().evaluate(function(n) {
  if (!n) { print('No matching image: use dated local packet; do not substitute an undated image.'); return; }
  var optical = ee.Image(scenes.sort('system:index',false).first());
  Map.addLayer(optical.multiply(0.0001),{bands:['B4','B3','B2'],min:0,max:0.3,gamma:1.2},'Dated optical reference');
  print('Exact optical source:', optical.get('system:id'), optical.date());
});
var tools = Map.drawingTools();
tools.setLinked(false); // Preserve unrelated Code Editor imports.
tools.setShown(true);
tools.setDrawModes(['polygon','rectangle']);
var classes = ['open_water','nonwater','exclude'];
var colors = ['238ee8','55a34a','efad2c'];
classes.forEach(function(name,i) {
  var fullName=selected.key+'::'+name;
  var found=false;
  tools.layers().forEach(function(layer) { if(layer.getName()===fullName) found=true; });
  if(!found) tools.layers().add(ui.Map.GeometryLayer([],fullName,colors[i]));
});
print('Choose a named drawing layer, then draw polygons. Unknown/unannotated space stays unknown.');
print('Cloud, optical shadow and uncertain boundaries should be excluded. Review both acquisition times.');
// This button only prepares an export task. The reviewer starts it in Tasks.
var exportButton = ui.Button('Prepare reviewed GeoJSON task', function() {
  if (REVIEWER==='ENTER_YOUR_NAME') { print('Fill REVIEWER first.'); return; }
  var features=[];
  tools.layers().forEach(function(layer) {
    var prefix=selected.key+'::';
    if(layer.getName().indexOf(prefix)!==0) return;
    var label=layer.getName().slice(prefix.length);
    if (classes.indexOf(label)<0) return;
    layer.geometries().forEach(function(geometry) {
      features.push(ee.Feature(geometry, {'class':label,'reviewer':REVIEWER,
        'independent_expert':INDEPENDENT_EXPERT,'reviewed_at_utc':new Date().toISOString(),
        'scene_id':selected.id,'scene_sha256':selected.scene_sha256,'radar_utc':selected.radar_utc,
        'optical_utc':selected.optical_utc,'reference_collection':'COPERNICUS/S2_SR_HARMONIZED'}));
    });
  });
  if (!features.length) { print('No polygons drawn.'); return; }
  Export.table.toDrive({collection:ee.FeatureCollection(features),description:selected.key.replace(/-/g,'_'),
    fileNamePrefix:selected.key+'_annotations',fileFormat:'GeoJSON'});
  print('Prepared only. Review the task, start it manually, then return the GeoJSON with this case ID.');
});
Map.add(exportButton);
'''
    (folder/('review_round3.js' if args.experiment==3 else 'review_round5.js')).write_text(js,encoding='utf-8')
    print('Prepared',len(cases),'Earth Engine review cases; no human labels or remote exports fabricated')

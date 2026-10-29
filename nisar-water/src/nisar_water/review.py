"""Blinded reference preparation and explicit human polygon import."""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject,Resampling,transform_geom
from rasterio.features import rasterize
from shapely.geometry import shape
from .common import read_json,write_json,sha256


def aligned_rgb(observation,sidecar):
    reference=sidecar.get('provenance',{}).get('reference',{})
    asset=reference.get('local_assets',{}).get('visual',{})
    if not asset:return None
    if sha256(asset['path'])!=asset['sha256']:raise ValueError('Optical visual hash mismatch')
    target=np.zeros((3,*observation.valid.shape),np.uint8)
    with rasterio.open(asset['path']) as src:
        for band in range(3):
            reproject(src.read(band+1),target[band],src_transform=src.transform,src_crs=src.crs,
                      dst_transform=observation.transform,dst_crs=observation.crs,resampling=Resampling.bilinear)
    return np.moveaxis(target,0,-1)


def annotation_labels(collection,observation):
    """Unannotated space is unknown, not implicitly nonwater."""
    if collection.get('type')!='FeatureCollection':raise ValueError('Expected WGS84 GeoJSON FeatureCollection')
    groups={'open_water':[],'nonwater':[],'exclude':[]}
    for feature in collection.get('features',[]):
        category=feature.get('properties',{}).get('class')
        geom=shape(feature.get('geometry'))
        if category not in groups or not geom.is_valid or geom.is_empty or geom.geom_type not in ('Polygon','MultiPolygon'):
            raise ValueError('Review features require valid polygons and class open_water/nonwater/exclude')
        groups[category].append((transform_geom('EPSG:4326',observation.crs,feature['geometry']),1))
    masks={k:(rasterize(v,out_shape=observation.valid.shape,transform=observation.transform,dtype='uint8').astype(bool)
              if v else np.zeros(observation.valid.shape,bool)) for k,v in groups.items()}
    if (masks['open_water']&masks['nonwater']&~masks['exclude']).any():raise ValueError('Conflicting review water/land polygons')
    labels=np.full(observation.valid.shape,255,np.uint8)
    labels[masks['nonwater']]=0;labels[masks['open_water']]=1
    labels[masks['exclude']|~observation.valid|(observation.weights<=0)|observation.exclusions]=255
    return labels


def create_review_pack(specs,directory):
    from .round3_evaluation import load_scene
    from .exports import write_raster
    from PIL import Image
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);index=[]
    for spec in specs:
        o,_,_,sidecar=load_scene(spec);key='review-'+spec['sha256'][:12];folder=directory/key;folder.mkdir(exist_ok=True)
        gray=np.uint8(np.clip(np.nan_to_num((10*np.log10(np.where(o.valid,o.power[0],1))+30)/33),0,1)*255)
        Image.fromarray(gray).save(folder/'radar-hh.png')
        rgb=aligned_rgb(o,sidecar)
        if rgb is not None:
            Image.fromarray(rgb).save(folder/'optical.png')
            write_raster(folder/'optical-rgb.tif',np.moveaxis(rgb,-1,0),o.transform,o.crs)
        write_raster(folder/'radar-power.tif',np.where(o.valid[None],o.power,np.nan),o.transform,o.crs,nodata=np.nan)
        template=folder/'annotations.geojson'
        if not template.exists():write_json(template,{'type':'FeatureCollection','features':[]})
        write_json(folder/'source.json',{'scene_id':spec['id'],'scene_sha256':spec['sha256'],'scene_path':spec['path'],
            'crs':str(o.crs),'transform':list(o.transform)[:6],'radar_time':spec['acquired_at'],
            'optical_time':sidecar.get('provenance',{}).get('reference',{}).get('reference_date'),
            'review_status':'unreviewed','independent_expert':False,'reviewer':None,
            'model_predictions_included':False,'class_properties':['open_water','nonwater','exclude']})
        index.append({'key':key,'scene_id':spec['id'],'source_sha256':spec['sha256'],'directory':str(folder.resolve()),'status':'unreviewed'})
    write_json(directory/'index.json',{'items':index,'expert_reviews_completed':0,
        'instructions':'Draw WGS84 polygons with class open_water/nonwater/exclude; leave uncertain shoreline and unannotated space unknown. Inspect optical/radar acquisition times and avoid predictions during reference creation.'})
    return index

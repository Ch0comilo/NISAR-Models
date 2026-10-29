from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import mapping,box
from rasterio.warp import transform_geom
from nisar_water.common import write_json,read_json,sha256
from nisar_water.review_import import import_review


def packet(tmp):
    scene=tmp/'scene.npz';transform=np.array([20,0,500000,0,-20,4400000])
    np.savez_compressed(scene,power=np.full((2,32,32),.01,np.float32),valid=np.ones((32,32),bool),
        transform=transform,epsg=np.array(32610),aoi_weights=np.ones((32,32),np.float32))
    source=write_json(tmp/'source.json',{'scene_id':'fixture','scene_path':str(scene),'scene_sha256':sha256(scene)})
    geometry=transform_geom('EPSG:32610','EPSG:4326',mapping(box(500100,4399500,500300,4399700)))
    feature={'type':'Feature','geometry':geometry,'properties':{'class':'open_water','reviewer':'Test reviewer',
        'reviewed_at_utc':'2026-10-28T15:00:00Z','scene_id':'fixture','scene_sha256':sha256(scene),'independent_expert':False}}
    return source,feature


def test_human_import_keeps_unannotated_land_unknown_and_does_not_change_source(tmp_path):
    source,feature=packet(tmp_path);scene=Path(read_json(source)['scene_path']);before=sha256(scene)
    annotation=write_json(tmp_path/'annotations.geojson',{'type':'FeatureCollection','features':[feature]})
    receipt=import_review(source,annotation,tmp_path/'imported')
    assert receipt['water_cells']==100 and receipt['land_cells']==0 and receipt['unknown_cells']==924
    assert receipt['label_kind']=='team_reviewed' and not receipt['frozen_experiments_changed']
    assert sha256(scene)==before
    with pytest.raises(FileExistsError):import_review(source,annotation,tmp_path/'imported')


def test_wrong_scene_attribution_and_unattributed_review_are_rejected(tmp_path):
    source,feature=packet(tmp_path);annotation=tmp_path/'annotations.geojson'
    feature['properties']['scene_id']='wrong'
    write_json(annotation,{'type':'FeatureCollection','features':[feature]})
    with pytest.raises(ValueError,match='different scene'):import_review(source,annotation,tmp_path/'out')
    feature['properties']['scene_id']='fixture';feature['properties']['reviewer']='ENTER_YOUR_NAME'
    write_json(annotation,{'type':'FeatureCollection','features':[feature]})
    with pytest.raises(ValueError,match='attribution'):import_review(source,annotation,tmp_path/'out')

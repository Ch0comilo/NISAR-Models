import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box,mapping
from rasterio.warp import transform_geom
from nisar_water.inputs import Observation
from nisar_water.opera_adapter import ancillary_sources,PublicRangeFile
from nisar_water.review import annotation_labels
from nisar_water.round3_evaluation import score,aggregate
from nisar_water.archive import all_candidates,process_aoi


def fixture_observation():
    return Observation(np.full((2,32,32),.01,np.float32),np.ones((32,32),bool),
                       Affine(20,0,500000,0,-20,4400000),32610,{})


def test_ancillary_tiles_cross_longitude_seams_without_duplicate_global_sources():
    sources=ancillary_sources((-120.04,39.05,-119.99,39.10))
    assert len(sources['dem'])==len(sources['hand'])==2
    assert len(sources['landcover'])==2 and len(sources['glad'])==2
    with pytest.raises(ValueError,match='bounded'):ancillary_sources((-120,39,-117,40))


def test_ancillary_reader_never_sends_a_request_to_an_unapproved_host():
    class NoNetwork:
        def head(self,*args,**kwargs):raise AssertionError('Unexpected network access')
    with pytest.raises(ValueError,match='host'):PublicRangeFile('https://example.com/file.tif',{}, {},NoNetwork())


def test_review_import_keeps_unannotated_space_unknown_and_rejects_conflicts():
    o=fixture_observation();geometry=transform_geom(o.crs,'EPSG:4326',mapping(box(500100,4399500,500300,4399700)))
    f={'type':'Feature','geometry':geometry,'properties':{'class':'open_water'}}
    collection={'type':'FeatureCollection','features':[f]}
    labels=annotation_labels(collection,o)
    assert (labels==1).sum()==100 and (labels==0).sum()==0 and (labels==255).any()
    land={**f,'properties':{'class':'nonwater'}}
    with pytest.raises(ValueError,match='Conflicting'):annotation_labels({'type':'FeatureCollection','features':[f,land]},o)


def test_unclassified_reference_is_reported_as_withheld_without_becoming_land():
    o=fixture_observation();truth=np.ones(o.valid.shape,np.uint8);evaluate=np.ones_like(o.valid)
    labels=np.ones_like(truth);labels[:16]=254;scores=np.ones(o.valid.shape,np.float32);scores[:16]=np.nan
    row=score({'id':'fixture','region':'r','acquired_at':'2026-01-01','split':'test'},o,truth,evaluate,labels,scores,.5)
    assert row['reference_coverage']==.5 and row['withheld_reference_ha']>0
    summary=aggregate([row]);assert summary['pooled']['recall']==1 and summary['regions']['r']['reference_coverage']==.5


def test_archive_splits_truncated_queries_and_selects_revision_without_pixel_access(tmp_path,monkeypatch):
    granule='NISAR_L2_PR_GCOV_022_034_A_021_4005_DHDH_A_20260604T130934_20260604T131009_P05023_N_F_J_001'
    base={'compatible':True,'coverage_fraction':1,'granule_id':granule,'acquired_at':'2026-06-04T13:09:34Z'}
    calls=[]
    def discover(aoi,start,end,work,reference):
        calls.append((start,end))
        if start=='2026-06-01' and end=='2026-06-04':return {'state':'ready','truncated':True,'candidates':[]}
        revision=1 if end=='2026-06-02' else 2
        return {'state':'ready','truncated':False,'candidates':[{**base,'id':'G123:'+str(revision),'revision_id':revision}]}
    monkeypatch.setattr('nisar_water.archive.discover',discover)
    selected,audit=all_candidates({},'2026-06-01','2026-06-04',tmp_path,'unused')
    assert len(calls)==3 and len(selected)==1 and selected[0]['revision_id']==2
    assert audit['superseded_processing_revisions']==['G123:1']


def test_archive_exhausted_transfer_budget_is_partial_without_fake_observation(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('nisar_water.archive.load_model',lambda path:SimpleNamespace(fingerprint='fixture'))
    monkeypatch.setattr('nisar_water.archive.all_candidates',lambda *args:([{'id':'fixture'}],{}))
    def no_network(*args,**kwargs):raise AssertionError('Budget must prevent acquisition')
    monkeypatch.setattr('nisar_water.archive.acquire',no_network)
    result=process_aoi({},'2026-01-01','2026-01-02','unused',tmp_path,'unused',transfer_cap_bytes=1)
    assert result['state']=='partial' and result['acquired_observations']==0
    assert result['receipts'][0]['state']=='transfer_budget_exhausted'

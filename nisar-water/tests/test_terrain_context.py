from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from nisar_water.common import write_json,sha256
from nisar_water import terrain_context as terrain


def test_constant_and_planar_dem_derivatives_have_physical_units():
    shape=(80,85)
    t=Affine(30,0,500000,0,-30,4200000)
    x=500000+(np.arange(shape[1])+.5)*30
    dem=np.tile((x-x[0])*.1,(shape[0],1)).astype(np.float32)
    valid=np.ones(shape,bool)
    hand=np.full(shape,8,np.float32)
    planes,available=terrain.physical_features(dem,valid,t,hand,valid,dem,valid,t,CRS.from_epsg(32611))
    point=(40,42)
    assert available[point]
    assert planes[0,*point]==pytest.approx(np.degrees(np.arctan(.1)),abs=1e-4)
    # Here the test grid is30m: thirty inter-centre steps *30m *0.1 rise/metre.
    assert planes[1,*point]==pytest.approx(90,abs=1e-4)
    assert planes[2,*point]==pytest.approx(np.log1p(8))
    assert planes[3,*point]==pytest.approx(8)
    assert planes[4,*point]==pytest.approx(8)


def test_missing_hand_and_dem_neighbors_remain_unknown_not_zero():
    shape=(80,85);t=Affine(20,0,500000,0,-20,4200000)
    dem=np.full(shape,100,np.float32);hand=np.full(shape,3,np.float32)
    valid=np.ones(shape,bool);dem_valid=valid.copy();hand_valid=valid.copy()
    dem_valid[40,42]=False;dem[40,42]=np.nan
    hand_valid[60,60]=False;hand[60,60]=np.nan
    planes,available=terrain.physical_features(dem,dem_valid,t,hand,hand_valid,dem,dem_valid,t,CRS.from_epsg(32611))
    assert not available[40,42] and np.isnan(planes[0,40,42]) and np.isnan(planes[1,40,42])
    assert not available[60,60] and np.isnan(planes[3,60,60])
    assert np.isfinite(planes[:,20,20]).all()
    assert not available[0].any()


def test_nonnegative_height_means_remain_nonnegative_after_running_filter_roundoff():
    shape=(90,100);t=Affine(20,0,500000,0,-20,4200000)
    dem=np.full(shape,100,np.float32);valid=np.ones(shape,bool)
    hand=np.zeros(shape,np.float32)
    hand[:,:35]=np.random.default_rng(20261028).uniform(0,1000,(90,35))
    planes,available=terrain.physical_features(dem,valid,t,hand,valid,dem,valid,t,CRS.from_epsg(32611))
    interior=available[20:70,45:80]
    assert interior.all()
    # Every source height in these neighborhoods is exactly zero; numerical
    # cancellation must not make their physically valid means negative/nodata.
    np.testing.assert_allclose(planes[3,20:70,45:80],0,atol=1e-9,rtol=0)
    assert not (planes[:,available]<0).any()


def test_zero_hand_is_a_real_measurement_and_historical_zero_does_not_mask_new_water():
    occurrence=np.zeros((160,160),np.float32)
    valid=np.ones_like(occurrence,bool)
    planes,distance_valid,occ_valid=terrain.prior_features(occurrence,valid)
    assert occ_valid.all()
    assert planes[0,80,80]==0
    assert planes[1,80,80]==1000
    assert distance_valid[80,80]
    # A valid0% occurrence value is retained, not changed to nodata or a water mask.
    assert np.isfinite(planes[0]).all()


def test_prior_distance_does_not_invent_water_at_crop_border_or_across_unknowns():
    occurrence=np.zeros((160,160),np.float32)
    valid=np.ones_like(occurrence,bool)
    occurrence[80,80]=80
    values,available,_=terrain.prior_features(occurrence,valid)
    assert values[1,80,80]==0
    assert values[1,80,83]==pytest.approx(60)
    valid[80,82]=False
    values,available,occ_valid=terrain.prior_features(occurrence,valid)
    assert not available[80,83] and np.isnan(values[1,80,83])
    assert occ_valid[80,83] and values[0,80,83]==0
    assert np.isnan(values[0,80,82])


def _observation():
    return SimpleNamespace(valid=np.ones((50,50),bool),weights=np.ones((50,50),np.float32),
              transform=Affine(20,0,500000,0,-20,4200000),crs=CRS.from_epsg(32611),metadata={"data_origin":"synthetic_fixture"})


def _sources(tmp_path,*,prior=False):
    crs=CRS.from_epsg(32611)
    t=Affine(30,0,498000,0,-30,4202000)
    result={}
    for name,value in (("dem",100),("hand",0),("reference_water",0)):
        if name=="reference_water" and not prior:continue
        path=tmp_path/(name+".tif")
        values=np.full((200,200),value,np.float32)
        if name=="reference_water":values[:,100:]=90
        with rasterio.open(path,"w",driver="GTiff",width=200,height=200,count=1,dtype="float32",
                           crs=crs,transform=t,nodata=np.nan) as dst:dst.write(values,1)
        record={"kind":name,"cropped_window_path":str(path),"cropped_window_sha256":sha256(path),
                "source_urls":["https://example.invalid/synthetic-fixture-only"],"reuse_origin":"synthetic_fixture"}
        result[name]=record
    return result


def test_covering_source_windows_provide_full_edge_context_and_cache_integrity(tmp_path,monkeypatch):
    source_dir=tmp_path/"source-files";source_dir.mkdir()
    sources=_sources(source_dir,prior=True)
    monkeypatch.setattr(terrain,"_candidate_records",lambda cache:{name:[sources[name]] for name in sources})
    def no_download(*args,**kwargs):raise AssertionError("Covering genuine/local windows must be reused")
    monkeypatch.setattr(terrain,"_download_source",no_download)
    result=terrain.load_context(_observation(),tmp_path/"cache",include_prior=True)
    assert result["names"]==list(terrain.TERRAIN_NAMES+terrain.PRIOR_NAMES)
    assert result["features"].shape==(7,50,50)
    assert result["available"].all()  # Available at target edge because source context was padded.
    assert result["features"][0].max()==0
    assert result["features"][1].max()==0
    assert result["features"][2].max()==0  # HAND=0 is observed, log1p=0.
    assert result["metadata"]["body_bytes"]==0
    assert result["metadata"]["native_nominal_resolution_m"]==30
    assert result["metadata"]["target_pixel_spacing_m"]==20
    cached=terrain.load_context(_observation(),tmp_path/"cache",include_prior=True)
    np.testing.assert_allclose(cached["features"],result["features"],equal_nan=True)
    assert cached["metadata"]["cache_hit"] and cached["metadata"]["body_bytes"]==0
    path=Path(cached["metadata"]["feature_cache_path"])
    with path.open("ab") as f:f.write(b"corruption")
    with pytest.raises(ValueError,match="integrity"):
        terrain.load_context(_observation(),tmp_path/"cache",include_prior=True)


def test_cache_only_missing_sources_cannot_trigger_network_or_fake_context(tmp_path,monkeypatch):
    monkeypatch.setattr(terrain,"DEFAULT_CACHE",tmp_path/"default-cache")
    monkeypatch.setattr(terrain,"_candidate_records",lambda cache:{name:[] for name in terrain.SOURCE_DESCRIPTIONS})
    def forbidden(*args,**kwargs):raise AssertionError("No implicit network")
    monkeypatch.setattr(terrain,"_download_source",forbidden)
    result=terrain.load_context(_observation())
    assert not result["available"].any()
    assert np.isnan(result["features"]).all()
    assert result["metadata"]["source_hashes"]=={}
    assert result["metadata"]["network_permitted"] is False
    assert result["metadata"]["observed_context_withheld_ha"]==pytest.approx(100)


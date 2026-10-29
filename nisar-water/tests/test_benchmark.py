import numpy as np
import rasterio
from fastapi.testclient import TestClient
from nisar_water.benchmark import confusion_metrics,reference_support
from nisar_water.online import select_dates
from nisar_water.api import create_app
from nisar_water.common import write_json
from nisar_water.exploration import to_notebook_cube
from test_science import obs,NAME


def test_water_only_support_does_not_invent_precision_or_land_accuracy():
    m=confusion_metrics([1,1,1],[0,0,0],[.04,.04,.04])
    assert m['recall']==0
    assert m['precision'] is None
    assert m['false_positive_rate'] is None
    assert m['missed_water_ha']==.12


def test_cloud_shadow_and_unknown_never_become_reference_negatives(tmp_path):
    o=obs()
    scl=np.full(o.valid.shape,9,np.uint8)
    scl[5:30,5:17]=6
    scl[5:30,23:39]=4
    scl[32:35,5:40]=3
    path=tmp_path/'scl.tif'
    with rasterio.open(path,'w',driver='GTiff',height=scl.shape[0],width=scl.shape[1],count=1,
                       dtype='uint8',crs=o.crs,transform=o.transform) as dst:dst.write(scl,1)
    raw,classes,evaluate=reference_support(o,{'local_assets':{'scl':{'path':str(path)}}})
    assert evaluate.any()
    assert set(classes[evaluate])=={4,6}
    assert not evaluate[np.isin(raw,[0,1,2,3,7,8,9,10,11])].any()
    assert not classes[5].any()


def test_selection_uses_coverage_and_one_geometry_before_model_scores():
    def candidate(day,track=135,coverage=1):
        name=NAME.replace('20260623','202606'+str(day).zfill(2)).replace('_135_A_',f'_{track:03d}_A_')
        return {'id':name,'granule_id':name.removesuffix('.h5'),'acquired_at':f'2026-06-{day:02d}T13:17:53Z',
                'coverage_fraction':coverage,'compatible':True,'revision_id':1,'record':{}}
    choices=select_dates([candidate(1),candidate(13),candidate(25),candidate(26,track=42),candidate(27,coverage=.3)])
    assert len(choices)==2
    assert [c['acquired_at'][:10] for c in choices]==['2026-06-01','2026-06-25']
    assert all('_135_A_' in c['granule_id'] for c in choices)


def test_benchmark_and_assessment_routes_have_explicit_missing_states(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/water/benchmark').status_code==404
        assert client.get('/water/assessment').status_code==404
        write_json(tmp_path/'report.json',{'origin':'synthetic_fixture','rows':[]})
        write_json(tmp_path/'assessment.json',{'state':'synthetic_fixture'})
        assert client.get('/water/benchmark').json()['origin']=='synthetic_fixture'
        assert client.get('/water/assessment').json()['state']=='synthetic_fixture'


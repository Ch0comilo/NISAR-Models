import numpy as np
from affine import Affine
from nisar_water.inputs import Observation
from nisar_water import round5_evaluation as ev


def test_recent_water_cannot_be_hidden_by_high_total_water_recall():
    shape=(100,100);truth=np.zeros(shape,np.uint8);truth.flat[:5000]=1
    scores=truth.astype(np.float32);scores.flat[:200]=0
    labels=(scores>=.5).astype(np.uint8);occurrence=np.full(shape,90,np.float32);occurrence.flat[:200]=0
    o=Observation(np.full((2,*shape),.01,np.float32),np.ones(shape,bool),Affine(20,0,500000,0,-20,4400000),32610,{})
    row=ev.score_scene({'id':'fixture','region':'r','acquired_at':'2026-01-01','split':'validation'},o,truth,np.ones(shape,bool),labels,scores,.5,occurrence)
    summary=ev.aggregate([row]);r=summary['regions']['r'];checks=ev.gate_checks(summary)['r']
    assert r['recall']==.96 and r['precision']==1 and r['reference_coverage']==1
    assert checks['full_reference_water_recovery'] and not checks['low_historical_occurrence_water_recovery']
    assert not ev.passes(summary)

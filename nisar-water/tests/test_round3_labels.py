import pytest
from nisar_water.round3_data import optical_calibration


def test_applied_provider_offset_is_not_subtracted_twice():
    scale,offset,basis=optical_calibration({'properties':{'earthsearch:boa_offset_applied':True}},
                                         {'scale':.0001,'offset':-.1})
    assert 382*scale+offset==pytest.approx(.0382)
    assert offset==0


def test_unapplied_offset_is_applied_exactly_once():
    scale,offset,_=optical_calibration({'properties':{'earthsearch:boa_offset_applied':False}},
                                     {'scale':.0001,'offset':-.1})
    assert 1382*scale+offset==pytest.approx(.0382)


def test_ambiguous_modern_reference_is_withheld():
    with pytest.raises(ValueError,match='Ambiguous'):
        optical_calibration({'properties':{'s2:processing_baseline':'05.13'}},{'scale':.0001,'offset':-.1})


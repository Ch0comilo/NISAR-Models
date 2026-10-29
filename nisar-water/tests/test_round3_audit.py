import importlib.util
from pathlib import Path
import numpy as np
import h5py

spec=importlib.util.spec_from_file_location("round3_audit",Path(__file__).resolve().parents[1]/"scripts/audit_round3_inputs.py")
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_independent_aggregation_preserves_power_and_requires_complete_native_support():
    raw=np.array([[[1,9,2,2],[1,9,2,2]],[[.1,.9,.2,.2],[.1,.9,.2,.2]]],np.float32)
    valid=np.ones((2,4),bool)
    result,good=audit.strict_aggregate(raw,valid,10)
    np.testing.assert_allclose(result[:,0,0],[5,.5])
    assert good.all()
    valid[0,0]=False
    result,good=audit.strict_aggregate(raw,valid,10)
    assert not good[0,0] and np.isnan(result[:,0,0]).all()
    assert good[0,1]


def test_context_interior_requires_full_300_m_radius_without_reflected_exterior():
    mask=audit.crop_interior_mask((45,50),15)
    assert mask.sum()==15*20
    assert not mask[:15].any() and not mask[:,:15].any()
    assert mask[15,15] and mask[29,34]
    assert not audit.crop_interior_mask((30,30),15).any()


def test_quantiles_do_not_invent_missing_measurements():
    assert audit.quantify(np.empty((0,41))) is None


def test_metadata_flags_remain_unknown_when_absent_and_cube_is_not_zero_height_feature(tmp_path):
    with h5py.File(tmp_path/"metadata-fixture.h5","w") as source:
        base="science/LSAR/GCOV/metadata"
        source[base+"/processingInformation/parameters/noiseCorrectionApplied"]=np.bytes_("False")
        source[base+"/processingInformation/parameters/radiometricTerrainCorrectionApplied"]=np.bytes_("True")
        source.create_dataset(base+"/radarGrid/incidenceAngle",shape=(3,4,5),dtype="float32")
        result=audit.inspect_extra_metadata(h5=source)
    assert result["processing_flags"]["noiseCorrectionApplied"] is False
    assert result["processing_flags"]["radiometricTerrainCorrectionApplied"] is True
    assert result["processing_flags"]["shadowMaskingApplied"] is None
    assert result["incidence_angle_status"]=="cube_available_not_interpolated_at_terrain_height"
    assert result["radiometric_arithmetic_applied"] is False
    assert "incidence_angle" not in result["small_scalar_metadata"]


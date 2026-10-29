import numpy as np
from nisar_water.agreement_model import agreement_labels


def test_disagreement_is_unknown_including_possible_new_water_and_possible_water_loss():
    a=np.array([[1,1,0,0,254,1]],np.uint8);b=np.array([[1,0,1,0,1,255]],np.uint8)
    observed=np.ones(a.shape,bool);labels,disagreement=agreement_labels(a,b,observed)
    np.testing.assert_array_equal(labels,[[1,254,254,0,254,254]])
    np.testing.assert_array_equal(disagreement,[[False,True,True,False,False,False]])
    observed[0,0]=False;labels,_=agreement_labels(a,b,observed)
    assert labels[0,0]==255

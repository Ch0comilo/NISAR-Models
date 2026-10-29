from copy import deepcopy
from nisar_water.context_provenance import scientific_context


def info():
    return {'ablation':'terrain','context_metadata':{'profile':'physical-v2','prior_profile':'prior-v1',
        'source_hashes':{'dem':'d','hand':'h','reference_water':'w'},'sources':{},'cache_hit':False,
        'target_epsg':32611,'target_transform6':[20,0,500000,0,-20,4400000]}}


def test_context_identity_binds_used_sources_without_cache_state_or_unused_prior():
    original=info();before=scientific_context(original)['input_sha256']
    changed=deepcopy(original);changed['context_metadata'].update(cache_hit=True,feature_cache_path='another/cache')
    changed['context_metadata']['source_hashes']['reference_water']='unused-different-prior'
    assert scientific_context(changed)['input_sha256']==before
    changed['context_metadata']['source_hashes']['dem']='different-dem'
    assert scientific_context(changed)['input_sha256']!=before
    changed['ablation']='terrain_prior';a=scientific_context(changed)['input_sha256']
    changed['context_metadata']['source_hashes']['reference_water']='new-prior'
    assert scientific_context(changed)['input_sha256']!=a
    assert scientific_context({'ablation':'radar'}) is None

"""Scientific input identity independent of cache hits and absolute cache paths."""
from .common import identity


def scientific_context(info):
    ablation=info.get('ablation')
    if ablation not in ('terrain','terrain_prior'):return None
    metadata=info.get('context_metadata',{})
    kinds=['dem','hand']+(['reference_water'] if ablation=='terrain_prior' else [])
    inputs={'ablation':ablation,'physical_profile':metadata.get('profile'),
            'prior_profile':metadata.get('prior_profile') if ablation=='terrain_prior' else None,
            'source_crop_sha256':{k:metadata.get('source_hashes',{}).get(k) for k in kinds},
            'target_epsg':metadata.get('target_epsg'),
            'target_transform':metadata.get('target_transform6',metadata.get('target_transform'))}
    return {'input_sha256':identity(inputs),'scientific_inputs':inputs,
            'sources':{k:metadata.get('sources',{}).get(k) for k in kinds},
            'source_hash_scope':'Actual cropped ancillary windows, not full global source files',
            'native_nominal_resolution_m':metadata.get('native_nominal_resolution_m'),
            'target_pixel_spacing_m':metadata.get('target_pixel_spacing_m'),
            'required_context_available_fraction':info.get('required_context_available_fraction'),
            'required_context_withheld_ha':info.get('required_context_withheld_ha'),
            'historical_prior_role':'Soft learned feature; never a hard water mask' if ablation=='terrain_prior' else 'Not used',
            'limitations':metadata.get('limitations',[])}

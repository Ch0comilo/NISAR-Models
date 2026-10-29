"""Post-test diagnostics only; no fitting, masking or threshold changes."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json
if __name__=='__main__':
    records=[]
    for spec in read_json(ROOT/'data/round5-prepared/manifest.json')['scenes']:
        if spec['split']!='test' or spec['region']=='alice-springs-airport':continue
        with np.load(spec['path'],allow_pickle=False) as z:power=z['power'];water=(z['labels']==1)&z['evaluate']&z['valid']
        with np.load(spec['context_path'],allow_pickle=False) as z:context=z['features']
        with np.load(ROOT/'outputs/round5/test-predictions/terrain_prior'/spec['id']/'prediction.npz',allow_pickle=False) as z:prediction=z['labels']
        for name,mask in [('all_reference_water',water),('correct_water',water&(prediction==1)),('missed_water',water&(prediction==0))]:
            if not mask.any():continue
            db=10*np.log10(power[:,mask]);values=context[:,mask]
            records.append({'scene':spec['id'],'support':name,'cells':int(mask.sum()),'median_HH_db':float(np.median(db[0])),
                'median_HV_db':float(np.median(db[1])),'median_static_dem_slope_deg':float(np.nanmedian(values[0])),
                'median_static_hand_log1p':float(np.nanmedian(values[2])),'median_historical_occurrence':float(np.nanmedian(values[5]))})
    write_json(ROOT/'evidence/recent-water-post-test-diagnostic.json',{'records':records,'used_for_training_or_selection':False,
        'interpretation':'Matched radar/weak-reference pixel diagnostics. Brightness/static-terrain differences do not establish wind, layover, RTC or registration as the cause; those require targeted geometry/calibration review.'})
    print(records)

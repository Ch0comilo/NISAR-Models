"""One fixed HGB fit on the same broadened geographic manifest as U-Net."""
from pathlib import Path
from collections import defaultdict
import argparse
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from nisar_water.common import read_json,write_json,sha256,utcnow
from nisar_water.features import FEATURE_NAMES,FEATURE_VERSION,tiles


def train(manifest_path,output):
    manifest_path,output=Path(manifest_path),Path(output)
    if output.exists():raise FileExistsError('Preserve previous fits; choose a new output')
    manifest=read_json(manifest_path);output.mkdir(parents=True)
    parameters={'max_iter':300,'max_leaf_nodes':31,'learning_rate':.05,'min_samples_leaf':40,
                'l2_regularization':10.,'early_stopping':False,'random_state':20261028}
    write_json(output/'plan.json',{'created_at':utcnow(),'manifest_sha256':sha256(manifest_path),'parameters':parameters,
        'sampling':'At most4096 labelled cells per scene (2048 per available class); equal total geographic-region weight; boundary weights retained',
        'test_paths_opened':False,'validation_not_used_for_fit':True})
    parts=[];rng=np.random.default_rng(20261028)
    for scene in manifest['scenes']:
        if scene['split']!='train':continue
        with np.load(scene['path'],allow_pickle=False) as z:
            power,valid,labels,weights=[z[k] for k in ('power','valid','labels','weights')]
        selected=[]
        for code in (0,1):
            indices=np.flatnonzero(valid&(labels==code)&(weights>0))
            if len(indices)>2048:indices=rng.choice(indices,2048,replace=False)
            selected.extend(indices.tolist())
        if not selected:continue
        selected=np.array(selected);rr,cc=np.unravel_index(selected,labels.shape)
        values=np.full((len(selected),len(FEATURE_NAMES)),np.nan,np.float32)
        for sl,f,support in tiles(power,valid):
            take=(rr>=sl[0].start)&(rr<sl[0].stop)&(cc>=sl[1].start)&(cc<sl[1].stop)
            values[take]=f[rr[take]-sl[0].start,cc[take]-sl[1].start]
        keep=np.isfinite(values).all(1)
        parts.append({'X':values[keep],'y':labels.ravel()[selected][keep],
                      'weights':weights.ravel()[selected][keep].astype(float),'region':scene['region']})
        print('HGB samples',scene['region'],int(keep.sum()),flush=True)
    sizes=defaultdict(int)
    for part in parts:sizes[part['region']]+=len(part['y'])
    X=np.concatenate([p['X'] for p in parts]);y=np.concatenate([p['y'] for p in parts])
    total=len(y);w=np.concatenate([p['weights']*total/(len(sizes)*sizes[p['region']]) for p in parts])
    model=HistGradientBoostingClassifier(**parameters)
    with threadpool_limits(limits=4):model.fit(X,y,sample_weight=w)
    joblib.dump(model,output/'estimator.joblib',compress=3)
    metadata={'model_type':'hgb','model_id':'nisar-water-hgb-round3-v1','feature_version':FEATURE_VERSION,'features':FEATURE_NAMES,
        'threshold':.5,'estimator_sha256':sha256(output/'estimator.joblib'),'created_at':utcnow(),
        'manifest_sha256':sha256(manifest_path),'training_counts':{'water':int((y==1).sum()),'nonwater':int((y==0).sum())},
        'training_regions':dict(sizes),'evaluation_status':'experimental_broadened_weak_validation','decision_policy':{'kind':'threshold','threshold':.5}}
    write_json(output/'model.json',metadata);print('HGB saved',output,metadata['training_counts'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('output');a=p.parse_args();train(a.manifest,a.output)


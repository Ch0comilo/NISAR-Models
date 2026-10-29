"""Old exposed failures are regression examples, never an unseen evaluation."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json
from nisar_water.registry import load_model
from nisar_water.inputs import load_catalog_observation
from nisar_water.benchmark import reference_support,confusion_metrics

if __name__=='__main__':
    model=load_model(ROOT/'models/round3/selected');old=read_json(ROOT/'outputs/new-images-20261028/report.json');rows=[]
    for row in old['rows']:
        if not row.get('observation_id'):continue
        folder=ROOT/'data/new-images-20261028'/row['case']
        entry=next(read_json(p)['observation'] for p in folder.glob('acquisition-*.json') if read_json(p)['observation']['id']==row['observation_id'])
        o=load_catalog_observation(entry);reference=read_json(folder/'references'/entry['id']/'reference.json')
        labels,_,_=model.predict(o);metric=None
        if reference['state']=='ready':
            _,classes,evaluate=reference_support(o,reference)
            if evaluate.any():
                if not (labels[evaluate]<=1).all():raise ValueError('Unexpected withheld reference')
                metric=confusion_metrics(classes[evaluate]==6,labels[evaluate]==1,o.weights[evaluate]*.04)
        rows.append({'region':row['case'],'id':row['observation_id'],'acquired_at':row['datetime'],
            'previous_spatial_v2':row['models']['spatial_v2']['metrics'],'round3':metric,
            'reference_state':row['reference_state'],'new_test':False,'used_in_training':row['case'] in ('lake-mead','lake-texoma','lake-okeechobee')})
    write_json(ROOT/'outputs/round3/regression-report.json',{'rows':rows,'model_sha256':model.fingerprint,
        'label_kind':'legacy_SCL_interiors_same_support','test_role':'Exposed regression examples; three scored scenes were used in new training'})

"""Review-only agreement policy; disagreement is unknown, never silently land."""
from pathlib import Path
import numpy as np
from .common import read_json,write_json,sha256,utcnow


def agreement_labels(first,second,observed):
    observed=np.asarray(observed,bool);a=np.asarray(first);b=np.asarray(second)
    if a.shape!=observed.shape or b.shape!=observed.shape:raise ValueError('Agreement grids differ')
    mutual=observed&(a<=1)&(b<=1);agree=mutual&(a==b)
    labels=np.full(a.shape,255,np.uint8);labels[observed]=254;labels[agree]=a[agree]
    return labels,mutual&~agree


def create_review_model(directory,radar,context):
    directory=Path(directory)
    if directory.exists():raise FileExistsError('Preserve the frozen review policy')
    components={}
    for name,path in [('radar',radar),('context',context)]:
        path=Path(path).resolve();metadata=read_json(path/'model.json')
        components[name]={'directory':str(path),'model_sha256':sha256(path/'model.json'),'model_id':metadata['model_id'],
                          'threshold':metadata['threshold']}
    metadata={'schema_version':'nisar-water-agreement-review-1','model_type':'agreement_review',
        'model_id':'nisar-water-fixed-agreement-review-v1','created_at':utcnow(),'components':components,
        'threshold':.5,'decision_policy':{'kind':'frozen_model_agreement','disagreement_code':254},
        'evaluation_status':'experimental_model_agreement_review_only','production_ready':False,
        'policy':'Classify only when current radar and prior-context decisions agree; otherwise unknown. Not a primary model selected by accuracy gates.',
        'test_used_for_policy_selection':False,'implementation_sha256':sha256(__file__)}
    write_json(directory/'model.json',metadata);return metadata


class AgreementReviewModel:
    def __init__(self,directory):
        from .registry import load_model
        self.directory=Path(directory);self.metadata=read_json(self.directory/'model.json')
        if self.metadata.get('schema_version')!='nisar-water-agreement-review-1':raise ValueError('Unsupported review policy')
        if sha256(__file__)!=self.metadata['implementation_sha256']:raise ValueError('Agreement implementation changed')
        self.models={}
        for role,source in self.metadata['components'].items():
            if sha256(Path(source['directory'])/'model.json')!=source['model_sha256']:raise ValueError('Review component changed')
            self.models[role]=load_model(source['directory'])
        self.model_id=self.metadata['model_id'];self.fingerprint=sha256(self.directory/'model.json')

    def predict(self,o,*,threshold=None,tile_size=256):
        if threshold is not None:raise ValueError('Review agreement uses frozen component operating points')
        first,_,_=self.models['radar'].predict(o,tile_size=tile_size)
        second,_,context=self.models['context'].predict(o,tile_size=tile_size)
        observed=o.valid&(o.weights>0);labels,disagreement=agreement_labels(first,second,observed)
        score=np.full(labels.shape,np.nan,np.float32);score[labels<=1]=labels[labels<=1]
        return labels,score,{**context,'threshold':.5,'threshold_override':False,'review_mode':True,
            'decision_policy':self.metadata['decision_policy'],'evaluation_status':self.metadata['evaluation_status'],
            'score_semantics':'deterministic binary agreement; not a calibrated water probability',
            'disagreement_withheld_ha':float((o.weights[disagreement]*.04).sum(dtype=float)),
            'components':self.metadata['components']}

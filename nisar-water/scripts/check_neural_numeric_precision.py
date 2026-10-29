"""Inspect actual full-float32 CUDA versus CPU bridge predictions; no fitting."""
from pathlib import Path
import sys
import numpy as np
import torch
from affine import Affine
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json
from nisar_water.neural import NeuralWaterModel
from nisar_water.inputs import Observation
if __name__=='__main__':
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    model=NeuralWaterModel(ROOT/'models/round3/unet',device='cuda');rows=[]
    for scene in read_json(ROOT/'data/round3-prepared/manifest.json')['scenes']:
        if scene['split']!='validation':continue
        with np.load(scene['path'],allow_pickle=False) as z:
            o=Observation(z['power'],z['valid'],Affine(*z['transform']),int(z['epsg']),{},z['aoi_weights'])
        labels,scores,_=model.predict(o,threshold=.1)
        with np.load(ROOT/'outputs/round3/validation-predictions/unet'/scene['id']/'prediction.npz',allow_pickle=False) as z:cpu=z['scores'];support=(z['labels']<=1)&(labels<=1)
        maximum=float(np.abs(scores[support]-cpu[support]).max())
        flips=int(np.count_nonzero((scores[support]>=.1)!=(cpu[support]>=.1)))
        rows.append({'scene':scene['id'],'full_fp32_cuda_vs_cpu_max_absolute_score_difference':maximum,'threshold_0_1_label_disagreements':flips})
    result={'fitting_performed':False,'test_images_used':False,'cuda_tf32':False,'records':rows,
            'passes':all(r['full_fp32_cuda_vs_cpu_max_absolute_score_difference']<1e-5 and r['threshold_0_1_label_disagreements']==0 for r in rows)}
    write_json(ROOT/'evidence/round3-neural-fp32-check.json',result);print(result)

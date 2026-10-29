"""Bounded inference in the existing Torch runtime; no credentials or live data."""
from pathlib import Path
import sys
import numpy as np
from affine import Affine
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json
from nisar_water.inputs import Observation
from nisar_water.neural import NeuralWaterModel

if __name__=='__main__':
    import torch
    torch.set_num_threads(4)
    request=read_json(sys.argv[1])
    with np.load(request['input'],allow_pickle=False) as z:
        o=Observation(z['power'],z['valid'],Affine(*z['transform']),int(z['epsg']),{},z['weights'],z['exclusions'])
    model=NeuralWaterModel(request['model'],device=request.get('device','cpu'))
    labels,scores,info=model.predict(o,threshold=request.get('threshold'),tile_size=request.get('tile_size',256))
    np.savez_compressed(request['output'],labels=labels,scores=scores);write_json(request['metadata'],info)


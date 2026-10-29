"""Load scientific model families while reusing the installed Torch runtime."""
from pathlib import Path
import importlib.util
import os
import subprocess
import tempfile
import numpy as np
from .common import read_json,write_json,sha256


class NeuralRuntimeModel:
    """CPU/API bridge when this Python environment has no compatible Torch."""
    def __init__(self,directory):
        self.directory=Path(directory).resolve();self.metadata=read_json(self.directory/'model.json')
        if sha256(self.directory/'weights.pt')!=self.metadata['weights_sha256']:raise ValueError('Neural weight hash mismatch')
        self.model_id=self.metadata['model_id'];self.fingerprint=sha256(self.directory/'model.json')
        self.root=Path(__file__).resolve().parents[2]
        default=self.root.parent/'nisar-models/.venv/Scripts/python.exe'
        self.runtime=Path(os.environ.get('NISAR_WATER_TORCH_PYTHON',str(default)))
        if not self.runtime.is_file():raise FileNotFoundError('Configure NISAR_WATER_TORCH_PYTHON to a compatible Torch Python runtime')

    def predict(self,observation,*,threshold=None,tile_size=256):
        cache=(self.root/'.cache/neural-inference').resolve();cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='job-',dir=cache) as directory:
            job=Path(directory).resolve()
            if not job.is_relative_to(cache):raise ValueError('Inference temporary path escaped its workspace')
            np.savez_compressed(job/'input.npz',power=observation.power,valid=observation.valid,weights=observation.weights,
                 exclusions=observation.exclusions,transform=np.array(list(observation.transform)[:6]),epsg=np.array(observation.crs.to_epsg()))
            write_json(job/'request.json',{'model':str(self.directory),'input':str(job/'input.npz'),'output':str(job/'output.npz'),
                       'metadata':str(job/'info.json'),'threshold':threshold,'tile_size':tile_size,'device':'cpu'})
            result=subprocess.run([str(self.runtime),str(self.root/'scripts/neural_worker.py'),str(job/'request.json')],
                cwd=self.root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=180,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode:raise RuntimeError('Neural inference runtime failed: '+result.stderr[-1200:])
            with np.load(job/'output.npz',allow_pickle=False) as archive:
                labels,scores=archive['labels'].copy(),archive['scores'].copy()
            return labels,scores,read_json(job/'info.json')


def load_model(directory,*,device='auto',context_cache_dir=None):
    metadata=read_json(Path(directory)/'model.json')
    if metadata.get('model_type')=='agreement_review':
        from .agreement_model import AgreementReviewModel
        model=AgreementReviewModel(directory)
        if context_cache_dir is not None:
            for component in model.models.values():
                if hasattr(component,'context_cache_dir'):component.context_cache_dir=context_cache_dir
        return model
    if metadata.get('model_type')=='context_hgb':
        from .context_model import ContextWaterModel
        return ContextWaterModel(directory,context_cache_dir=context_cache_dir if context_cache_dir is not None else os.environ.get('NISAR_WATER_CONTEXT_CACHE'))
    if metadata.get('model_type')=='opera':
        from .opera_model import OperaWaterModel
        return OperaWaterModel(directory)
    if metadata.get('model_type')=='unet':
        if importlib.util.find_spec('torch') is not None:
            from .neural import NeuralWaterModel
            return NeuralWaterModel(directory,device=device)
        return NeuralRuntimeModel(directory)
    from .model import WaterModel
    return WaterModel(directory)


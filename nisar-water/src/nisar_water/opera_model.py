"""Headless wrapper for pinned OPERA NI open-water science on bounded crops."""
from pathlib import Path
import hashlib
import os
import subprocess
import numpy as np
import rasterio
from rasterio.warp import reproject,Resampling
from .common import read_json,write_json,sha256,identity,utcnow
from .features import tiles
from .opera_adapter import SOURCE_COMMIT,prepare_job


def implementation_hashes(root):
    root=Path(root)
    paths=[root/'src/nisar_water/opera_adapter.py',root/'src/nisar_water/opera_model.py']
    paths+=sorted((root/'vendor/DSWX-SAR/src/dswx_sar').rglob('*.py'))
    paths+=sorted((root/'vendor/DSWX-SAR/src/dswx_sar/defaults').glob('*ni.yaml'))
    return {str(p.relative_to(root)).replace('\\','/'):sha256(p) for p in paths}


def create_model(directory,root):
    directory,root=Path(directory),Path(root).resolve();directory.mkdir(parents=True,exist_ok=True)
    config={'model_id':'nisar-water-opera-ni-crop-default-v1','model_type':'opera',
            'threshold':.5,'decision_policy':{'kind':'official_opera_ni_default_open_water'},
            'created_at':utcnow(),'source_commit':SOURCE_COMMIT,'implementation_hashes':implementation_hashes(root),
            'evaluation_status':'experimental_official_algorithm_crop_adapter',
            'feature_version':'official-opera-ni-30m-linear-hhhv-open-water-v1',
            'is_official_daac_product':False,'inundated_vegetation':False,
            'limitations':['Crop context differs from full-frame official processing',
                          'Historical ancillary water and landcover can constrain newly appearing water',
                          'No invented shadow mask or uncalibrated noise subtraction'],
            'runtime':{'gdal':'3.10.3','numpy':'2.2.3','prefix':'.venv-opera'}}
    write_json(directory/'model.json',config);return config


class OperaWaterModel:
    def __init__(self,directory):
        self.directory=Path(directory).resolve();self.metadata=read_json(self.directory/'model.json')
        self.root=Path(__file__).resolve().parents[2]
        if self.metadata.get('source_commit')!=SOURCE_COMMIT:raise ValueError('Unsupported OPERA source revision')
        if self.metadata['implementation_hashes']!=implementation_hashes(self.root):raise ValueError('OPERA implementation hash mismatch')
        self.model_id=self.metadata['model_id'];self.fingerprint=sha256(self.directory/'model.json')
        self.cache=self.root/'.cache/opera-inference';self.cache.mkdir(parents=True,exist_ok=True)

    def predict(self,o,*,threshold=None,tile_size=256):
        if threshold is not None and threshold!=.5:raise ValueError('Official binary decision has no tunable probability threshold')
        digest=identity({'power':hashlib.sha256(o.power.tobytes()).hexdigest(),
                         'valid':hashlib.sha256(o.valid.tobytes()).hexdigest(),
                         'transform':list(o.transform)[:6],'crs':str(o.crs),'model':self.fingerprint})
        job=self.cache/digest
        execution=job/'execution.json'
        if execution.exists():
            receipt=read_json(execution)
        else:
            job.mkdir(parents=True,exist_ok=True)
            prepare_job(o.metadata,job,self.root/'vendor/DSWX-SAR',observation=o)
            prefix=Path(os.environ.get('NISAR_WATER_OPERA_PREFIX',str(self.root/'.venv-opera')))
            conda=os.environ.get('NISAR_WATER_CONDA',str(Path.home()/'anaconda3/condabin/conda.bat'))
            command=[conda,'run','--prefix',str(prefix),'--no-capture-output','python',
                     str(self.root/'scripts/run_opera_comparison.py'),'worker',str(job/'request.json')]
            with (job/'runtime.log').open('w',encoding='utf-8') as log:
                run=subprocess.run(command,cwd=self.root,stdout=log,stderr=subprocess.STDOUT,timeout=600,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if run.returncode or not execution.exists():raise RuntimeError('OPERA runtime failed; inspect '+str(job/'runtime.log'))
            receipt=read_json(execution)
        if receipt['state']!='completed':raise RuntimeError('OPERA science stages failed; inspect '+str(execution))
        if sha256(receipt['output_path'])!=receipt['output_sha256']:raise ValueError('OPERA cached output integrity failure')
        raw=np.full(o.valid.shape,255,np.uint8)
        with rasterio.open(receipt['output_path']) as src:
            values=src.read(1)
            if not np.isin(values,[0,1,255]).all():raise ValueError('Unexpected official binary output codes')
            reproject(values,raw,src_transform=src.transform,src_crs=src.crs,dst_transform=o.transform,dst_crs=o.crs,
                      resampling=Resampling.nearest,src_nodata=255,dst_nodata=255)
        supported=np.zeros(o.valid.shape,bool)
        for sl,_,eligible in tiles(o.power,o.valid,tile_size):supported[sl]=eligible
        observed=o.valid&(o.weights>0);classified=observed&supported&~o.exclusions&(raw<=1)
        labels=np.full(o.valid.shape,255,np.uint8);labels[observed]=254;labels[classified]=raw[classified]
        scores=np.full(o.valid.shape,np.nan,np.float32);scores[classified]=raw[classified]
        return labels,scores,{'threshold':.5,'threshold_override':False,'decision_policy':self.metadata['decision_policy'],
                'evaluation_status':self.metadata['evaluation_status'],
                'score_semantics':'deterministic binary official-algorithm decision; not a probability',
                'is_official_daac_product':False,'source_commit':SOURCE_COMMIT,'execution_receipt':str(execution),
                'ancillary_provenance':read_json(job/'request.json')['ancillary']}

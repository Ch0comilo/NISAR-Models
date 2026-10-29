"""Official OPERA NISAR algorithm stages on bounded, georeferenced GCOV crops.

This adapter does not claim to produce a DAAC-validated OPERA product. It replaces
only full-frame HDF5 mosaicking/MGRS publication with scientific COG crop I/O.
All six open-water science stages are imported from the pinned official source.
"""
from pathlib import Path
import io
import json
import math
import os
import subprocess
import sys
import time
from types import SimpleNamespace

SOURCE_COMMIT="d9ae8ed5178417dec22a6fc549c4368cfd51ebe9"
PUBLIC_HOSTS={"copernicus-dem-30m.s3.amazonaws.com","glo-30-hand.s3.us-west-2.amazonaws.com",
              "esa-worldcover.s3.eu-central-1.amazonaws.com","storage.googleapis.com"}


def _namespace(value):
    if isinstance(value,dict):return SimpleNamespace(**{k:_namespace(v) for k,v in value.items()})
    if isinstance(value,list):return [_namespace(x) for x in value]
    return value


def ancillary_urls(bounds):
    west,south,east,north=bounds
    if math.floor(west)!=math.floor(east) or math.floor(south)!=math.floor(north):
        raise ValueError("This bounded comparator requires an AOI within one one-degree DEM/HAND tile")
    lon,lat=math.floor(west),math.floor(south)
    northing=f"{'N' if lat>=0 else 'S'}{abs(lat):02d}"
    easting=f"{'E' if lon>=0 else 'W'}{abs(lon):03d}"
    tile=f"Copernicus_DSM_COG_10_{northing}_00_{easting}_00"
    wc_lat,wc_lon=3*math.floor(lat/3),3*math.floor(lon/3)
    wc=f"{'N' if wc_lat>=0 else 'S'}{abs(wc_lat):02d}{'E' if wc_lon>=0 else 'W'}{abs(wc_lon):03d}"
    jrc_lon=10*math.floor(lon/10);jrc_lat=10*math.ceil((lat+.00001)/10)
    jrc=f"{abs(jrc_lon)}{'E' if jrc_lon>=0 else 'W'}_{abs(jrc_lat)}{'N' if jrc_lat>=0 else 'S'}"
    glad_lat=10*math.ceil((lat+.00001)/10);glad_lon=10*math.floor(lon/10)
    glad=f"{abs(glad_lat):02d}{'N' if glad_lat>=0 else 'S'}_{abs(glad_lon):03d}{'E' if glad_lon>=0 else 'W'}"
    return {"dem":f"https://copernicus-dem-30m.s3.amazonaws.com/{tile}_DEM/{tile}_DEM.tif",
            "hand":f"https://glo-30-hand.s3.us-west-2.amazonaws.com/v1/2021/{tile}_HAND.tif",
            "landcover":f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_{wc}_Map.tif",
            "reference_water":f"https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{jrc}v1_4_2021.tif",
            "glad":f"https://storage.googleapis.com/earthenginepartners-hansen/GLCLU2000-2020/v2/2020/{glad}.tif"}


def ancillary_sources(bounds):
    """Enumerate intersecting public tiles, including crops crossing tile seams."""
    west,south,east,north=bounds
    if not (-180<=west<east<=180 and -90<=south<north<=90) or east-west>1 or north-south>1:
        raise ValueError('Ancillary crop must be a bounded, non-wrapping region smaller than one degree')
    result={}
    for lon in range(math.floor(west),math.ceil(east)):
        for lat in range(math.floor(south),math.ceil(north)):
            for name,url in ancillary_urls((lon+.01,lat+.01,lon+.02,lat+.02)).items():
                result.setdefault(name,[])
                if url not in result[name]:result[name].append(url)
    return result


class PublicRangeFile(io.RawIOBase):
    def __init__(self,url,cache,ledger,session):
        from urllib.parse import urlparse
        super().__init__()
        p=urlparse(url)
        if p.scheme!="https" or p.hostname not in PUBLIC_HOSTS:
            raise ValueError("Unsupported public ancillary host")
        self.name,self.cache,self.ledger,self.session=url,cache,ledger,session
        self.position=0
        response=session.head(url,timeout=30,allow_redirects=False);response.raise_for_status()
        self.size=int(response.headers["Content-Length"])
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0:raise ValueError("Negative ancillary seek")
        return self.position
    def read(self,count=-1):
        if count<0:count=self.size-self.position
        if count>8*1024**2:raise ValueError("Unbounded ancillary read rejected")
        blocksize=256*1024;end=min(self.position+count,self.size);chunks=[]
        while self.position<end:
            block=self.position//blocksize;start=block*blocksize;stop=min(start+blocksize,self.size)
            if block not in self.cache:
                if self.ledger['body_bytes']+stop-start>self.ledger['cap_bytes']:raise ValueError("Ancillary range budget exceeded")
                with self.session.get(self.name,headers={'Range':f'bytes={start}-{stop-1}','Accept-Encoding':'identity'},
                                      timeout=30,stream=True,allow_redirects=False) as response:
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{stop-1}/{self.size}':
                        raise ValueError("Ancillary server did not honor exact ranges")
                    data=response.raw.read(stop-start+1)
                if len(data)!=stop-start:raise ValueError("Ancillary byte count mismatch")
                self.cache[block]=data;self.ledger['body_bytes']+=len(data)
                self.ledger['requests'].append({'url':self.name,'start':start,'end_exclusive':stop,'body_bytes':len(data)})
            length=min(end-self.position,stop-self.position);offset=self.position-start
            chunks.append(self.cache[block][offset:offset+length]);self.position+=length
        return b''.join(chunks)


def prepare_job(entry,job,source_root,*,observation=None):
    import numpy as np
    import rasterio
    from rasterio.warp import reproject,Resampling,transform_geom
    from rasterio.transform import array_bounds,from_origin
    from rasterio.windows import Window,from_bounds
    from rasterio.io import MemoryFile
    from rasterio.merge import merge
    from shapely.geometry import shape,box,mapping
    import requests
    from .inputs import load_catalog_observation
    from .common import write_json,sha256
    job=Path(job).resolve();job.mkdir(parents=True,exist_ok=True)
    scratch=job/'scratch';scratch.mkdir(exist_ok=True)
    ancillary=job/'ancillary';ancillary.mkdir(exist_ok=True)
    o=load_catalog_observation(entry) if observation is None else observation
    h,w=o.valid.shape;left,bottom,right,top=array_bounds(h,w,o.transform)
    transform=from_origin(math.floor(left/30)*30,math.ceil(top/30)*30,30,30)
    width=math.ceil((right-transform.c)/30);height=math.ceil((transform.f-bottom)/30)
    def save(path,data,crs,affine,nodata=None):
        if data.ndim==2:data=data[None]
        with rasterio.open(path,'w',driver='GTiff',width=data.shape[2],height=data.shape[1],count=len(data),
                           dtype=data.dtype,crs=crs,transform=affine,nodata=nodata,compress='DEFLATE') as dst:dst.write(data)
    validity=np.zeros((height,width),np.uint8)
    reproject(o.valid.astype(np.uint8),validity,src_transform=o.transform,src_crs=o.crs,dst_transform=transform,
              dst_crs=o.crs,resampling=Resampling.min,dst_nodata=0)
    for channel,name in enumerate(('HH','HV')):
        power=np.full((height,width),np.nan,np.float32)
        reproject(np.where(o.valid,o.power[channel],np.nan),power,src_transform=o.transform,src_crs=o.crs,
                  dst_transform=transform,dst_crs=o.crs,resampling=Resampling.average,src_nodata=np.nan,dst_nodata=np.nan)
        power[validity==0]=np.nan
        save(scratch/f'mosaic_{name}.tif',power,o.crs,transform,np.nan)
    save(scratch/'mosaic_mask.tif',validity,o.crs,transform,0)
    bbox=rasterio.warp.transform_bounds(o.crs,'EPSG:4326',*array_bounds(height,width,transform),densify_pts=21)
    urls=ancillary_sources(bbox)
    ledger={'body_bytes':0,'cap_bytes':128*1024**2,'requests':[]}
    source_info={};session=requests.Session();session.trust_env=False
    crop_geom=mapping(box(*bbox))
    try:
        for name,source_urls in urls.items():
            path=ancillary/(name+'.tif')
            if not path.exists():
                memory_files=[];readers=[]
                try:
                    for url in source_urls:
                        cache={}
                        def opener(p,mode='rb'):
                            if p!=url:raise FileNotFoundError(p)
                            return PublicRangeFile(url,cache,ledger,session)
                        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
                            with rasterio.open(url,opener=opener) as src:
                                geom=transform_geom('EPSG:4326',src.crs,crop_geom)
                                raw=from_bounds(*shape(geom).bounds,transform=src.transform)
                                c0=max(0,math.floor(raw.col_off)-3);r0=max(0,math.floor(raw.row_off)-3)
                                c1=min(src.width,math.ceil(raw.col_off+raw.width)+3);r1=min(src.height,math.ceil(raw.row_off+raw.height)+3)
                                if c1<=c0 or r1<=r0:continue
                                window=Window(c0,r0,c1-c0,r1-r0);data=src.read(window=window)
                                memory=MemoryFile();memory_files.append(memory)
                                with memory.open(driver='GTiff',width=data.shape[2],height=data.shape[1],count=len(data),
                                     dtype=data.dtype,crs=src.crs,transform=src.window_transform(window),nodata=src.nodata) as dst:dst.write(data)
                                readers.append(memory.open())
                    if not readers:raise ValueError('Ancillary tiles did not cover crop')
                    data,affine=merge(readers)
                    save(path,data,readers[0].crs,affine,readers[0].nodata)
                finally:
                    for reader in readers:reader.close()
                    for memory in memory_files:memory.close()
            source_info[name]={'source_urls':source_urls,'path':str(path),'sha256':sha256(path)}
    finally:
        session.close()
        ledger_path=job/'ancillary-transfer.json'
        if ledger_path.exists():
            from .common import read_json
            old=read_json(ledger_path);ledger['body_bytes']+=old['body_bytes'];ledger['requests']=old['requests']+ledger['requests']
        write_json(ledger_path,ledger)
    request={'observation':entry,'source_root':str(Path(source_root).resolve()),'job':str(job),
             'ancillary':source_info,'source_commit':SOURCE_COMMIT,
             'adapter_notes':['Unmodified official open-water NI stages; bounded COG input and output adapters',
                              '30m average linear power; minimum validity aggregation',
                              'No full-frame HDF5 mosaicking or MGRS product publication',
                              'Current valid-subsample mask supplied, not a fabricated geometry shadow mask',
                              'Inundated vegetation disabled: open-water comparison only',
                              'Default scientific parameters; CPU counts restricted to one'],
             'comparison_profile':'official_opera_ni_open_water_stages_crop_adapter'}
    write_json(job/'request.json',request)
    return request


def run_official_worker(request_path):
    """Runs inside the isolated GDAL conda environment, never the training runtime."""
    import yaml
    from .common import read_json,write_json,utcnow,sha256
    request=read_json(request_path);job=Path(request['job'])
    os.chdir(job)
    vendor=Path(request['source_root'])
    sys.path.insert(0,str(vendor/'src'))
    from dswx_sar.nisar import pre_processing,region_growing,masking_with_ancillary,refine_with_bimodality
    from dswx_sar.common import _initial_threshold,_fuzzy_value_computation
    groups=yaml.safe_load((vendor/'src/dswx_sar/defaults/dswx_ni.yaml').read_text())['runconfig']['groups']
    processing=yaml.safe_load((vendor/'src/dswx_sar/defaults/algorithm_parameter_ni.yaml').read_text())['runconfig']['processing']
    processing.update(polarizations=['HH','HV'],copol='HH',crosspol='HV',polarization_mode='DH_POL',nisar_uni_mode=True)
    processing['filter']['n_cpus']=1
    processing['initial_threshold']['number_cpu']=1
    processing['masking_ancillary']['number_cpu']=1
    processing['refine_with_bimodality']['number_cpu']=1
    processing['inundated_vegetation']['enabled']=False
    groups['processing']=processing
    groups['input_file_group']['input_file_path']=['scratch/mosaic_HH.tif','scratch/mosaic_HV.tif']
    groups['dynamic_ancillary_file_group'].update(dem_file='ancillary/dem.tif',hand_file='ancillary/hand.tif',
        worldcover_file='ancillary/landcover.tif',reference_water_file='ancillary/reference_water.tif',glad_classification_file='ancillary/glad.tif')
    groups['product_path_group'].update(scratch_path='scratch',sas_output_path='product',product_path='product')
    groups['static_ancillary_file_group']['static_ancillary_inputs_flag']=False
    cfg=_namespace({'groups':groups})
    previous=job/'execution.json'
    if previous.exists():
        import shutil
        number=len(list(job.glob('execution-attempt-*.json')))+1
        shutil.copyfile(previous,job/f'execution-attempt-{number}.json')
        if (job/'traceback.txt').exists():shutil.copyfile(job/'traceback.txt',job/f'traceback-attempt-{number}.txt')
    write_json(job/'effective-config.json',groups)
    stages=[('pre_processing',pre_processing),('initial_threshold',_initial_threshold),
            ('fuzzy_value',_fuzzy_value_computation),('region_growing',region_growing),
            ('masking_ancillary',masking_with_ancillary),('refine_bimodality',refine_with_bimodality)]
    receipt={'state':'running','started_at':utcnow(),'source_commit':request['source_commit'],'stages':[]}
    try:
        for name,module in stages:
            print('OPERA stage:',name,flush=True);start=time.perf_counter();module.run(cfg)
            receipt['stages'].append({'name':name,'seconds':time.perf_counter()-start})
            write_json(job/'execution.json',receipt)
        output=job/'scratch/bimodality_output_binary_HH_HV.tif'
        if not output.exists():raise FileNotFoundError('Official final binary raster missing')
        receipt.update(state='completed',finished_at=utcnow(),output_path=str(output),output_sha256=sha256(output))
    except Exception as error:
        import traceback
        receipt.update(state='failed',finished_at=utcnow(),error_type=type(error).__name__,message=str(error)[:1000])
        (job/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print(type(error).__name__,str(error)[:500],flush=True)
    write_json(job/'execution.json',receipt)
    return receipt


def compare_job(request_path,reference_support,output):
    import numpy as np
    import rasterio
    from rasterio.warp import reproject,Resampling
    from .common import read_json,write_json
    from .inputs import load_catalog_observation
    from .benchmark import confusion_metrics
    request=read_json(request_path);job=Path(request['job']);execution=read_json(job/'execution.json')
    if execution['state']!='completed':return execution
    o=load_catalog_observation(request['observation'])
    labels=np.full(o.valid.shape,255,np.uint8)
    with rasterio.open(execution['output_path']) as src:
        reproject(src.read(1),labels,src_transform=src.transform,src_crs=src.crs,dst_transform=o.transform,
                  dst_crs=o.crs,resampling=Resampling.nearest,dst_nodata=255)
    labels[~o.valid | (o.weights<=0)]=255
    with np.load(reference_support,allow_pickle=False) as z:classes=z['classes'];evaluate=z['evaluate']
    common=evaluate & (labels<=1)
    measure=confusion_metrics(classes[common]==6,labels[common]==1,o.weights[common].astype(float)*.04) if common.any() else None
    result={'state':'completed','profile':request['comparison_profile'],'observation_id':o.metadata['id'],
            'source_commit':request['source_commit'],'metrics':measure,
            'reference_support_ha':float((o.weights[evaluate]*.04).sum(dtype=float)),
            'classified_reference_support_ha':float((o.weights[common]*.04).sum(dtype=float)),
            'withheld_reference_ha':float((o.weights[evaluate & ~common]*.04).sum(dtype=float)),
            'is_official_daac_product':False,'execution':execution,'ancillary':request['ancillary']}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'),labels=labels,evaluation_mask=common)
    write_json(output,result)
    return result


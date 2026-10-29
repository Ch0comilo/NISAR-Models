"""Create split-preserving image supervision with explicit unknown shore pixels."""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject,Resampling
from rasterio.transform import rowcol
from scipy.ndimage import binary_erosion
from PIL import Image,ImageDraw
from .common import read_json,write_json,sha256,utcnow,safe_id
from .inputs import load_catalog_observation
from .features import tiles
from .round3_data import fetch_spectral


def aligned_reference(o,reference_dir):
    reference_dir=Path(reference_dir)
    ref=read_json(reference_dir/"reference.json")
    spectral=fetch_spectral(reference_dir,o.metadata["aoi"])
    if ref["state"]!="ready" or spectral["state"]!="ready":raise ValueError("No calibrated spectral reference")
    scl=np.zeros(o.valid.shape,np.uint8)
    with rasterio.open(ref["local_assets"]["scl"]["path"]) as src:
        reproject(src.read(1),scl,src_transform=src.transform,src_crs=src.crs,dst_transform=o.transform,dst_crs=o.crs,resampling=Resampling.nearest)
    bands={};all_valid=np.ones(o.valid.shape,bool)
    for key,asset in spectral["assets"].items():
        with rasterio.open(asset["path"]) as src:
            raw=src.read(1);valid=(raw!=asset["nodata"])
            reflectance=raw.astype(np.float32)*asset["scale"]+asset["offset"]
            reflectance[~valid]=np.nan
            target=np.full(o.valid.shape,np.nan,np.float32)
            reproject(reflectance,target,src_transform=src.transform,src_crs=src.crs,dst_transform=o.transform,dst_crs=o.crs,
                      src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.average)
            support=np.zeros(o.valid.shape,np.uint8)
            reproject(valid.astype(np.uint8),support,src_transform=src.transform,src_crs=src.crs,dst_transform=o.transform,dst_crs=o.crs,
                      resampling=Resampling.min,dst_nodata=0)
            all_valid&=(support==1)&np.isfinite(target)
            bands[key]=np.maximum(target,0)  # Clipping only for index screening, originals retained.
    green,nir,swir=[bands[k] for k in ("green","nir","swir16")]
    ndwi=np.full(o.valid.shape,np.nan,np.float32);mndwi=np.full_like(ndwi,np.nan)
    np.divide(green-nir,green+nir,out=ndwi,where=(green+nir)>.001)
    np.divide(green-swir,green+swir,out=mndwi,where=(green+swir)>.001)
    support=np.zeros_like(o.valid)
    for sl,_,v in tiles(o.power,o.valid):support[sl]=v
    known=all_valid&o.valid&support&(o.weights>=1-1e-6)
    water=known&(scl==6)&(ndwi>0)&(mndwi>.1)
    land=known&np.isin(scl,[4,5])&((ndwi<0)|(mndwi<0))
    labels=np.full(o.valid.shape,255,np.uint8);labels[land]=0;labels[water]=1
    interior=(binary_erosion(water,iterations=2)|binary_erosion(land,iterations=2))&known
    weights=np.zeros(o.valid.shape,np.float32);weights[water|land]=.3;weights[interior]=1
    rgb=np.zeros((3,*o.valid.shape),np.uint8)
    with rasterio.open(ref["local_assets"]["visual"]["path"]) as src:
        for i in range(3):
            reproject(src.read(i+1),rgb[i],src_transform=src.transform,src_crs=src.crs,dst_transform=o.transform,dst_crs=o.crs,resampling=Resampling.bilinear)
    rgb=np.moveaxis(rgb,0,-1);rgb[~np.isin(scl,[4,5,6])]=35
    return labels,weights,interior,rgb,{"reference":ref,"spectral":spectral,"index_policy":"DN scale+offset applied; nonnegative reflectance for NDWI/MNDWI; source nodata excluded"}


def save_scene(o,labels,weights,evaluate,output,spec,provenance,rgb=None):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    # Bound old cached crops to full 31x31 source context where the source halo was smaller.
    complete=np.ones(o.valid.shape,bool);complete[:15]=False;complete[-15:]=False;complete[:,:15]=False;complete[:,-15:]=False
    labels=labels.copy();weights=weights.copy();evaluate=evaluate.copy()
    labels[~complete]=255;weights[~complete]=0;evaluate&=complete
    np.savez_compressed(output,power=o.power,valid=o.valid,labels=labels,weights=weights,evaluate=evaluate,
                        transform=np.array(list(o.transform)[:6]),epsg=np.array(o.crs.to_epsg()),aoi_weights=o.weights)
    metadata={**spec,"path":str(output.resolve()),"sha256":sha256(output),"source_observation":o.metadata,
              "label_kind":"weak","provenance":provenance,"counts":{"water":int((labels==1).sum()),"nonwater":int((labels==0).sum()),
              "unknown":int((labels==255).sum()),"interior":int(evaluate.sum())},"human_reviewed":False,
              "assistant_visual_qa":"pending" if spec['split']!='test' else "withheld_until_freeze"}
    write_json(output.with_suffix('.json'),metadata)
    if spec["split"]!="test":
        if rgb is None:
            gray=np.uint8(np.clip(np.nan_to_num((10*np.log10(np.where(o.valid,o.power[0],1))+30)/33),0,1)*255)
            rgb=np.repeat(gray[...,None],3,axis=2)
        base=Image.fromarray(rgb).convert("RGBA")
        overlay=np.zeros((*labels.shape,4),np.uint8)
        overlay[labels==1]=[25,145,255,145];overlay[labels==0]=[245,120,40,95]
        overlay[(labels!=255)&~evaluate]=[255,220,60,130]
        shown=Image.alpha_composite(base,Image.fromarray(overlay)).convert("RGB")
        canvas=Image.new("RGB",(rgb.shape[1]*2,rgb.shape[0]+30),"#14202c")
        canvas.paste(base.convert("RGB"),(0,30));canvas.paste(shown,(rgb.shape[1],30))
        ImageDraw.Draw(canvas).text((8,8),spec['region']+' '+spec['acquired_at'][:10]+' | optical/SAR + weak labels; no expert review',fill="white")
        canvas.save(output.with_suffix('.qa.png'))
    return {k:metadata[k] for k in ("id","path","split","group","region","acquired_at","sha256")}


def build_scenes(data_root,output,legacy_root):
    data_root,output,legacy_root=map(Path,(data_root,output,legacy_root));output.mkdir(parents=True,exist_ok=True)
    scenes=[];audit=[]
    for path in sorted(data_root.glob('*/collection.json')):
        collection=read_json(path)
        for selected in collection.get('selected',[]):
            region=selected['region'];split=selected['split'];entry=selected['observation']
            oid=safe_id(region+'-'+entry['acquired_at'])
            print('PREPARE',region,split,flush=True)
            o=load_catalog_observation(entry)
            try:labels,weights,evaluate,rgb,provenance=aligned_reference(o,selected['reference_dir'])
            except Exception as error:
                audit.append({'region':region,'split':split,'state':'reference_preparation_failed','error':str(error)[:300]});continue
            spec={'id':oid,'split':split,'region':region,'group':region,'acquired_at':entry['acquired_at']}
            saved=save_scene(o,labels,weights,evaluate,output/'scenes'/(oid+'.npz'),spec,provenance,rgb)
            scenes.append(saved)
            if split!='test':audit.append({'region':region,'split':split,'known_cells':int((labels!=255).sum())})
    # Previously exposed failures are now explicit TRAIN examples, never a new test.
    old=Path('data/new-images-20261028')
    for region in ('lake-mead','lake-texoma','lake-okeechobee','gatun-lake'):
        for receipt in sorted((old/region).glob('acquisition-*.json')):
            acquired=read_json(receipt);entry=acquired['observation']
            ref_dir=old/region/'references'/entry['id'];ref=read_json(ref_dir/'reference.json')
            if ref['state']!='ready':continue
            with rasterio.open(ref['local_assets']['scl']['path']) as src:
                if not np.isin(src.read(1),[4,5,6]).any():continue
            print('PREPARE exposed TRAIN',region,entry['acquired_at'],flush=True)
            o=load_catalog_observation(entry)
            try:labels,weights,evaluate,rgb,provenance=aligned_reference(o,ref_dir)
            except Exception as error:
                audit.append({'region':region,'state':'reference_preparation_failed','error':str(error)[:300]});continue
            oid=safe_id(region+'-'+entry['acquired_at'])
            spec={'id':oid,'split':'train','group':region,'region':region,'acquired_at':entry['acquired_at']}
            scenes.append(save_scene(o,labels,weights,evaluate,output/'scenes'/(oid+'.npz'),spec,provenance,rgb))
    catalog={e['id']:e for e in read_json(legacy_root/'data/catalog.json')['observations']}
    archive=legacy_root/'data/labels/water-correction-v9/samples.npz'
    with np.load(archive,allow_pickle=False) as z:
        ids=z['train__observation_ids'];xy=z['train__xy'];y=z['train__y']
    for oid in np.unique(ids):
        entry=catalog[str(oid)];o=load_catalog_observation(entry);take=ids==oid
        rr,cc=map(np.asarray,rowcol(o.transform,*xy[take].T))
        labels=np.full(o.valid.shape,255,np.uint8);labels[rr,cc]=(y[take]==1).astype(np.uint8)
        weights=(labels!=255).astype(np.float32);evaluate=labels!=255
        region='iowa-control' if 'iowa-' in oid else 'tapajos-control' if 'tapajos-' in oid else 'oroville'
        spec={'id':str(oid),'split':'train','group':region,'region':region,'acquired_at':entry['acquired_at']}
        scenes.append(save_scene(o,labels,weights,evaluate,output/'scenes'/(safe_id(oid)+'.npz'),spec,
                      {'kind':'existing_weak_training_cells','source_npz_sha256':sha256(archive),'source_split':'train_only'}))
    manifest={'created_at':utcnow(),'data_origin':'observed','label_kind':'weak','scenes':scenes,
              'plan_sha256':sha256(data_root/'plan.json'),'test_rule':'No test scene opened by fitting or validation selection',
              'supplement_plan_sha256':sha256(data_root/'supplement-plan.json') if (data_root/'supplement-plan.json').exists() else None,
              'shoreline_review_status':'Spectral/SCL consensus weak boundary labels; assistant QA does not establish expert shoreline truth'}
    write_json(output/'manifest.json',manifest);write_json(output/'preparation-audit.json',{'records':audit})
    return manifest


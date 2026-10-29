"""Reuse corrected context; prepare fresh reservoir context and release TRAIN."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json,sha256,utcnow
from nisar_water.round5_evaluation import load_scene
from nisar_water.terrain_context import load_context
from nisar_water.context_model import load_context_sidecar

if __name__=='__main__':
    directory=ROOT/'data/round5-prepared';source=directory/'scenes-manifest.json';manifest=read_json(source);scenes=[];records=[]
    if (ROOT/'models/round5').exists():raise ValueError('Prepared inputs are closed after fitting starts')
    for spec in manifest['scenes']:
        if spec.get('context_path'):
            for a,b in [('path','sha256'),('context_path','context_sha256'),('context_metadata_path','context_metadata_sha256')]:
                if sha256(spec[a])!=spec[b]:raise ValueError('Exposed source changed')
            scenes.append(spec);continue
        o,_,_,_=load_scene(spec);context=load_context(o,cache_dir=ROOT/'.cache/terrain-context',include_prior=True)
        path=directory/'context'/(spec['id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,features=context['features'],available=context['available'],prior_available=context['prior_available'],
             prior_occurrence_available=context['prior_occurrence_available'],names=np.array(context['names']),
             transform=np.array(list(o.transform)[:6]),transform6=np.array(list(o.transform)[:6]),epsg=np.array(o.crs.to_epsg()),
             scene_sha256=np.array(spec['sha256']))
        metadata={**context['metadata'],'scene_sha256':spec['sha256'],'scene_id':spec['id'],
                  'target_transform':list(o.transform)[:6],'target_epsg':o.crs.to_epsg()};write_json(path.with_suffix('.json'),metadata)
        scenes.append({**spec,'context_path':str(path.resolve()),'context_sha256':sha256(path),
                       'context_metadata_path':str(path.with_suffix('.json').resolve()),'context_metadata_sha256':sha256(path.with_suffix('.json'))})
        print('CONTEXT round5',spec['split'],spec['id'],flush=True)
    final={**manifest,'scenes':scenes,'context_prepared_at':utcnow(),'context_source_manifest_sha256':sha256(source),
           'new_test_predictions_inspected':False,'emerging_water_validation_gate':{'low_JRC_occurrence_max':10,'minimum_water_reference_ha':4,
                'minimum_full_reference_recovery':.90,'minimum_class_coverage':.98}}
    write_json(directory/'manifest.json',final)
    validation=[s for s in scenes if s['split']=='validation'];width,height=620,330;cols=3
    canvas=Image.new('RGB',(width*cols,height*((len(validation)+cols-1)//cols)),'#14202c')
    for i,s in enumerate(validation):
        im=Image.open(Path(s['path']).with_suffix('.qa.png')).convert('RGB');im.thumbnail((width-10,height-35))
        x=(i%cols)*width;y=(i//cols)*height;canvas.paste(im,(x,y+30));ImageDraw.Draw(canvas).text((x+5,y+5),s['id'],fill='white')
    canvas.save(ROOT/'evidence/round5-validation-label-qa.png')
    for s in scenes:
        if s['split']=='test':continue
        with np.load(s['path'],allow_pickle=False) as z:
            available=z['valid']&(z['aoi_weights']>0);shape=available.shape;transform=z['transform'];epsg=int(z['epsg'])
        c=load_context_sidecar(s,shape,transform,epsg,scene_sha256=s['sha256'])
        fraction=float(c['available'][available].mean()) if available.any() else None
        if fraction is None or fraction<.98:raise ValueError('Insufficient development context: '+s['id'])
        records.append({'id':s['id'],'split':s['split'],'physical_observed_coverage':fraction})
    write_json(directory/'development-context-check.json',{'records':records,'new_test_pixels_read_by_check':False})
    print('Prepared round5 manifest',len(scenes),'scenes; pending visual release')

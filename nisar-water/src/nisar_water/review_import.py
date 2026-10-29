"""Versioned human annotation import; frozen training/evaluation never rewritten."""
from pathlib import Path
from datetime import datetime
import shutil
import numpy as np
from affine import Affine
from .common import read_json,write_json,sha256,utcnow
from .inputs import Observation
from .review import annotation_labels


def import_review(packet_source,annotations,output):
    packet_source,annotations,output=Path(packet_source),Path(annotations),Path(output)
    source=read_json(packet_source);collection=read_json(annotations)
    scene=Path(source['scene_path'])
    if sha256(scene)!=source['scene_sha256']:raise ValueError('Packet scientific source hash changed')
    if collection.get('type')!='FeatureCollection' or not collection.get('features'):raise ValueError('Supply nonempty reviewed polygons')
    reviewers=set();dates=set();expert=[]
    for feature in collection['features']:
        p=feature.get('properties',{})
        if p.get('scene_id')!=source['scene_id'] or p.get('scene_sha256')!=source['scene_sha256']:
            raise ValueError('Annotation belongs to a different scene or source revision')
        name=str(p.get('reviewer','')).strip()
        if not name or name=='ENTER_YOUR_NAME':raise ValueError('Reviewer attribution is required')
        try:date=datetime.fromisoformat(str(p['reviewed_at_utc']).replace('Z','+00:00'))
        except (ValueError,KeyError):raise ValueError('A valid reviewed_at_utc is required')
        if date.tzinfo is None:raise ValueError('Review time must have a UTC offset')
        reviewers.add(name);dates.add(date.isoformat());expert.append(p.get('independent_expert') is True)
    with np.load(scene,allow_pickle=False) as z:
        o=Observation(z['power'],z['valid'],Affine(*z['transform']),int(z['epsg']),{},z['aoi_weights'])
    labels=annotation_labels(collection,o)
    if not (labels<=1).any():raise ValueError('No reviewed class pixels overlap valid observation support')
    destination=output/('review-'+source['scene_sha256'][:12])/sha256(annotations)[:16]
    if destination.exists():raise FileExistsError('Preserve imported review versions; this annotation hash already exists')
    destination.mkdir(parents=True)
    shutil.copyfile(annotations,destination/'annotations.geojson')
    np.savez_compressed(destination/'labels.npz',labels=labels,evaluate=labels<=1,
                        transform=np.array(list(o.transform)[:6]),epsg=np.array(o.crs.to_epsg()),
                        scene_sha256=np.array(source['scene_sha256']))
    result={'imported_at':utcnow(),'scene_id':source['scene_id'],'scene_sha256':source['scene_sha256'],
        'annotations_sha256':sha256(annotations),'labels_sha256':sha256(destination/'labels.npz'),
        'label_kind':'independent_expert_declared' if all(expert) else 'team_reviewed',
        'independent_expert_declaration_externally_verified':False,'reviewers':sorted(reviewers),'reviewed_at':sorted(dates),
        'water_cells':int((labels==1).sum()),'land_cells':int((labels==0).sum()),'unknown_cells':int((labels==255).sum()),
        'frozen_experiments_changed':False,'destination':str(destination.resolve()),
        'scope':'Human-annotated support only; unknown area is not implicitly land; qualify future accuracy with reference uncertainty'}
    write_json(destination/'receipt.json',result);return result

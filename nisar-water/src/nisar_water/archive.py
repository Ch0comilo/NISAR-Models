"""Process all compatible acquisitions for an AOI, without optical-data filtering."""
from pathlib import Path
from datetime import date,timedelta
from collections import defaultdict
from .common import read_json,write_json,identity,sha256,utcnow
from .online import discover,acquire
from .inputs import parse_granule,load_catalog_observation
from .registry import load_model
from .pipeline import run_series


def all_candidates(aoi,start,end,work,reference_root):
    start,end=date.fromisoformat(start),date.fromisoformat(end)
    if start>end or (end-start).days>366:raise ValueError('Use an ordered interval of at most 366 days')
    work=Path(work);queries=[];candidates=[]
    def query(first,last):
        folder=work/(first.isoformat()+'_'+last.isoformat());path=folder/'discovery.json'
        found=read_json(path) if path.exists() else discover(aoi,first.isoformat(),last.isoformat(),folder,reference_root)
        write_json(path,found);queries.append({'path':str(path.resolve()),'sha256':sha256(path),'truncated':found.get('truncated')})
        if found.get('state') not in ('ready','no_compatible_data'):raise ValueError('Discovery failed: '+str(found.get('message'))[:200])
        if found.get('truncated'):
            if first==last:raise ValueError('More than 200 records on one date; refine the query rather than silently dropping observations')
            middle=first+timedelta(days=(last-first).days//2)
            query(first,middle);query(middle+timedelta(days=1),last)
        else:candidates.extend(found.get('candidates',[]))
    query(start,end)
    # A single acquisition can have several processing revisions. Select from
    # public metadata, never from its radar appearance or prediction.
    unique={};unsupported=[];superseded=[]
    for candidate in candidates:
        if not candidate.get('compatible') or candidate.get('coverage_fraction',0)<=0:
            unsupported.append(candidate['id']);continue
        meta=parse_granule(candidate['granule_id']+'.h5')
        key=(candidate['acquired_at'],meta['track'],meta['pass_direction'],meta['frame'])
        previous=unique.get(key)
        rank=lambda c:(c.get('revision_id') or 0,c['granule_id'])
        if previous is None or rank(candidate)>rank(previous):
            if previous:superseded.append(previous['id'])
            unique[key]=candidate
        else:superseded.append(candidate['id'])
    return sorted(unique.values(),key=lambda c:(c['acquired_at'],c['granule_id'])),{
        'queries':queries,'unsupported_or_no_footprint':unsupported,'superseded_processing_revisions':superseded,
        'revision_rule':'Highest CMR revision, then lexical granule identity per exact acquisition time and geometry'}


def process_aoi(aoi,start,end,model_dir,output,reference_root,*,transfer_cap_bytes=1024**3,limit=0,context_cache_dir=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    model=load_model(model_dir) if context_cache_dir is None else load_model(model_dir,context_cache_dir=context_cache_dir)
    if aoi.get('type')=='Feature':aoi=aoi['geometry']
    scientific_request={'aoi':aoi,'start':start,'end':end,'model_sha256':model.fingerprint,'context_halo_m':320,
                        'reference_required':False,'observation_limit':limit,'transfer_cap_bytes':transfer_cap_bytes}
    if context_cache_dir is not None:scientific_request['context_cache_dir']=str(Path(context_cache_dir).resolve())
    request_path=output/'request.json'
    if request_path.exists() and read_json(request_path)!=scientific_request:raise ValueError('Existing AOI job has different parameters; use a new output directory')
    write_json(request_path,scientific_request)
    candidates,discovery=all_candidates(aoi,start,end,output/'discovery',reference_root)
    selected=candidates[:limit] if limit else candidates
    write_json(output/'selection.json',{'candidate_count':len(candidates),'selected_count':len(selected),'limited':len(selected)<len(candidates),
              'optical_or_prediction_selection':False,**discovery})
    entries=[];receipts=[];transferred=0
    for candidate in selected:
        key=identity({'candidate':candidate['id'],'aoi':aoi,'halo':320})[:24];folder=output/'acquisitions'/key
        path=folder/'receipt.json'
        if path.exists():result=read_json(path)
        else:
            remaining=transfer_cap_bytes-transferred
            if remaining<1024**2:
                receipts.append({'candidate':candidate['id'],'state':'transfer_budget_exhausted'});break
            print('ACQUIRE',candidate['acquired_at'],candidate['granule_id'],flush=True)
            result=acquire(candidate,aoi,folder,reference_root,max_bytes=min(128*1024**2,remaining),halo_metres=320)
            write_json(path,result)
        transferred+=int(result.get('transfer_bytes') or 0)
        receipts.append({'candidate':candidate['id'],'state':result['state'],'receipt':str(path)})
        if result['state'] in ('acquired','cached'):entries.append(result['observation'])
        write_json(output/'progress.json',{'processed':len(receipts),'selected':len(selected),'radar_body_bytes':transferred,'receipts':receipts})
    write_json(output/'catalog.json',{'observations':entries})
    groups=defaultdict(list)
    for entry in entries:
        key=identity({k:entry.get(k) for k in ('track','pass_direction','frame','frequency','aoi')})[:16]
        groups[key].append(entry)
    series=[]
    for key,observations in groups.items():
        destination=output/'series'/key
        if (destination/'series.json').exists():saved=read_json(destination/'series.json')
        else:
            # Preserve any failed partial export; a new attempt remains reviewable.
            if destination.exists():destination=destination/('attempt-'+utcnow().replace(':','-'))
            saved=run_series((load_catalog_observation(e) for e in observations),model,destination,series_id=key)
        series.append({'id':key,'path':str((destination/'series.json').resolve()),'observations':len(saved['observations'])})
    complete=len(entries)==len(candidates) and len(selected)==len(candidates)
    result={'state':('no_compatible_data' if not candidates else 'completed' if complete else 'partial'),'complete_for_compatible_query':complete,
            'candidate_observations':len(candidates),'acquired_observations':len(entries),'radar_body_bytes':transferred,
            'model_sha256':model.fingerprint,'series':series,'receipts':receipts,
            'scope':'Current supported NISAR L-band frequency-A HH/HV GCOV collection, bounded UTM AOI; no cloud or optical filter'}
    write_json(output/'batch.json',result);return result

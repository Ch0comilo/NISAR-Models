"""Standalone bounded scientific NISAR GCOV range reader.

Adapted from the first-party nisar_models/acquisition.py reader. Original source
SHA256: 2dedcfa4e9ab15e0da808faff4ebb324377ca65f36d2aa39a5c019c2d4b1c0c9
The physical crop, power and native-validity algorithms are retained. Credentials
and transfer accounting are injected through a client; no sibling runtime,
credential store, full-download fallback or network client is imported here.
This version additionally requires complete in-file identity metadata and the
current classifier's native north-up UTM 10/20 m square grid profile.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Protocol
from urllib.parse import urlsplit

import numpy as np

CMR_URL = 'https://cmr.earthdata.nasa.gov/search/granules.umm_json'
COLLECTION = 'NISAR_L2_GCOV_PROVISIONAL_V1'
COLLECTION_ID = 'C2854338529-ASF'

class AcquisitionError(ValueError):
    """An acquisition failure with a non-sensitive, machine-readable category."""
    def __init__(self, code, message, details=None):
        self.code = code
        self.details = details or {}
        super().__init__(message)


class RangeClient(Protocol):
    def stat(self, url: str) -> dict: ...
    def read_range(self, url: str, start: int, end: int) -> bytes: ...


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_product_id(product_id):
    """Decode exact official single-acquisition filename fields, never infer dates."""
    name = Path(product_id).name.removesuffix('.h5')
    pattern = (r'^NISAR_L2_(?P<processing_type>\w+)_GCOV_(?P<cycle>\d+)_(?P<track>\d+)_(?P<pass_direction>[AD])_'
               r'(?P<frame>\d+)_(?P<mode>\d+)_(?P<polarization_mode>\w+)_(?P<source_mode>[AM])_'
               r'(?P<start>\d{8}T\d{6})_(?P<end>\d{8}T\d{6})_(?P<crid>[^_]+)_'
               r'(?P<tail>.+)_(?P<product_counter>\d+)$')
    matched = re.fullmatch(pattern, name)
    if not matched:
        raise AcquisitionError('UNSUPPORTED_PRODUCT_ID', 'Unrecognized exact NISAR GCOV product identity')
    fields = matched.groupdict()
    for key in ('cycle', 'track', 'frame'):
        fields[key] = int(fields[key])
    for key in ('start', 'end'):
        fields[key] = datetime.strptime(fields[key], '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc).isoformat().replace('+00:00','Z')
    return fields


def _bbox_geometry(bbox):
    west,south,east,north = map(float,bbox)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90) or east-west > 0.2 or north-south > 0.2:
        raise AcquisitionError('AOI_BOUNDS', 'Use a small ordered WGS84 AOI no larger than 0.2 degrees per side')
    return {'type':'Polygon','coordinates':[[[west,south],[east,south],[east,north],[west,north],[west,south]]]}


def _summarize_granule(item):
    umm = item['umm']
    identity = umm['GranuleUR']
    fields = parse_product_id(identity)
    links = [x['URL'] for x in umm.get('RelatedUrls', []) if x.get('Type') == 'GET DATA' and
             x.get('URL','').startswith('https://') and urlsplit(x['URL']).path.lower().endswith('.h5')]
    temporal = umm['TemporalExtent'].get('RangeDateTime',{})
    fields.update({'granule_id': identity, 'concept_id':item['meta']['concept-id'],
        'revision_id':item['meta']['revision-id'], 'collection':umm.get('CollectionReference',{}),
        'cmr_start':temporal.get('BeginningDateTime'), 'cmr_end':temporal.get('EndingDateTime'),
        'https_urls':links, 'spatial_extent':umm.get('SpatialExtent'),
        'additional_attributes':umm.get('AdditionalAttributes',[]),
        'archive_information':umm.get('DataGranule',{}).get('ArchiveAndDistributionInformation',[])})
    return fields


class HTTPSRangeReader(io.RawIOBase):
    """Seekable bounded range reader suitable for h5py's fileobj driver.

    Every request stays a range. RAM is bounded by an LRU cache. No read-all API
    and no non-range download path exist. The client enforces the global ledger.
    """
    def __init__(self, client: RangeClient, url, *, size=None, block_size=1024*1024,
                 cache_blocks=16, max_bytes=1024**3):
        super().__init__()
        self.cache=OrderedDict()
        parsed=urlsplit(url)
        if parsed.scheme!='https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise AcquisitionError('UNSAFE_SOURCE_URL','Only unsigned HTTPS granule URLs are accepted')
        if block_size < 4096 or block_size > 16*1024**2 or cache_blocks < 1 or max_bytes < block_size:
            raise AcquisitionError('RANGE_LIMIT','Invalid bounded range cache configuration')
        self.client,self.url=client,url
        metadata=client.stat(url) if size is None else {'size':size}
        self.size=int(metadata.get('size',metadata.get('content_length',0)))
        if self.size<=0:
            raise AcquisitionError('REMOTE_SIZE_UNAVAILABLE','Exact remote object length is required')
        self.position=0
        self.block_size,self.cache_blocks,self.max_bytes=block_size,cache_blocks,max_bytes
        self.bytes_transferred=0
        self.requests=0
        self.cache=OrderedDict()
        self.disk_cache=None
        if hasattr(client,'ledger'):
            identity=hashlib.sha256(json.dumps({'url':url,'size':self.size,'etag':metadata.get('etag')},sort_keys=True).encode()).hexdigest()
            self.disk_cache=Path(client.ledger.storage_root)/'_range-cache'/identity
            self.disk_cache.mkdir(parents=True,exist_ok=True)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if target < 0 or target > self.size:
            raise AcquisitionError('RANGE_BOUNDS','HDF5 requested a seek outside the source')
        self.position=target
        return target

    def read(self, size=-1):
        if size is None or size < 0:
            raise AcquisitionError('UNBOUNDED_READ','Unbounded source reads are prohibited')
        size=min(size,self.size-self.position)
        if size > 128*1024**2:
            raise AcquisitionError('READ_LIMIT','A single scientific read exceeds 128 MiB')
        parts=[]
        while size:
            index=self.position//self.block_size
            start=index*self.block_size
            end=min(start+self.block_size,self.size)
            block=self.cache.pop(index,None)
            cache_path=self.disk_cache/f'{index:012d}.bin' if self.disk_cache is not None else None
            if block is None and cache_path is not None and cache_path.is_file() and cache_path.with_suffix('.sha256').is_file():
                block=cache_path.read_bytes()
                if len(block)!=end-start or hashlib.sha256(block).hexdigest()!=cache_path.with_suffix('.sha256').read_text().strip():
                    raise AcquisitionError('CACHE_CHECKSUM_MISMATCH','Cached scientific range failed its recorded SHA256')
            if block is None:
                if self.bytes_transferred+end-start>self.max_bytes:
                    raise AcquisitionError('TRANSFER_LIMIT','Per-granule bounded transfer ceiling reached')
                block=self.client.read_range(self.url,start,end)
                if len(block)!=end-start:
                    raise AcquisitionError('RANGE_RESPONSE_MISMATCH','Source range has an unexpected byte count')
                self.bytes_transferred+=len(block)
                self.requests+=1
                if cache_path is not None:
                    temporary=cache_path.with_suffix('.part')
                    temporary.write_bytes(block)
                    temporary.replace(cache_path)
                    checksum=cache_path.with_suffix('.sha256')
                    temporary_checksum=checksum.with_suffix('.sha256.part')
                    temporary_checksum.write_text(hashlib.sha256(block).hexdigest(),encoding='ascii')
                    temporary_checksum.replace(checksum)
            self.cache[index]=block
            while len(self.cache)>self.cache_blocks:
                self.cache.popitem(last=False)
            offset=self.position-start
            count=min(size,len(block)-offset)
            parts.append(block[offset:offset+count])
            self.position+=count
            size-=count
        return b''.join(parts)

    def readinto(self, buffer):
        block=self.read(len(buffer))
        buffer[:len(block)]=block
        return len(block)

    def close(self):
        self.cache.clear()
        super().close()


def _scalar(value):
    value=np.asarray(value)
    if value.size!=1:
        raise AcquisitionError('METADATA_SHAPE','Expected a scalar source metadata field')
    result=value.reshape(()).item()
    return result.decode('utf-8') if isinstance(result,bytes) else result


def _metadata_value(group, key):
    if key not in group:
        return None
    value=np.asarray(group[key][()])
    if value.size==1:
        return _scalar(value)
    return [x.decode('utf-8') if isinstance(x,bytes) else x.item() if hasattr(x,'item') else x for x in value.reshape(-1)]


def _grid_window(group, bbox):
    """Read bounded coordinate axes and cover the complete AOI, including no-data."""
    from affine import Affine
    from rasterio.warp import transform_bounds
    projection=group['projection']
    epsg=int(_scalar(projection.attrs['epsg_code']) if 'epsg_code' in projection.attrs else _scalar(projection[()]))
    if group['xCoordinates'].size>1000000 or group['yCoordinates'].size>1000000:
        raise AcquisitionError('GRID_LIMIT','Source coordinate axes exceed the inspection bound')
    x,y=np.asarray(group['xCoordinates'][:]),np.asarray(group['yCoordinates'][:])
    if len(x)<2 or len(y)<2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise AcquisitionError('INVALID_GRID','Source grid has unavailable coordinate values')
    dx,dy=float(x[1]-x[0]),float(y[1]-y[0])
    if dx<=0 or dy>=0 or not np.allclose(np.diff(x),dx) or not np.allclose(np.diff(y),dy):
        raise AcquisitionError('INVALID_GRID','Require a regular north-up source grid')
    if not (32601<=epsg<=32660 or 32701<=epsg<=32760) or not np.isclose(dx,-dy) or not any(np.isclose(dx,v) for v in (10,20)):
        raise AcquisitionError('UNSUPPORTED_GRID','Current profile requires native square UTM 10 or 20 m power pixels')
    crs=f'EPSG:{epsg}'
    west,south,east,north=transform_bounds('EPSG:4326',crs,*bbox,densify_pts=41)
    origin_x,origin_y=float(x[0]-dx/2),float(y[0]-dy/2)
    c0,c1=math.floor((west-origin_x)/dx),math.ceil((east-origin_x)/dx)
    r0,r1=math.floor((north-origin_y)/dy),math.ceil((south-origin_y)/dy)
    shape=(r1-r0,c1-c0)
    if min(shape)<=0 or math.prod(shape)>4_000_000:
        raise AcquisitionError('AOI_PIXEL_LIMIT','Native AOI window is empty or exceeds four million pixels')
    sc0,sc1=max(0,c0),min(len(x),c1)
    sr0,sr1=max(0,r0),min(len(y),r1)
    if sc1<=sc0 or sr1<=sr0:
        raise AcquisitionError('NO_GRID_COVERAGE','Exact granule grid does not intersect the requested AOI')
    return {'crs':crs,'transform':Affine(dx,0,origin_x+c0*dx,0,dy,origin_y+r0*dy),
        'shape':shape,'source_slices':(slice(sr0,sr1),slice(sc0,sc1)),
        'destination_slices':(slice(sr0-r0,sr1-r0),slice(sc0-c0,sc1-c0)),
        'source_shape':[len(y),len(x)],'source_window':[sc0,sr0,sc1-sc0,sr1-sr0],
        'pixel_spacing_m':[dx,-dy]}


def extract_gcov_window(client, record, bbox, *, frequency='A', max_bytes=1024**3):
    """Extract real RTC gamma0 intensities and raw mask through HTTPS ranges."""
    import h5py
    _bbox_geometry(bbox)
    if frequency not in {'A','B'}:
        raise AcquisitionError('FREQUENCY_REQUIRED','Choose exact GCOV frequency A or B')
    url=record['https_urls'][0]
    fields=parse_product_id(record['granule_id'])
    if Path(urlsplit(url).path).name!=Path(record['granule_id']).name.removesuffix('.h5')+'.h5':
        raise AcquisitionError('SOURCE_IDENTITY_MISMATCH','Source HTTPS path must identify the exact selected granule')
    with HTTPSRangeReader(client,url,max_bytes=max_bytes) as reader:
        with h5py.File(reader,'r') as source:
            group_path=f'science/LSAR/GCOV/grids/frequency{frequency}'
            if group_path not in source:
                raise AcquisitionError('FREQUENCY_MISSING','Exact source product lacks the selected frequency')
            group=source[group_path]
            if any(key not in group for key in ('HHHH','HVHV','mask')):
                raise AcquisitionError('CHANNEL_OR_MASK_MISSING','Both HH/HV intensities and product quality mask are required')
            grid=_grid_window(group,bbox)
            arrays={}
            raw_mask=np.full(grid['shape'],255,dtype=np.uint8)
            raw_mask[grid['destination_slices']]=np.asarray(group['mask'][grid['source_slices']],dtype=np.uint8)
            valid=(raw_mask>0)&(raw_mask<255)
            mask_attributes={str(k):str(v) for k,v in group['mask'].attrs.items()}
            mask_values,mask_counts=np.unique(raw_mask,return_counts=True)
            support_inspection={'source_granule':record['granule_id'],'shape':list(grid['shape']),
                'crs':grid['crs'],'source_window':grid['source_window'],
                'raw_mask_counts':{str(int(value)):int(count) for value,count in zip(mask_values,mask_counts)},
                'mask_attributes':mask_attributes}
            if not valid.any():
                raise AcquisitionError('NO_VALID_OBSERVATIONS','Source mask has no fully focused support in the exact AOI',support_inspection)
            layers={}
            for channel in ('HHHH','HVHV'):
                dataset=group[channel]
                if any(str(dataset.attrs.get(key,'False')).lower().strip("b'\"") in {'true','1'}
                       for key in ('isReceiveOnlyChannel1','isReceiveOnlyChannel2')):
                    raise AcquisitionError('RECEIVE_ONLY_INPUT','Receive-only channel is not an active backscatter observation')
                if dataset.shape!=tuple(grid['source_shape']) or dataset.dtype.kind!='f':
                    raise AcquisitionError('SCIENTIFIC_LAYER_MISMATCH','Intensity must be a real floating raster on the identified grid')
                values=np.full(grid['shape'],np.nan,dtype=np.float32)
                values[grid['destination_slices']]=np.asarray(dataset[grid['source_slices']],dtype=np.float32)
                valid &= np.isfinite(values)&(values>0)
                arrays[channel]=values
                layers[channel]={'dataset':'/'+group_path+'/'+channel,'dtype':str(dataset.dtype),
                    'chunk_shape':list(dataset.chunks) if dataset.chunks else None,'compression':dataset.compression,
                    'attributes':{str(k):str(v) for k,v in dataset.attrs.items()}}
            for values in arrays.values():
                values[~valid]=np.nan
            identification={}
            if 'science/LSAR/identification' not in source:
                raise AcquisitionError('SOURCE_METADATA_UNAVAILABLE','GCOV identification metadata is required')
            if 'science/LSAR/identification' in source:
                identity=source['science/LSAR/identification']
                for key in ('granuleId','zeroDopplerStartTime','zeroDopplerEndTime','trackNumber','frameNumber',
                            'orbitPassDirection','productVersion','productSpecificationVersion','compositeReleaseId','productType','radarBand','missionId'):
                    value=_metadata_value(identity,key)
                    if value is not None:
                        identification[key]=value
                required=('granuleId','zeroDopplerStartTime','trackNumber','frameNumber',
                          'productVersion','compositeReleaseId','productType','radarBand','missionId')
                if any(key not in identification or identification[key] in (None,'') for key in required):
                    raise AcquisitionError('SOURCE_METADATA_UNAVAILABLE','Complete in-file GCOV identity and processing metadata are required')
                if 'trackNumber' in identification and int(identification['trackNumber'])!=fields['track']:
                    raise AcquisitionError('SOURCE_IDENTITY_MISMATCH','HDF5 track differs from selected catalog identity')
                if 'frameNumber' in identification and int(identification['frameNumber'])!=fields['frame']:
                    raise AcquisitionError('SOURCE_IDENTITY_MISMATCH','HDF5 frame differs from selected catalog identity')
                expected_identity={'granuleId':record['granule_id'].removesuffix('.h5'),
                    'compositeReleaseId':fields['crid'],'radarBand':'L','missionId':'NISAR','productType':'GCOV'}
                for key,expected_value in expected_identity.items():
                    if key in identification and str(identification[key]).removesuffix('.h5')!=expected_value:
                        raise AcquisitionError('SOURCE_IDENTITY_MISMATCH','HDF5 identity differs from the exact selected product')
                if not identification.get('productVersion'):
                    raise AcquisitionError('PROCESSING_VERSION_UNAVAILABLE','Actual in-file productVersion is required; filename counter is not a software version')
                if 'zeroDopplerStartTime' in identification:
                    actual=datetime.fromisoformat(str(identification['zeroDopplerStartTime']).replace('Z','+00:00'))
                    if actual.tzinfo is None:
                        # NISAR product-spec timestamps are UTC, including the
                        # in-file form without an explicit Z suffix.
                        actual=actual.replace(tzinfo=timezone.utc)
                    expected=datetime.fromisoformat(fields['start'].replace('Z','+00:00'))
                    if abs((actual-expected).total_seconds())>=1:
                        raise AcquisitionError('SOURCE_DATE_MISMATCH','HDF5 acquisition time differs from the exact selected second')
            metrics={'transferred_bytes':reader.bytes_transferred,'range_requests':reader.requests,'remote_size_bytes':reader.size}
    if not valid.any():
        raise AcquisitionError('NO_VALID_OBSERVATIONS','No common valid positive HH/HV support exists in the requested AOI')
    return {**{k:v for k,v in grid.items() if k not in ('source_slices','destination_slices')},
        'arrays':arrays,'raw_mask':raw_mask,'valid':valid,'layers':layers,'mask_attributes':mask_attributes,
        'identification':identification,'source':record,'frequency':frequency,'metrics':metrics}


def _write_cog(path, values, window, *, channel, granule_id, units, nodata):
    import rasterio
    from rasterio.shutil import copy as rio_copy
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.working.tif')
    partial=path.with_suffix('.part.tif')
    with rasterio.open(temporary,'w',driver='GTiff',height=values.shape[0],width=values.shape[1],count=1,
                       dtype=values.dtype,crs=window['crs'],transform=window['transform'],nodata=nodata,
                       tiled=True,blockxsize=256,blockysize=256,compress='deflate') as output:
        output.write(values,1)
        output.set_band_description(1,channel)
        output.update_tags(granule_id=granule_id,units=units,polarization=channel,
                           provenance='observed NISAR GCOV bounded HTTPS scientific window')
    rio_copy(temporary,partial,driver='COG',compress='DEFLATE',blocksize=256,
             overview_resampling='AVERAGE' if units=='gamma0_power' else 'NEAREST')
    temporary.unlink()
    partial.replace(path)
    with rasterio.open(path) as source:
        if source.crs.to_string()!=window['crs'] or source.shape!=values.shape or source.tags(ns='IMAGE_STRUCTURE').get('LAYOUT')!='COG':
            raise AcquisitionError('OUTPUT_VERIFICATION_FAILED','Derived scientific COG grid or layout failed verification')
    return {'reference':str(path.resolve()),'sha256':sha256(path),'band':1,'dataset':None}

"""Public, dated Sentinel-2 reference windows with strict byte-accounted ranges."""
import io
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_geom
import requests
from shapely.geometry import shape
from .common import read_json, write_json, sha256

HOST = "sentinel-cogs.s3.us-west-2.amazonaws.com"


class RangeFile(io.RawIOBase):
    def __init__(self, url, cache, ledger, session):
        super().__init__()
        if urlparse(url).hostname != HOST or urlparse(url).scheme != "https":
            raise ValueError("Reference host is not the supported public Sentinel COG host")
        self.name, self.cache, self.ledger, self.session = url, cache, ledger, session
        self.position = 0
        response = session.head(url, timeout=30, allow_redirects=False)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, count=-1):
        if count < 0:
            count = self.size-self.position
        if count > 8*1024**2:
            raise ValueError("Unbounded reference read rejected")
        block_size = 256*1024
        end, chunks = min(self.size, self.position+count), []
        while self.position < end:
            block = self.position//block_size
            start, stop = block*block_size, min((block+1)*block_size, self.size)
            if block not in self.cache:
                if self.ledger["body_bytes"]+stop-start > self.ledger["cap_bytes"]:
                    raise ValueError("Reference transfer cap reached")
                with self.session.get(self.name, headers={"Range":f"bytes={start}-{stop-1}", "Accept-Encoding":"identity"},
                                      timeout=30, stream=True, allow_redirects=False) as response:
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop-1}/{self.size}":
                        raise ValueError("Exact reference range was not honored")
                    data = response.raw.read(stop-start+1)
                if len(data) != stop-start:
                    raise ValueError("Reference range byte count mismatch")
                self.cache[block] = data
                self.ledger["body_bytes"] += len(data)
                self.ledger["requests"].append({"url":self.name,"start":start,"end_exclusive":stop,"body_bytes":len(data)})
            offset = self.position-start
            length = min(end-self.position,stop-self.position)
            chunks.append(self.cache[block][offset:offset+length])
            self.position += length
        return b"".join(chunks)


def crop_public_cog(url, aoi, ledger, session):
    cache = {}
    def opener(path, mode="rb"):
        if path != url:
            raise FileNotFoundError(path)
        return RangeFile(url,cache,ledger,session)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(url,opener=opener) as src:
            geometry = transform_geom("EPSG:4326",src.crs,aoi)
            window = from_bounds(*shape(geometry).bounds,transform=src.transform).round_offsets().round_lengths()
            window = window.intersection(Window(0,0,src.width,src.height))
            return src.read(window=window),src.window_transform(window),src.crs


def fetch_reference(entry, output, *, max_hours=24):
    output = Path(output)
    receipt = output/"reference.json"
    if receipt.exists():
        saved = read_json(receipt)
        for asset in saved.get("local_assets",{}).values():
            if sha256(asset["path"]) != asset["sha256"]:
                raise ValueError("Cached optical reference hash mismatch")
        return saved
    output.mkdir(parents=True,exist_ok=True)
    ledger = {"body_bytes":0,"cap_bytes":64*1024**2,"requests":[]}
    session = requests.Session()
    session.trust_env = False
    radar = datetime.fromisoformat(entry["acquired_at"].replace("Z","+00:00"))
    result = {"observation_id":entry["id"],"radar_date":entry["acquired_at"],"state":"unavailable",
              "selection":"nearest acquisition within24h, coverage>=80% when available; before model prediction; clouds/unknown excluded per pixel"}
    try:
        query={"collections":["sentinel-2-l2a"],"intersects":entry["aoi"],"limit":30,
               "datetime":f"{(radar-timedelta(hours=max_hours)).isoformat()}/{(radar+timedelta(hours=max_hours)).isoformat()}"}
        response=session.post("https://earth-search.aws.element84.com/v1/search",json=query,timeout=30)
        response.raise_for_status()
        items=response.json()["features"]
        write_json(output/"search.json",{"query":query,"items":items})
        geometry=shape(entry["aoi"])
        covered=[i for i in items if shape(i["geometry"]).intersection(geometry).area/geometry.area>=.8]
        if covered:
            items=covered
        if not items:
            result["reason"]="no_optical_acquisition_within_24_hours"
        else:
            item=min(items,key=lambda i:abs((datetime.fromisoformat(i["properties"]["datetime"].replace("Z","+00:00"))-radar).total_seconds()))
            optical=datetime.fromisoformat(item["properties"]["datetime"].replace("Z","+00:00"))
            result.update(source_id=item["id"],reference_date=item["properties"]["datetime"],
                          date_difference_hours=(optical-radar).total_seconds()/3600,
                          tile_cloud_percent=item["properties"].get("eo:cloud_cover"),local_assets={})
            for key in ("scl","visual"):
                url=item["assets"][key]["href"]
                data,transform,crs=crop_public_cog(url,entry["aoi"],ledger,session)
                path=output/(key+".tif")
                with rasterio.open(path,"w",driver="GTiff",count=len(data),height=data.shape[1],width=data.shape[2],
                                   dtype=data.dtype,crs=crs,transform=transform,compress="DEFLATE") as dst:
                    dst.write(data)
                result["local_assets"][key]={"path":str(path.resolve()),"sha256":sha256(path),"source_url":url}
            result["state"]="ready"
    except Exception as error:
        # These are public requests without authorization headers; record a bounded diagnostic.
        result.update(state="failed",reason=type(error).__name__,message=str(error)[:500])
    finally:
        session.close()
        write_json(output/"transfer-ledger.json",ledger)
    write_json(receipt,result)
    return result


"""Bounded live acquisition using the existing protected Earthdata runtime.

Only public metadata and paths enter worker jobs. Authentication remains in the
existing current-user encrypted credential helper, never in notebook cells.
"""
from pathlib import Path
import importlib.util
import sys
import uuid
from .common import read_json, write_json


def reference_adapter(reference_root):
    reference_root = Path(reference_root).resolve()
    source = reference_root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from nisar_landcover import acquisition
    return acquisition


def discover(aoi, start, end, work_dir, reference_root):
    adapter = reference_adapter(reference_root)
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    job = work_dir / ("discover-"+uuid.uuid4().hex)
    job.mkdir()
    request = {"operation": "discover", "validated": adapter.validate_request_aoi(aoi),
               "temporal": adapter._dates(start, end), "page_size": 100, "max_pages": 2,
               "storage_root": str(work_dir), "progress_path": str(job/"progress.json"),
               "cancel_path": str(job/"cancel"), "result_path": str(job/"result.json")}
    write_json(job/"request.json", request)
    try:
        result = adapter._discover_worker(request)
    except Exception as error:
        result = adapter._failure(error)
        result["candidates"] = []
    write_json(job/"result.json", result)
    return result


def acquire(candidate, aoi, work_dir, reference_root, max_bytes=128*1024**2, *, halo_metres=None):
    adapter = reference_adapter(reference_root)
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    job = work_dir / ("acquire-"+uuid.uuid4().hex)
    job.mkdir()
    old_halo,old_recipe=adapter.HALO_METRES,adapter.PREPROCESSING
    if halo_metres is not None:
        if not 80<=int(halo_metres)<=1000:
            raise ValueError("Context halo must be between 80 and 1000 metres")
        adapter.HALO_METRES=int(halo_metres)
        adapter.PREPROCESSING=f"round3-gcov-power-halo{int(halo_metres)}m-v1"
    request = {"operation": "acquire", "candidate": candidate,
               "validated": adapter.validate_request_aoi(aoi), "storage_root": str(work_dir/"crops"),
               "catalog_path": str(work_dir/"catalog.json"), "max_transfer_bytes": max_bytes,
               "progress_path": str(job/"progress.json"), "cancel_path": str(job/"cancel"),
               "result_path": str(job/"result.json")}
    write_json(job/"request.json", request)
    try:
        result = adapter._acquire_worker(request)
    except Exception as error:
        result = adapter._failure(error)
    finally:
        adapter.HALO_METRES,adapter.PREPROCESSING=old_halo,old_recipe
    if halo_metres is not None and result.get('observation'):
        observation=result['observation']
        observation['context_halo_metres']=int(halo_metres)
        original=observation.get('provenance',[])
        observation['reader_provenance_original']=original.copy()
        observation['provenance']=[note.replace('80m texture/alignment halo',f'{int(halo_metres)}m texture/alignment halo') for note in original]
    write_json(job/"result.json", result)
    return result


def select_dates(candidates, max_dates=2):
    """Prefer a single fully covering geometry, independent of model outputs."""
    groups = {}
    for candidate in candidates:
        if not candidate.get("compatible") or candidate.get("coverage_fraction", 0) < .8:
            continue
        record = candidate["record"]
        key = (record.get("track"), record.get("pass_direction"), record.get("frame"))
        # Summarized metadata keys vary; parse the exact granule as the authority.
        from .inputs import parse_granule
        meta = parse_granule(candidate["granule_id"]+".h5")
        key = (meta["track"], meta["pass_direction"], meta["frame"])
        groups.setdefault(key, []).append(candidate)
    if not groups:
        return []
    key = max(groups, key=lambda k: (len({x["acquired_at"] for x in groups[k]}),
                                    max(x["acquired_at"] for x in groups[k])))
    unique = {}
    for candidate in groups[key]:
        stamp = candidate["acquired_at"]
        old = unique.get(stamp)
        if old is None or (candidate.get("revision_id") or 0) > (old.get("revision_id") or 0):
            unique[stamp] = candidate
    ordered = sorted(unique.values(), key=lambda x: x["acquired_at"])
    if len(ordered) <= max_dates:
        return ordered
    return [ordered[0], ordered[-1]] if max_dates == 2 else ordered[-max_dates:]


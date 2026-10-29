"""Adapters for the existing crop catalog, GCOV HDF5 and notebook xarray cubes.

All adapters retain linear gamma0 power until after aggregation. No imagery RGB,
place name, optical reference or future observation is an inference feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely.geometry import shape, box, mapping
from shapely.ops import transform as geom_transform

from .common import sha256, safe_id

CHANNELS = ("HHHH", "HVHV")
MAX_INPUT_PIXELS = 16_000_000


@dataclass
class Observation:
    power: np.ndarray
    valid: np.ndarray
    transform: Affine
    crs: object
    metadata: dict
    weights: np.ndarray | None = None
    exclusions: np.ndarray | None = None

    def __post_init__(self):
        self.power = np.asarray(self.power, dtype=np.float32)
        self.valid = np.asarray(self.valid, dtype=bool)
        if self.power.shape != (2, *self.valid.shape) or self.valid.ndim != 2:
            raise ValueError("Expected ordered HH/HV power (2,H,W) and validity (H,W)")
        if self.valid.size > MAX_INPUT_PIXELS:
            raise ValueError("Input exceeds 16 million pixels; provide a bounded crop")
        crs = CRS.from_user_input(self.crs)
        if not crs.is_projected or crs.axis_info[0].unit_name != "metre":
            raise ValueError("Use a projected grid in metres")
        epsg = crs.to_epsg() or 0
        if not (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
            raise ValueError("Current area/feature profile requires a local UTM grid")
        t = self.transform
        if t.b or t.d or t.a <= 0 or t.e >= 0 or not np.isclose(t.a, -t.e):
            raise ValueError("Only north-up square grids are supported")
        if not np.isclose(t.a, 20):
            raise ValueError("The trained profile needs 20 m power pixels; call prepare first")
        self.crs = rasterio.crs.CRS.from_user_input(crs)
        self.valid &= np.isfinite(self.power).all(0) & (self.power > 0).all(0)
        if self.weights is None:
            self.weights = np.ones(self.valid.shape, np.float32)
        if self.weights.shape != self.valid.shape or not np.isfinite(self.weights).all():
            raise ValueError("AOI weights must match the grid and be finite")
        if (self.weights < 0).any() or (self.weights > 1).any():
            raise ValueError("AOI weights must be in [0,1]")
        if self.exclusions is None:
            self.exclusions = np.zeros(self.valid.shape, bool)
        self.exclusions = np.asarray(self.exclusions, bool)
        if self.exclusions.shape != self.valid.shape:
            raise ValueError("Exclusion mask must match the grid")


def aoi_weights(aoi, transform, crs, shape_hw):
    """Fractional pixel coverage, preserving polygon holes and partial boundary cells."""
    if aoi is None:
        return np.ones(shape_hw, np.float32)
    if aoi.get("type") == "Feature":
        aoi = aoi["geometry"]
    geom = shape(aoi)
    if geom.is_empty or not geom.is_valid or geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("AOI must be a valid WGS84 Polygon or MultiPolygon")
    geom = geom_transform(Transformer.from_crs(4326, crs, always_xy=True).transform, geom)
    footprint = box(*rasterio.transform.array_bounds(*shape_hw, transform))
    geom = geom.intersection(footprint)
    weights = np.zeros(shape_hw, np.float32)
    if geom.is_empty:
        return weights
    touched = rasterize([(mapping(geom), 1)], out_shape=shape_hw, transform=transform,
                        all_touched=True, dtype="uint8").astype(bool)
    interior = geom.buffer(-np.hypot(transform.a, transform.e) / 2 - 1e-7)
    if not interior.is_empty:
        weights[rasterize([(mapping(interior), 1)], out_shape=shape_hw,
                          transform=transform, dtype="uint8").astype(bool)] = 1
    cell_area = abs(transform.a * transform.e)
    for row, col in zip(*np.where(touched & (weights == 0))):
        left, top = transform @ (int(col), int(row))
        right, bottom = transform @ (int(col) + 1, int(row) + 1)
        weights[row, col] = min(1, geom.intersection(box(left, bottom, right, top)).area / cell_area)
    return weights


def prepare(power, valid, transform, crs, metadata, *, aoi=None, exclusions=None):
    metadata = dict(metadata)
    if aoi is not None:
        metadata["aoi"] = aoi.get("geometry", aoi)
        requested = geom_transform(Transformer.from_crs(4326, crs, always_xy=True).transform,
                                   shape(metadata["aoi"]))
        metadata["requested_aoi_area_ha"] = float(requested.area/10000)
    power = np.asarray(power, np.float32)
    valid = np.asarray(valid, bool).copy()
    if valid.size > MAX_INPUT_PIXELS or power.shape != (2, *valid.shape):
        raise ValueError("Invalid or oversized input shape")
    if not np.isclose(transform.a, -transform.e) or transform.b or transform.d:
        raise ValueError("Input needs north-up square pixels")
    if not (np.isclose(transform.a, 10) or np.isclose(transform.a, 20)):
        raise ValueError("Only native 10 m or 20 m data are supported; no upsampling")
    valid &= np.isfinite(power).all(0) & (power > 0).all(0)
    if exclusions is not None:
        exclusions = np.asarray(exclusions, bool)
        if exclusions.shape != valid.shape:
            raise ValueError("Exclusion mask shape differs from source")
    if np.isclose(transform.a, 10):
        h, w = valid.shape
        h2, w2 = h // 2, w // 2
        if h2 == 0 or w2 == 0:
            raise ValueError("Insufficient pixels for 20 m aggregation")
        trimmed = power[:, :h2*2, :w2*2]
        good = valid[:h2*2, :w2*2].reshape(h2, 2, w2, 2).all((1, 3))
        power = np.where(valid[:h2*2, :w2*2][None], trimmed, np.nan)
        power = power.reshape(2, h2, 2, w2, 2).mean((2, 4))
        if exclusions is not None:
            exclusions = exclusions[:h2*2, :w2*2].reshape(h2, 2, w2, 2).any((1, 3))
        valid = good
        transform = transform @ Affine.scale(2)
        metadata = {**metadata, "aggregation": "2x2 mean linear power; all four samples valid; incomplete edges trimmed"}
    weights = aoi_weights(aoi, transform, crs, valid.shape)
    return Observation(power, valid, transform, crs, metadata, weights, exclusions)


def load_catalog_observation(entry, base_dir=None, *, verify=True):
    for key, expected in {"sensor": "NISAR", "radar_band": "L", "product_type": "GCOV",
                          "frequency": "A", "units": "gamma0_power", "rtc_applied": True}.items():
        if entry.get(key) != expected:
            raise ValueError(f"This model requires {key}={expected}")
    if entry.get("polarizations") != list(CHANNELS):
        raise ValueError("Ordered HHHH, HVHV covariance powers are required")
    paths = []
    for key in (*CHANNELS, "mask"):
        asset = entry["assets"][key]
        path = Path(asset["reference"])
        if not path.is_absolute():
            path = Path(base_dir or ".") / path
        if verify and (not asset.get("sha256") or sha256(path) != asset["sha256"]):
            raise ValueError(f"Integrity check failed for {key}")
        paths.append((path, asset.get("band", 1)))
    with rasterio.open(paths[0][0]) as hh, rasterio.open(paths[1][0]) as hv, rasterio.open(paths[2][0]) as mask:
        for src in (hv, mask):
            if (src.shape, src.crs, src.transform) != (hh.shape, hh.crs, hh.transform):
                raise ValueError("Input channels/mask are not aligned")
        if hh.width * hh.height > MAX_INPUT_PIXELS:
            raise ValueError("Catalog crop exceeds pixel limit")
        power = np.stack([hh.read(paths[0][1], masked=True).filled(np.nan),
                          hv.read(paths[1][1], masked=True).filled(np.nan)])
        m = mask.read(paths[2][1], masked=True)
        valid = ~np.ma.getmaskarray(m) & np.isin(m.data, entry.get("mask_valid_values", [1]))
        meta = {k: entry.get(k) for k in ("id", "granule_id", "acquired_at", "track", "frame",
                 "pass_direction", "frequency", "processing_version", "crid", "data_origin")}
        meta["source_sha256"] = {k: entry["assets"][k]["sha256"] for k in (*CHANNELS, "mask")}
        meta["mask_policy"] = "verified derived source validity; geometry shadow/layover not supplied"
        return prepare(power, valid, hh.transform, hh.crs, meta, aoi=entry.get("aoi"))


def parse_granule(name):
    m = re.search(r"NISAR_L2_[^_]+_GCOV_\d{3}_(\d{3})_([AD])_(\d{3})_.*?_(\d{8}T\d{6})_", name)
    if not m:
        raise ValueError("Need an exact NISAR GCOV granule name to identify acquisition geometry")
    stamp = m[4]
    return {"track": int(m[1]), "pass_direction": m[2], "frame": int(m[3]),
            "acquired_at": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}Z",
            "granule_id": name.removesuffix(".h5"), "id": safe_id(name.removesuffix(".h5"))}


def from_dataset(dataset):
    """Yield every acquisition of the exact `pila` returned by construir_pila.

    Pass the in-memory Dataset, or save it as NetCDF. We never open pickle files.
    Notebook multilooking can average partially valid blocks; that source policy
    is explicitly retained in provenance rather than represented as native QA.
    """
    if not set(CHANNELS).issubset(dataset.data_vars) or dataset.attrs.get("frecuencia", "A") != "A":
        raise ValueError("Need frequency A HHHH and HVHV variables")
    desc = str(dataset.attrs.get("descripcion", "")).lower()
    if dataset.attrs.get("units") != "gamma0_power" and not ("gamma0" in desc and "lineal" in desc):
        raise ValueError("Dataset must document linear gamma0 power in units or descripcion")
    epsg = dataset.attrs.get("epsg")
    x, y = np.asarray(dataset.x), np.asarray(dataset.y)
    if epsg is None or len(x) < 2 or len(y) < 2:
        raise ValueError("Need epsg and at least two x/y pixel centres")
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    if not np.allclose(np.diff(x), dx) or not np.allclose(np.diff(y), dy):
        raise ValueError("Notebook grid is irregular")
    transform = Affine(dx, 0, float(x[0] - dx/2), 0, dy, float(y[0] - dy/2))
    from shapely import wkt
    aoi = mapping(wkt.loads(dataset.attrs["aoi_wkt"])) if dataset.attrs.get("aoi_wkt") else None
    seen = set()
    for i in range(dataset.sizes.get("time", 0)):
        ds = dataset.isel(time=i)
        if "nombre" not in ds.coords:
            raise ValueError("Notebook cube must retain its nombre acquisition coordinate")
        name = str(ds.nombre.item())
        meta = parse_granule(name)
        if meta["granule_id"] in seen:
            raise ValueError("Duplicate acquisition in notebook cube")
        seen.add(meta["granule_id"])
        actual = np.datetime_as_string(ds.time.values.astype("datetime64[s]"), unit="s") + "Z"
        if actual != meta["acquired_at"]:
            raise ValueError("Notebook time disagrees with granule acquisition time")
        meta.update(frequency="A", data_origin="observed", mask_policy="notebook NaN mask; inherited multilook validity",
                    notebook_multilook=int(dataset.attrs.get("multilook", 1)))
        power = np.stack([ds[c].transpose("y", "x").values for c in CHANNELS])
        yield prepare(power, np.isfinite(power).all(0), transform, epsg, meta, aoi=aoi)


def from_hdf5(path, *, aoi=None, window=None):
    """Read a bounded local GCOV window; never materialize a whole large HDF5."""
    import h5py
    path = Path(path)
    meta = parse_granule(path.name)
    with h5py.File(path, "r") as h5:
        ident = h5.get("science/LSAR/identification")
        if ident is None:
            raise ValueError("GCOV identification metadata is required")
        for key, expected in (("productType", "GCOV"), ("radarBand", "L")):
            if key not in ident:
                raise ValueError(f"Missing HDF5 {key} identification")
            actual = ident[key][()]
            actual = actual.decode() if isinstance(actual, bytes) else str(actual)
            if actual != expected:
                raise ValueError(f"This model requires HDF5 {key}={expected}")
        for source_key, target_key in (("productVersion", "processing_version"), ("compositeReleaseId", "crid")):
            if source_key in ident:
                value = ident[source_key][()]
                meta[target_key] = value.decode() if isinstance(value, bytes) else str(value)
        for source_key, expected in (("trackNumber", meta["track"]), ("frameNumber", meta["frame"])):
            if source_key in ident and int(ident[source_key][()]) != expected:
                raise ValueError("Filename and HDF5 acquisition geometry disagree")
        g = h5["science/LSAR/GCOV/grids/frequencyA"]
        x, y = np.asarray(g["xCoordinates"]), np.asarray(g["yCoordinates"])
        projection = g["projection"]
        epsg = int(projection.attrs.get("epsg_code", projection[()]))
        dx, dy = float(x[1]-x[0]), float(y[1]-y[0])
        transform = Affine(dx, 0, x[0]-dx/2, 0, dy, y[0]-dy/2)
        if aoi is not None:
            geom = shape(aoi.get("geometry", aoi))
            geom = geom_transform(Transformer.from_crs(4326, epsg, always_xy=True).transform, geom)
            # 320 m context for the largest spatial window.
            raw = from_bounds(*geom.buffer(320).bounds, transform)
            c0 = max(0, int(np.floor(raw.col_off/2))*2)
            r0 = max(0, int(np.floor(raw.row_off/2))*2)
            c1 = min(len(x), int(np.ceil((raw.col_off+raw.width)/2))*2)
            r1 = min(len(y), int(np.ceil((raw.row_off+raw.height)/2))*2)
        elif window is not None:
            c0, r0, width, height = map(int, window)
            c1, r1 = c0 + width, r0 + height
        else:
            raise ValueError("Provide a WGS84 AOI or pixel window for HDF5 input")
        if c0 < 0 or r0 < 0 or c1 > len(x) or r1 > len(y) or c1 <= c0 or r1 <= r0:
            raise ValueError("Window does not intersect the source grid")
        if (c1-c0)*(r1-r0) > MAX_INPUT_PIXELS:
            raise ValueError("HDF5 crop exceeds 16 million source pixels")
        power = np.stack([g[c][r0:r1, c0:c1] for c in CHANNELS])
        if "mask" not in g:
            raise ValueError("GCOV native subswath mask is required")
        m = np.asarray(g["mask"][r0:r1, c0:c1])
        meta.update(frequency="A", data_origin="observed", source_file=str(path.resolve()),
                    source_window=[c0, r0, c1-c0, r1-r0], mask_policy="native GCOV 0/255 rejected; not a shadow/layover mask")
        return prepare(power, (m > 0) & (m < 255),
                       window_transform(Window(c0, r0, c1-c0, r1-r0), transform), epsg, meta, aoi=aoi)


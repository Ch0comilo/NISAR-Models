"""Scientific rasters plus independently georeferenced web overlays."""
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.warp import reproject, transform_bounds
from affine import Affine
from scipy.ndimage import binary_erosion
from shapely.geometry import shape, mapping, box
from shapely.ops import transform as geom_transform
from pyproj import Transformer

from .common import write_json, sha256, identity, utcnow

WATER = (35, 153, 245, 180)
BORDER = (126, 233, 255, 255)


def write_raster(path, values, transform, crs, *, nodata=None):
    values = np.asarray(values)
    if values.ndim == 2:
        values = values[None]
    with rasterio.open(path, "w", driver="COG", width=values.shape[2], height=values.shape[1],
                       count=values.shape[0], dtype=values.dtype, crs=crs, transform=transform,
                       nodata=nodata, compress="DEFLATE", blocksize=256,
                       overview_resampling="NEAREST") as dst:
        dst.write(values)


def _web_grid(observation):
    h, w = observation.valid.shape
    bounds = transform_bounds(observation.crs, "EPSG:4326",
                             *rasterio.transform.array_bounds(h, w, observation.transform), densify_pts=21)
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3] or bounds[2]-bounds[0] > 180:
        raise ValueError("This web overlay profile does not support antimeridian-crossing crops")
    # Web pixels cover exactly these WGS84 bounds; native UTM PNGs cannot use these bounds directly.
    scale = min(1, 1600/max(h, w))
    wh, ww = max(1, round(h*scale)), max(1, round(w*scale))
    transform = Affine((bounds[2]-bounds[0])/ww, 0, bounds[0], 0, -(bounds[3]-bounds[1])/wh, bounds[3])
    return list(bounds), transform, (wh, ww)


def render(observation, labels):
    bounds, web_transform, web_shape = _web_grid(observation)
    web_labels = np.full(web_shape, 255, np.uint8)
    reproject(labels, web_labels, src_transform=observation.transform, src_crs=observation.crs,
              dst_transform=web_transform, dst_crs="EPSG:4326", src_nodata=255, dst_nodata=255,
              resampling=Resampling.nearest)
    db = np.full(observation.valid.shape, np.nan, np.float32)
    good = observation.valid & (observation.weights > 0)
    db[good] = 10*np.log10(observation.power[0, good])
    web_db = np.full(web_shape, np.nan, np.float32)
    reproject(db, web_db, src_transform=observation.transform, src_crs=observation.crs,
              dst_transform=web_transform, dst_crs="EPSG:4326", src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    water = web_labels == 1
    border = water & ~binary_erosion(water, structure=np.ones((3, 3)), border_value=0)
    fill = np.zeros((*web_shape, 4), np.uint8)
    fill[water] = WATER
    outlines = np.zeros_like(fill)
    outlines[border] = BORDER
    gray = np.uint8(np.clip(np.nan_to_num((web_db+30)/33), 0, 1)*255)
    sar = np.stack([gray, gray, gray, np.where(np.isfinite(web_db), 255, 0).astype(np.uint8)], axis=-1)
    composite = np.array(Image.alpha_composite(Image.fromarray(sar), Image.fromarray(fill)))
    composite[border] = BORDER
    uncertain = np.zeros_like(fill)
    uncertain[web_labels == 254] = (255, 174, 55, 160)
    return bounds, {"sar": sar, "water": fill, "border": outlines,
                    "composite": composite, "uncertain": uncertain}


def export_observation(observation, labels, score, info, model, output, *, asset_root=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_raster(output / "water-mask.tif", labels, observation.transform, observation.crs, nodata=255)
    write_raster(output / "water-score.tif", score, observation.transform, observation.crs, nodata=np.nan)
    write_raster(output / "aoi-fraction.tif", observation.weights.astype(np.float32), observation.transform, observation.crs)
    bounds, images = render(observation, labels)
    if info.get('review_mode'):
        images['review-composite']=np.array(Image.alpha_composite(Image.fromarray(images['composite']),Image.fromarray(images['uncertain'])))
    for name, image in images.items():
        Image.fromarray(image).save(output / f"{name}.png")
    to_wgs84 = Transformer.from_crs(observation.crs, 4326, always_xy=True).transform
    features = []
    cell_area = abs(observation.transform.a*observation.transform.e)
    # Fractional AOI cells are clipped using the source AOI geometry when supplied.
    aoi_native = None
    if observation.metadata.get("aoi"):
        aoi_native = geom_transform(Transformer.from_crs(4326, observation.crs, always_xy=True).transform,
                                    shape(observation.metadata["aoi"]))
    for geom, value in shapes((labels == 1).astype(np.uint8), mask=labels == 1, transform=observation.transform):
        if value != 1:
            continue
        polygon = shape(geom)
        if aoi_native is not None:
            polygon = polygon.intersection(aoi_native)
        if polygon.is_empty:
            continue
        features.append({"type": "Feature", "geometry": mapping(geom_transform(to_wgs84, polygon)),
                         "properties": {"observationId": observation.metadata["id"], "class": "open_water",
                                        "areaHa": polygon.area/10000, "perimeterM": polygon.length}})
    write_json(output / "water.geojson", {"type": "FeatureCollection", "features": features})
    from .context_provenance import scientific_context
    context=scientific_context(info)
    if context is not None:write_json(output/'context-provenance.json',context)
    weight_ha = observation.weights.astype(np.float64)*cell_area/10000
    observed = observation.valid & (observation.weights > 0)
    captured_ha = float(weight_ha.sum())
    requested_ha = max(captured_ha, observation.metadata.get("requested_aoi_area_ha", captured_ha))
    states = {"waterHa": float(weight_ha[labels == 1].sum()),
              "nonwaterHa": float(weight_ha[labels == 0].sum()),
              "uncertainHa": float(weight_ha[labels == 254].sum()),
              "unobservedHa": float(weight_ha[(labels == 255) & (observation.weights > 0)].sum()) + requested_ha-captured_ha,
              "observedHa": float(weight_ha[observed].sum()),
              "requestedHa": requested_ha}
    files = [*sorted(output.glob("*.png")), *sorted(output.glob("*.tif")), output / "water.geojson"]
    if context is not None:files.append(output/'context-provenance.json')
    assets = {}
    for path in files:
        relative = path.relative_to(asset_root).as_posix() if asset_root is not None else path.name
        key = "water-polygons" if path.suffix == ".geojson" else path.stem
        assets[key] = {"url": "/assets/"+relative, "sha256": sha256(path), "bytes": path.stat().st_size}
    result = {"schemaVersion": "nisar-water-observation-1", "id": observation.metadata["id"],
              "datetime": observation.metadata["acquired_at"], "granuleId": observation.metadata.get("granule_id"),
              "kind": "water_extent", "isDemo": observation.metadata.get("data_origin") == "synthetic_fixture",
              "modelId": model.model_id, "modelSha256": model.fingerprint,
              "evaluationStatus": info["evaluation_status"], "threshold": info["threshold"],
              "decisionPolicy": info.get("decision_policy", {"kind": "threshold", "threshold": info["threshold"]}),
              "thresholdOverride": info["threshold_override"], "scoreSemantics": info["score_semantics"],
              "crs": str(observation.crs), "displayCrs": "EPSG:4326", "bounds": bounds,
              "shape": list(labels.shape), "pixelSpacingM": [20, 20], "displayRangeDb": [-30, 3],
              "classCodes": {"0": "nonwater", "1": "open_water", "254": "uncertain_or_excluded", "255": "unobserved_or_outside"},
              "statistics": states, "assets": assets, "provenance": observation.metadata,
              "generatedAt": utcnow(), "limitations": [
                  "Experimental weak-reference model; global accuracy and independent shoreline accuracy unvalidated",
                  "Dark dry ground and radar shadows can still resemble water",
                  "Native subswath validity does not remove all terrain shadow or layover",
                  "Wind-roughened water and inundated vegetation may be missed",
                  "Water extent change alone does not establish human activity or a flood cause"]}
    if context is not None:
        result['contextInputSha256']=context['input_sha256']
        result['contextAblation']=info['ablation']
        result['requiredContextWithheldHa']=info.get('required_context_withheld_ha')
    if info.get('review_mode'):
        result['reviewMode']=True
        result['disagreementWithheldHa']=info['disagreement_withheld_ha']
    write_json(output / "result.json", result)
    return result


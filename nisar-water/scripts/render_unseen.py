"""Render the complete radar-covered footprint and frozen reference errors."""
import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from PIL import Image, ImageDraw, ImageFont
from nisar_water.common import read_json
from nisar_water.development import TEST_ID
from nisar_water.inputs import load_catalog_observation
from nisar_water.exports import _web_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = read_json(args.results / "unseen-evaluation.json")
    entry = next(e for e in read_json(args.source / "data/catalog.json")["observations"] if e["id"] == TEST_ID)
    o = load_catalog_observation(entry)
    _, wt, wh = _web_grid(o)
    with np.load(args.results / "reference-support.npz") as z:
        scl, source_class, evaluated = z["raw_scl"], z["source_class"], z["evaluated"]
    with np.load(args.results / "improved_v2-prediction.npz") as z:
        pred = z["labels"]
    results = {}
    for name in ("previous_v1", "improved_v2"):
        series = read_json(args.results / name / "series.json")
        record = series["observations"][0]
        for key in ("sar", "composite"):
            path = args.results / name / record["assets"][key]["url"].removeprefix("/assets/")
            results[name+"_"+key] = Image.open(path).convert("RGBA")
    sar = results["improved_v2_sar"]
    coverage = np.asarray(sar)[..., 3] > 0
    rr, cc = np.where(coverage)
    crop = (int(cc.min()), int(rr.min()), int(cc.max()+1), int(rr.max()+1))
    def warp(array):
        dst = np.zeros(wh, np.uint8)
        reproject(array, dst, src_transform=o.transform, src_crs=o.crs,
                  dst_transform=wt, dst_crs="EPSG:4326", resampling=Resampling.nearest)
        return dst
    ref_dir = args.source / "data/labels/water-correction-v4/red-rock-reference"
    optical = np.zeros((*wh, 3), np.uint8)
    with rasterio.open(ref_dir / "visual.tif") as src:
        for band in range(3):
            reproject(src.read(band+1), optical[..., band], src_transform=src.transform, src_crs=src.crs,
                      dst_transform=wt, dst_crs="EPSG:4326", resampling=Resampling.bilinear)
    clear = np.isin(warp(scl), [4, 5, 6])
    optical[~clear] = (40, 48, 59)
    optical[~coverage] = (0, 0, 0)
    errors = np.zeros((*o.valid.shape, 3), np.uint8)
    errors[evaluated & (source_class != 6) & (pred == 0)] = (91, 105, 116)
    errors[evaluated & (source_class == 6) & (pred == 1)] = (35, 153, 245)
    errors[evaluated & (source_class != 6) & (pred == 1)] = (235, 66, 194)
    errors[evaluated & (source_class == 6) & (pred == 0)] = (255, 160, 44)
    web_error = np.stack([warp(errors[..., i]) for i in range(3)], axis=-1)
    panels = [("Radar input", sar), ("Optical reference", Image.fromarray(optical)),
              ("Previous water model", results["previous_v1_composite"]),
              ("Improved water model", results["improved_v2_composite"]),
              ("Improved model errors", Image.fromarray(web_error))]
    canvas = Image.new("RGB", (1450, 1025), "#101923")
    draw = ImageDraw.Draw(canvas)
    font = lambda size: ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", size)
    draw.text((25, 15), "UNSEEN LOCATION | Lake Red Rock, Iowa", font=font(32), fill="white")
    draw.text((25, 60), "NISAR: 19 Oct 2026 11:21 UTC   |   Optical: same day, 5 h 51 min later", font=font(21), fill="#b6cadc")
    draw.text((25, 92), "72 km from the nearest development labels; model and decision policy frozen before test prediction", font=font(20), fill="#a8dfff")
    for i, (title, panel) in enumerate(panels):
        left = 25+i*285
        draw.text((left, 140), title, font=font(20), fill="white")
        panel = panel.crop(crop)
        backing = Image.new("RGBA", panel.size, (0, 0, 0, 255))
        panel = Image.alpha_composite(backing, panel.convert("RGBA")).convert("RGB")
        scale = min(260/panel.width, 670/panel.height)
        panel = panel.resize((round(panel.width*scale), round(panel.height*scale)), Image.Resampling.NEAREST)
        canvas.paste(panel, (left+(260-panel.width)//2, 178))
    old, new = [report["models"][key]["metrics"] for key in ("previous_v1", "improved_v2")]
    draw.text((25, 870), f"Precision: {old['precision']:.1%} -> {new['precision']:.1%}     Water recall: {old['recall']:.1%} -> {new['recall']:.1%}     IoU: {old['iou']:.1%} -> {new['iou']:.1%}", font=font(24), fill="white")
    draw.text((25, 909), "Errors: BLUE correct water   ORANGE missed water   MAGENTA false water   GRAY correct nonwater", font=font(20), fill="#c8d7e5")
    draw.text((25, 944), "Test support: 376.44 ha / 9,411 labelled cells. Unknown/cloud/boundary pixels are not scored.", font=font(20), fill="#b6cadc")
    draw.text((25, 978), "Entire radar-covered strip shown. Machine optical reference, one region/date; global and shoreline accuracy unvalidated.", font=font(18), fill="#ffd18a")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()


"""Verify actual saved outputs, static HTTP access, geometry and area reconciliation."""
import argparse
from pathlib import Path
import numpy as np
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from nisar_water.api import create_app
from nisar_water.common import read_json, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    records = []
    with TestClient(create_app(root)) as client:
        for item in client.get("/water/series").json():
            response = client.get(item["url"])
            assert response.status_code == 200
            series = response.json()
            assert series["isDemo"] is False
            for result in series["observations"]:
                assets = result["assets"]
                assert set(("sar", "water", "border", "composite", "water-mask", "water-score", "water-polygons")).issubset(assets)
                for asset in assets.values():
                    relative = asset["url"].removeprefix("/assets/")
                    path = root / relative
                    assert sha256(path) == asset["sha256"]
                    assert client.get(asset["url"]).status_code == 200
                def path_for(name):
                    return root / assets[name]["url"].removeprefix("/assets/")
                with rasterio.open(path_for("water-mask")) as mask, rasterio.open(path_for("water-score")) as score:
                    labels, scores = mask.read(1), score.read(1)
                    assert mask.crs == score.crs and mask.transform == score.transform and mask.shape == score.shape
                    assert set(np.unique(labels)).issubset({0, 1, 254, 255})
                    assert np.isfinite(scores[labels <= 1]).all()
                    assert not np.isfinite(scores[labels == 255]).any()
                    assert mask.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
                for name in ("water", "border", "composite"):
                    with Image.open(path_for(name)) as image:
                        assert image.mode == "RGBA"
                stats = result["statistics"]
                assert np.isclose(stats["waterHa"]+stats["nonwaterHa"]+stats["uncertainHa"], stats["observedHa"])
                assert np.isclose(stats["observedHa"]+stats["unobservedHa"], stats["requestedHa"])
                polygons = read_json(path_for("water-polygons"))
                assert polygons["type"] == "FeatureCollection"
                assert np.isclose(sum(f["properties"]["areaHa"] for f in polygons["features"]), stats["waterHa"], atol=1e-4)
                records.append({"observation": result["id"], "assets_verified": len(assets), "waterHa": stats["waterHa"]})
    write_json(args.report, {"state": "passed", "series_count": len(list(root.rglob("series.json"))),
                            "observations": records, "checks": ["COG layout", "native raster alignment", "finite classified scores",
                            "observed/unknown/invalid area partition", "vector/raster area equality", "RGBA transparency",
                            "all asset hashes", "all HTTP asset routes"], "scientific_accuracy_claim": False})
    print("Verified", len(records), "actual observations")


if __name__ == "__main__":
    main()


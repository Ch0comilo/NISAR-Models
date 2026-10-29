# Third-party software and data

This repository contains original pipeline code and three locally fitted
scikit-learn HGB checkpoints. They are not NASA, ASF, ESA, JRC or OPERA products
and are not endorsed by those organizations. No open-source license has been
chosen for this project's original code/checkpoints; see [LICENSE](LICENSE).

Dependencies retain their upstream licenses. `uv.lock` records the resolved
Python packages. The current HGB runtime uses NumPy, SciPy, scikit-learn,
rasterio/GDAL, PROJ/pyproj, Shapely, Pillow, xarray, h5py, joblib, FastAPI,
Uvicorn and their dependencies. Installing dependencies does not replace their
upstream license notices.

Training and evaluation used NISAR L-band GCOV scientific data distributed
through NASA Earthdata/ASF and dated Sentinel-2 optical weak references.
Physical features use Copernicus DEM GLO-30, HAND 2021 and, optionally, JRC Global
Surface Water occurrence 2021 v1.4. Remote data are fetched only when explicitly
requested; credentials and raw satellite/ancillary crops are not distributed.
Preserve source URLs, acquisition identities and derived-window SHA256 receipts
when redistributing generated maps. Consult the upstream data providers' terms
and attribution requirements for your intended use.

- [NISAR data guide](https://nisar-docs.asf.alaska.edu/)
- [Copernicus DEM collection](https://registry.opendata.aws/copernicus-dem/)
- [Global HAND collection](https://registry.opendata.aws/glo-30-hand/)
- [JRC Global Surface Water](https://global-surface-water.appspot.com/)
- [Sentinel-2 Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)

The historical OPERA comparator used DSWX-SAR commit
`d9ae8ed5178417dec22a6fc549c4368cfd51ebe9`. Its vendored source/runtime is excluded
from this repository bundle. See [the historical comparator documentation](docs/round3-opera.md)
and the [upstream source and license](https://github.com/opera-adt/DSWX-SAR).
Current HGB inference does not run that comparator or require its environment.

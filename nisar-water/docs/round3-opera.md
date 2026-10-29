# Pinned official OPERA open-water comparator

Source: [OPERA DSWX-SAR](https://github.com/opera-adt/DSWX-SAR), commit
`d9ae8ed5178417dec22a6fc549c4368cfd51ebe9`, in `vendor/DSWX-SAR`.
Source/configuration hashes are part of the frozen model metadata. Official
processing modules and NI scientific parameter YAML were not edited.

The six invoked stages are pre-processing, initial threshold, fuzzy value,
region growing, ancillary masking and bimodality refinement. Full-frame HDF5
mosaicking, MGRS tiling and DAAC product publication are replaced by bounded
crop I/O. Inundated vegetation is disabled for this open-water comparison.
CPU counts are one. The input is mean linear gamma0 HH/HV power reprojected from
20m to30m, with conservative native-validity aggregation. Source invalidity is
not misrepresented as an independently estimated geometry shadow mask.

This is an **official-algorithm crop adapter**, not an official OPERA product
or proven equivalent to a full-frame run. Small homogeneous all-water crops
may lack the threshold-estimation context available to full-frame processing.
The observed Constance/open-water failures must remain visible.

Actual ancillary crops use bounded authenticated-free HTTPS ranges from:

* Copernicus GLO30 DEM and the GLO30 HAND2021 public archive.
* ESA WorldCover2020 v100.
* JRC Global Surface Water2021 occurrence v1.4.
* GLAD GLCLU2020 classification v2, described on the
  [GLAD data page](https://glad.umd.edu/dataset/GLCLUC2020).

JRC, WorldCover and GLAD contain historical water information. OPERA therefore
has extra inference context that the radar-only learned models do not use.
This is a practical system comparison, not an architecture-only controlled
comparison. Historical priors may constrain new water; emerging-water behavior
was not validated here. Retain each source's upstream attribution/license
requirements; the GLAD dataset specifies CC BY4.0 and its Potapov2022 citation.

DEM/HAND1-degree tile seams and the larger landcover/water tile seams are
enumerated and cropped individually, then merged on their native grids. This
was exercised on the Tahoe crop crossing longitude-120°. No zero-filled fake
ancillary was substituted. Each job retains exact source URLs, local SHA256,
transfer-byte ledger, effective configuration, runtime log and final output hash.

The isolated `.venv-opera` conda prefix uses GDAL3.10.3 and numpy2.2.3. An initial
GDAL3.13 environment incorrectly exposed16-bit floating point source pixels
and caused threshold overflow; this failed development attempt was retained.
Pinning GDAL3.10.3 restored finite float32 processing before any validation or
test selection. A required real GLAD ancillary was also added during development.
Neither change tuned scientific parameters against test labels.

The prefix is about2.6GiB. It is separate from the existing Torch/CUDA runtime.
Windows conda activation can emit an OpenCL temporary-path warning because the
workspace contains spaces; completed execution receipts, output codes and
hashes were checked rather than assuming that warning meant a science failure.

`src/nisar_water/opera_adapter.py` owns bounded crop I/O and the worker.
`opera_model.py` supplies the common model protocol and verified prediction cache.
`scripts/run_opera_comparison.py worker REQUEST.json` runs inside the isolated
conda runtime. The model registry exposes this comparator through the same
prediction/export interface. Its score is a deterministic binary decision,
not a calibrated probability. Normal selected-model HGB inference needs neither
this runtime nor an ancillary download.

# Independent preprocessing audit before round3 training

This audit opens only the exposed `data/new-images-20261028` observations,
their already-exposed reference masks and predictions, the old phase1 catalog,
and the old `spatial-v2/development/train.npz` archive. It does not open new
round3 test geography, fit a classifier or change checkpoints. Numeric evidence
is saved in `evidence/round3-input-audit.json`.

## Result

**No demonstrated channel-order, double-logarithm, amplitude/power or 10-to-20 m
aggregation bug explains the large Mead/Okeechobee failures.** Independent
aggregation, in-file channel identities, derived COG tags and native-mask
comparisons passed for both water-positive training observations and all eight
exposed regression observations. The largest absolute difference between the
independent float64 power average cast to float32 and the production aggregate
was 5.96e-8; all other inspected cases were exactly equal on retained pixels.

There is a real context limitation: the online crop recipe preserves an 80 m
halo, while the largest feature needs a full 15-cell radius at 20 m, or 300 m.
The larger window correctly reports partial support rather than reflecting
invented pixels, but its boundary statistics can differ from interior examples.
New round3 acquisition should retain at least 320 m context. Preserve old
experiments instead of silently changing their inputs or identities.

This limitation cannot explain the principal misses:

| Exposed reference water | All-water recall | Recall >300 m inside cached crop | Water cells >300 m inside | Fraction near crop edge |
|---|---:|---:|---:|---:|
| Mead, Oct 6 | 11.33% | 11.36% | 47,050 | 0.65% |
| Texoma, Oct 21 | 98.57% | 98.56% | 36,574 | 0.66% |
| Okeechobee, Oct 15 | 0.00% | 0.00% | 86,736 | 10.33% |

The interior criterion refers to cached source-crop borders. It does not claim
that every interior pixel is free from native masked gaps or terrain effects.
The reference metrics are agreement with the old screened SCL interiors, not
independent field/shoreline truth.

## Source identities and missing context

Both water-positive training observations and seven exposed images use mode
`4005`, native 10 m pixels and ordered L-band frequency-A HHHH/HVHV diagonals.
Texoma Oct 21 uses mode `2005` and native 20 m pixels. Every inspected cached
source records product version `1.0.11`, specification `1.5.0` and release
`P05023`. Thus the scored Texoma success differs in bandwidth/posting, but the
failure/success pattern alone does not establish bandwidth as its cause.

The old crop metadata did not retain incidence geometry, noise-equivalent
backscatter LUTs or processing correction flags. Bounded authenticated reads
of the exposed source HDF5 files now retain descriptors/flags without reading
whole geometry/noise arrays. Both original positive training observations,
Mead and Okeechobee have the same inspected flags: noise correction false,
radiometric terrain correction true, shadow masking false, preprocessing
multilooking false and postprocessing filtering false. A differential change
in those flags therefore is not demonstrated as the failure explanation.

Verified paths to retain in new recipes:

- `/science/LSAR/GCOV/metadata/radarGrid/incidenceAngle` with its
  `xCoordinates`, `yCoordinates`, `heightAboveEllipsoid` and `projection`.
- `/science/LSAR/GCOV/metadata/calibrationInformation/frequencyA/noiseEquivalentBackscatter/HH`
  and `HV`, plus that group's `xCoordinates`, `yCoordinates` and `projection`.
- `/science/LSAR/GCOV/metadata/processingInformation/parameters/noiseCorrectionApplied`,
  `radiometricTerrainCorrectionApplied`, `shadowMaskingApplied`,
  `postProcessingFilteringApplied`, `preprocessingMultilookingApplied`,
  `validSamplesSubSwathMaskingApplied`, `polarimetricSymmetrizationApplied`,
  `polarizationOrientationCorrectionApplied` and `faradayRotationApplied`.
- Under that parameters group, `geocoding/isSourceDataUpsampled` and
  `preprocessing/frequencyA/{numberOfAzimuthLooks,numberOfRangeLooks}`.

Incidence angle is a three-dimensional metadata cube, not an already aligned
two-dimensional image. An actual angle feature requires interpolation at
terrain height and explicit grid support; this audit does not replace missing
height with zero. ASF describes the cube and its coordinate axes in its
[metadata guide](https://nisar-docs.asf.alaska.edu/metadata/).

The inspected source noise LUT descriptions state linear `DN^2`. Do not subtract
them from gamma0 or compute SNR until the relevant calibration/radiometry
convention has been verified. This audit performs no noise arithmetic. A false
noise-correction flag is evidence about source processing, not proof that
noise causes the observed classifier error.

## Measured distribution and supervision limitations

All 6,774 old water-positive training cells come from two Oroville-area
observations: 5,215 from June 23 ascending and 1,559 from October 10 descending.
Many adjacent pixels are correlated; their count is not geographic diversity.

| Reference-water feature median | Old training water | Mead | Texoma | Okeechobee |
|---|---:|---:|---:|---:|
| HH dB | -15.84 | -20.24 | -17.72 | -19.57 |
| HV dB | -21.57 | -23.73 | -21.81 | -25.67 |
| HH 31x31 mean-power dB | -12.77 | -19.88 | -17.36 | -19.44 |
| HV 31x31 mean-power dB | -19.45 | -23.66 | -21.57 | -25.58 |

42.7% of the Mead reference-water HH values lie outside the old training-water
1st–99th percentile interval. For Okeechobee, 62.9% of HV values lie outside
that interval; Texoma is much closer at 3.5% HH and 1.3% HV. These are
diagnostic, correlated distribution comparisons, not calibrated novelty scores.

The larger context around old training-water cells is substantially brighter
than the water pixels themselves. Large homogeneous water interiors have
different neighborhood statistics. This supports adding broad water interiors,
narrow rivers, shorelines and different surrounding land cover. It is not proof
that any one feature is causally responsible for the learned decisions.

## Callable acquisition helper

`scripts/audit_round3_inputs.py` provides:

```python
extra = inspect_extra_metadata(client, record, max_bytes=16 * 1024**2)
# Or reuse an existing open h5py File without another fetch:
extra = inspect_extra_metadata(h5=open_hdf5)
```

It returns typed processing flags, source scalar values, small looks/projection
metadata, and geometry/noise dataset shapes, dtypes and selected attributes.
Missing flags remain `None`. Incidence status explicitly records that the cube
has not been interpolated at terrain height. Noise values are not materialized,
normalized or subtracted. Source records and clients contain no serialized
credential. The legacy reader and shared model modules were not modified.

Reproduce the audit and focused tests:

```powershell
.\.venv\Scripts\python.exe scripts\audit_round3_inputs.py --remote-metadata
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests\test_round3_audit.py `
  --basetemp evidence\round3-audit-tests-NEW
```

Four focused software tests cover independent power aggregation, complete
native validity, honest large-window edge support and missing/cube metadata
semantics. The actual-source audit is separate evidence, not a substitute for
new-region model validation.


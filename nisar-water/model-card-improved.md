# Improved NISAR water experiment

**Research only: no candidate accepted for general use.**

Use the exact frozen directories to reproduce reported operating points:

* `models/round5/selected`: validation-selected diagnostic radar-only HGB0.20.
  This is a fallback diagnostic choice when all validation gates failed.
* `outputs/round5/frozen/terrain`: terrain feature comparison at0.10.
* `outputs/round5/frozen/terrain_prior`: conservative soft-prior comparison at0.10.
* `models/agreement-review`: separately frozen review aid; classify agreement,
  mark disagreement unknown254. Its score is binary agreement, not probability.

No variant was selected or tuned using the fresh test results. Current optical
SCL/spectral labels are weak reference; no expert polygons were imported into
these fits. All71 previously exposed scenes became round5 training; six new
validation scenes and five geographically reserved test scenes remained separate.
The three round5 fits share203,678 sampled cells and fixed parameters.

The source profile is NISAR L-GCOV frequency-A HHHH/HVHV gamma0 linear power,
prepared on20m local UTM grids. Features use up to31-cell context. Physical
variants use genuine nominal30m DEM/HAND and optional JRC2021 occurrence/distance.
They do not take contemporary optical labels, coordinates, RGB previews or
future observations as inference features. Historical water is not a coded
hard mask; learned reliance can still suppress genuinely current water.

On the same five fresh test images, the prior variant reaches99.95% precision,
75.49% water recall and0.000577% land FPR. Nam Theun1 recall is92.28%, but
Yusufeli recall is19.73%. The validation-selected radar-only variant has27.18%
precision/74.14% recall and6.08% airport land FPR. These correlated regional
weak-reference results are not worldwide or expert shoreline accuracy.

The review aid classified95.76% of all reference area but recovered only18.40ha
of442.40ha water, withheld315.04ha water and878.84ha land, and still missed
water where both models agreed incorrectly. Its conditional100% precision
does not establish complete detection or readiness. It fails the full gate.

Outputs remain compatible with the team's JSON/PNG/COG/GeoJSON contract. Native
mask codes are0 land,1 open water,254 unknown/excluded,255 unobserved/outside.
Scores are uncalibrated. `context-provenance` records actually used public source
hashes and feature profiles. Missing required terrain is explicit unknown.
Configure `--context-cache` or `NISAR_WATER_CONTEXT_CACHE` for bounded public
ancillary fetching on a new AOI; existing cached sources need no extra download.

Known limits include narrow/mountainous water, static terrain predating changed
water surfaces, roughened water, layover/shadow, inundated vegetation and weak
optical reference uncertainty. The cause of the Yusufeli errors is not established.
No human-impact, flood-cause, wildfire, farm or deforestation accuracy is claimed.

Authoritative reports: `outputs/round5/assessment.json` and
`docs/improved-classifier-results.md`. Open the executed
`notebooks/NISAR_Water_Improved_Classifier.ipynb`. Prefer the five fresh-region
reference packets in `review/earthengine-round5` for independent team review;
original round3 packets now represent exposed training regions.

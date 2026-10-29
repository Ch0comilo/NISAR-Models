# Contract for nisar-hazard-watch

The team's current observed imagery contract is in
`src/features/observations/observationApi.ts`. Water extent should be an additional
observation layer, with its own service URL, not a wildfire/deformation response.
No frontend files were changed. This is an integration-ready output contract;
the existing UI still needs to fetch and render it.

Portable models are under `checkpoints/round5/`. Generate a fresh result root
with the CLI/Python worker, then run the read-only backend:

```sh
uv run --frozen nisar-water serve --root outputs/my-river-01 --port 8010 --assessment benchmarks/round5/assessment.json
```

The current assessment is **not_ready_for_general_use**. Always display it
beside mapped areas. The historical `evaluationStatus` retained inside frozen
metadata must not override the current assessment. `/health` reports server
availability, not scientific acceptance. Without an explicit `--assessment`
or root assessment file, `/water/assessment` returns 404.

Original workspace results under `outputs/round5` remain available locally on
port 8014, including historical review-only series. They are excluded from the
portable repository. The three portable profiles use the same schema below.
See [API integration](api-integration.md) for the implemented endpoints and
proposed future inference jobs; there is no POST inference route today.

Suggested frontend environment setting: `VITE_WATER_API_URL=http://127.0.0.1:8010`.
This is a proposal for your team to implement; the current frontend does not
already consume that variable. Production hosting must serve these same JSON
and PNG/COG/GeoJSON files with the appropriate origin and CORS policy.

```ts
export type WaterAsset = {
  url: string; // Resolve against the WATER service origin.
  sha256: string;
  bytes: number;
};

export type WaterDecisionPolicy =
  | { kind: 'threshold'; threshold: number }
  | { kind: 'region_growing'; seed_threshold: number; grow_threshold: number; minimum_seed_pixels: number }
  | { kind: 'official_opera_ni_default_open_water' }
  | { kind: 'frozen_model_agreement'; disagreement_code: 254 };

export type WaterObservation = {
  schemaVersion: 'nisar-water-observation-1';
  kind: 'water_extent';
  id: string;
  datetime: string; // Exact UTC acquisition time.
  granuleId: string;
  isDemo: boolean;
  modelId: string;
  modelSha256: string;
  evaluationStatus: 'experimental_acceptance_failed' | 'local_weak_validation_passed'
    | 'experimental_broadened_weak_validation' | 'experimental_unet_weak_validation'
    | 'experimental_official_algorithm_crop_adapter' | 'experimental_round4_validation_pending'
    | 'experimental_model_agreement_review_only';
  threshold: number;
  thresholdOverride: boolean;
  decisionPolicy: WaterDecisionPolicy;
  scoreSemantics: string;
  crs: string; // Scientific GeoTIFF CRS, usually UTM.
  displayCrs: 'EPSG:4326';
  bounds: [number, number, number, number]; // W, S, E, N in degrees.
  shape: [number, number]; // Native analysis raster rows, columns.
  pixelSpacingM: [20, 20];
  statistics: {
    waterHa: number;
    nonwaterHa: number;
    uncertainHa: number;
    unobservedHa: number;
    observedHa: number;
    requestedHa: number;
  };
  assets: Record<string, WaterAsset>;
  contextInputSha256?: string;
  contextAblation?: string;
  requiredContextWithheldHa?: number;
  reviewMode?: boolean;
  disagreementWithheldHa?: number;
  limitations: string[];
};

export type WaterTransition = {
  beforeId: string;
  afterId: string;
  before: string;
  after: string;
  state: 'comparable' | 'no_common_support';
  commonClassifiedHa: number;
  gainHa: number | null;
  lossHa: number | null;
  netChangeHa: number | null;
  beforeWaterHa: number | null;
  afterWaterHa: number | null;
  changeUrl: string;
};

export type WaterSeries = {
  schemaVersion: 'nisar-water-series-1';
  kind: 'water_extent_time_series';
  id: string;
  isDemo: boolean;
  modelId: string;
  evaluationStatus: WaterObservation['evaluationStatus'];
  track: number;
  pass: 'A' | 'D';
  frame: number;
  frequency: 'A';
  observations: WaterObservation[];
  timeline: Array<WaterObservation['statistics'] & {
    id: string;
    datetime: string;
    commonAllDatesHa: number;
    waterOnCommonAllDatesHa: number | null;
  }>;
  transitions: WaterTransition[];
  animationUrl: string | null;
  occurrenceUrl: string;
  observationCountUrl: string;
};
```

`GET /water/series` returns `{ key, id, url }[]`.
Fetch a returned URL against the service origin to get `WaterSeries`.
404 means no computed series. Service errors never produce demonstration data.
Use acquisition IDs/times to match the existing observed timeline; do not match
by nearest calendar day or by the observation plan.

To render a selected exact observation in Cesium:

```ts
const origin = 'http://127.0.0.1:8010';
const [west, south, east, north] = observation.bounds;
const rectangle = Cesium.Rectangle.fromDegrees(west, south, east, north);
for (const key of ['water', 'border']) {
  const asset = observation.assets[key];
  if (!asset) continue;
  const provider = await Cesium.SingleTileImageryProvider.fromUrl(
    new URL(asset.url, origin).href,
    { rectangle },
  );
  viewer.imageryLayers.addImageryProvider(provider);
}
```

Assets: `sar`, `water`, `border`, `composite`, `uncertain`, `water-mask`,
`water-score`, `water-polygons`, `aoi-fraction`. `water-polygons` contains GeoJSON;
`water` contains an RGBA PNG. Water and nonwater scores are uncalibrated and must
not be displayed as independently verified percentage certainty.

Physical-context exports also include `context-provenance` JSON and optional
`contextInputSha256`, `contextAblation` and `requiredContextWithheldHa` fields.
Only actually used source hashes contribute to the context identity; cache
paths/hit flags and unused priors do not. Review-only series add `reviewMode`,
`disagreementWithheldHa` and a `review-composite` PNG with amber unknowns. An
unknown disagreement can be possible water gain, possible loss or model error;
it is not confirmed water or confirmed change. Agreement is not independent
reference verification. The primary schema and existing class codes remain
compatible. Never graph unknown area as zero water.

Display the experimental status and unknown area alongside mapped water area.
For graphs of evolution, use `waterOnCommonAllDatesHa` from the manifest's
`timeline` and `commonAllDatesHa` as its comparable support. Raw total water area
can vary solely because scene coverage varies. Transitions are water extent
estimates, not confirmed floods, wildfire damage or human-caused change.


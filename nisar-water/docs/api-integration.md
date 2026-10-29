# API integration with the team's application

The core is a Python inference worker plus a read-only saved-result FastAPI
service. The current schema already supplies the image, colored water, boundary,
vectors, native rasters, dates, comparable-support metrics and provenance needed
by the separate UI. Current model status remains `not_ready_for_general_use`.

## Implemented today

Run local scientific inference first, using the README or
`examples/predict_hdf5.py`. Then serve that output root:

```sh
uv run --frozen nisar-water serve --root outputs/my-river-01 --port 8010 --assessment benchmarks/round5/assessment.json --origin http://localhost:5173
```

Bind is localhost by default. Add `--host 0.0.0.0` when deliberately deploying
behind your hosting/reverse-proxy setup. `--origin` is repeatable; the development
localhost/127.0.0.1 port 5173 origins remain allowed by default. Authentication,
TLS, rate limits and a job queue belong to the hosting layer; they are not
provided by this local result server.

| Method/path | Behavior |
|---|---|
| `GET /health` | Service availability, not scientific acceptance |
| `GET /water/assessment` | Explicit `--assessment` JSON or root `assessment.json`; otherwise 404 |
| `GET /water/benchmark` | Root `report.json`; otherwise 404 |
| `GET /water/series` | List `{key, id, url}` entries discovered under the result root |
| `GET /water/series/{key}` | Saved manifest, with asset URLs rewritten for that series |
| `GET /assets/{path}` | Saved result assets |
| `GET /docs`, `GET /openapi.json` | Actual FastAPI interface |

The server exposes files under its result root. Mount only intended result
artifacts there; keep credentials and raw inputs elsewhere. `--assessment`
can point outside the served root and does not expose other files beside it.
The explicit assessment takes precedence over an older root assessment.

No POST inference or upload route exists. No endpoint automatically downloads
the NISAR archive. The `aoi` CLI now uses the standalone bounded Earthdata reader
by default. A future job worker can provide the user's/server's credential in
memory or `EARTHDATA_TOKEN`; the public notebook demonstrates this path without
requiring sibling projects. The older adapter requires `--reference-root`.

## Frontend consumption

```javascript
const base = 'http://127.0.0.1:8010';
const assessment = await fetch(`${base}/water/assessment`).then(r => {
  if (!r.ok) throw new Error('Model assessment unavailable');
  return r.json();
});
const entries = await fetch(`${base}/water/series`).then(r => r.json());
if (entries.length) {
  const series = await fetch(new URL(entries[0].url, base)).then(r => r.json());
  const observation = series.observations[0];
  const fillUrl = new URL(observation.assets.water.url, base).href;
  const borderUrl = new URL(observation.assets.border.url, base).href;
  // Place both overlays using observation.bounds: [west, south, east, north].
  // Display assessment.state, unknown area and acquisition datetime beside them.
}
```

Use exact acquisition IDs/timestamps to join with the UI timeline, rather than
nearest-day matching. Web PNGs are in EPSG:4326; native COGs retain UTM. Graph
`waterOnCommonAllDatesHa` and its `commonAllDatesHa` support. Missing/unknown
coverage is not zero water. Scores are uncalibrated. The assessment overrides
the historical pre-evaluation label retained inside frozen model metadata.
The bundled assessment's pooled metrics describe the diagnostic radar model;
it records that none of the three shipped candidates is accepted.

[frontend-contract.md](frontend-contract.md) contains the TypeScript/map-layer
contract. [openapi.json](openapi.json) is an exported snapshot of the actual
routes; response object details are described by that contract and real
`series.json` artifacts.

## Proposed next step: asynchronous inference jobs

These routes are a design proposal and **are not implemented**:

1. `POST /water/jobs` validates a bounded AOI and server-managed scientific
   input reference, selects an allowlisted trusted checkpoint and enqueues work.
   Return `202` with a job ID/status URL.
2. A worker invokes `from_hdf5`/`predict_dataset` and `run_series` in a new
   job-owned output directory. Use one worker per process or enforce isolation;
   acquire context explicitly and retain native validity, transfer receipts,
   model/input hashes and partial failures.
3. `GET /water/jobs/{id}` reports queued/running/completed/partial/failed state
   and links to existing result manifests. The frontend renders the same assets.

Prefer server-managed local/object-store data references over accepting arbitrary
filesystem paths or URL fetches from clients. Keep Earthdata credentials in the
worker's protected configuration. Large HDF5/window inference should not block
an HTTP request handler. Version checkpoint and schema IDs so future farm,
forest/fire specialists can share observation/result plumbing without being
presented as water predictions.

## Optional container

The Dockerfile targets Linux/Python 3.13 and installs the pinned base runtime
from `pyproject.toml`; use the lockfile workflow for exact development resolution.
It includes the compact checkpoints and assessment, runs as a non-root user,
and serves `/results`. A Docker runtime was not available for local validation;
the Python inference/API path was tested directly.

```sh
docker build -t nisar-water .
docker run --rm -p 8010:8010 --mount type=bind,source=/absolute/path/to/results,target=/results,readonly nisar-water
```

The host result directory must exist and contain completed exported series.
This container starts a result server, not an inference job service. Override
its command to invoke `nisar-water hdf5 ...` with separately mounted inputs,
writable outputs and writable context cache when integrating a worker.

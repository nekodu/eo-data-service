# eo-data-service

A small Earth observation data service built to learn the pieces a real one needs:
satellite data ingested into a cloud-native cube, an API behind OpenID Connect and an
API gateway, Kubernetes deployment with security defaults, monitoring, CI, and a
nowcasting model that has to beat a naive baseline.

Everything runs on a laptop with [kind](https://kind.sigs.k8s.io/).

## Status, honestly

| Part | State |
|---|---|
| Ingest from the EUMETSAT Data Store with EUMDAC (MSG Cloud Mask, `EO:EUM:DAT:MSG:CLM`) | Written. Token and product search work against the live Data Store; product downloads from my account time out (open with EUMETSAT support), so no real slots are in the cube yet |
| Zarr cube, API, auth, gateway, monitoring | Working on kind; end-to-end checks below pass |
| Nowcasting model | Trained and evaluated on **synthetic** cloud masks only. The API flags that (`"synthetic": true`) |

Numbers on synthetic data say the code works, not that the model is useful on real clouds.

## Architecture

```
                     ┌──────────────── kind cluster ─────────────────────────────┐
client ──HTTP──▶ Envoy Gateway ──▶ eods-api (FastAPI, 2 replicas) ──▶ Zarr cube (PVC)
   │              • JWT check (Keycloak JWKS)   • checks the token again          ▲
   │              • rate limit, timeout         • enforces scopes                 │
   │              • only /v1/* and the token    • /metrics for Prometheus   ingest CronJob
   │                endpoint are routed                                         (EUMDAC → GRIB →
   └──token──▶ Keycloak (client credentials)                                     cube, every 30 min)
                 eods-reader:      cube:read
                 eods-forecaster:  cube:read nowcast:run
   Prometheus + Grafana: scrape the API, alert on stale data, auth failures, latency
   NetworkPolicy: API reachable only from the gateway and Prometheus
```

- **Data**: each MSG cloud-mask slot (15 min, GRIB2 in a zip) is decoded with ecCodes,
  cropped to a Europe window and mapped to `0 clear / 1 cloudy / 255 no data`. The pixel
  window comes from the grid's own lat/lon (ecCodes' space-view geoiterator), so the cube
  keeps native geometry. Stored as Zarr v3 chunked `(time 8, y 256, x 256)`; appends skip
  slots already present and refuse out-of-order writes.
- **API**: `GET /v1/cube`, `GET /v1/cube/cloud-fraction`, `GET /v1/cube/window`,
  `POST /v1/nowcast`. Windows are size-capped (413) and validated (422).
- **Auth**: Keycloak realm with two machine clients and OAuth scopes. The gateway rejects
  unsigned or expired tokens; the API verifies signature, issuer, audience, expiry and
  scope itself, so a misrouted request can't skip authorisation. `alg=none`, foreign
  keys, wrong audience/issuer and missing scopes are covered by tests.
- **Kubernetes**: Pod Security `restricted`, non-root, read-only root filesystem, all
  capabilities dropped, no service-account tokens mounted, resource limits, probes,
  default-deny NetworkPolicy. Keycloak runs in dev mode (no TLS, H2): fine for a lab,
  not for production. Secrets are generated at deploy time and never committed.
- **Observability**: request rate/latency by route, auth failures by reason, cells
  served, cube freshness; JSON logs with request ids and client ids (never tokens).

## Nowcasting baseline

Question: will a pixel be cloudy 60 minutes (4 slots) from now? Three answers, scored on
exactly the same test pixels:

- **persistence**: the mask stays as it is. The reference any model has to beat.
- **gradient boosting** (`HistGradientBoostingClassifier`, served by the API): per-pixel
  features from the last 4 slots, its own history plus cloud cover and its trend in 5×5 and
  21×21 neighbourhoods.
- **CNN** (PyTorch, `eods compare`): a small fully convolutional net on the last 4 masks plus
  the valid-data mask. Four dilated 3×3 convolutions (dilation 1, 2, 4, 8) give a receptive
  field of about 31 pixels without pooling away resolution; trained on random 64×64 crops
  with a masked BCE loss so no-data pixels don't count. Saved as a `state_dict` only, so
  loading never unpickles arbitrary objects.

Train/test is split by time with a gap of one horizon, never by pixel, because neighbouring
pixels in one frame are near-copies.

Synthetic cube (96 slots, blobs drifting with a fixed wind), test block = last 25 %,
276,736 valid pixels:

| | accuracy | F1 (cloud) | CSI | Brier |
|---|---|---|---|---|
| persistence | 0.891 | 0.722 | 0.565 | 0.109 |
| gradient boosting | 0.935 | 0.828 | 0.707 | 0.050 |
| CNN | 0.961 | 0.899 | 0.817 | 0.032 |

The synthetic clouds move with a constant wind, which is the easy case and exactly what a
convolution can learn. Real clouds grow, decay and change direction, so all gaps to
persistence will shrink, and the CNN's lead may not survive. That is the first thing to
check once real slots are in.

## Run it

Needs Docker, kind, kubectl, helm, uv (Linux or WSL2).

```bash
uv sync --extra ingest --extra torch
uv run pytest -q                                   # 24 tests
uv run eods compare --cube ~/eods-data/cube.zarr   # persistence vs boosting vs CNN

mkdir -p ~/eods-data/models                        # synthetic data until real slots exist
uv run eods synth --out ~/eods-data/cube.zarr --steps 96
uv run eods train --cube ~/eods-data/cube.zarr --model ~/eods-data/models/nowcast.joblib

bash scripts/up.sh                                 # cluster, gateway, Keycloak, API, monitoring
bash scripts/smoke.sh                              # end-to-end checks through the gateway
```

Real data (needs a Data Store API key in `.env`, see `.env.example`):

```bash
uv run eods fetch --start 2026-11-17T00:00 --end 2026-11-18T00:00 --raw data/raw
uv run eods build --raw data/raw --out data/cube.zarr
```

## End-to-end checks (`scripts/smoke.sh`)

```
PASS reader token issued
PASS forecaster token issued
PASS no token -> gateway 401
PASS garbage token -> 401
PASS reader reads cube (200)
PASS reader cannot nowcast (403)
PASS forecaster nowcasts (200)
PASS window outside cube -> 422
PASS admin console not routed (404)
PASS metrics not routed (404)
PASS wrong client secret refused (401)
PASS direct pod-to-API blocked by NetworkPolicy
```

## Things that went wrong on the way

- `kind load` fails on Keycloak's multi-arch image; the node pulls it instead.
- A Gateway on kind never becomes `Programmed` with a LoadBalancer service (no address);
  an `EnvoyProxy` resource switches it to ClusterIP.
- After the WSL VM restarted, kindnet lost its NetworkPolicy watch and the policy
  silently stopped being enforced. The smoke test caught it; restarting kindnet fixed it.
  A policy that isn't tested isn't a control.
- `/v1/nowcast` returned 500 for a window outside the cube; now 422, with a test.

## Not done yet

- Real cloud-mask slots in the cube and a model trained on them.
- TLS everywhere, Keycloak on a real database, per-client rate limits (needs the
  gateway's global rate limiting with Redis).
- Serving the CNN through the API (it only runs offline in `eods compare` so far).

## Licence

MIT. EUMETSAT data is subject to EUMETSAT's data policy (MSG Cloud Mask: free and
unrestricted).

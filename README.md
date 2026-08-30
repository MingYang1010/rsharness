# EO Harness Deployment

The stack contains TerriaMap, the stateful EO Harness Environment API, and an internal deterministic renderer. All three run in the `yangm` user's Rootless Docker daemon. Images, container snapshots, build cache, and daemon metadata are stored under `/sata`; the shared system daemon and `/var/lib/docker` are not used by this project.

The Rootless Docker systemd service has a user-only proxy drop-in at `~/.config/systemd/user/docker.service.d/proxy.conf`. Proxy values are not stored in this project.

## Paths

- Project configuration: `/sata/yangm/eo-harness`
- Rootless Docker data root: `/sata/yangm/docker-rootless/data`
- Docker context: `rootless`
- Docker socket: `/run/user/1017/docker.sock`
- TerriaMap port while running: `3001`
- Harness API port: `127.0.0.1:8000`
- Renderer port: internal Compose network only, `8090`
- Episode state: `/sata/yangm/eo-harness/state/episodes.sqlite3`
- V2 artifact store: `/sata/yangm/eo-harness/artifacts`
- V2 immutable task packs: `/sata/yangm/eo-harness/tasks`
- Local WorldCover directory: `/sata/yangm/eo-harness/datasets/worldcover-2021`

## Local Dataset

The active EO layer is the local RGB COG `ESA_WorldCover_10m_2021_v200_N30E120_Map_RGB.tif`, covering `120-123 E` and `30-33 N`. Compose mounts the dataset directory read-only at `/app/wwwroot/data/worldcover-2021`.

TerriaMap serves the COG with HTTP Range support; the browser does not contact Terrascope or ESA S3 for WorldCover. The default basemap is the Natural Earth texture bundled with Cesium, so the deterministic renderer does not depend on OpenStreetMap.

## Interface Language

The interface opens in English for a fresh browser session so the renderer profile has a fixed language. The globe button in the upper-right menu switches between `简体中文` and `English`; the selected language is stored in browser local storage and survives reloads. A URL can explicitly select a language with `?lng=zh_Hans` or `?lng=en`.

This uses TerriaJS 8.12.2's bundled translations and built-in language panel. Project-scoped Chinese and English overrides fill the upstream keys used by upload, workbench, footer, and drag/drop controls. Both override directories are mounted read-only; no custom frontend image is required.

## Responsive Controls

The project mounts a version-pinned stylesheet and a small TerriaJS 8.12.2 bridge from `/sata/yangm/eo-harness/ui`. Desktop and mobile zoom controls have `46 x 44` CSS-pixel hit targets. The bridge exposes the native zoom control on small screens and requests render frames while Terria's native Cesium zoom tween runs.

At `390 x 844`, the coordinate readout and secondary footer links are hidden so data attribution and the scale remain on one line. At `1280 x 720`, the language code is visually replaced by the globe icon while retaining the tooltip and accessible name `切换语言 / Change language`.

## Harness API V1

Environment API `0.2.0` owns episode state, action validation, budgets, observations, and append-only traces. V1 supports `set_view`, `pan`, `zoom`, `set_layer_visibility`, `set_layer_opacity`, and `submit_answer` through `reset`, `step`, `state`, and `trace` endpoints.

All success responses use a typed `meta + data` envelope; validation, domain, routing, and internal errors use `meta + error`. SQLite state is stored on SATA and survives API container recreation. `client_action_id` makes response `data` retry-stable, while budget exhaustion truncates an episode and `submit_answer` terminates it. Equivalent episodes expose the same `semantic_trace_hash` even when IDs and timestamps differ.

OpenAPI documentation is available at `http://127.0.0.1:8000/docs` on the server. The committed contract is [contracts/openapi-v1.json](contracts/openapi-v1.json), with golden examples under [contracts/fixtures](contracts/fixtures). See [docs/harness-api-v1.md](docs/harness-api-v1.md) for usage and [docs/api-compatibility-policy.md](docs/api-compatibility-policy.md) for the V1 change boundary.

## Harness API V2 M2

Environment API implementation `0.4.0` exposes independent V2 body schema `2.0.0` without changing the frozen V1 body contract or migrating V1 episodes. SQLite schema `2` adds observation, artifact, and evaluation associations through additive `v2_*` tables in the existing SATA database.

The immutable M1 task `worldcover-grounded-vqa@1.0.0` remains structural. M2 adds `worldcover-grounded-vqa@1.1.0`: map actions are projected into TerriaMap, read back, checked for stable nonblank output, and captured as content-addressed PNG artifacts. The WorldCover evaluator computes `task.accuracy`, `evidence.faithfulness`, and `process.efficiency` from a fixed AOI and a checksum-pinned canonical class raster. `/v2/capabilities` reports the renderer and evaluator as available; executable raster tools remain a later milestone.

The V2 OpenAPI document is served at `http://127.0.0.1:8000/v2/openapi.json` and committed at [contracts/v2/openapi-v2.json](contracts/v2/openapi-v2.json). Golden V2 requests and responses are under [contracts/v2/fixtures](contracts/v2/fixtures). See [docs/harness-api-v2.md](docs/harness-api-v2.md) for the endpoint and retry contract.

Set `EO_HARNESS_V2_ENABLED=0` on the API container to disable the V2 runtime. V2 routes remain registered and return typed HTTP `503 v2_disabled`; this does not remove V2 tables or artifacts and leaves V1 available. The default Compose configuration enables V2 and mounts `tasks/`, `config/v2/`, and datasets read-only while mounting the artifact store read-write.

## Start

```bash
/sata/yangm/eo-harness/scripts/start-harness.sh
```

The script verifies the Rootless Docker data root, builds missing API and renderer images, starts all three services, and waits for TerriaMap, renderer, and API health responses. `start-terriamap.sh` remains as a compatibility wrapper and starts the complete stack. After changing source, rebuild the affected service explicitly:

```bash
docker --context rootless compose \
  -f /sata/yangm/eo-harness/compose.yaml build harness-api
docker --context rootless compose \
  -f /sata/yangm/eo-harness/compose.yaml build renderer
```

## Test

Run the complete V1 and V2 regression suite from the source root:

```bash
PYTHON=python3 /sata/yangm/eo-harness/scripts/test-harness.sh
```

The current Python gate contains 15 frozen V1 tests and 29 V2 tests. It checks contract snapshots, typed errors, idempotency, optimistic concurrency, cursor pagination, semantic hashes, additive migration rollback, evidence selectors, rendered observations, artifact integrity and Range reads, WorldCover evaluation, restart persistence, and structural replay. The renderer has four Node tests for map-state validation, capture-quality rejection, PNG hashing, and offline request routing.

## Inspect

```bash
docker --context rootless compose \
  -f /sata/yangm/eo-harness/compose.yaml ps

docker --context rootless info --format \
  'root={{.DockerRootDir}} security={{json .SecurityOptions}}'

curl --noproxy '*' -fsS http://127.0.0.1:3001/
curl --noproxy '*' -fsS http://127.0.0.1:8000/healthz
```

## Stop

```bash
/sata/yangm/eo-harness/scripts/stop-all.sh
```

`stop-all.sh` removes only this Compose project's TerriaMap, renderer, API containers, and networks. It deliberately leaves the shared Rootless Docker service running because other user projects may use it. Episode state and artifacts under `/sata/yangm/eo-harness` are retained.

## Persistence

The service is enabled, but the host currently reports `Linger=no`. Run once with administrator permission so Rootless Docker can start at boot and remain available without an SSH session:

```bash
sudo loginctl enable-linger yangm
loginctl show-user yangm -p Linger
```

The expected result is `Linger=yes`.

## Legacy Data

The superseded bootstrap-daemon data at `/sata/yangm/docker/eo-harness` was deleted on 2026-07-14 after verifying that neither the rootless nor system daemon used it. The current scripts use only `/sata/yangm/docker-rootless/data`.

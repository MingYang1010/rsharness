# EO Harness Deployment

The running stack contains TerriaMap and the stateful EO Harness Environment API. Both run in the `yangm` user's Rootless Docker daemon. Images, container snapshots, build cache, and daemon metadata are stored under `/sata`; the shared system daemon and `/var/lib/docker` are not used by this project.

The Rootless Docker systemd service has a user-only proxy drop-in at `~/.config/systemd/user/docker.service.d/proxy.conf`. Proxy values are not stored in this project.

## Paths

- Project configuration: `/sata/yangm/eo-harness`
- Rootless Docker data root: `/sata/yangm/docker-rootless/data`
- Docker context: `rootless`
- Docker socket: `/run/user/1017/docker.sock`
- TerriaMap port while running: `3001`
- Harness API port: `127.0.0.1:8000`
- Episode state: `/sata/yangm/eo-harness/state/episodes.sqlite3`
- V2 artifact store: `/sata/yangm/eo-harness/artifacts`
- V2 immutable task packs: `/sata/yangm/eo-harness/tasks`
- Local WorldCover directory: `/sata/yangm/eo-harness/datasets/worldcover-2021`

## Local Dataset

The active EO layer is the local RGB COG `ESA_WorldCover_10m_2021_v200_N30E120_Map_RGB.tif`, covering `120-123 E` and `30-33 N`. Compose mounts the dataset directory read-only at `/app/wwwroot/data/worldcover-2021`.

TerriaMap serves the COG with HTTP Range support; the browser does not contact Terrascope or ESA S3 for WorldCover. The OpenStreetMap basemap is still external.

## Interface Language

The interface opens in Simplified Chinese for a fresh browser session. The globe button in the upper-right menu switches between `简体中文` and `English`; the selected language is stored in browser local storage and survives reloads. A URL can explicitly select a language with `?lng=zh_Hans` or `?lng=en`.

This uses TerriaJS 8.12.2's bundled translations and built-in language panel. Project-scoped Chinese and English overrides fill the upstream keys used by upload, workbench, footer, and drag/drop controls. Both override directories are mounted read-only; no custom frontend image is required.

## Responsive Controls

The project mounts a version-pinned stylesheet and a small TerriaJS 8.12.2 bridge from `/sata/yangm/eo-harness/ui`. Desktop and mobile zoom controls have `46 x 44` CSS-pixel hit targets. The bridge exposes the native zoom control on small screens and requests render frames while Terria's native Cesium zoom tween runs.

At `390 x 844`, the coordinate readout and secondary footer links are hidden so data attribution and the scale remain on one line. At `1280 x 720`, the language code is visually replaced by the globe icon while retaining the tooltip and accessible name `切换语言 / Change language`.

## Harness API V1

Environment API `0.2.0` owns episode state, action validation, budgets, observations, and append-only traces. V1 supports `set_view`, `pan`, `zoom`, `set_layer_visibility`, `set_layer_opacity`, and `submit_answer` through `reset`, `step`, `state`, and `trace` endpoints.

All success responses use a typed `meta + data` envelope; validation, domain, routing, and internal errors use `meta + error`. SQLite state is stored on SATA and survives API container recreation. `client_action_id` makes response `data` retry-stable, while budget exhaustion truncates an episode and `submit_answer` terminates it. Equivalent episodes expose the same `semantic_trace_hash` even when IDs and timestamps differ.

OpenAPI documentation is available at `http://127.0.0.1:8000/docs` on the server. The committed contract is [contracts/openapi-v1.json](contracts/openapi-v1.json), with golden examples under [contracts/fixtures](contracts/fixtures). See [docs/harness-api-v1.md](docs/harness-api-v1.md) for usage and [docs/api-compatibility-policy.md](docs/api-compatibility-policy.md) for the V1 change boundary.

## Harness API V2 M1

Environment API implementation `0.3.0` adds an independent V2 schema without changing the frozen V1 body contract or migrating V1 episodes. V2 schema version `2.0.0` provides immutable task lookup, reset, state, optimistic and idempotent step execution, cursor-paginated event traces, and structural replay. Metadata is stored in additive `v2_*` SQLite tables in the existing SATA database.

The registered M1 task is `worldcover-grounded-vqa@1.0.0`. It pins the local WorldCover asset by SHA-256 and currently emits structural map-state observations. TerriaMap projection, deterministic rendered observations, tool execution, artifact HTTP endpoints, and evaluator execution remain gated to M2 or M3 and are reported as unavailable by `/v2/capabilities`.

The V2 OpenAPI document is served at `http://127.0.0.1:8000/v2/openapi.json` and committed at [contracts/v2/openapi-v2.json](contracts/v2/openapi-v2.json). Golden V2 requests and responses are under [contracts/v2/fixtures](contracts/v2/fixtures). See [docs/harness-api-v2.md](docs/harness-api-v2.md) for the endpoint and retry contract.

Set `EO_HARNESS_V2_ENABLED=0` on the API container to disable V2 route registration. This does not remove V2 tables or artifacts and leaves V1 available. The default Compose configuration enables V2 and mounts `tasks/` and `config/v2/` read-only.

## Start

```bash
/sata/yangm/eo-harness/scripts/start-harness.sh
```

The script verifies the Rootless Docker data root, builds the pinned API image when it is absent, starts both services, and waits for TerriaMap and API health responses. `start-terriamap.sh` remains as a compatibility wrapper and now starts the complete stack. After changing API source, rebuild explicitly with `docker --context rootless compose -f /sata/yangm/eo-harness/compose.yaml build harness-api`.

## Test

Run the complete V1 and V2 regression suite from the source root:

```bash
PYTHON=python3 /sata/yangm/eo-harness/scripts/test-harness.sh
```

The current gate contains 15 frozen V1 tests and 23 V2 tests. It checks contract snapshots, typed errors, idempotency, optimistic concurrency, cursor pagination, semantic hashes, additive migration rollback, evidence selectors, artifact integrity, and structural replay.

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

`stop-all.sh` removes only this Compose project's TerriaMap/API containers and network. It deliberately leaves the shared Rootless Docker service running because other user projects may use it. Episode state under `/sata/yangm/eo-harness/state` is retained.

## Persistence

The service is enabled, but the host currently reports `Linger=no`. Run once with administrator permission so Rootless Docker can start at boot and remain available without an SSH session:

```bash
sudo loginctl enable-linger yangm
loginctl show-user yangm -p Linger
```

The expected result is `Linger=yes`.

## Legacy Data

The superseded bootstrap-daemon data at `/sata/yangm/docker/eo-harness` was deleted on 2026-07-14 after verifying that neither the rootless nor system daemon used it. The current scripts use only `/sata/yangm/docker-rootless/data`.

# EO Harness A800 Deployment Report

Status: TerriaMap and frozen Environment API V1 contract verified and running on 2026-07-15.

## Outcome

TerriaMap `v0.4.6` and Environment API `v0.2.0` are configured on A800 host `lamda12` and start reproducibly from `/sata/yangm/eo-harness`. The deployment uses the `yangm` user's Rootless Docker daemon directly; the temporary privileged bootstrap daemon and private TCP Docker API are no longer part of the runtime path.

## Environment API V1

The API is the source of truth for episode state. It exposes `reset`, `step`, `state`, `trace`, `action-space`, health, and OpenAPI endpoints. Its deterministic V1 action space contains six actions: `set_view`, `pan`, `zoom`, `set_layer_visibility`, `set_layer_opacity`, and `submit_answer`.

Episode state and append-only transitions are stored in `/sata/yangm/eo-harness/state/episodes.sqlite3`. A completed reference episode recorded all six actions with sequences `1-6`, terminated through `submit_answer`, rejected a later step with HTTP `409`, and retained the same six-transition trace hash after the API container was recreated. A separate one-step episode exhausted its budget, returned `truncated=true`, and rejected both a later step and a conflicting idempotency key with HTTP `409`.

Release `0.2.0` freezes `/v1` body schema `1.0.0`. Success responses use `meta + data`; domain, validation, routing, and internal errors use `meta + error`. Every route declares a concrete response model. A projection layer converts persisted `0.1.0` dictionaries into public DTOs without changing the SQLite schema. `state_hash` and `trace_hash` remain instance-specific, while `semantic_state_hash` and `semantic_trace_hash` allow equivalent replays with different IDs and timestamps to compare equal.

The committed machine contract is `contracts/openapi-v1.json`, with eight golden JSON fixtures. Fifteen tests cover domain behavior, SQLite behavior, real HTTP response bodies, `201/404/409/422/500`, request-ID echo, idempotent response data, old persisted response projection, semantic hash equivalence, UTF-8 request size, and exact OpenAPI snapshot equality.

## Single-Source Inspection Demo

The current running demo exposes one EO source, `ESA WorldCover 2021 (10 m, Local N30E120)`. The official palette COG and a browser-compatible RGB COG are stored under `/sata/yangm/eo-harness/datasets/worldcover-2021` and mounted read-only into TerriaMap. The item is added to the workbench at startup and opens over Shanghai and surrounding areas.

Verification on 2026-07-14 confirmed HTTP 200, exactly one workbench dataset, correct WorldCover classification colors, and a fully rendered overlay. The browser made four same-origin RGB COG Range requests, all returning HTTP 206 `image/tiff`; it made zero Terrascope WMS and zero ESA S3 requests. Zooming changed the scale from `30 km` to `10 km`, and dragging changed the map center. The `eo-harness-terriamap` container is intentionally left running on port `3001` for user inspection.

## Interface Language

TerriaJS's bundled `zh_Hans` translation and built-in globe menu are enabled without rebuilding the image. A fresh browser session opens in Simplified Chinese, and the menu switches between `简体中文` and `English`. URL selection and the browser's saved language take precedence over the Chinese fallback.

Headless Chrome verified both switch directions and reload persistence. Read-only project overrides fill the upstream upload, workbench, footer, and drag/drop keys in both languages, avoiding raw keys in Chinese and Chinese fallback text in English.

## Responsive Map Controls

The deployment mounts a local stylesheet and a TerriaJS 8.12.2 bridge without rebuilding the image. Desktop and mobile zoom buttons measure `46 x 44` CSS pixels. The bridge exposes Terria's native ZoomControl on small screens and supplies the repaint frames omitted by the upstream idle Cesium zoom path.

At `1280 x 720`, the workbench, toolbar, and map controls had no detected overlap or horizontal overflow. At `390 x 844`, the mobile header, language panel, zoom controls, data-attribution footer, and scale remained inside the viewport. Story, About, AR, pedestrian, and the mobile coordinate readout are hidden because they are not needed by this harness.

## Pinned Artifact

- Image: `ghcr.io/terriajs/terriamap:0.4.6`
- Linux/amd64 digest: `sha256:0853ec153c53cef6ae926c99698ccc5cf4cc1a89906550068d3a36d95c3d54e0`
- Rootless local tag: `eo-harness/terriamap:0.4.6`
- License: Apache-2.0
- Dataset preparation image: `osgeo/gdal:alpine-small-3.6.3`
- GDAL image digest: `sha256:4b58dab500d110a850deec1c88ee9a8da1142dafe075a96099ce8251b89814bf`

## Local Dataset

- Official source COG: `ESA_WorldCover_10m_2021_v200_N30E120_Map.tif`
- Source size: `93,443,932` bytes
- Source SHA256: `f3859c80b7bd8d61a82f0adafee02378d418a2f1bafe32d8229b84301329c814`
- RGB runtime COG: `ESA_WorldCover_10m_2021_v200_N30E120_Map_RGB.tif`
- Runtime size: `152,280,029` bytes
- Runtime SHA256: `9f376abaca38815c5c743126147aeffd1916bb1907ad98929d341d4e6c87381c`
- Coverage: `120-123 E`, `30-33 N`
- Mount: `/sata/yangm/eo-harness/datasets/worldcover-2021` to `/app/wwwroot/data/worldcover-2021`, read-only

## Rootless Runtime

- Engine: Docker `29.1.3`
- Context: `rootless`
- Socket: `/run/user/1017/docker.sock`
- Data root: `/sata/yangm/docker-rootless/data`
- Storage driver: `overlayfs` with the containerd snapshotter
- Security options: `rootless`, `seccomp`, and `cgroupns`
- Network: RootlessKit with `slirp4netns`
- Proxy: user systemd drop-in, mode `600`
- Current rootless image usage after TerriaMap, the API, Python, GDAL, and `hello-world`: `1.726GB`; all image and build-cache storage remains under `/sata`

The shared system daemon still uses `/var/lib/docker`, contains no TerriaMap image, and retained its nine unrelated containers unchanged.

## Verification

The rootless image pull resolved to the pinned digest. HTTP checks returned:

| Endpoint | Status | Bytes |
| --- | ---: | ---: |
| `/` | 200 | 1229 |
| `/config.json` | 200 | 756 |
| `/init/eo-harness.json` | 200 | 1678 |
| `/eo-harness.css` | 200 | 3044 |
| `/eo-harness-ui.js` | 200 | 2569 |
| API `/healthz` | 200 | dynamic JSON |
| API `/openapi.json` | 200 | dynamic JSON |
| RGB COG Range (`0-16383`) | 206 | 16384 |

Browser verification through a temporary SSH tunnel confirmed:

- document title `EO Harness Map`;
- a fresh browser session opened in Simplified Chinese;
- the language menu contained `简体中文` and `English`;
- English and Chinese selections both persisted after reload;
- nonblank Cesium map canvas at desktop and mobile breakpoints;
- three `46 x 44` zoom targets at both `1280 x 720` and `390 x 844`;
- no visible overflow or untranslated project keys at either breakpoint;
- correct WorldCover classification colors from the local RGB COG;
- same-origin local COG resource entries and HTTP 206 Range delivery;
- zero Terrascope WMS and ESA S3 requests;
- camera-height checks passed for zoom-in, reset, zoom-out, and final reset.
- API action-space count was `6`, and all 15 domain/store/HTTP contract tests passed locally and inside the final read-only API image;
- API container recreation preserved episode `ep-9e5676cb1de740238bbb803a715eed63` with six transitions and trace hash `ef8940c3dc3a3e0bbb1e2a8b00d0ee04aee46c2c387afe061da4d3d870b3b791`;
- deployment smoke episode `ep-6fcceef7c3fc40b294d5dcf07d74fccf` terminated after two transitions, returned identical idempotent `data`, and produced semantic trace hash `e739b4a6234b55a2080800d643b44c94da997e2034da0a93115c24c4e1ae37af`;
- the live `/openapi.json` object exactly matched the committed snapshot, and a reset request exceeding 64 KiB after UTF-8 encoding returned the common HTTP `422` envelope;
- the API is healthy, uses a read-only root filesystem, drops all capabilities, enables `no-new-privileges`, and publishes port `8000` only on `127.0.0.1`.

## Commands

```bash
/sata/yangm/eo-harness/scripts/start-harness.sh

docker --context rootless compose \
  -f /sata/yangm/eo-harness/compose.yaml ps

curl --noproxy '*' -fsS http://127.0.0.1:3001/

/sata/yangm/eo-harness/scripts/stop-all.sh
```

## Current State

- `eo-harness-terriamap` is running on port `3001`.
- `eo-harness-api` is healthy on `127.0.0.1:8000`.
- The API image is `eo-harness/harness-api:0.2.0`, linux/amd64 image ID `sha256:18cb4bd23a3cc1587e02bb349b4e281e2df884c11445e5530dfd9e4b4535423d`, `48,608,004` bytes.
- The API bind mount is the only writable container path; it maps `/sata/yangm/eo-harness/state` to `/app/state`.
- The TerriaMap container was not recreated while the API was added.

## Remaining Actions and Limits

- `loginctl show-user yangm -p Linger` still reports `Linger=no`. Run `sudo loginctl enable-linger yangm` for boot and logout persistence.
- The unused legacy daemon data at `/sata/yangm/docker/eo-harness` was explicitly approved for cleanup and deleted. Post-cleanup verification confirmed the path is absent, Rootless Docker remains healthy, and the nine unrelated system-daemon containers are unchanged.
- The WorldCover layer is local. The OpenStreetMap basemap remains external and depends on network availability.
- The UI bridge depends on TerriaJS 8.12.2 class names and React container internals. Revalidate the zoom and language controls before upgrading TerriaMap/TerriaJS.
- API state is not yet projected into the open TerriaMap browser, and V1 observations contain typed map state rather than a rendered viewport image.
- Retrieval, perception, measurement, evaluator, artifact provenance, and geospatial memory actions remain to be implemented.

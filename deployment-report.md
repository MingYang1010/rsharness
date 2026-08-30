# EO Harness A800 Deployment Report

Status: V1 remains frozen; V2 M2 was accepted on A800 on 2026-08-30. All EO Harness containers are intentionally stopped and removed at delivery.

## Outcome

TerriaMap `v0.4.6`, Environment API `v0.4.0`, and deterministic renderer `v1.0.0` are configured on A800 host `lamda12` and start reproducibly from `/sata/yangm/eo-harness`. The deployment uses the `yangm` user's Rootless Docker daemon directly; the temporary privileged bootstrap daemon and private TCP Docker API are not part of the runtime path.

## V2 M2 Acceptance

M2 adds immutable task `worldcover-grounded-vqa@1.1.0` while leaving structural task `1.0.0` unchanged. The API projects fixed map state into an internal TerriaMap renderer, checks read-back and stable nonblank output, stores content-addressed PNG artifacts, freezes source and rendered evidence, and persists a three-metric WorldCover evaluation. The renderer is attached only to the internal Compose network and has no published host port.

The accepted live episode was `ep2-868bee31fc5249818999626bc196cd49`:

- fixed AOI: `121.45-121.55 E`, `31.20-31.30 N`;
- four successful actions, two frozen evidence references, 19 append-only events, zero failed actions;
- rendered observation `obs-331b3ebab7444379a9082f254a434cda`;
- PNG artifact `art-f2e4fdfa2c18e787167305437eee88c72b4c656b559d412ffa12c2f61594c538`, `175,307` bytes;
- artifact SHA-256 `f2e4fdfa2c18e787167305437eee88c72b4c656b559d412ffa12c2f61594c538`;
- full content returned HTTP `200`; `bytes=0-31` returned HTTP `206`, 32 bytes, and the correct PNG signature;
- evaluation `eval-c174db7d049644dc91c4b5caa7d76276`, status `completed`, aggregate reward `0.9840972`;
- metrics: accuracy `1.0`, evidence faithfulness `1.0`, process efficiency `0.840972`;
- canonical class distribution: class 50 built-up `1,128,426 / 1,440,000` pixels, or `78.36%`;
- structural replay passed all 19 events.

The deterministic profile captured two consecutive identical PNGs. A second independent renderer session for the same semantic map state also produced the same artifact SHA-256. Capture QA recorded `1024 x 768`, opaque fraction `1.0`, gray fraction `0.019259`, 4,096 sampled colors, consistent camera/layers, and stable frame state.

Recorded renderer provenance is TerriaMap `0.4.6`, TerriaJS `8.12.2`, Cesium `23.0.2`, Chromium `140.0.7339.186`, Playwright `1.55.1`, bridge `2.0.0`, and renderer `1.0.0`.

Before recreation, the API container was `67c1de5a...be155` and renderer was `740687f0...f9c1`. After forced recreation, they were `c598b6da...a0a` and `d84a97f8...df30`; TerriaMap remained `7594ec3d...7e67`. The episode, artifact bytes, evaluation ID, and all hashes remained exact:

- state hash `44947cc1e7503c38379764a681f685271a81d247d70bfa0004b9dfd66627cc64`;
- semantic state hash `3726b0f344288412fd664511006f5b50fffc3e5b98a7449465757066f810cbef`;
- trace hash `2d70793911da06b4c05422aabd3fc491a37c077519e7f1c1bb63a06c36a86f51`;
- semantic trace hash `ad3347e4fb514d6edf161da9ef78ea16e883794a9297aaab14e709ce7eb2718c`.

The schema `1 -> 2` migration preserved three V1 episodes and two pre-existing V2 episodes. The pre-M2 database backup is `/sata/yangm/eo-harness/state/backups/episodes.sqlite3.pre-m2-20260830T130935Z`, SHA-256 `09bce5caf4d86ff4d70a919e035a97bd0119a58ee715b5b4b6e7cdb43bc46279`.

Final verification passed 44/44 Python tests locally and in the A800 API image, 4/4 renderer tests locally and in the A800 renderer image, npm audit with zero vulnerabilities, live V1 and V2 OpenAPI object equality, and typed `503 v2_disabled` behavior while V1 remained healthy. `contracts/openapi-v1.json` retains SHA-256 `6b694787231cf815873b1902e3e6322168adb63f2b872fc28f5d73e0120c166f`; all eight V1 fixtures remain byte-identical to baseline parent `835e765`.

## Environment API V1

The API is the source of truth for episode state. It exposes `reset`, `step`, `state`, `trace`, `action-space`, health, and OpenAPI endpoints. Its deterministic V1 action space contains six actions: `set_view`, `pan`, `zoom`, `set_layer_visibility`, `set_layer_opacity`, and `submit_answer`.

Episode state and append-only transitions are stored in `/sata/yangm/eo-harness/state/episodes.sqlite3`. A completed reference episode recorded all six actions with sequences `1-6`, terminated through `submit_answer`, rejected a later step with HTTP `409`, and retained the same six-transition trace hash after the API container was recreated. A separate one-step episode exhausted its budget, returned `truncated=true`, and rejected both a later step and a conflicting idempotency key with HTTP `409`.

Release `0.2.0` freezes `/v1` body schema `1.0.0`. Success responses use `meta + data`; domain, validation, routing, and internal errors use `meta + error`. Every route declares a concrete response model. A projection layer converts persisted `0.1.0` dictionaries into public DTOs without changing the SQLite schema. `state_hash` and `trace_hash` remain instance-specific, while `semantic_state_hash` and `semantic_trace_hash` allow equivalent replays with different IDs and timestamps to compare equal.

The committed machine contract is `contracts/openapi-v1.json`, with eight golden JSON fixtures. Fifteen tests cover domain behavior, SQLite behavior, real HTTP response bodies, `201/404/409/422/500`, request-ID echo, idempotent response data, old persisted response projection, semantic hash equivalence, UTF-8 request size, and exact OpenAPI snapshot equality.

## Single-Source Inspection Demo

The configured inspection demo exposes one EO source, `ESA WorldCover 2021 (10 m, Local N30E120)`. The official palette COG and a browser-compatible RGB COG are stored under `/sata/yangm/eo-harness/datasets/worldcover-2021` and mounted read-only into TerriaMap. The item is added to the workbench at startup and opens over Shanghai and surrounding areas.

Verification on 2026-07-14 confirmed HTTP 200, exactly one workbench dataset, correct WorldCover classification colors, and a fully rendered overlay. The browser made four same-origin RGB COG Range requests, all returning HTTP 206 `image/tiff`; it made zero Terrascope WMS and zero ESA S3 requests. Zooming changed the scale from `30 km` to `10 km`, and dragging changed the map center. The container was left running after that historical check; the 2026-08-30 M2 delivery state is stopped.

## Interface Language

TerriaJS's bundled `zh_Hans` translation and built-in globe menu are enabled without rebuilding the image. A fresh browser session now opens in English for deterministic rendering, and the menu switches between `简体中文` and `English`. URL selection and the browser's saved language take precedence over the English fallback.

Headless Chrome verified both switch directions and reload persistence. Read-only project overrides fill the upstream upload, workbench, footer, and drag/drop keys in both languages, avoiding raw keys in Chinese and Chinese fallback text in English.

## Responsive Map Controls

The deployment mounts a local stylesheet and a TerriaJS 8.12.2 bridge without rebuilding the image. Desktop and mobile zoom buttons measure `46 x 44` CSS pixels. The bridge exposes Terria's native ZoomControl on small screens and supplies the repaint frames omitted by the upstream idle Cesium zoom path.

At `1280 x 720`, the workbench, toolbar, and map controls had no detected overlap or horizontal overflow. At `390 x 844`, the mobile header, language panel, zoom controls, data-attribution footer, and scale remained inside the viewport. Story, About, AR, pedestrian, and the mobile coordinate readout are hidden because they are not needed by this harness.

## Pinned Artifact

- Image: `ghcr.io/terriajs/terriamap:0.4.6`
- Linux/amd64 digest: `sha256:0853ec153c53cef6ae926c99698ccc5cf4cc1a89906550068d3a36d95c3d54e0`
- Rootless local tag: `eo-harness/terriamap:0.4.6`
- API image: `eo-harness/harness-api:0.4.0`, image ID `sha256:8ccf4297fd87281c2a837048772fd82bc22bb90c82f8d287ea692e864264b243`
- Renderer image: `eo-harness/renderer:0.4.0`, image ID `sha256:4b36d6c0b8ccfaf7c7c1c2db6e69b2f89ac3286e767a505b55b82290a7092333`
- Renderer base: Playwright `v1.55.1-noble`, pinned manifest digest `sha256:2f29369043d81d6d69a815ceb80760f55e85f5020371ad06a4d996f18503ad1c`
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

The following browser checks were recorded on 2026-07-14 before M2 changed the fresh-session fallback from Chinese to English:

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

- `eo-harness-terriamap`, `eo-harness-renderer`, and `eo-harness-api` are stopped and removed after acceptance.
- The Rootless Docker service remains available for other user projects; EO Harness images remain under `/sata/yangm/docker-rootless/data`.
- Episode state, artifacts, datasets, and the pre-M2 backup remain under `/sata/yangm/eo-harness`.
- The API and renderer use read-only root filesystems, drop all capabilities, and enable `no-new-privileges`. Only state and artifact bind mounts are writable.

## Remaining Actions and Limits

- `loginctl show-user yangm -p Linger` still reports `Linger=no`. Run `sudo loginctl enable-linger yangm` for boot and logout persistence.
- The unused legacy daemon data at `/sata/yangm/docker/eo-harness` was explicitly approved for cleanup and deleted. Post-cleanup verification confirmed the path is absent, Rootless Docker remains healthy, and the nine unrelated system-daemon containers are unchanged.
- The WorldCover layer and deterministic Natural Earth basemap are local. Renderer requests to external origins remain blocked; the TerriaMap Google Fonts CSS import is replaced by an empty local response.
- The UI bridge depends on TerriaJS 8.12.2 class names and React container internals. Revalidate the zoom and language controls before upgrading TerriaMap/TerriaJS.
- M2 implements state projection, deterministic rendered observation, artifact provenance/content, frozen evidence, and one WorldCover evaluator. V1 remains structural by design.
- Executable raster tools, catalog/STAC access, Sentinel-2 temporal tasks, perception models, agent adapters, batch runner, and resume remain to be implemented.

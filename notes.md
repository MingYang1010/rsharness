# Notes: TerriaMap Docker Deployment

## A800 Baseline

- Host: `lamda12`
- User: `yangm`
- Docker: `29.1.3`
- Docker Compose: `2.40.3`
- User groups include `docker`
- Target root: `/sata/yangm/eo-harness`
- Ports `3000`, `3001`, `8080`, `8081`, and `8888` were not listening at preflight.
- Nine pre-existing stopped containers exist and are out of scope.

## Sources

- Official repository: `https://github.com/TerriaJS/TerriaMap`
- Selected release: `v0.4.6`, published 2026-03-31
- Source HEAD at inspection: `5d75980b4967695e230727754828bf18fadb4764`
- Official image: `ghcr.io/terriajs/terriamap:0.4.6`
- TerriaMap linux/amd64 digest: `sha256:0853ec153c53cef6ae926c99698ccc5cf4cc1a89906550068d3a36d95c3d54e0`
- Docker dind image: `docker:29.1.3-dind`
- Docker dind linux/amd64 digest: `sha256:64d6ee47ea821c986467199baa162f5ac8cde3f57b719f18e23f3ed7a7444131`
- License: Apache-2.0

## Verification Log

- The first local raster is the official ESA WorldCover 2021 v200 `N30E120` COG: `ESA_WorldCover_10m_2021_v200_N30E120_Map.tif`.
- The file is stored at `/sata/yangm/eo-harness/datasets/worldcover-2021/`, is `93,443,932` bytes (`90M` allocated), and has SHA256 `f3859c80b7bd8d61a82f0adafee02378d418a2f1bafe32d8229b84301329c814`.
- File inspection reports a `36000 x 36000`, 8-bit, deflate-compressed palette TIFF in EPSG:4326. It covers longitude `120-123` and latitude `30-33`.
- TerriaJS 8.12.2 has a native `cog` catalog item and uses HTTP Range requests through `terriajs-tiff-imagery-provider`. No TiTiler container is needed for this single local COG.
- Native COG rendering is unavailable in TerriaJS's Leaflet 2D viewer, so the local demo uses a top-down `3dSmooth` Cesium viewer.
- TerriaJS rendered the official single-band palette COG as grayscale. Its exposed COG traits cannot pass a single-band custom color map, and the underlying provider rejects `convertToRGB` when the source has fewer than three samples.
- A pinned temporary GDAL image, `osgeo/gdal:alpine-small-3.6.3@sha256:4b58dab500d110a850deec1c88ee9a8da1142dafe075a96099ce8251b89814bf`, expanded the official palette to an RGB COG without modifying the source file.
- The RGB derivative is `ESA_WorldCover_10m_2021_v200_N30E120_Map_RGB.tif`, `152,280,029` bytes, SHA256 `9f376abaca38815c5c743126147aeffd1916bb1907ad98929d341d4e6c87381c`. It has COG layout, DEFLATE compression, three RGB bands, 512-pixel blocks, and seven overview levels.
- Final browser verification showed one local WorldCover workbench item, no model errors, a full `1440 x 813` Cesium canvas, and the official green/red/blue/pink land-cover class colors.
- The browser made four byte-range requests to the same-origin RGB COG; every response was HTTP 206 with `image/tiff`. Requests to `titiler.terrascope.be`, `wmts.terrascope.be`, and `esa-worldcover.s3` were all zero.
- Zoom and pan verification changed the scale from `30 km` to `10 km` and moved the displayed center to approximately `31.39145 N, 121.87443 E`.
- The dataset directory is mounted read-only from `/sata/yangm/eo-harness/datasets/worldcover-2021` to `/app/wwwroot/data/worldcover-2021`.
- The TerriaMap static server returns `Content-Type: image/tiff`, `Content-Length: 93443932`, and `Accept-Ranges: bytes`. A `bytes=0-16383` request returned HTTP 206 and SHA256 `82b25f947422a877b9db038eb78d58c24974fded0102ed30cde049900605935e`, matching the same range from the official S3 object.
- The single-source inspection demo uses ESA WorldCover 2021 at 10 m resolution.
- The former layer identifier `WORLDCOVER_2021_MAP` is not present in the live Terrascope WMS capabilities. The current layer identifier is `esa-worldcover-map-10m-2021-v2_map`.
- The live layer advertises a single `TIME` value and default of `2021-01-01`. A direct `512 x 512` GetMap request with that value returned HTTP 200 and a valid RGBA PNG.
- TerriaJS 8.12.2 source tests confirm that top-level `workbench` accepts catalog item IDs. The demo now auto-loads `esa-worldcover-2021` and uses `initialTimeSource: start` so TerriaJS selects the exact WMS time tag from capabilities.
- The initial and home camera cover the Yangtze River Delta (`116,28.5,123.5,34`) so the land-cover classes are visible immediately instead of starting at global extent.
- The running single-source demo returned HTTP 200 for `/`, `/config.json`, and `/init/eo-harness.json`; the served init file SHA256 matched the local file exactly.
- Headless Chrome verification at `1440 x 900` showed `DATASETS (1)`, one visible `ESA WorldCover 2021 (10 m)` workbench card, no `Terrascope Open WMS Catalog`, and a fully rendered land-cover overlay.
- Browser resource entries confirmed TerriaMap proxied WMS GetMap requests with `time=2021-01-01`, `layers=esa-worldcover-map-10m-2021-v2_map`, and `crs=EPSG:3857`; returned tile bodies were non-empty PNG imagery.
- Interaction verification clicked zoom-in and dragged the map. The scale changed from `100 km` to `50 km`, the center changed, and the overlay re-rendered after the new WMS tiles settled.

- The official image tag was visible through `docker manifest inspect`.
- First image pull failed due to a transient GHCR TCP 443 timeout.
- Shared daemon root is `/var/lib/docker` on the root filesystem.
- Rootless Docker prerequisites are absent; a project-scoped dind daemon will keep the actual TerriaMap image and layers under `/sata/yangm/docker/eo-harness`.
- The first Docker Hub pull attempt for the pinned dind bootstrap image timed out before layer download.
- Root cause: user shell HTTP(S) proxy is configured, but the system Docker service has no proxy environment.
- Workaround: Google `go-containerregistry/crane` `v0.21.7`, release asset SHA256 `1a57bc98207fa1c0d04bf760699099e26f8383499bfd55b99c1b919a928a7230`, downloads digest-pinned linux/amd64 image archives to SATA.
- Docker dind is fetched from `mirror.gcr.io/library/docker:29.1.3-dind`; both Google mirror and AWS Public ECR resolve to the expected Docker Hub amd64 digest.
- Image preparation is single-writer guarded by `flock`; `.tmp` archives are removed on normal exit or signal termination.
- Final bootstrap choice: pre-existing `python:3.12-slim` outer image plus host `/usr/bin/dockerd` in a privileged isolated namespace. The project daemon inherits the user shell proxy and stores its complete data root on SATA.
- Feasibility test confirmed Docker `29.1.3`, `overlay2`, and `/sata/yangm/docker/eo-harness` as the inner Docker root.
- Final daemon runtime state uses `/run/eo-harness` on container tmpfs; it does not write runtime sockets or pid files to the system disk or project directory.
- The outer namespace must mask the host `/run` entirely; otherwise the nested dockerd attaches to the shared host containerd and shim paths cross namespace boundaries.
- The chroot must bind the outer namespace's `/proc` and writable cgroup hierarchy over the read-only host-root bind. Without these mounts, inner container startup fails while adjusting shim OOM state or creating its cgroup.
- Nested `docker exec` does not reliably join the inner container mount namespace in this chroot design. The Compose healthcheck was removed and readiness is verified from the host with `curl --noproxy '*'` instead.
- The first readiness-loop run exposed an unset `TERRIA_PORT`; `start-terriamap.sh` now defines a default of `3001`.
- TerriaMap initially rendered a gray Leaflet canvas because the nested container could not resolve external tile hosts and did not inherit the shell proxy. Compose now sets explicit DNS servers and passes proxy variables at runtime without persisting their values.
- Final runtime checks on 2026-07-14 confirmed:
  - private daemon root `/sata/yangm/docker/eo-harness`, Docker `29.1.3`, `overlay2`;
  - TerriaMap writable and merged layer paths under the SATA data root;
  - shared daemon has no TerriaMap image;
  - pinned repo digest `sha256:0853ec153c53cef6ae926c99698ccc5cf4cc1a89906550068d3a36d95c3d54e0`;
  - SATA Docker data uses approximately `2.2G`;
  - HTTP 200 for `/`, `/config.json`, `/init/eo-harness.json`, and an OSM proxy tile;
  - browser title `EO Harness Map`, `24/24` OSM tiles loaded at `256 x 256`, map viewport `1280 x 720`;
  - data catalog contains `EO Harness Open Data`, ESA WorldCover, and Terrascope WMS;
  - zoom interaction changed requested OSM tiles from level 2 to level 3.
- Shutdown verification confirmed no project containers, no listeners on `3001` or `23750`, no project dockerd, and no local SSH tunnel on `13001`. All nine unrelated stopped containers remained untouched.

## Rootless Docker Migration

- Rootless Docker `29.1.3` was installed with RootlessKit `2.3.5`, `slirp4netns`, and an AppArmor profile scoped to `/sata/yangm/docker-rootless/bin/rootlesskit`.
- Rootless daemon checks passed with data root `/sata/yangm/docker-rootless/data`, storage driver `overlayfs`, and security options `rootless`, `seccomp`, and `cgroupns`.
- The `hello-world` image was pulled and executed successfully through the rootless context.
- A mode-`600` systemd drop-in persists the existing proxy environment without storing proxy values in this project.
- `start-project-docker.sh` now validates and starts the user service; it no longer creates a privileged bootstrap container.
- `start-terriamap.sh` and `stop-all.sh` now use `docker --context rootless` explicitly.
- Rootless TerriaMap verification repeated the four HTTP 200 checks, loaded `24/24` visible OSM tiles, and changed tile level from 2 to 3 after zoom-in.
- Rootless data usage after installation is approximately `1.4G`; the shared daemon still has no TerriaMap image.
- Final rootless shutdown removed the TerriaMap container and Compose network while leaving the reusable user daemon active.
- `Linger=no` remains the only service-persistence gap. It requires `sudo loginctl enable-linger yangm`.
- The user explicitly approved deletion of the superseded `/sata/yangm/docker/eo-harness` directory. Safety checks confirmed it was a real directory, not a symlink or mountpoint, and neither active daemon used it.
- Because noninteractive sudo was unavailable, cleanup used the pre-existing `python:3.12-slim` image with `--pull never`, `--read-only`, `--network none`, and exact bind mounts. The target contents were deleted first; the empty target directory was then removed with `rmdir`.
- Post-cleanup checks confirmed the old path is absent, Rootless Docker still uses `/sata/yangm/docker-rootless/data`, the TerriaMap digest is intact, project containers remain at zero, and the nine unrelated system-daemon containers are unchanged.

## Chinese Interface

- TerriaJS 8.12.2 already bundles Simplified Chinese at `/app/wwwroot/build/TerriaJS/languages/zh_Hans/translation.json` and includes the desktop/mobile `LangPanel`; no custom React component or image rebuild is required.
- `config/config.json` enables `zh_Hans` and `en`, uses `zh_Hans` as the fresh-session fallback, and detects `querystring` before `localStorage` so explicit links and saved user choices take precedence.
- The local and remote config SHA256 is `584b28b16952b9d5fdbd1d9dc5879f7e5bfa10ec82264d403015b7ece76b708c`.
- Only `eo-harness-terriamap` was force-recreated. The page returned HTTP 200 after restart, and no Compose service, image, or dataset mount was changed.
- A fresh Chrome profile opened in Chinese and showed the upper-right globe button. Its menu contained exactly `简体中文` and `English`.
- Switching to English updated the UI immediately and survived a reload with `i18nextLng=en`. Switching back to Chinese survived a reload with `i18nextLng=zh_Hans`.
- After both reloads, the WorldCover item remained visible, the Cesium canvas measured `1440 x 813`, and two local RGB COG resource requests were present.
- The bundled Simplified Chinese translation is incomplete. The observed gaps in `models.catalog.upload`, `workbench.disableAll`, and two footer link keys are now covered by project-scoped overrides.

## Responsive UI Verification

- Local overrides now cover the missing upload, workbench, footer, and drag/drop keys in both `zh_Hans` and `en`. English no longer falls back to Chinese for these controls.
- The local `ui/index.html`, `ui/eo-harness.css`, and `ui/eo-harness-ui.js` files are mounted read-only. A checksum-mode `rsync` dry run reported no differences between the local project and `/sata/yangm/eo-harness` after deployment.
- At `1280 x 720`, all three native zoom buttons measured `46 x 44` CSS pixels, the language button measured `44 x 32`, the body width remained `1280`, and no visible button/link overlap was detected.
- At `390 x 844`, all three native zoom buttons measured `46 x 44` CSS pixels. No visible element crossed the viewport boundary after the mobile footer fix; the footer retained Cesium attribution, `数据归属`/`Data attribution`, and the `50 km` scale.
- Story, About, AR, and pedestrian controls were absent at both breakpoints. The four original translation keys were absent, and browser console errors were zero.
- TerriaJS 8.12.2's native `zoomIn` and `zoomOut` methods create a 200 ms Cesium tween but do not call `notifyRepaintRequired()`. The version-pinned bridge requests frames for 350 ms after native zoom-control clicks.
- Final camera-height checks passed: `472,504.6 m -> 157,501.5 m` after zoom-in, back to `472,504.6 m` after reset, and `1,417,513.7 m` after zoom-out. A final reset returned to `472,504.6 m`.
- English persisted across reload with `i18nextLng=en`; Chinese persisted with `i18nextLng=zh_Hans`. Both languages retained three mobile zoom buttons and had zero visible overflow.
- Browser resource timing listed only the same-origin RGB COG under `worldcover-2021`; requests to `titiler.terrascope.be` were zero. A direct byte-range check returned HTTP `206`.
- The running container has nine read-only mounts, including both language override directories, all three UI files, and the WorldCover dataset. Rootless Docker still reports `/sata/yangm/docker-rootless/data` as its data root.

## Harness Environment API V1

- Environment API `v0.2.0` owns episode state independently of TerriaMap. V1 exposes `/v1/reset`, `/v1/action-space`, `/v1/episodes/{id}/step`, `/state`, and `/trace`, plus `/healthz` and OpenAPI docs.
- The deterministic action space has six actions: `set_view`, `pan`, `zoom`, `set_layer_visibility`, `set_layer_opacity`, and `submit_answer`.
- Core and contract behavior is covered by 15 `unittest` cases. The same suite passed inside the final read-only linux/amd64 image.
- SQLite uses WAL mode and transactional `BEGIN IMMEDIATE` updates. The durable path is `/sata/yangm/eo-harness/state/episodes.sqlite3`.
- The completed reference episode is `ep-9e5676cb1de740238bbb803a715eed63`. It contains six ordered transitions, terminates at step 6, and has trace hash `ef8940c3dc3a3e0bbb1e2a8b00d0ee04aee46c2c387afe061da4d3d870b3b791` before and after API container recreation.
- A separate `max_steps=1` episode returned `truncated=true`; later actions and a reused action ID with a different payload both returned HTTP `409`.
- The API image uses pinned `python:3.12-slim` base digest `sha256:c3d81d25b3154142b0b42eb1e61300024426268edeb5b5a26dd7ddf64d9daf28` and pinned FastAPI `0.116.1`, HTTPX `0.28.1`, Pydantic `2.11.7`, and Uvicorn `0.35.0`.
- Final image ID is `sha256:18cb4bd23a3cc1587e02bb349b4e281e2df884c11445e5530dfd9e4b4535423d`, size `48,608,004` bytes, architecture `amd64`.
- Security checks passed: host binding `127.0.0.1:8000`, read-only root filesystem, all Linux capabilities dropped, `no-new-privileges`, and one SATA-backed writable state mount.
- TerriaMap remained running with its original container ID while the API was built and recreated. WorldCover data and the nine TerriaMap read-only mounts were unchanged.
- Current boundary: API actions update deterministic typed state but do not yet move the open TerriaMap camera or emit a rendered image observation. The next milestone is a renderer/state projection adapter, followed by explicit geospatial memory actions.

## Harness API V1 Contract Freeze

- Service release `0.2.0` freezes `/v1` body schema `1.0.0` with `meta + data` success and `meta + error` failure envelopes.
- Every route has an explicit FastAPI `response_model`; request validation, domain errors, route errors, and internal failures share `ErrorResponse` without leaking exception text.
- `contracts.py` projects persisted dictionaries into public Pydantic DTOs. The SQLite schema is unchanged, and a contract test removes `0.2.0`-only fields from a persisted transition to verify that an old `0.1.0` response is still readable.
- `reward` is required and nullable. `final_answer` and `client_action_id` remain required nullable fields, while list/object fields remain present when empty.
- `trace_hash` remains the instance-specific hash. New `semantic_state_hash` and `semantic_trace_hash` exclude episode IDs and timestamps; equivalent episodes with different idempotency keys pass the equality test.
- The committed OpenAPI snapshot is `49,176` bytes. Eight request/response fixtures cover reset, step, state, trace, action-space, and validation failure.
- The local suite now has 15 passing tests: six domain, four store, and five HTTP contract tests. The HTTP tests also cover `201`, `404`, `409`, `422`, `500`, request-ID echo, idempotent data equality, and OpenAPI snapshot equality.
- Compatibility policy: `0.1.0` was exploratory; from `0.2.0`, existing V1 body fields, enums, status meanings, and hash semantics are frozen. Retrieval, perception, memory, evaluator, and rendered-observation bodies require `/v2` or separate versioned endpoints.
- Final A800 image is `eo-harness/harness-api:0.2.0`, ID `sha256:18cb4bd23a3cc1587e02bb349b4e281e2df884c11445e5530dfd9e4b4535423d`, size `48,608,004` bytes, architecture `amd64`.
- Live `/openapi.json` exactly equals the committed snapshot. The reference six-transition trace retained hash `ef8940c3dc3a3e0bbb1e2a8b00d0ee04aee46c2c387afe061da4d3d870b3b791` after both API-only recreations.
- Deployment smoke episode `ep-6fcceef7c3fc40b294d5dcf07d74fccf` terminated with two transitions, identical idempotent `data`, `409` closed behavior, `422` strict validation, and semantic trace hash `e739b4a6234b55a2080800d643b44c94da997e2034da0a93115c24c4e1ae37af`.
- Final SQLite count is 3 episodes and 9 transitions. TerriaMap retained container ID `58a97bcc2b50b71c20ae4fc971798c6e539f28103bae50fe051490ae2b494acb`; WorldCover and all unrelated containers were untouched.

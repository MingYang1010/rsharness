# Task Plan: TerriaMap Docker Deployment on A800

## Goal

Configure a reproducible TerriaMap Docker deployment under `/sata/yangm/eo-harness`, verify the running web application, then stop all containers created by this project.

## Phases

- [x] Phase 1: Verify the official TerriaMap source and supported build path
- [x] Phase 2: Create the SATA-backed project and Docker Compose configuration
- [x] Phase 3: Build and run TerriaMap
- [x] Phase 4: Verify HTTP responses, static assets, and rendered UI
- [x] Phase 5: Stop the stack and confirm no project containers remain running
- [x] Phase 6: Record commands and update the Obsidian research note
- [x] Phase 7: Migrate TerriaMap from the bootstrap daemon to Rootless Docker
- [x] Phase 8: Re-run storage, HTTP, browser, and shutdown verification
- [x] Phase 9: Delete the explicitly approved legacy Docker data root
- [x] Phase 10: Reduce the catalog to one ESA WorldCover source and auto-load it
- [x] Phase 11: Sync the single-source configuration and start TerriaMap for inspection
- [x] Phase 12: Verify HTTP, imagery rendering, catalog scope, and pan/zoom interaction
- [x] Phase 13: Select a bounded local WorldCover COG and verify native TerriaJS support
- [x] Phase 14: Download the official WorldCover COG into the SATA-backed dataset directory
- [x] Phase 15: Mount the dataset read-only and replace the external WMS catalog item
- [x] Phase 16: Verify local Range delivery, rendering, interaction, and absence of WorldCover WMS requests
- [x] Phase 17: Inspect TerriaJS language assets, language detection, and built-in switcher behavior
- [x] Phase 18: Enable Simplified Chinese and English in the mounted TerriaMap configuration
- [x] Phase 19: Recreate only TerriaMap and verify Chinese, English switching, and preference persistence
- [x] Phase 20: Re-check the local WorldCover layer and update deployment documentation
- [x] Phase 21: Capture desktop and mobile layout baselines and identify stable UI selectors
- [x] Phase 22: Add a version-pinned local UI stylesheet and Chinese translation overrides
- [x] Phase 23: Sync UI assets, recreate only TerriaMap, and verify responsive layout and zoom controls
- [x] Phase 24: Re-check language persistence and local WorldCover rendering, then update documentation
- [x] Phase 25: Define the V1 episode, state, action, observation, budget, and trace contract
- [x] Phase 26: Implement the deterministic operation engine and SQLite trace store with unit tests
- [x] Phase 27: Expose reset, step, state, trace, action-space, and health endpoints through FastAPI
- [x] Phase 28: Add the API service to the SATA-backed Rootless Docker stack and run an end-to-end episode
- [x] Phase 29: Verify replay evidence, document the API, and update the Obsidian project note
- [x] Phase 30: Freeze the V1 request, success, error, state, trace, and hash contracts
- [x] Phase 31: Add typed public response models and isolate them from SQLite representations
- [x] Phase 32: Add HTTP contract, validation, idempotency, semantic-hash, and OpenAPI snapshot tests
- [x] Phase 33: Build and recreate only harness-api, then verify old and new episodes on A800
- [x] Phase 34: Publish the compatibility policy, fixtures, deployment evidence, and Obsidian update

## Acceptance Criteria

- Source is pinned to a specific upstream commit or release.
- Docker configuration is stored under `/sata/yangm/eo-harness`.
- Project data, build context, and logs do not use `/data` or `/lamda`.
- The TerriaMap page returns HTTP 200 and renders non-blank map UI.
- `docker compose down` is run after testing.
- No `eo-harness` container remains running in the background.
- Existing unrelated containers are untouched.
- The inspection demo contains exactly one EO source: ESA WorldCover 2021 (10 m).
- The WorldCover layer is visible without requiring the user to add it manually.
- The verified demo container remains running for the user's inspection.
- The local demo stores its WorldCover raster under `/sata/yangm/eo-harness/datasets/worldcover-2021`.
- The local raster is mounted read-only into TerriaMap and served with HTTP Range support.
- The rendered WorldCover layer makes no request to `titiler.terrascope.be`.
- A fresh browser session opens in Simplified Chinese.
- The built-in language menu exposes both `简体中文` and `English`.
- Switching languages updates the interface immediately and persists across reloads.
- At `1280 x 720`, the Chinese workbench and top toolbar have no clipped or overlapping controls.
- At `390 x 844`, the mobile header, menu, and map controls remain inside the viewport.
- Zoom in, reset view, and zoom out each have a visible hit target of at least `44 x 44` CSS pixels.
- `POST /v1/reset` creates an active episode with a deterministic initial map state and step budget.
- `POST /v1/episodes/{episode_id}/step` validates and executes the six V1 map/control actions.
- Repeating a `client_action_id` returns the original transition without consuming another step.
- `GET /v1/episodes/{episode_id}/state` and `/trace` survive an API container restart.
- Reaching the step budget truncates the episode; `submit_answer` terminates it; later steps return HTTP 409.
- The harness API binds only to host loopback and stores its SQLite state under `/sata/yangm/eo-harness/state`.
- Every public endpoint declares a typed response model and returns one stable `meta + data` or `meta + error` envelope.
- Request validation, domain conflicts, unknown routes, and internal errors share one documented error shape.
- Public state is projected explicitly from persisted state, so SQLite-only fields can change without silently changing V1 output.
- Equivalent episodes have the same semantic trace hash even when episode IDs, timestamps, and idempotency keys differ.
- The committed OpenAPI snapshot and golden JSON fixtures match the running `0.2.0` service.
- The pre-upgrade reference episode remains readable after only `harness-api` is recreated.

## Decisions Made

- Treat the requested open-source platform as TerriaMap/TerriaJS based on the established project decision.
- Use CPU-only containers for this phase; NVIDIA Container Toolkit is not required.
- Do not delete pre-existing exited containers or images.
- Use a project-scoped Docker-in-Docker daemon because rootless prerequisites are absent and the shared daemon stores layers under `/var/lib/docker` on the system disk.
- Bind the inner daemon data root to `/sata/yangm/docker/eo-harness`; remove the outer bootstrap container after testing.
- Pin TerriaMap `0.4.6` to its linux/amd64 image digest and reuse the installed host Docker `29.1.3` binary for the isolated project daemon.
- Use the pre-existing `python:3.12-slim` image as a temporary privileged namespace shell and execute the host `dockerd` through a read-only chroot. This avoids downloading a new dind image to the shared daemon.
- Supersede the bootstrap design after Rootless Docker installation. Current scripts use the `rootless` context and require data root `/sata/yangm/docker-rootless/data`.
- Keep the Rootless Docker user service running after project shutdown; `stop-all.sh` removes only the EO Harness Compose resources.
- For the inspection demo, keep OpenStreetMap only as geographic context and expose exactly one remote-sensing source in the catalog.
- Supersede the earlier shutdown acceptance criterion for this phase: leave the verified TerriaMap container running because the user requested an interactive inspection.
- Use the official `N30E120` WorldCover 2021 v200 COG as the first local dataset. It covers Shanghai and surrounding areas in a single 3 by 3 degree, approximately 93.4 MB file.
- Use TerriaJS's native `cog` catalog item instead of adding TiTiler for this single-file demo. This removes an unnecessary service while retaining byte-range access to the local COG.
- Switch from Leaflet `2d` to top-down `3dSmooth`, because TerriaJS 8.12.2 does not render COG catalog items in the 2D viewer.
- Preserve the official palette COG and generate a separate three-band RGB COG with GDAL 3.6.3. TerriaJS's TIFF provider does not apply embedded palettes to single-band COGs, while the RGB derivative preserves the official class colors.
- Use TerriaJS 8.12.2's built-in `LangPanel` and bundled `zh_Hans` translation instead of maintaining a custom frontend component or image.
- Use `zh_Hans` as the fallback for a fresh session while detecting `querystring` first and `localStorage` second, so explicit links and saved user choices take precedence.
- Keep TerriaJS's native ZoomControl behavior. A small pinned bridge changes its `screenSize` from `medium` to `any`, because TerriaJS 8.12.2 otherwise omits zoom controls on mobile.
- Use a stylesheet loaded after `TerriaMap.css` to enlarge zoom hit targets and compact only the desktop language button; do not patch the minified application bundle.
- Hide Story, About, AR, and pedestrian controls because they are unused in the EO harness and consume the space needed by core map actions.
- Add matching English overrides for the upstream keys missing from both language bundles; otherwise the Chinese fallback produces mixed-language English screens.
- Keep the Cesium credit, data-attribution link, and scale visible on mobile while hiding the coordinate readout and secondary terms/credits links that cannot fit in 390 CSS pixels.
- Use a document-level click bridge to request map frames for 350 ms after native zoom-control actions. TerriaJS 8.12.2 leaves `notifyRepaintRequired()` commented out in `zoomIn` and `zoomOut`, so an idle request-render scene otherwise receives the click without advancing its 200 ms tween.
- Treat the V1 API as the episode-state source of truth; TerriaMap remains a renderer and is not allowed to own benchmark state.
- Limit V1 to `set_view`, `pan`, `zoom`, `set_layer_visibility`, `set_layer_opacity`, and `submit_answer`. Retrieval, perception, memory, and evaluator actions remain later phases.
- Use a transactional SQLite store and append-only transitions for the first service milestone. PostGIS is unnecessary before multiple workers or spatial memory queries exist.
- Publish the API only on `127.0.0.1:8000`; use SSH or VSCode port forwarding for clients until authentication is designed.
- Treat service `0.2.0` as the V1 contract-freeze release. The new response envelope is intentionally breaking relative to the exploratory `0.1.0` API; later V1 changes must be additive.
- Keep the SQLite schema unchanged during contract stabilization and translate stored dictionaries through an explicit public projection layer.
- Keep `trace_hash` as an episode-instance integrity hash and add `semantic_trace_hash` for cross-episode replay equivalence.
- Cap a V1 trace at the existing `max_steps <= 1000` bound instead of adding pagination before traces can exceed that size.

## Errors Encountered

- The first live `0.2.0` smoke script created its reset episode and then stopped because `dict(response.headers)` exposed `X-Request-Id` while the assertion requested `X-Request-ID`. The retry normalized header names to lowercase and reused the same zero-step episode, avoiding an extra record; all intended HTTP checks then passed.
- A pre-deployment Python `urllib` request to API loopback inherited the shell proxy and returned HTTP `502`. The retry used an empty `ProxyHandler`, confirmed the six-transition reference trace, and made no service or file changes.
- The original WorldCover layer identifier `WORLDCOVER_2021_MAP` was absent from the current Terrascope WMS capabilities. It was replaced with the verified live identifier `esa-worldcover-map-10m-2021-v2_map`; the required discrete time value is selected from capabilities with `initialTimeSource: start`.
- The first three-file `rsync` placed `config/eo-harness.json` at the remote project root instead of the mounted `config/` directory. Its SHA256 was checked against the local source, the exact accidental path `/sata/yangm/eo-harness/eo-harness.json` was removed, and the config was re-synced to the mounted path before recreating only the TerriaMap service.
- The in-app browser runtime could not initialize because it attempted to redefine the Node `process` property. Verification used the installed Chrome headless/CDP interface instead; no deployment files were changed by this fallback.
- The first local COG browser check rendered the palette source as grayscale. TerriaJS exposes only a restricted subset of the TIFF provider render options and rejects `convertToRGB` for single-band inputs, so the source was preserved and a three-band RGB COG derivative was generated.
- The first GDAL conversion container ran as the host UID inside Rootless Docker and could not write the bind mount because container root maps to the host user. It exited before creating output. The retry used container root, completed successfully, and the exited helper container was removed.
- The first RGB browser retry hit `ERR_CONNECTION_REFUSED` because the temporary SSH port forward had exited while TerriaMap was recreated. Remote HTTP remained 200; the tunnel was re-established, local HTTP was checked, and the complete browser audit was rerun successfully.
- Direct `du` of Rootless Docker snapshot internals emitted permission warnings for remapped overlay work directories. Storage reporting used `docker --context rootless system df` instead; no permissions or files were changed.
- The first language-verification headless Chrome was reaped with its short command session, while the SSH forwarding state was not reliable enough for testing. Remote TerriaMap remained healthy; verification was restarted with explicit long-lived foreground sessions.
- A combined CDP script did not return cleanly across an in-page reload. The switch, reload, and state checks were split into independent CDP connections; both English and Chinese persistence then passed.
- The test tunnel reused an existing SSH ControlMaster, so stopping the foreground client did not remove the forwarded port. The exact forward was cancelled with `ssh -O cancel -L 13001:127.0.0.1:3001 a800x4_197_via_win`; ports `13001` and `9224` were both closed afterward.
- A later `ssh -O forward` retry found no ControlMaster socket, while a background `ssh -fN` command returned success without leaving port `13001` listening. The verification tunnel was restarted with `ControlMaster=no`, `ControlPath=none`, and a monitored foreground session.
- The first responsive mobile audit measured a `559 px` bottom bar in a `390 px` viewport. Hiding only the mobile coordinate readout and Terria software logo reduced the bar to the viewport width while retaining Cesium attribution and the scale; secondary terms/credits links are now desktop-only.
- The first native zoom audit showed reset and zoom-out movement but no idle zoom-in movement. TerriaJS 8.12.2 starts the Cesium tween without requesting repaint frames; the local bridge now drives the native tween without replacing the native control.
- English initially fell back to Chinese for the locally supplied footer and drag/drop keys. A separate read-only English override removes the mixed-language fallback.
- The first harness API image build reached the pinned Python base image but pip could not access PyPI because build containers did not inherit the SSH shell proxy. The build was stopped before an image was emitted; Compose now passes the existing proxy variables as ephemeral build arguments without storing their values in project files.
- The first API smoke episode stopped after its idempotency assertion because the first response and SQLite-replayed response had equivalent JSON objects but different key order. The store now canonicalizes the first response through the same serialization path before returning it; the incomplete development-only episode is removed before the final persistence test.
- The first canonical `start-harness.sh` run rebuilt an unchanged API image because Compose generated a new provenance manifest, then printed one expected readiness retry error. The script now disables local provenance generation and silences transient curl errors so unchanged builds remain content-stable and startup output stays actionable.
- A two-file rsync omitted `--relative` and created the extra path `/sata/yangm/eo-harness/start-harness.sh` instead of updating `scripts/start-harness.sh`. The correct relative path was synchronized, the exact accidental root-level file was removed, and both path absence and script executability were verified.
- `--provenance=false` was ignored by the current Compose fallback builder because buildx is absent, so an attestation manifest was still emitted. Canonical startup now builds the API image only when its pinned local tag is missing; source updates use an explicit Compose build before recreation.
- An image-inspection format referenced absent `.Config.Image` metadata and failed before printing the requested values. The retry used only supported `.Id`, `.Size`, and `.Architecture` fields and succeeded.
- The first Obsidian static assertion expected the reference episode ID in the experiment note, while it had only been recorded in the canonical Knowledge note. The experiment record now carries the same episode ID and trace hash, and the assertion passes against both notes.

- First `docker pull ghcr.io/terriajs/terriamap:0.4.6` attempt failed while issuing the registry manifest HEAD request: TCP 443 timeout to `20.205.243.164`. The image manifest had resolved successfully during preflight, so this is being treated as a transient registry path failure rather than a missing tag.
- Standard rootless Docker cannot be installed without administrator action because `newuidmap`, `newgidmap`, `rootlesskit`, and `slirp4netns` are absent. `/etc/subuid` and `/etc/subgid` are configured, but the required setuid helpers are not installed.
- Pulling the pinned `docker:29.1.3-dind` image also timed out at the Docker Hub manifest HEAD request. No image layers or project containers were created. Registry routing/proxy behavior is under investigation.
- Shell HTTP(S) requests use a configured proxy, while the shared Docker service has no proxy environment. The deployment now downloads digest-pinned image archives with `crane` through the user proxy and loads them into the intended daemon.
- Docker Hub layer transfer stalled after writing 26 MB for about four minutes. The temporary archive was discarded, and the same digest is now fetched from Google's official `mirror.gcr.io` Docker Hub mirror; AWS Public ECR independently reported the same digest.
- The first remote Ctrl-C disconnected the SSH client but left the old transfer process alive. Both temporary process groups were terminated, the shared `.tmp` was removed, and `prepare-images.sh` now uses `flock` plus signal cleanup to prevent concurrent writers and stale partial archives.
- The dind archive route was superseded after a successful isolated test using the existing host `dockerd` binary. The tested nested daemon reported Docker `29.1.3`, `overlay2`, and data root `/sata/yangm/docker/eo-harness`.
- The feasibility daemon left root-owned `exec-root` runtime files under the project directory. The final configuration moves `exec-root` and pidfile to a 64 MB tmpfs inside the bootstrap namespace; only durable Docker data remains on SATA.
- First formal start set `DOCKER_HOST` before launching the inner daemon, so the bootstrap image check queried a nonexistent inner endpoint and falsely reported the pre-existing image as missing. `DOCKER_HOST` is now exported only after project daemon startup.
- The first tmpfs destination was a nonexistent child under the read-only `/host` bind, so OCI setup could not create its mountpoint. Runtime tmpfs now overlays the existing `/host/tmp`, and the command creates `/host/tmp/eo-harness` before chroot.
- With only `/tmp` isolated, dockerd discovered and reused the host `/run/containerd/containerd.sock`; the host containerd could not see the outer tmpfs shim paths. The complete `/host/run` is now an isolated 128 MB tmpfs, forcing a private managed containerd and keeping all runtime sockets namespace-local.
- After private containerd isolation, inner startup exposed mismatched read-only `/proc` and cgroup views. The bootstrap command now bind-mounts the outer namespace's `/proc` and writable cgroup tree into the chroot before starting dockerd.
- Nested `docker exec` did not reliably enter the TerriaMap mount namespace, so a Compose healthcheck produced false failures even while the server was running. Readiness now uses a host-side `curl --noproxy '*'` loop.
- The first readiness-loop run failed because `TERRIA_PORT` was unset in `start-terriamap.sh`. The script now defaults it to `3001`.
- Browser testing initially showed a gray map. Inner DNS could not use the host loopback resolver, and the Terria service did not inherit proxy variables. Compose now supplies reachable DNS servers and runtime proxy environment without storing proxy values in files.
- Ubuntu AppArmor initially blocked RootlessKit user namespaces for the custom SATA binary path. A path-scoped AppArmor profile fixed the restriction without disabling the global security control.

## Status

**Complete through Phase 34** - API `0.2.0` is healthy on A800 with V1 schema `1.0.0` frozen. Fifteen local/container tests, exact live OpenAPI equality, old-episode compatibility, idempotent data, semantic hashes, UTF-8 size validation, API-only recreation, documentation, and Obsidian write-back are verified.

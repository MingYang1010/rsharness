# EO-Gym integration

## Ownership and storage

A800 `/sata/yangm/eo-harness` is authoritative. Keep external source, download
receipts, indexes, metadata and execution outputs under ignored `runtime/`.
Existing datasets and models remain in their existing SATA directories.
Only verified source/configuration/tests/documentation commits are synchronized
to the local checkout. Do not commit dataset inventories or downloaded source.

## Repository guard

Install the versioned hook with `git config core.hooksPath .githooks`.
`python3 scripts/check_git_payload.py` checks the staged Git blobs, not working
tree sizes. It rejects data/runtime paths, model/archive/raster formats,
bulk JSONL, secret environment files, symlinks, files above 2 MiB and aggregate
staged payload above 10 MiB. Explicitly stage named source files, never runtime.

## Pinned upstream acquisition

`config/eo-gym-source.json` pins the HF revision and archive SHA-256 values.
Run `python3 scripts/fetch_eo_gym.py --destination
/sata/yangm/eo-harness/runtime/eo-gym/upstream` for source only. No training
trajectories, metadata indexes, models or imagery are included in that action.
Connections default to the A800 system proxy, as explicitly requested by the
operator; `--network direct` is available for diagnostics. Proxy addresses and
credentials are never stored in source or acquisition receipts. Each source file
is verified against its pinned Git blob ID, and a runtime receipt records SHA-256.

Archive acquisition is an explicit separate `--archive` action. It uses the selected
network mode, bounded size, resumable partial files and SHA-256 verification.
It does not extract archives; extraction needs a separate path/link/size audit.
A shared 3 TB runtime reservation ledger now gates the acquisition, packed-image
extraction and inventory CLIs. See [storage-quota.md](storage-quota.md) for recovery
and enforcement boundaries. The opt-in storage broker also gates persistent
artifacts; provider work/cache uses bounded tmpfs. Other writers still require
integration; this is not yet whole-system quota enforcement or a kernel quota.

## Trust and completion boundaries

EO-Gym software licensing remains unresolved: external local research use only;
no upstream code is redistributed in this repository. Dataset license terms
must be recorded separately. Ground-truth-backed simulators must not enter the
real-agent tool allowlist or count as real perception. Even the upstream crop
module imports ground-truth helpers: mounting no label files is mandatory.

## Implemented CPU integration (2026-09-17)

- `app.eo_gym_bridge` exposes only the upstream CPU crop class, using original
  hash-verified source. It skips the upstream package's eager GPU-model imports
  through a namespace shim, without modifying upstream source files.
- `/execute` accepts only an approved opaque asset ID and normalized AOI. It
  rejects raw paths/URLs and simulation tools. The deployment mounts only a
  staged input image; original dataset directories, labels and indexes are absent.
- `EO_HARNESS_EO_GYM_URL` opts the Harness into `eo_gym.crop`. With the variable
  absent and catalog disabled, frozen V1/M2 behavior and capability fixtures are unchanged.
- `tool.invoke` reserves one in-flight action per episode, releases the SQLite
  write transaction during provider I/O, and atomically finalizes observations,
  artifacts, budgets and cached responses. Retries do not rerun successful tools.
- A 90-second stale lease becomes a recorded interruption, never an automatic
  duplicate execution. Executed failures consume one step and tool call. Input
  accounting is logical asset bytes; failed-call physical I/O is not measured.
- PNG outputs retain pixel coordinates and source lineage, not invented spatial
  bounds. TaskManifest supports explicit PixelAssetRef inputs with audited width,
  height and channels. Pixel-only/mixed episodes have no geographic map state and
  reject geographic actions/evidence. Crop tool version 1.1.0 records decoded
  dimensions/channels in PixelArtifactRef; new evidence windows are bounds-checked
  from persisted metadata. PNG payloads are fully decoded before registration.
- New headless tasks can explicitly enable [derivation identities](artifact-derivations.md):
  different lineage receives distinct artifact/evidence references while content
  storage still deduplicates bytes. Old tasks retain content-only IDs and conflict
  refusal; no old metadata, task versions or frozen contracts are rewritten.

## Reproducible smoke deployment

Use existing rootless image `eo-harness/harness-api:0.4.0` as the dependency base,
with current source mounted read-only. Isolated runtime dependencies are Pillow
11.3.0 and requests 2.32.5; download wheels on A800 using its system proxy and
install offline into `runtime/eo-gym/python`. Do not upgrade shared environments.

1. Run `scripts/inventory_datasets.py --config config/local-sources.json --output
   /sata/yangm/eo-harness/runtime/<new-inventory-directory>` in the project image,
   mounting existing data read-only. Output is a private candidate inventory, not
   permission to expose those assets to an Agent.
2. Run `scripts/prepare_eo_gym_smoke.py --root /sata/yangm/eo-harness --inventory
   runtime/inventory-20260917-01/coverage.json`. It stages one audited FAIR1M2
   georeferenced image and creates separate task/state/artifact directories. It
   refuses to overwrite an existing smoke directory.
3. Initialize the private credential/store with `python3 scripts/prepare_storage_broker.py`.
   Use a fresh smoke directory; existing local artifact stores are not migrated.
   See [storage-broker.md](storage-broker.md). Run
   `docker --context rootless compose -f compose.eo-gym-smoke.yaml up -d
   --wait provider harness`, then `... run --rm --no-deps verify`.
4. Recreate only the smoke Harness, then run `... run --rm --no-deps verify
   python /verify.py --resume-check` to verify persisted state, trace and content.
5. Run `docker --context rootless compose -f compose.eo-gym-smoke.yaml down`.
   Runtime evidence remains on SATA; no ports are published and the network has
   no external egress. Only this smoke project's resources are stopped.

BLAS/OMP/GDAL threads are limited to one inside smoke containers. Without this,
OpenBLAS attempted 64 threads, hit the 64-PID container limit and the API exited
139. Do not compensate by changing shared host process limits.

## Verified results and unfinished scope

- A800 isolated source gate: 290 Python tests discovered, 256 passed and 34
  environment-dependent tests skipped. This includes the 44 original regressions;
  synthetic test fixtures are not counted as dataset coverage.
- Live HTTP smoke: real FAIR1M2 image, 400x300 crop, one tool call, frozen evidence,
  answer submission and passed structural checks. Container recreation preserves
  state, trace hash and artifact hash. Reports are under ignored
  `runtime/eo-gym-smoke/reports/`.
- Inventory: 31 roots; 15 sampled-readable, 7 partial, 9 unavailable to the current
  image-file scanner. All 65 selected samples read successfully. Counts include
  duplicate dataset versions; they are not unique observations. These are the
  original file-scanner counts, not current packed-source admission counts.
  The approximately 651 MB index is not in Git.
- Reviewed Arrow/Parquet image-only sample admission is implemented; see
  [packed-data.md](packed-data.md). Three real XLRS Arrow sources each supplied
  three distinct images; all nine real HTTP crop/submission/restart tests passed.
  Parquet has fixture acceptance only. Oversized images remain coverage gaps.
- Full EO-Gym archives are not downloaded: both HF and HF-mirror range probes
  reset through the current system proxy. PyPI succeeds through the same proxy.
- Reviewed [Sentinel-2 STAC admission](stac-admission.md) now reads bounded COG
  windows via the A800 system proxy. Three dated Nanjing scenes produced 12 native
  windows; only 3 display-RGB inputs enter the Agent catalog. Twelve scripted
  actions, four-service recovery and fresh-provider execution replay pass.
  A separate [native NDVI task](native-raster-tools.md) now admits six red/NIR
  inputs, with 14 scripted actions, exact independent pixel checks, five-service
  recovery and fresh-provider replay. The bounded
  [Sentinel-2 temporal benchmark](temporal-benchmark-v1.md) now covers
  deterministic pair selection, aligned red/SCL stacks, typed lineage,
  hidden semantic scoring, correct abstention, false-confidence detection and
  fresh-provider execution replay. The original RGB and NDVI tasks have not changed.
- These smoke runs are scripted interaction acceptance, NOT Qwen inference,
  semantic task accuracy or full dataset integration. The existing HTTP `replay`
  remains structural. The separate operator [execution replay](execution-replay.md)
  reruns supported terminal actions with a fresh provider, compares full normalized
  trace/state and regenerated artifact hashes, and never reads old output blobs.
  Its three-image12-action positive and provider-offline negative checks pass.
  The smoke task intentionally has no semantic score; historical runtime identity
  and model reasoning replay are not established.
- Episode-local `catalog.search`/`catalog.inspect_asset` are opt-in; see
  [catalog.md](catalog.md) for query semantics, logical metadata costs and the
  Agent gateway/privacy boundary. They do not expose the private index.
- A separate [Agent gateway](agent-gateway.md) binds one credential to one pinned
  episode, projects public fields and denies raw operator routes. A three-image
  split-network smoke denies backend DNS/IP access and preserves12 cached actions
  and artifact hashes after recreating gateway/provider/storage/Harness. Reports:
  `runtime/agent-xlrs-20260917-01/reports/`. This is not real model inference.
- A800 also has an explicit `raster.resample@1.0.0` categorical path: reviewed
  same-scene SCL is aligned to a pinned reference grid with nearest-neighbor only.
  Three dates passed exact independent pixel/mask comparison,14-step Agent
  interaction, five-service recovery and positive/offline execution replay. This
  is not a cloud mask or general-purpose reprojection facility.
- Still needed: independent cloud ground truth, broader packed-source coverage,
  production multi-session/auth lifecycle, general continuous raster reprojection,
  zonal statistics, other STAC providers/public imagery subsets, cross-task
  evidence memory and the Qwen runner. Latest GPU preflight reports DRAIN and
  NVML library/driver mismatch; no driver or scheduler changes were made.

The latest broker smoke uses three XLRS caption images. After the crop, all three
services (provider, broker and Harness) were recreated; cached action, durable
artifact bytes, evidence validation and submission remained valid for all samples.
Sample 0 additionally recreated broker/Harness after submission. Reports remain
under `runtime/broker-xlrs-20260917-{0,1,2}/reports/`. Provider cache and old local
artifact directories contain no persistent content; checksum-verified blobs live
only in `runtime/managed-artifacts/`. The three dedicated projects were removed.

The subsequent catalog smoke combined those three images in one episode:
two search pages, three inspections/crops, three evidence saves and submission
(12 actions). Recreating provider/storage/Harness preserved all cached responses,
state/budgets and three artifact hashes. Reports are in
`runtime/catalog-xlrs-20260917-01/reports/`; the dedicated project is removed.
This uses catalog output to select crop inputs, not the raw task manifest route.

## Pixel-only smoke variants

The preparer accepts `--dataset-id`, `--sample-index` and a fresh `--output-name`
under runtime. It verifies the pinned checksum and actual image dimensions and
does not discard existing CRS. Pixel-only source images receive no invented bbox.
Use `EO_SMOKE_ROOT=./runtime/<output-name> docker --context rootless compose
-p <distinct-smoke-project> -f compose.eo-gym-smoke.yaml ...` for every command of
that run. This preserves previous smoke databases and reports. The verifier
checks the null map, pixel contract, real crop, retry, evidence and submission;
the existing `--resume-check` verifies persisted hashes after Harness recreation.
Only staged approved images are mounted, never dataset labels or whole roots.

Verified on A800: three seeded DOTA_V2_patched_nilsleh images (512x512 RGB,
without CRS) each produced a real 256x256 crop, idempotent retry, frozen evidence
and answer submission. All three Harness recreations preserved state, trace and
artifact hashes. Reports remain under `runtime/pixel-dota-20260917-{0,1,2}/reports/`;
all three smoke projects were stopped and removed. No dataset-wide admission,
Qwen inference or semantic accuracy is claimed. Original imagery was read-only.
The four renderer tests also pass; V1 contracts, old V2 fixtures and immutable
WorldCover tasks have no Git diff.

## Typed artifact acceptance and client migration

Artifact metadata/content reads require the episode_id query parameter and verify
the persisted episode-artifact association. This fixes an observed cross-episode
read bug; it is a required client change, not an authentication system. Tests
cover omitted/invalid scope, unrelated episodes and foreign evidence. Old
artifact JSON and hashes remain unchanged. Legacy sources without dimensions
cannot support new pixel evidence. Re-execution producing the same content with
different metadata still fails explicitly for legacy-policy tasks; opt-in
derivation-policy tasks retain both independent references without overwriting.

For a live restart check before evidence submission, run the verifier with
`--pause-after-crop`, recreate only the Harness, then `--complete-after-restart`.
This checks cached crop equality, persisted pixel metadata, rejection of an
oversized evidence window, valid evidence and answer submission. Recreate again
and run `--resume-check` to verify final state, trace, content and metadata.
Use fresh runtime names; do not overwrite earlier reports.

Three real DOTA samples passed this mid-episode restart flow on A800, including
post-restart oversized-window rejection and a second recreation after submission.
Reports are under `runtime/typed-dota-20260917-{0,1,2}/reports/`. All dedicated
containers/networks were removed; original inputs and earlier reports remain.

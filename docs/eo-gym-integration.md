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
New total storage is capped at 3 TB by the acquisition workflow; individual
archive space checks alone do not constitute a global quota manager.

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
  absent, frozen V1/M2 behavior and capability fixtures are unchanged.
- `tool.invoke` reserves one in-flight action per episode, releases the SQLite
  write transaction during provider I/O, and atomically finalizes observations,
  artifacts, budgets and cached responses. Retries do not rerun successful tools.
- A 90-second stale lease becomes a recorded interruption, never an automatic
  duplicate execution. Executed failures consume one step and tool call. Input
  accounting is logical asset bytes; failed-call physical I/O is not measured.
- PNG outputs retain pixel coordinates and source lineage, not invented spatial
  bounds. The current main TaskManifest still requires georeferenced input assets;
  pixel-only dataset task integration remains future work.
- Same content with conflicting provenance is rejected and recorded, rather than
  overwriting existing artifact metadata. Multiple derivations per content object
  require a future provenance model extension.

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
3. Run `docker --context rootless compose -f compose.eo-gym-smoke.yaml up -d
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

- A800: 64 Python tests pass, including the 44 original regressions and real
  upstream CPU tests. Synthetic test fixtures are not counted as dataset coverage.
- Live HTTP smoke: real FAIR1M2 image, 400x300 crop, one tool call, frozen evidence,
  answer submission and passed structural checks. Container recreation preserves
  state, trace hash and artifact hash. Reports are under ignored
  `runtime/eo-gym-smoke/reports/`.
- Inventory: 31 roots; 15 sampled-readable, 7 partial, 9 unavailable to the current
  image-file scanner. All 65 selected samples read successfully. Counts include
  duplicate dataset versions; they are not unique observations. Parquet/Arrow
  packages need dedicated adapters. The approximately 651 MB index is not in Git.
- Full EO-Gym archives are not downloaded: both HF and HF-mirror range probes
  reset through the current system proxy. PyPI succeeds through the same proxy.
- This is scripted interaction acceptance, NOT Qwen inference, semantic task
  accuracy, full dataset integration or execution replay. `replay` remains the
  existing structural verifier. The smoke task intentionally has no semantic score.
- Still needed: pixel-only tasks, packed-dataset adapters, reviewed asset access,
  remaining raster/STAC tools, public imagery subsets, semantic evaluators and
  the Qwen runner. A800 GPU allocation remains blocked by occupied cards/DRAIN.

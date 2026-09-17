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

Source acquisition is not an executable adapter, deployed tool server or model
acceptance. Those stages require independent tests and explicit status reporting.

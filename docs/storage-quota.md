# Runtime storage reservations

The operator ledger lives in ignored `runtime/.quota/ledger.sqlite3`. It caps
cooperative acquisition jobs at **3,000,000,000,000 bytes (decimal 3 TB)**. The
policy is persistent: opening the same ledger with a different limit fails.
Previously created runtime files are charged, not silently excluded as a baseline.
Original datasets/models outside runtime are not scanned or modified.

## Accounting and recovery

Before granting work, one SQLite `BEGIN IMMEDIATE` transaction scans runtime and
combines current usage with all unfinished reservations:

`charge = current data + control allowance + sum(max(0, scope capacity - current scope usage))`

The control allowance is at least 16 MiB (or actual control usage if greater).
File and directory charges use the larger of logical size and allocated blocks;
sparse files cannot hide logical growth. Hardlinks are conservatively counted per
pathname. Symbolic links are counted without following their targets, but a
writer's scope/ancestors cannot contain links. Cross-filesystem entries and
unreadable scans fail closed. The managed root must be on one local filesystem;
do not put this SQLite/lock protocol on NFS or replace its files while jobs run.

Reservations are maximum **total** scope sizes, not increments; existing partial
files are not counted twice on resume. Parent/child active scopes are rejected.
Per-scope kernel locks prevent two live writers from sharing a scope. A process
crash releases that lock but leaves its persisted headroom charged indefinitely.
There is no age-based cleanup. Another cooperating process can resume the same
scope after acquiring its lock. Successful completion releases unused headroom;
failure retains it and all physical partial files. No data is automatically deleted.

Inspect the ledger on A800:

```bash
cd /sata/yangm/eo-harness
python3 scripts/storage_quota.py
```

To abandon a failed operation, first establish that its writer has stopped, then
use `--reconcile /sata/yangm/eo-harness/runtime/<exact-scope>`. Reconciliation also
requires the kernel writer lock and fails if a cooperative writer is live. It
releases unused headroom only; residual files remain charged. Never delete/reset
the ledger to recover space. The CLI does not delete files or raise the ceiling.

## Connected writers

- `fetch_eo_gym.py`: mandatory reservation before network/payload writes. All
  destinations must now be below this project's runtime; the former external
  destination escape is no longer accepted. Source acquisition reserves twice its
  pinned source byte budget plus 16 MiB for temporary files/metadata. Archive
  acquisition reserves six times compressed size plus 16 MiB while downloading.
  This is a conservative temporary reservation, not evidence that extraction is
  safe. No archive extraction is implemented here; a later extractor must reserve
  and enforce its actual file-count/expanded-size limits independently. Existing
  partial symlinks are rejected. Downloads default to the existing system proxy;
  this change does not configure a proxy or perform a download.
- `extract_packed_images.py`: reserves 512 MiB image payload plus 16 MiB overhead
  for a fresh sample directory. The adapter bounds the combined receipt/coverage/
  allowlist payload to 8 MiB before publishing any metadata. Failure leaves the
  incomplete scope charged; original tables/labels stay read-only and private.
- `inventory_datasets.py`: reserves space for an 8 GiB SQLite catalog plus its
  rollback journal and 16 MiB overhead. SQLite `max_page_count` enforces catalog
  size, coverage JSON is bounded to 8 MiB, and temporary SQL work stays in memory.
  Old inventories are preserved; new runs still require a fresh output directory.
- `prepare_eo_gym_smoke.py`: reserves 256 MiB while staging a bounded input and
  task tree. This does not reserve later episode-state or report growth.
- Internal artifact broker: authenticated, bounded, checksum-verified uploads
  reserve before persistent writes. The new smoke deployment routes ArtifactStore
  through this service and keeps provider work/cache in bounded tmpfs. See
  [storage-broker.md](storage-broker.md) for deployment and recovery semantics.

These are operator CLIs. Low-level Python helpers are not standalone secure write
interfaces. Operators running extraction in Docker must mount the same complete
runtime at the project path, writable for the shared ledger/output; mount selected
source roots read-only. Do not expose this operator container, ledger, original
tables or private receipts to an Agent/provider.

## Enforcement limits — still required

This is an application-level reservation protocol, **not a kernel disk quota**.
Each connected writer has its own pre-write byte bound. An arbitrary process can
write without reserving; its files are charged at the next scan, but the ledger
cannot stop that process. A detected writer overrun is recorded as failed and
future grants are denied when over quota; detection does not undo an overrun.

API SQLite/WAL/logs, Docker logs, dependency installers and future extraction/STAC
writers still need integration. Existing local ArtifactStore deployments remain
ungated until explicitly configured/migrated to the broker; old content is not
automatically moved. Do not claim whole-system enforcement or mount the entire
runtime into an untrusted provider. Unmanaged runtime bytes are nevertheless
included in every new reservation; unmanaged writes outside runtime are not.

## Physical usage audit

The operator command scripts/audit_physical_usage.py is an independent report,
not another reservation mechanism. It counts allocated bytes as st_blocks * 512,
deduplicates hard links by (st_dev, st_ino), counts a symlink entry without
following its target, and separates SQLite main/WAL/SHM, reports, logs, Docker
logs, checkpoints, runtime artifacts, and uncategorized bytes. Directory
scan/stat/count failures and cross-filesystem entries are returned in errors
with complete=false, and the CLI exits 2. It never deletes or rewrites data.
Keep this audit separate from StorageQuota: the ledger remains a cooperative
logical reservation, while this command answers how many physical bytes are
actually occupied.

## Verified acceptance (2026-09-17)

Tests cover simultaneous competing processes, actual child-process exit, retained
headroom/resume, live-writer refusal, partial download without duplicated bytes,
quota denial before network/output, persistent policy, nested scopes, outside
paths/symlinks, sparse files, free disk checks, output overrun and metadata/index
limits. Download transport uses bounded fixtures, not a real HF archive.

The A800 ledger was initialized against existing runtime files (1,057,411,072
charged data bytes at that snapshot). A request for the entire 3 TB ceiling was
rejected before creating its output. One real XLRS caption image was then extracted
under a 553,648,128-byte reservation; its 1,766,801 image bytes and original shard
SHA were verified, and the reservation became complete with zero unused headroom
retained. Evidence is in `runtime/quota-packed-20260917-01/` and the ledger. This
is storage/extraction acceptance, not another model or semantic evaluation.

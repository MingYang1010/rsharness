# Internal artifact storage broker

The broker makes the existing 3 TB runtime ledger the write gate for durable
artifact bytes. The Harness retains artifact metadata, provenance, episode scope,
evidence and evaluation. The broker only stores verified content by SHA-256;
it cannot list datasets, run tools or take over episode state.

## Trust boundaries

| Component | Read access | Persistent writes |
| --- | --- | --- |
| Trusted storage service | Runtime accounting tree, private credential, managed objects | Ledger and managed object directory only |
| Harness | Task inputs/metadata, its state DB, private broker credential | Episode DB; artifact bytes via authenticated broker |
| EO-Gym provider | Staged image and allowlist, pinned source/dependencies | None in the smoke deployment; work/results are bounded tmpfs |
| Agent / verifier | Harness HTTP interface | No direct broker or filesystem access |

All service ports are internal to a no-egress Docker network. Do not publish the
broker port. The global accounting root is mounted read-only in the trusted
broker, with only `.quota/` and `managed-artifacts/` writable. It is never mounted
in the provider or Harness. The private token is mounted only in broker/Harness,
not in provider/verifier. Unauthorized object PUT/GET returns 401. The health
endpoint exposes no ledger, filesystem paths or credentials.

The broker credential is a trusted-service credential, not per-user or per-episode
authorization. Public artifact reads still pass through Harness `episode_id`
association checks. Knowing an object hash does not authorize a broker request.

## Write, failure and retry semantics

- Maximum object: 64 MiB. A bounded Content-Length is required. Streamed bytes
  cannot exceed it; length and SHA-256 must match before publication. Uploads have
  a 30-second receive deadline. Temporary files are fsynced, renamed atomically,
  then their directory is fsynced.
- New uploads reserve `2 * object length + 64 KiB` under an opaque hash scope.
  The reservation covers a crash-left partial plus a fresh attempt. Normal failed
  requests remove only their own uniquely named temporary file; failed headroom
  remains charged until retry or explicit operator reconciliation. Process-killed
  partial files may remain and are charged. Never erase the ledger to clear them.
- Repeated valid content is a no-write operation. A cached object is rehashed,
  and repeated request content is still checked. A corrupt stored object returns
  409 and is preserved for operator review; there is no silent remote repair.
- Quota denial returns 507 before payload writes. A concurrently active upload
  for the same hash returns 503. The client does not follow redirects, inherit
  proxy variables for internal traffic, expose credentials, or fall back to local
  disk after broker failure. External acquisition still uses A800's system proxy.
- Broker reads and client downloads both verify SHA-256. The client limits output
  bytes before returning them. Harness Range reads retain their existing semantics
  but currently fetch/hash the whole bounded object; this is not remote range I/O.

Provider worker files retain the existing 64 MiB file-size resource limit. Work
now lives in a 256 MiB `/tmp` tmpfs; the output cache lives in a separate 128 MiB
tmpfs. Publication uses a process-safe file lock, conservative cumulative byte
check and atomic copy between those filesystems. A full cache returns 507, not a
retryable network error. Equal content is not copied twice. tmpfs pages count
toward the container's memory limit; do not treat this as extra free GPU/host RAM.
Recreating the provider intentionally loses its cache; finalized Harness artifacts
remain in the broker, and episode idempotency prevents rerunning finished crops.

## Deployment and migration

On A800, before starting the updated smoke Compose file:

```bash
cd /sata/yangm/eo-harness
python3 scripts/prepare_storage_broker.py
```

This retains or creates an owner-only random credential at
`runtime/.quota/broker-token`. Its contents must never enter Git, logs or reports.
Run normal smoke preparation with a **fresh** output name; it now reserves 256 MiB
for staging and rejects a task tree over 4 MiB/1024 entries or containing links.
Then use `EO_SMOKE_ROOT=./runtime/<fresh-name>` and a distinct Compose project name
for every `up`, `run`, `recreate` and `down` command. Storage is a health-checked
dependency of the Harness.

`EO_HARNESS_ARTIFACT_BROKER_URL` and `EO_HARNESS_ARTIFACT_TOKEN_FILE` select the
remote backend. Without them, existing local ArtifactStore behavior/tests remain
unchanged. Existing artifact URI/JSON/checksums are not rewritten. **Old local
artifact stores are not automatically migrated:** retain their prior deployment
configuration or implement an explicit checksum-verified migration. Do not switch
an old episode database to a fresh broker and assume its content has moved.

## Remaining coverage

The new smoke deployment controls persistent artifact writes and provider disk
growth. It does not make arbitrary uncooperative writes safe. API SQLite/WAL,
runtime logs, Docker logs, dependency installers and future archive extraction or
STAC acquisition still need compatible reservations/bounds. Staging reservations
cover preparation only, not later writes into the episode state/report directory.
Old/direct local ArtifactStore deployments are not globally gated unless moved to
the broker. There is no semantic evaluator, geographic memory or execution replay
implemented by this storage change.

## A800 acceptance (2026-09-17)

- Full Python regression: 120 tests. The 13 added tests cover broker
  authorization, size/checksum rejection, quota denial, concurrent same-hash
  refusal, duplicate writes, restart, corruption preservation, path safety,
  client bounds/redirect refusal, no local fallback and provider cache limits.
- Three real XLRS caption samples passed crop, simultaneous provider/broker/
  Harness recreation, cached action, invalid-evidence rejection, valid evidence
  and answer submission. All final state/trace/content checks passed; sample 0
  additionally recreated broker/Harness after submission. No model inference or
  semantic accuracy is implied.
- Three persistent crops total 17,082,599 bytes. Their hashes match raw broker
  files and Harness API results. Smoke-local artifact/provider directories are
  empty. Upload reservations are complete and global reserved headroom is zero.
- Real container inspection confirmed no published ports, no writable SATA mount
  or credential in provider, and no artifact/runtime mount in Harness. A direct
  provider-to-broker request without the token returned 401. The token's contents
  were never printed and its host file mode is 0600. All three dedicated Compose
  projects/networks were removed; reports/verified artifacts remain.

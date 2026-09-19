# Governed Agent credential control plane

This control plane adds pinned issuance authorization and tamper-evident lifecycle
records to the existing episode-scoped Agent gateway. It is for an internal
research operator. `actor_id` and `subject_id` are operator-supplied identifiers,
not identities authenticated by an external IdP, TLS client certificate or
workload-identity system.

## Policy and registry contract

- `AgentIssuancePolicy@1.0.0` fixes a UTC validity window, allowed issuers and
  subjects, exact task/version grants, per-grant maximum TTL and a per-subject
  active-session limit. The issuer requires the operator-provided SHA-256 of the
  policy file and fails closed on a changed pin, wide permissions, symlinks,
  invalid time, identity, task or limit.
- The committed `config/agent-issuance-policy-v1.json` is a narrow example for
  `worldcover-grounded-vqa@1.0.0` and `@1.1.0`. Its SHA-256 is
  `3e5f19138aca8d9e9fd5cf5e8d76259776817112b4e66c822c6999b2ba02bd7e`
  and it expires at `2027-01-01T00:00:00Z`. It must be replaced and newly pinned,
  not silently broadened, for another task or validity period.
- Governed sessions use `AgentCredentialRegistry@1.1.0` and persist issuer,
  subject, policy ID and policy SHA-256 beside the existing hash-only binding.
  Version 1.0 remains readable for legacy internal smoke runs. A non-empty legacy
  registry cannot be silently upgraded or mixed with governed records.
- Issuance performs an early policy check and repeats the active-session check
  while holding the registry writer lock. This prevents concurrent issuers from
  bypassing the configured subject limit. Revocation remains permitted after a
  pinned policy expires; rotation requires a currently valid policy and grant.

## Audit contract

The audit is owner-private `0600` JSONL, bounded to 64 MiB and 16 KiB per event.
Each event has a sequence, UTC time, operation ID, previous-event hash and its own
SHA-256 over canonical content. Issuance, rotation and revocation write paired
`*_started`/`*_completed` events with the same operation ID; binding reuse is a
single event. Records contain task/policy/session identifiers and generation but
never a plaintext bearer token, label, source path or raw task payload.

`scripts/query_control_plane_audit.py` opens the audit read-only, verifies the
entire chain before returning data, checks an optional expected policy pin and
supports bounded filtering by operation, episode, subject and event type.
Malformed JSON, a partial final line, a broken hash/sequence, symlink, non-regular
file, wide permissions or an oversized log fails closed.

This is application-level append-only and tamper-evident, not external WORM
storage. A trusted host administrator can replace both the file and its expected
head unless the head is anchored elsewhere. Backups, retention, off-host
anchoring and recovery are still required for production use.

## Failure and reconciliation boundary

The lifecycle spans backend reset, token files, registry replacement and audit
append; it is not one cross-file transaction. A crash can therefore leave a
`*_started` event without a matching completion, a pending reset receipt, an
orphan rotation token or a changed registry without the completion event. The
shared operation ID is the reconciliation key. Preserve all files, stop automatic
retry, inspect backend/registry/token state privately, then record the operator's
recovery action. An unmatched start is a crash/reconciliation signal, not proof
that the protected mutation did or did not happen.

## Operator use

Governed issuance supplies all five arguments together:

```sh
python scripts/issue_agent_session.py \
  --job runtime/<run>/job.json \
  --output runtime/<run>/session \
  --registry runtime/<run>/credentials/registry.json \
  --issuance-policy config/agent-issuance-policy-v1.json \
  --issuance-policy-sha256 <reviewed-sha256> \
  --actor-id trusted-operator \
  --subject-id approved-runner \
  --audit-log runtime/<run>/audit/events.jsonl \
  --reviewed-public-task
```

Pass the same pinned policy, actor, subject and audit path to
`manage_agent_registry.py` for rotation or revocation. Query from a trusted,
read-only mount:

```sh
python scripts/query_control_plane_audit.py \
  --audit-log runtime/<run>/audit/events.jsonl \
  --policy-sha256 <reviewed-sha256> \
  --limit 100
```

The existing `compose.agent-smoke.yaml` intentionally keeps legacy issuance for
backward-compatible isolation regression. It is not a governed deployment
example and does not weaken the opt-in policy path.

## A800 acceptance and limits

The accepted run is retained outside Git at
`runtime/control-plane-20260920-02`. It used the real immutable
`worldcover-grounded-vqa@1.1.0` task and episode
`ep2-45c070cedbe44573aafaf2fabf22ff09`. Issuance, generation-2 rotation and
revocation produced six valid events; all three started/completed operation IDs
paired and the audit head is
`d91d60f4b103cdd71996f5077c341d5475f2c7df1793c8d512d4a893dd333daf`.
Registry, audit and both token files were `0600`; neither registry nor audit
contained either plaintext token. An unauthorized subject was rejected, after
which registry, audit and episode DB SHA-256 values remained unchanged. The
dedicated container and network were removed; runtime evidence remains ignored.

Focused tests pass 25/25, the full Python suite passes 304/304 with 34 existing
environment-dependent skips, and renderer tests pass 4/4. This proves the pinned
internal lifecycle and audit path. It does **not** provide TLS/service identity,
authenticated external user or workload identity, distributed rate limiting,
backup/recovery, off-host audit anchoring or public multitenant acceptance.

# Independent artifact derivations

## Two identities, one content store

New headless tasks can opt into `task.metadata.artifact_identity =
"derivation-sha256-v1"`. `artifact_id` then identifies a complete derivation,
while `sha256`, `size_bytes` and the content URI identify its bytes. Two executions
with different parameters or inputs may therefore have different artifact IDs
and identical content checksums. Existing broker/local stores keep only one blob
per checksum. Metadata is never replaced just because the bytes match.

The existing `ArtifactRef` / `PixelArtifactRef` fields, artifact ID format,
evidence schema, API headers, frozen V1/V2 OpenAPI and SQLite schema remain
unchanged. No old database rows, responses, task versions or traces are rewritten.
Tasks without the explicit policy retain `content-sha256-v1`: their artifact ID
is the content checksum and conflicting metadata is still refused. Do not enable
the policy by editing an immutable existing task; create a new task ID/version.
Unknown policies and use with non-headless observation profiles are rejected.

This separates logical artifact identity from physical byte deduplication without
an ambiguous "choose a different ID only on conflict" fallback. The result does
not depend on which task writes first or what another episode has produced.

## Identity recipe and validation

The tool result is converted during the normal atomic finalization path, before
artifact registration and observation/trace/evidence references are created.
The ID suffix is SHA-256 of canonical UTF-8 JSON:

```json
{
  "domain": "eo-harness/artifact-derivation/v1",
  "artifact": "the complete serialized ArtifactRef, excluding only artifact_id"
}
```

The illustrative string above stands for an object. Canonical encoding uses
sorted keys, no whitespace separators, non-ASCII UTF-8 and disallows NaN. Null
fields and array order remain significant. The object includes content checksum,
size, URI, kind/media type, spatial/temporal/pixel metadata and all lineage fields.
Current EO-Gym crop lineage binds tool version, input ID and a parameter hash
containing normalized arguments, input checksum and pinned upstream revision.
The URI must independently agree with the content checksum.

Repeated identical derivations have the same ID; different lineage or metadata
has a different ID. Repeated executions remain separate action/tool-run and
observation records even when they reuse a derivation ID. Exact client-action
retries retain the existing no-reexecution behavior. Hashes establish identity
and tamper detection against the stored record, not proof that an untrusted
provider told the truth; provider input/output validation remains necessary.

Existing artifact and episode/observation association tables store each derived
reference separately. This supports precise `EvidenceRef.source_ref` values;
`frozen_sha256` remains the content checksum, not the artifact ID suffix.
Successful derivations are recorded; failed executions remain tool-run/error
records rather than being invented as completed artifacts.

## Agent and client contract

The public Agent session declares `task.artifact_identity`, pinned by the trusted
issuer from the task manifest. Old private bindings omit it and default to the
legacy scheme. Gateway metadata reads verify the requested ID and the declared
identity recipe. Content reads separately verify SHA-256/size and still require
the bound episode's artifact association. Metadata returned to Agents omits the
storage URI as before.

Clients must use the returned `artifact_id` verbatim and check bytes against the
explicit `sha256` field. Do not reconstruct IDs from a checksum or derive content
checksums by stripping `art-`. `scripts/verify_agent_smoke.py` now follows this
rule. The old single-image legacy smoke remains specific to legacy-policy tasks.
The operator API remains private; this is not new authentication for that API.

Execution replay uses the pinned task policy and new disposable storage, so both
derivations must be reproduced independently. It compares exact metadata/lineage
and bytes without treating equal content as equal provenance. The identity
module is included in the replay runtime source receipt.

## A800 real-image acceptance

Prepare a new task from one previously reviewed pixel-image smoke:

```bash
python scripts/prepare_derivation_smoke.py \
  --sample-run /sata/yangm/eo-harness/runtime/broker-xlrs-20260917-0 \
  --output-name derivation-new-run
```

Run inside the existing CPU image with rasterio, read-only source, reviewed input
and writable quota-managed runtime, as for the catalog preparer. Preparation
refuses an existing destination and reuses the existing bounded/hash-checking
preparer. It creates a distinct task ID, never modifies the original task or image.

Use all three Compose files with one dedicated project and matching variables:

```bash
export EO_SMOKE_ROOT=./runtime/derivation-new-run
export EO_AGENT_RUN=derivation-new-run
export EO_HARNESS_CATALOG_ENABLED=1
docker --context rootless compose -p eo-harness-derivation-new \
  -f compose.eo-gym-smoke.yaml -f compose.agent-smoke.yaml \
  -f compose.derivation-smoke.yaml up -d --wait provider storage harness
# Operator has reviewed the generated task prompt/schema and single image input.
docker --context rootless compose -p eo-harness-derivation-new \
  -f compose.eo-gym-smoke.yaml -f compose.agent-smoke.yaml \
  -f compose.derivation-smoke.yaml run --rm --no-deps issue-agent
docker --context rootless compose -p eo-harness-derivation-new \
  -f compose.eo-gym-smoke.yaml -f compose.agent-smoke.yaml \
  -f compose.derivation-smoke.yaml up -d --wait agent-gateway
docker --context rootless compose -p eo-harness-derivation-new \
  -f compose.eo-gym-smoke.yaml -f compose.agent-smoke.yaml \
  -f compose.derivation-smoke.yaml run --rm --no-deps agent-check
```

The frontend-only client searches/inspects one image, requests normalized windows
`[.25,.25,.75,.75]` and `[.25000001,.25,.75,.75]`, verifies equal pixel windows/bytes
but distinct parameter hashes/IDs, saves two precise evidence refs and submits.
Recreate only these four services, rerun the issuer (must preserve the episode),
and run `agent-check python /verify.py --resume`. Seven responses/state and both
artifact metadata/content records must match without new provider executions.
Reports refuse overwrite and are capped at 1 MiB. Existing runtime/log/API writer
quota limitations still apply; do not claim this adds whole-system hard quota.

Stop the original task services and perform a separate fresh-provider execution
audit using [execution-replay.md](execution-replay.md), the new episode ID and
an unused report name. Original task/DB mounts must stay read-only. Afterwards
run Compose down for each exact dedicated project; retain runtime evidence.

Acceptance on 2026-09-17 uses `runtime/derivation-xlrs-20260917-01/` and task
`catalog-smoke-fe00e742208d633e-derivation@1.0.0`, pinned by
`28ef1312b6772a54d18bc60152dd056e938f24b101363003202f8a2cdf2014a3`.
The original Agent episode is `ep2-c4b1e167443842dfae08f4d16efb9268`.
The first real run passed seven actions and two actual provider crop calls;
both crops have SHA-256
`f765200e20519829e02ab067d642a5fd8a689a69f1f7fe64f7d0f1e58917ee4d`.

Four-service recreation and issuer retry preserved the same episode, seven
cached responses and both derivations; the new provider logged zero crop calls
during this recovery check. Reports: `reports/derivation-checkpoint.json` and
`reports/derivation-resume.json`. The database contains two artifact rows with
one distinct content checksum; the broker has one 3,824,932-byte blob.

Fresh-provider replay report:
`runtime/execution-derivation-20260917-01/execution.json`. It reexecuted7/7 actions
and two crop workers, reproducing final state, full normalized trace and both IDs:

- `art-dc9dd5333d7ba9cff03a19c91ebaab49a3f4ed79d92346f97770e588a5b294ff`
- `art-e878dc10069e3788d68d97789117ce5a7c6df4f229d9649e7ce1d6df6c82058f`

The original DB SHA before/after replay is
`fcfdd6f05e2114cd7bd6206c6b64e7a29e3fc5ea807a77675e82159d77cfe3fe`.
This audit does not import the old content-only artifact as either derivation;
old artifacts remain intact and share bytes only at the content-storage layer.

179 Python tests pass (13 new derivation tests), including frozen V1/V2 contracts,
distinct parameters/inputs with equal bytes, precise evidence, restart/retry,
foreign episode denial, pixel dimensions, tampered lineage and execution replay.
These are scripted real-image and fixture checks, not Qwen inference, semantic
task scoring, renderer derivation support or historical runtime reconstruction.

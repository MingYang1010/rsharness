# Cross-task evidence memory v1

Current research plan: [remote-sensing challenges and roadmap](remote-sensing-challenges-and-roadmap.md).
This filename and the contract identifiers below are historical. Basic Qwen
interaction was recorded on 2026-09-22; autonomous use of this memory benchmark
has not been established by that run.

## Acceptance status and claim boundary

The isolated A800 lifecycle run at
`runtime/evidence-memory-acceptance-20260920-01` passed on 2026-09-20. Runtime
task packs, SQLite databases, certificates, private keys and reports are ignored
and must not be committed. The repository contains the fail-closed memory store,
Agent tool, matched benchmark generator, evaluator and acceptance driver needed
to reproduce the run from the separately reviewed WorldCover inputs.

This checkpoint proves that one evaluated task can publish frozen geographic
evidence and that a later immutable task can retrieve it through a pinned,
budgeted snapshot. It also proves restart idempotency, execution replay and
historical/latest invalidation semantics. The accepted actions were injected by
an operator script. No Qwen/model reasoning was executed or replayed.

## Store, policy and provenance contract

`EvidenceMemoryPolicy@1.0.0` binds all of the following:

- one policy ID and geographic memory scope;
- a UTC validity window;
- writer actor IDs to exact certificate SHA-256 pins;
- source task/version and evaluator grants with a minimum aggregate reward;
- reader task/version grants;
- maximum record TTL, active-record count and query-result count.

The policy file is checksum-pinned, bounded to 256 KiB, must be a regular file
and must not be group/world writable. Publication and invalidation are trusted
operator operations. The Agent cannot write the store.

The evidence-memory SQLite file is separate from episode state, owner-private
`0600`, bounded to 256 MiB and uses full synchronous writes. Each scope is an
append-only event sequence. Every event contains the previous event SHA-256 and
its own canonical SHA-256. A snapshot is identified by both sequence and head
hash; a reader task pins both values. Invalidation appends a new event rather
than deleting or rewriting the published record, so an old task can replay its
pinned snapshot while a new task sees the latest active set.

A record freezes task and manifest identity, episode and evaluation identity,
evidence and source identity, source SHA-256, bbox, time range, platform,
instrument, availability, expiry and a provenance SHA-256. The memory ID is a
content-derived SHA-256 over the complete record. Publication additionally
requires a terminated answered episode, a cited evidence reference, a frozen
source hash, WGS84 spatial metadata and an evaluation that satisfies the exact
policy grant.

`memory.search@1.0.0` returns only the Agent-safe projection: memory ID, object
type, public summary, bbox/time, platform/instrument, source task, source hash,
provenance hash and validity window. It does not expose source episode ID,
evidence ID, source reference or local path. The API starts only when store,
policy and policy SHA-256 are configured together:

```text
EO_HARNESS_EVIDENCE_MEMORY_STORE
EO_HARNESS_EVIDENCE_MEMORY_POLICY
EO_HARNESS_EVIDENCE_MEMORY_POLICY_SHA256
```

A partial configuration fails startup. A changed policy pin, unreadable or
tampered event chain, ungranted task, mismatched snapshot, expired record or
over-limit query returns the same public policy denial.

## Budget and benchmark contract

A search is metadata-only but is not free. It reserves one tool call and one
step, and charges logical input bytes equal to the canonical bytes scanned from
all records in the pinned snapshot. The policy can return at most 20 records and
the accepted policy lowers this to five. The current logical cost model does not
measure SQLite page reads, filesystem cache effects or physical I/O.

`config/evidence-memory-benchmark-v1.json` is the committed template (SHA-256
`eb8d7b34adc7899c1ceea1fed1e7bbc50cba9036bb431e7ee378ba0b310969f2`).
`scripts/prepare_evidence_memory_benchmark.py` creates two immutable tasks with
the same prompt, inputs, answer schema and budget:

| Variant | Allowed actions | Memory binding |
| --- | --- | --- |
| `evidence-memory-benchmark@1.0.0` | `memory.search`, then answer | exact policy, scope, sequence, snapshot hash and `as_of` |
| `evidence-memory-benchmark@1.0.1` | answer only | none |

The preparer requires exactly one active record for the query. The runtime
acceptance config is generated after publication so `as_of` is not older than
the record, and it replaces the template bbox with the fixed WorldCover
evaluation AOI. Generated task files and their manifest live only in runtime.

`evidence-memory-v1@1.0.0` scores these independent metrics:

| Metric | Weight | Requirement for full credit |
| --- | ---: | --- |
| `task.accuracy` | 0.6 | submitted label equals the fixed expected label |
| `evidence.memory_faithfulness` | 0.3 | one valid pinned retrieval and exactly the returned memory ID cited |
| `process.efficiency` | 0.1 | treatment protocol, step/wall-time bounds, no failed action or renderer call |

The control can receive accuracy credit if it guesses the label, but without a
real retrieval it cannot receive memory-faithfulness credit; a correct guess is
therefore capped at 0.7. The accepted control abstained and scored 0.1.

## Operator commands

Use a fresh dedicated directory under project `runtime/`. Generate the writer
certificate and private key there; do not add either file to Git. Publish only
after the source episode has a completed granted evaluation:

```bash
python scripts/manage_evidence_memory.py \
  --database runtime/<run>/memory/events.sqlite3 \
  --policy runtime/<run>/policy/evidence-memory-policy.json \
  --policy-sha256 <reviewed-policy-sha256> \
  publish \
  --actor-id trusted-memory-curator \
  --actor-certificate-file runtime/<run>/credentials/curator.crt \
  --episode-database runtime/<run>/source/episodes.sqlite3 \
  --tasks runtime/<run>/source-tasks \
  --episode-id <source-episode-id> \
  --evidence-id <source-evidence-id> \
  --object-type land-cover-assessment \
  --public-summary <reviewed-public-summary> \
  --ttl-seconds 1209600
```

Freeze a matched task pair from the returned sequence and hash:

```bash
python scripts/prepare_evidence_memory_benchmark.py \
  --config runtime/<run>/benchmark/config.json \
  --policy runtime/<run>/policy/evidence-memory-policy.json \
  --policy-sha256 <reviewed-policy-sha256> \
  --store runtime/<run>/memory/events.sqlite3 \
  --tasks runtime/<run>/source-tasks \
  --output runtime/<run>/benchmark/tasks
```

Inspect or invalidate through the same pinned policy and certificate:

```bash
python scripts/manage_evidence_memory.py \
  --database runtime/<run>/memory/events.sqlite3 \
  --policy runtime/<run>/policy/evidence-memory-policy.json \
  --policy-sha256 <reviewed-policy-sha256> snapshot

python scripts/manage_evidence_memory.py \
  --database runtime/<run>/memory/events.sqlite3 \
  --policy runtime/<run>/policy/evidence-memory-policy.json \
  --policy-sha256 <reviewed-policy-sha256> \
  invalidate \
  --actor-id trusted-memory-curator \
  --actor-certificate-file runtime/<run>/credentials/curator.crt \
  --memory-id <memory-id> \
  --reason <reviewed-reason>
```

`scripts/accept_evidence_memory.py` composes those commands with the real
episode store, WorldCover evaluator, benchmark episodes, backend recreation and
execution replay. It refuses to overwrite an existing report, and all mutable
paths must stay under project `runtime/`.

## Accepted A800 result

The report is
`runtime/evidence-memory-acceptance-20260920-01/evidence-memory-acceptance.json`
with SHA-256
`2fed59fe2a90ced37dc3c3ffcffed33a22c146cf65aeeb8431f7a20892038359`.

The source task was a runtime copy of
`worldcover-grounded-vqa@1.1.0` with the input instrument reviewed as
`WorldCover-map`. The source task manifest hash was
`14cf8ef7f2909a2fc9eaa94047311427963f7be2428b460bd88f6c717df40930`.
Episode `ep2-c35804debc6244ec92774ac66f36e761` cited
`ev-worldcover-memory-source`, submitted `built-up` and received accuracy 1.0,
faithfulness 0.0 and efficiency 0.0, for aggregate 0.6. The isolated acceptance
policy intentionally used 0.6 as its publication threshold because no renderer
was configured. This verifies lifecycle mechanics, not the stronger rendered
source-evidence threshold expected for a production policy.

Publication under policy hash
`b8aa31591b3761e68ec573ba102280ad3f5f2a955273cce491ed4c893006a967`
created:

- memory ID
  `mem-867a03b8e4eddc94ea30ba87c7188ac2b0266982137f19119455dae8f4b23bb0`;
- sequence 1 snapshot
  `3556b6482a894afbba21678bde2bf4b64b42eb5e0ab9ef407199fa60c163fce1`;
- with-memory task manifest
  `c17f224dc519d59fb8d2b957398839827d990da537d0023fab5586d5326a6d9e`;
- without-memory task manifest
  `d1d2cb9996de8766f18610e1a189fcf3d4b6b0c27b1598fbfee64c9416a2e3b7`.

The matched result was:

| Variant | Episode | Accuracy | Memory faithfulness | Efficiency | Aggregate |
| --- | --- | ---: | ---: | ---: | ---: |
| with memory | `ep2-6c21445ce06e4884a5e4f65f2d5acf90` | 1.0 | 1.0 | 1.0 | 1.0 |
| without memory | `ep2-8262f7cbb80e4b688e8856d020915267` | 0.0 | 0.0 | 1.0 | 0.1 |

Recreating the backend preserved the search, answer and control responses and
both evaluation objects exactly. Execution replay re-executed both with-memory
actions, re-ran the semantic evaluator, matched final state and semantic trace,
and left the original snapshot unchanged. The replay snapshot hash was
`b97b5bac67ebc551fbeda25ddd76f8259ed794b62a26e630efe34f91579aeff8`.

Invalidation created sequence 2 snapshot
`24916134d85f1bc9b4172df11f07250d1bd9155663d8731d95a85b6e2205b3f8`.
The sequence-1 task still returned its one pinned record; a new sequence-2
binding returned zero matches and the latest snapshot had zero active records.

Focused memory/replay tests pass 34/34. The full Python suite runs 325 tests:
291 pass and 34 environment-dependent tests skip. Runtime data, generated task
packs, databases, report, certificate and private key remain outside Git.

## Remaining boundary

This version does not establish autonomous memory use, multi-record planning,
memory conflict resolution, learned write selection, cross-region or
cross-sensor generalization, concurrent distributed writers, physical I/O
accounting, backup/recovery, off-host anchoring or public multitenant security.
The accepted writer certificate is a locally generated static pin, not an
external CA/IdP lifecycle. Application-level append-only hashing detects local
tampering when the expected head is retained; a trusted host administrator can
replace both database and expected state unless the head is anchored elsewhere.

The benchmark measures the causal availability of one governed memory record,
not whether a model learns when to retrieve, ignore or invalidate memory.
Autonomous Qwen memory retrieval, conflict handling and controlled comparisons
remain a separate experiment from the completed basic interaction run.

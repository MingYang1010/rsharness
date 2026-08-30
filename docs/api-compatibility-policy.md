# EO Harness API Compatibility Policy

## Frozen Contract

Service release `0.2.0` freezes HTTP API path `/v1` at body schema version `1.0.0`. The earlier `0.1.0` service was exploratory and returned unwrapped dictionaries; clients must update once to the `meta + data` and `meta + error` envelopes before relying on V1 compatibility.

The machine-readable contract is stored in:

- `contracts/openapi-v1.json`;
- `contracts/fixtures/*.json`;
- `harness_api/app/schemas.py`.

The OpenAPI snapshot and fixtures are checked by HTTP-level tests. Regenerating them is not sufficient to approve a contract change; the changed JSON must be reviewed against this policy.

## V1 Guarantees

- Every success body contains exactly `meta` and `data`.
- Every error body contains exactly `meta` and `error`.
- `meta.api_version` is `v1`; `meta.schema_version` is `1.0.0`.
- `X-Request-ID` accepts `^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`. A valid caller value is echoed in the response body and header; otherwise the server generates one.
- Episode IDs match `^ep-[a-f0-9]{32}$`.
- Timestamps are RFC 3339 UTC strings ending in `Z`, with zero to six fractional digits.
- Numeric values are finite JSON numbers. Coordinates use WGS84 decimal degrees; clients must compare coordinates numerically rather than by their textual JSON representation.
- Required nullable fields remain present: `reward`, `final_answer`, and `client_action_id` use JSON `null` when unavailable.
- Collection fields remain present as arrays or objects, including empty `evidence_refs`, `metadata`, `visible_layers`, and error `details`.
- A V1 trace contains at most `1000` transitions because `max_steps <= 1000`; V1 does not paginate traces.
- Retrying the same `client_action_id` and action returns identical `data`. Per-request `meta.request_id` may differ.

## Hash Semantics

- `state_hash` identifies one episode state. It excludes `created_at` and `updated_at`, but includes `episode_id`.
- `trace_hash` identifies one persisted episode trace, including instance IDs and transition timestamps.
- `semantic_state_hash` excludes episode ID and state timestamps.
- `semantic_trace_hash` hashes the initial semantic state, ordered action/resulting-state pairs, and final semantic state. It excludes episode IDs, transition timestamps, and `client_action_id`, so equivalent replays have the same value.
- All hashes use lowercase SHA-256 hex and should be treated as opaque identifiers by clients.

## Change Rules

Allowed without a new API version:

- implementation, performance, logging, and storage changes that preserve the committed HTTP bodies;
- correcting documentation that does not change runtime behavior;
- adding optional HTTP response headers;
- adding a new endpoint whose path does not alter existing endpoint behavior.

Requires `/v2`:

- removing, renaming, adding, or changing a request or response body field;
- changing required, nullable, enum, validation, default, or numeric-bound behavior;
- adding an action type to traces accepted by the environment;
- changing termination, truncation, idempotency, ordering, or hash semantics;
- changing an existing HTTP status code or error-code meaning.

Future perception, retrieval, memory, evaluator, and rendered-observation contracts should therefore be designed under `/v2` or separate versioned endpoints. Existing `/v1` episodes remain readable through the V1 projection layer.

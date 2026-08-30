# EO Harness Environment API V1

## Contract

Environment API release `0.2.0` is the source of truth for episode state. TerriaMap remains a renderer and does not own benchmark state. HTTP path `/v1` is frozen at body schema version `1.0.0`; compatibility rules are in [api-compatibility-policy.md](api-compatibility-policy.md).

Every success uses:

```json
{
  "meta": {
    "api_version": "v1",
    "schema_version": "1.0.0",
    "schema": "eo-harness.reset.response",
    "request_id": "req-example-1"
  },
  "data": {}
}
```

Every domain, validation, routing, and internal error uses:

```json
{
  "meta": {
    "api_version": "v1",
    "schema_version": "1.0.0",
    "schema": "eo-harness.error.response",
    "request_id": "req-example-1"
  },
  "error": {
    "code": "validation_error",
    "message": "Request validation failed",
    "details": []
  }
}
```

## Endpoints

| Method | Path | Success schema | Purpose |
| --- | --- | --- | --- |
| `GET` | `/healthz` | `HealthResponse` | Service and SQLite readiness |
| `GET` | `/v1/action-space` | `ActionSpaceResponse` | Frozen V1 action definitions |
| `POST` | `/v1/reset` | `ResetResponse` | Create an episode |
| `GET` | `/v1/episodes/{episode_id}/state` | `StateResponse` | Read current state |
| `POST` | `/v1/episodes/{episode_id}/step` | `StepResponse` | Execute one idempotent action |
| `GET` | `/v1/episodes/{episode_id}/trace` | `TraceResponse` | Read the ordered replay trace |
| `GET` | `/docs` | n/a | Interactive OpenAPI documentation |

The six V1 actions are `set_view`, `pan`, `zoom`, `set_layer_visibility`, `set_layer_opacity`, and `submit_answer`. Unknown fields, string values supplied for numeric fields, invalid coordinates, unknown layers, and invalid opacity or zoom values return the common HTTP `422` envelope.

## Example Episode

The API is published only on A800 loopback. Run these commands on the server, or forward remote port `8000` through SSH or VSCode.

```bash
RESET=$(curl --noproxy '*' -fsS \
  -X POST http://127.0.0.1:8000/v1/reset \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: worldcover-reset-1' \
  -d '{
    "task_id": "worldcover-inspection",
    "prompt": "Inspect the local WorldCover tile.",
    "seed": 42,
    "max_steps": 10
  }')

EPISODE_ID=$(printf '%s' "$RESET" | jq -r '.data.episode_id')

curl --noproxy '*' -fsS \
  -X POST "http://127.0.0.1:8000/v1/episodes/$EPISODE_ID/step" \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: worldcover-step-1' \
  -d '{
    "client_action_id": "zoom-1",
    "action": {
      "type": "zoom",
      "direction": "in",
      "factor": 2
    }
  }'

curl --noproxy '*' -fsS \
  "http://127.0.0.1:8000/v1/episodes/$EPISODE_ID/trace" | jq '.data'
```

`client_action_id` makes retries idempotent. Reusing the ID with the same action returns identical `data` without consuming another step. Reusing it with a different action returns HTTP `409`.

## State, Reward, And Hashes

- `submit_answer` sets `terminated=true`.
- Consuming the final budgeted step sets `truncated=true`.
- Later actions return HTTP `409`.
- `reward` is required but currently `null`; an evaluator is not implemented in V1.
- `state_hash` and `trace_hash` identify a concrete episode instance.
- `semantic_state_hash` and `semantic_trace_hash` identify equivalent state and replay content independently of IDs and timestamps.
- State persists in `/sata/yangm/eo-harness/state/episodes.sqlite3` across API recreation.

## Contract Artifacts

`contracts/openapi-v1.json` is the committed OpenAPI snapshot. `contracts/fixtures/` contains normalized request, success, trace, action-space, and validation-error examples. Regenerate them only after intentional contract review:

```bash
python harness_api/scripts/export_contracts.py
python -m unittest discover -s harness_api/tests -p 'test_*.py' -v
```

## Current Boundary

V1 implements deterministic map and control operations. It does not render viewport images, project API state into an open TerriaMap browser, retrieve imagery by AOI/time/sensor, run perception tools, expose memory operations, or compute evaluator rewards. Those interfaces require a separately versioned contract rather than mutation of the frozen V1 body.

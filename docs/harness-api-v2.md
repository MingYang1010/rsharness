# EO Harness Environment API V2 M1

## Contract Boundary

Implementation release `0.3.0` exposes V2 body schema `2.0.0` alongside the frozen V1 API. V2 uses independent DTOs, routes, task manifests, events, action results, and `v2_*` SQLite tables. It does not convert or modify V1 episodes.

Every success uses a typed `meta + data` envelope:

```json
{
  "meta": {
    "api_version": "v2",
    "schema_version": "2.0.0",
    "schema": "eo-harness.v2.reset.response",
    "request_id": "req-v2-reset-1"
  },
  "data": {}
}
```

Every validation, policy, state, routing, and internal error uses `meta + error`. The error object includes `code`, `message`, `retryable`, `phase`, and structured `details`; internal exception text is not returned.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v2/capabilities` | Read implemented and declared capabilities |
| `GET` | `/v2/tasks/{task_id}/versions/{task_version}` | Read an immutable task manifest |
| `POST` | `/v2/reset` | Create an episode from a registered task reference |
| `GET` | `/v2/episodes/{episode_id}/state` | Read current state and hashes |
| `POST` | `/v2/episodes/{episode_id}/step` | Execute an optimistic, idempotent action |
| `GET` | `/v2/episodes/{episode_id}/trace` | Read an append-only event page |
| `POST` | `/v2/episodes/{episode_id}/replay` | Run structural replay checks |
| `GET` | `/v2/openapi.json` | Read the standalone V2 OpenAPI document |

The M1 action set is `map.set_view`, `map.pan`, `map.zoom`, layer visibility and opacity, map time range, `memory.save_evidence`, and the three answer outcomes. `memory.bookmark_aoi` and `tool.invoke` are declared contract variants but return a typed policy rejection until their milestone is implemented.

## Example Episode

Run on A800 or through an SSH or VSCode forward of loopback port `8000`:

```bash
RESET=$(curl --noproxy '*' -fsS \
  -X POST http://127.0.0.1:8000/v2/reset \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: req-v2-reset-1' \
  -d '{
    "task_ref": {
      "task_id": "worldcover-grounded-vqa",
      "task_version": "1.0.0"
    },
    "seed": 42
  }')

EPISODE_ID=$(printf '%s' "$RESET" | jq -r '.data.episode_id')

curl --noproxy '*' -fsS \
  -X POST "http://127.0.0.1:8000/v2/episodes/$EPISODE_ID/step" \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: req-v2-zoom-1' \
  -d '{
    "client_action_id": "action-zoom-1",
    "expected_state_version": 0,
    "action": {
      "type": "map.zoom",
      "direction": "in",
      "factor": 2
    }
  }'

curl --noproxy '*' -fsS \
  "http://127.0.0.1:8000/v2/episodes/$EPISODE_ID/trace?limit=2" | jq '.data'

curl --noproxy '*' -fsS \
  -X POST "http://127.0.0.1:8000/v2/episodes/$EPISODE_ID/replay" | jq '.data'
```

## Retry And Trace Semantics

- Send the latest `state_version` as `expected_state_version`; stale writes return retryable HTTP `409 state_version_conflict`.
- Reusing `client_action_id` with the same expected version and action returns identical `data` without consuming budget or appending events.
- Reusing the ID with a different request returns HTTP `409 idempotency_conflict`.
- Policy and domain failures append one `action.accepted` and one `action.failed` event; retrying the same request does not append duplicates.
- Trace cursors use `seq:<integer>`. `limit` defaults to 100 and is capped at 1000.
- Concrete hashes retain instance IDs and timestamps. Semantic hashes remove instance-specific values while retaining the task manifest and behaviorally relevant content.

## Storage And Rollback

V2 startup applies additive, transactional, idempotent migrations to `/sata/yangm/eo-harness/state/episodes.sqlite3`. Migration failure does not advance the V2 schema version or leave partial V2 tables. The artifact mount is `/sata/yangm/eo-harness/artifacts:/app/artifacts`, but M1 does not expose artifact content endpoints.

Set `EO_HARNESS_V2_ENABLED=0` to start the application in V1-only mode without running V2 migration or registering V2 routes. Do not delete V2 tables or artifact files when rolling back the application version.

## Current M1 Limit

M1 returns structural map-state observations only. It does not prove TerriaMap state projection, nonblank deterministic screenshots, WorldCover evaluation, tool sandboxing, or Sentinel-2 temporal tasks. Those are M2-M4 acceptance items and must not be reported as complete from the M1 contract tests.

## Contract Verification

```bash
python harness_api/scripts/export_v2_contracts.py
python -m unittest discover -s harness_api/tests -p 'test_*.py' -v
```

The committed V1 OpenAPI and eight V1 fixtures must remain byte-identical while V2 changes are developed.

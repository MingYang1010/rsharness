# Episode-local catalog

Set `EO_HARNESS_CATALOG_ENABLED=1` to enable `catalog.search` and
`catalog.inspect_asset`. A provider is not required. With `EO_HARNESS_EO_GYM_URL`
also set, the same dispatcher supports EO-Gym crops. Defaults and frozen V1/M2
capability fixtures are unchanged. Each task must allow `tool.invoke` and name
the individual tools in `scenario.allowed_tools`.

## Scope and trust

The catalog is a deterministic query of the **pinned task manifest**, not a STAC
client or a search of the private dataset index. Only image assets in both
`task.inputs` and the current episode's `accessible_asset_refs` are eligible.
Known label/annotation/target roles are excluded case-insensitively. Unknown,
hidden, label and inaccessible IDs receive the same inspection error. An operator
must review roles, IDs and public descriptive strings; this is not automatic
detection of mislabeled ground truth. Search never grants access to new assets.

Query results include ID, content/snapshot hashes, media type, byte size, bands,
platform/instrument, quality, license and declared spatial/temporal/pixel metadata.
They omit URI, raw source, roles, geometry and evaluator configuration. No paths
are dereferenced and no imagery is read. Pixel-only images retain `spatial=null`.
The enclosing observation pins `task_manifest_hash`.

**This is not an authenticated Agent gateway.** Existing GET task routes return
the full trusted manifest; other state/trace APIs also have no user authentication.
Do not give an untrusted Agent unrestricted HTTP access to this deployment.
A separate [episode-scoped Agent gateway](agent-gateway.md) now provides reviewed
task/observation projections and a split-network deployment; use that entry point
for an Agent. Catalog projection alone does not prove system-wide label or
server-path isolation. Scenario cutoff/freshness admission remains the task
builder's responsibility; search time filters do not enforce new data permissions.

## Actions

```json
{
  "client_action_id": "search-1",
  "expected_state_version": 0,
  "action": {
    "type": "tool.invoke",
    "tool_id": "catalog.search",
    "arguments": {"limit": 20, "offset": 0, "bands": ["visual"]}
  }
}
```

`catalog.search` accepts only:

- `bbox`: `{west,south,east,north}` WGS84 box; intersects including boundaries.
  Only assets explicitly tagged EPSG:4326 or OGC:CRS84 are compared; other CRS
  are not reinterpreted. Antimeridian-crossing queries are rejected.
- `time_range`: `{start,end}` UTC ISO timestamps ending in Z; inclusive overlap,
  valid calendar dates, chronological rather than lexical comparison.
- `platform`: exact case-sensitive match, at most 128 characters.
- `bands`: all requested bands must exist; at most 32 names, each at most 128 chars.
- `max_cloud_cover_percent`: inclusive upper bound between 0 and 100.
- `limit`: integer 1–50, default 20; `offset`: integer 0–1000, default 0.

Unknown fields and invalid types/bounds are rejected. Missing spatial/time/cloud
metadata never satisfies the corresponding filter. Without that filter, unknown
values remain visible as null; no coordinates or timestamps are synthesized.
Results sort by asset ID and return `assets`, `matched_count`, `next_offset`,
and `cost`. Paging is stable for a pinned manifest and unchanged episode access.

`catalog.inspect_asset` accepts only `{"asset_id":"<returned ID>"}` and returns
`asset` plus `cost`. The metadata is identical to the search record.

## Cost and persistence

Both use the existing reserve/run/finalize, version check, idempotency and 90s
interruption lease. A successful query consumes one step, one tool call and
logical public metadata bytes. Search charges every eligible record scanned on
every page; inspection charges one record. Canonical UTF-8 JSON byte lengths
exclude private fields and do **not** represent physical I/O or image bytes.
The result reports `logical-public-metadata-v1`, record count and charged bytes.

One record is capped at 8 KiB, one result at 64 KiB, and task inputs at the existing
1,000 limit. Oversized results fail with an instruction to lower page size. There
is no raster artifact or artifact-byte reservation for metadata queries. Results
and their metadata hash are persisted in observations/events; API SQLite/report
growth is still outside the shared hard-growth gate. Invalid/preflight-denied
queries create no tool run. Executed failures/expired leases consume a step and
tool call once, with unknown input cost recorded as unknown (budget increment 0),
not measured zero I/O. Cached retries return the original result without recharge.

## Real acceptance

`scripts/prepare_catalog_smoke.py` combines at most three previously reviewed
pixel-image smoke runs into a fresh task under a 512 MiB staging reservation.
It verifies image hashes/size/dimensions and copies no label fields. Original runs
and imagery are retained. Use the project dependency image on A800, with project
source read-only and runtime writable; no dependency download is required.

```sh
python scripts/prepare_catalog_smoke.py \
  --sample-run runtime/broker-xlrs-20260917-0 \
  --sample-run runtime/broker-xlrs-20260917-1 \
  --sample-run runtime/broker-xlrs-20260917-2 \
  --output-name catalog-xlrs-20260917-01
export EO_SMOKE_ROOT=./runtime/catalog-xlrs-20260917-01
export EO_HARNESS_CATALOG_ENABLED=1
export EO_SMOKE_VERIFY_SCRIPT=./scripts/verify_catalog_smoke.py
docker --context rootless compose -p eo-harness-catalog-01 -f compose.eo-gym-smoke.yaml up -d storage provider harness
docker --context rootless compose -p eo-harness-catalog-01 -f compose.eo-gym-smoke.yaml run --rm verify
docker --context rootless compose -p eo-harness-catalog-01 -f compose.eo-gym-smoke.yaml up -d --force-recreate storage provider harness
docker --context rootless compose -p eo-harness-catalog-01 -f compose.eo-gym-smoke.yaml run --rm verify python /verify.py --resume
docker --context rootless compose -p eo-harness-catalog-01 -f compose.eo-gym-smoke.yaml down
```

Use new runtime/project names for another run. Verification checks two search
pages, three inspections/crops, evidence and submission, then cached response,
state/budget and artifact hash equality after recreating all three services. This
is **scripted real-image interaction**, not Qwen inference, semantic scoring,
execution replay or dataset-wide admission. Runtime reports stay outside Git.

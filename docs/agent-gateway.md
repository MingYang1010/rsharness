# Multi-session, episode-scoped Agent gateway

`app.agent_gateway` is a separate service, not an authentication change to frozen
V1/V2 operator APIs. One gateway instance can resolve up to256 hash-only session
records; each opaque bearer credential maps to exactly one already-created episode
and its task manifest hash. The model runner only receives the gateway URL and its
credential. Never expose the operator API to that runner or mount tasks, indexes,
original data, databases or broker tokens.

## Trust boundaries

- The operator reviews the task prompt, answer schema, input IDs and descriptive
  metadata as public. The issuer rejects known label/non-image input roles, but
  cannot detect a mislabeled ground-truth image or a secret embedded in prose.
- The trusted issuer calls `/v2/reset` once and writes a binding and random
  256-bit token, mode0600 inside a mode0700 runtime directory. It atomically adds
  the binding, token hash, UTC issuance/expiry, state and generation to an
  owner-private registry. The gateway mounts the registry directory read-only;
  each runner mounts only its own plaintext token. Plaintext tokens never enter
  the registry, source, prompts, logs or Git.
- The backend still owns state, budgets, idempotency, evidence and evaluation.
  The gateway never writes the episode DB and checks episode/task/hash pins on
  each state/action/artifact path. A client cannot supply a different episode ID.
- Deploy the gateway on separate internal front/back Docker networks. The Agent
  receives only the front network. Harness/provider/storage receive only the back
  network. No host ports are published. Application routing alone is insufficient
  if the runner can directly reach an unauthenticated operator port.
- This is an internal research deployment, not a production multitenant security
  service. External TLS, user identity, distributed rate limiting and a durable
  control-plane audit service remain separate work. Host/Docker administrator
  access is trusted. Do not put a model-runner shell on the backend network or give
  it host filesystem/Docker access.

## Credential lifecycle

- `AgentCredentialRegistry@1.0.0` admits at most256 records and2MiB. Token hashes
  and episode IDs must both be unique. Registry files must be owner-private regular
  files reached without symlinks; malformed, oversized, duplicate or unavailable
  registries fail closed before request-body or backend access.
- Every record has UTC `issued_at`, `expires_at`, `status`, optional `revoked_at`
  and a monotonic generation. TTL is bounded to60 seconds through7 days. Unknown,
  not-yet-valid, expired and revoked credentials are rejected by middleware before
  routing. A file-backed gateway reloads the registry on every authenticated
  request, so an atomic host-side registry update takes effect without restart.
- `scripts/manage_agent_registry.py` revokes a bound episode or rotates it to a
  newly generated token and increments the generation. Rotation replaces the only
  token hash for that episode, so the old token stops resolving. New plaintext
  tokens are written mode0600 to a new dedicated runtime directory and are never
  printed. The management tool and issuer serialize writes through the shared
  storage ledger and atomically replace the registry file.
- `EO_AGENT_BINDING_FILE` remains an explicit compatibility mode for one legacy
  session. New deployments should set `EO_AGENT_REGISTRY_FILE`. Compatibility mode
  has no expiry/revocation semantics and must not be described as multi-session.

## Agent API

Every route except `/healthz` requires `Authorization: Bearer <token>`.
Unknown routes do not proxy to the backend. All query parameters are rejected;
scope is taken exclusively from the binding. OpenAPI/docs, raw task, evaluation,
trace, replay and arbitrary reset endpoints are absent.

| Route | Public result |
| --- | --- |
| GET `/agent/session` | Public task, current state/latest observation, audited tool argument schemas |
| GET `/agent/state` | Current scoped state, excluding evaluation/diagnostics |
| GET `/agent/observations/{id}` | One observation from the bound episode, projected fields only |
| POST `/agent/step` | Existing V2 StepRequest, same client_action_id/version semantics |
| GET `/agent/artifacts/{id}` | Episode-associated artifact metadata, without storage URI/geometry |
| GET `/agent/artifacts/{id}/content` | Bounded PNG or specifically admitted scientific NDVI TIFF bytes after metadata, size and SHA-256 verification |

The task view contains prompt, answer schema, input IDs, budget and the intersection
of reviewed task/runtime actions and tools. It excludes raw AssetRefs, evaluator
config, arbitrary task metadata and server paths. Prompt/schema are operator-public,
not automatically redacted text. Map layer names are replaced with their public
asset IDs. State excludes evaluation metrics/diagnostics even after submission;
the trusted experiment runner may fetch evaluation separately.

Observations omit raw provenance and warnings. Known map/asset/image references
and explicitly projected outputs of `catalog.search`, `catalog.inspect_asset`,
`eo_gym.crop`, and `raster.band_math` NDVI are supported. Scientific outputs have
explicit numeric/grid/mask-policy projection; both input IDs must be bound to the
task. TIFF downloads require the reviewed raster tool/version/lineage and8MiB
limit, not just a media-type claim. Unknown observation/tool output variants fail closed
until a public projection is reviewed. Catalog results omit URI/source/roles;
arbitrary extra provider fields are not forwarded. Errors expose only audited
codes and a generic message, not upstream messages, headers, paths or details.

Actions are additionally checked against the binding's expanded task action/tool
allowlist. Tool arguments are validated before forwarding. Artifact/observation
reads use the bound episode, never a caller-provided scope. No redirects, inherited
system proxies or caller-supplied outbound headers are used for internal requests.
Downloads/acquisition elsewhere continue to use the A800 system proxy.

## Retry and resource behavior

- The gateway is stateless across restart; reload the same registry. Successful
  actions are cached by the backend, not repeated by the gateway. On timeout or
  ambiguous delivery, retain and retry the same action ID and exact request.
- Issuance is different from an Agent action. A completed binding is preserved
  on rerun, with token/hash/task checks. Before reset, the issuer writes a pending
  receipt. If reset succeeded but binding publication did not complete, rerun
  refuses automatic reset: an operator must reconcile the existing episode.
  There is no exactly-once reset guarantee across that failure window.
- Request bodies require Content-Length and actual bounded length, max128KiB.
  Backend JSON max2MiB, public image max64MiB; images are verified before serving.
  Four active requests maximum through the final response byte; request/response
  deadline50s, backend timeout40s, request-body deadline10s. Oversized/malformed
  upstream content fails closed. Responses use no-store/nosniff.
- Reads do not consume extra tool budget; backend actions retain existing costs.
  Issuance can publish multiple session bindings to the same registry; rerunning a
  completed issuance preserves its existing token and never repeats reset. A
  different credential for an existing episode requires explicit rotation.
- This does not provide physical-I/O accounting or quota enforcement for API DB,
  reports and Docker logs. Issuance metadata has a2MiB shared-ledger reservation.

## A800 acceptance recipe

First prepare a fresh three-image catalog smoke directory as in `catalog.md`.
Review its generated public prompt/schema/image IDs before issuing. For example:

```sh
export EO_SMOKE_ROOT=./runtime/agent-xlrs-20260917-01
export EO_AGENT_RUN=agent-xlrs-20260917-01
export EO_AGENT_TTL_SECONDS=3600
export EO_HARNESS_CATALOG_ENABLED=1
docker --context rootless compose -p eo-harness-agent-01 -f compose.eo-gym-smoke.yaml -f compose.agent-smoke.yaml up -d storage provider harness
docker --context rootless compose -p eo-harness-agent-01 -f compose.eo-gym-smoke.yaml -f compose.agent-smoke.yaml run --rm issue-agent
docker --context rootless compose -p eo-harness-agent-01 -f compose.eo-gym-smoke.yaml -f compose.agent-smoke.yaml up -d agent-gateway
docker --context rootless compose -p eo-harness-agent-01 -f compose.eo-gym-smoke.yaml -f compose.agent-smoke.yaml run --rm agent-check
docker --context rootless compose -p eo-harness-agent-01 -f compose.eo-gym-smoke.yaml -f compose.agent-smoke.yaml up -d --force-recreate storage provider harness agent-gateway
docker --context rootless compose -p eo-harness-agent-01 -f compose.eo-gym-smoke.yaml -f compose.agent-smoke.yaml run --rm agent-check python /verify.py --resume
docker --context rootless compose -p eo-harness-agent-01 -f compose.eo-gym-smoke.yaml -f compose.agent-smoke.yaml down
```

For multi-session acceptance, issue a second output directory against the same
`credentials/registry.json`, then run `agent-registry-check`. The checker is on
`agent-front` only; it performs two different real crops, verifies own artifact
reads, and denies cross-token observation/artifact reads. Use the trusted setup
container to call `manage_agent_registry.py --revoke`, then `--rotate` with a new
dedicated `--token-output`. Re-run the checker with phases `revoked-a`,
`rotated-a`, and after four-service recreation, `restarted`. A third issuance with
`--ttl-seconds 60` is checked once with `active-c` and again after expiry with
`expired-c`. Rotation checks mount the new token only for that checker run; never
mount the registry, task files, state DB, bindings, or another session's token in
the model runner.

The accepted A800 run is retained under
`runtime/agent-registry-smoke-20260919-01/reports/`: two episode scopes remained
distinct, both real crop artifacts were readable only by their owner, revoke and
rotation took effect without gateway restart, the old token became unauthorized,
the 60-second token expired, and all states survived four-service recreation.
These are credential/isolation checks, not TLS, external user authentication,
Qwen inference, or task semantic accuracy.

For a direct-IP network denial check, the trusted operator sets
`EO_TEST_BACKEND_IP` from the actual Harness container's network inspection before
running the verifier. This variable is only a diagnostic target, not routing or
model input. The verifier has no raw task/job/database mounts and uses only the
gateway's session/catalog/observation/artifact APIs. Its reports remain under the
new runtime directory, separate from credentials. Preserve both for operator
audit/resume; do not copy them into Git or model context.

The acceptance is three real XLRS caption images,12 actions and four-service
recreation, including authentication/route rejection and backend DNS/IP denial.
This is **scripted Agent interaction**, not Qwen inference, semantic accuracy,
execution replay or a claim that all future tools are automatically safe.

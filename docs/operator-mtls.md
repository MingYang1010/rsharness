# Certificate-bound operator control plane

This opt-in profile gives credential issuance a separate operator-facing Harness
and a certificate-bound operator identity. It removes the need to expose reset,
task, capability, evaluation, trace or replay routes through the Agent-facing
Harness.

```text
trusted operator -- mTLS --> operator Harness -- shared state --> Agent backend
model runner -- mTLS + bearer --> Agent ingress --> gateway -- mTLS --> Agent backend
```

The implementation is an internal research control plane. It proves explicit
service and leaf-certificate separation plus a manual CA overlap/cutover. It is
not an external CA/IdP, OCSP/CRL service, automated renewal/revocation system or
production PKI.

## Operator identity contract

- `AgentIssuancePolicy@1.2.0` binds every allowed subject and issuer to exactly
  one SHA-256 of the certificate DER. Subject and issuer pins must each be
  complete and unique, and the two role sets must be disjoint.
- Operator-bound sessions use `AgentCredentialRegistry@1.3.0`. Each record
  persists both `subject_certificate_sha256` and
  `issuer_certificate_sha256`; a non-empty older registry cannot be silently
  mixed with this schema.
- `issue_agent_session.py` derives the issuer pin from the actual operator mTLS
  client certificate supplied to its HTTPS connection. There is no CLI option
  for an operator to type an arbitrary actor pin.
- `manage_agent_registry.py` derives the same pin from
  `--actor-certificate-file` before authorizing rotation or revocation. The
  certificate pin, actor ID, subject ID, task grant and policy pin must all
  match the stored session and current policy.
- Operator-authenticated audit records use `ControlPlaneEvent@1.1.0` and include
  `actor_certificate_sha256`. Schema-1.0 parsing, canonical serialization and
  hash verification remain compatible with existing audit files.

The policy pins leaf certificates in addition to TLS CA validation. Trusting a
new CA at the server is therefore insufficient by itself: the reviewed policy
must also grant the exact new operator leaf pin.

## Interface and key separation

`compose.operator-mtls.yaml` adds `operator-harness`, listening on TLS and
requiring a certificate issued by its operator-client CA. It runs with
`EO_HARNESS_INTERFACE_ROLE=operator`, has no published host port and shares only
the state, task, artifact and internal provider/storage dependencies needed for
the operator workflow.

The ordinary backend in `compose.backend-mtls.yaml` runs with
`EO_HARNESS_INTERFACE_ROLE=agent-backend`. That role permits only:

- `GET /healthz`;
- episode state and observation reads;
- episode `step`;
- artifact metadata and content reads.

All other routes return HTTP 404 with `interface_route_denied`, including reset,
task, capability, evaluation, trace, replay, OpenAPI and interactive docs. This
is defense in depth around the Agent gateway; it does not replace bearer scope,
mTLS or network isolation.

Use separate operator server/client volumes, CAs and keys. Never reuse the
gateway backend certificate as an operator certificate, and never mount an
operator CA private key into either Harness service.

| Volume variable | Exact contents |
| --- | --- |
| `EO_BACKEND_OPERATOR_SERVER_TLS_VOLUME` | `operator-harness.crt`, mode-0400 `operator-harness.key`, `operator-client-ca.crt` |
| `EO_BACKEND_OPERATOR_CLIENT_TLS_VOLUME` | `operator-harness-server-ca.crt`, `operator.crt`, mode-0400 `operator.key` |

## Compose use

Create reviewed external volumes that contain only the files referenced by the
Compose overlays, then validate the five-layer configuration before starting it:

```sh
export EO_BACKEND_HARNESS_TLS_VOLUME=<agent-harness-server-volume>
export EO_BACKEND_GATEWAY_TLS_VOLUME=<gateway-client-volume>
export EO_BACKEND_OPERATOR_SERVER_TLS_VOLUME=<operator-harness-server-volume>
export EO_BACKEND_OPERATOR_CLIENT_TLS_VOLUME=<operator-client-volume>
export EO_OPERATOR_POLICY_FILE=<reviewed-policy-file>
export EO_OPERATOR_POLICY_SHA256=<reviewed-policy-sha256>
export EO_OPERATOR_ACTOR_ID=<issuer-id-pinned-by-policy>
export EO_AGENT_SUBJECT_ID=<subject-id-pinned-by-policy>
export EO_AGENT_SUBJECT_CERTIFICATE_SHA256=<runner-leaf-der-sha256>

docker --context rootless compose \
  -f compose.eo-gym-smoke.yaml \
  -f compose.agent-smoke.yaml \
  -f compose.agent-mtls.yaml \
  -f compose.backend-mtls.yaml \
  -f compose.operator-mtls.yaml \
  --profile agent --profile agent-mtls --profile operator-mtls config --quiet
```

The operator client volume contains the operator Harness server CA and exactly
one operator certificate/key pair. The operator Harness volume contains its
server certificate/key and the currently accepted operator-client CA bundle.
The committed schema-1.0 policy is only a narrow legacy example; an
operator-bound run needs a separately reviewed schema-1.2 policy.

## Manual CA rotation boundary

The accepted manual rotation used two independently signed operator client
certificates and this order:

1. add the new issuer ID and exact leaf pin to the reviewed policy;
2. serve an overlap CA bundle that verifies both old and new operator chains;
3. verify that both certificate-bound issuance paths succeed and produce
   separately audited sessions;
4. replace the server bundle with the new-only CA;
5. verify the old certificate fails at TLS while the new certificate still
   succeeds; then remove the obsolete issuer grant in the next policy revision.

Do not remove the old CA before the new identity succeeds, and do not treat CA
overlap alone as policy authorization. This runbook is manual test-fixture
rotation, not automatic renewal or revocation.

## A800 acceptance and limits

Ignored evidence is retained at `runtime/operator-mtls-20260920-01`. The summary
is `reports/acceptance-summary.json`, SHA-256
`d72a9b2ad711543eeecd70368ce2a0d5caff41c6bc440c4d5135a3932ed0474a`.
The reviewed policy SHA-256 is
`42b54da5bed91dcde6b8b17d594865fecd863e0f0908051185e1761cc74a2958`.

Observed results:

- the overlap bundle accepted both old and new operator identities, creating
  episodes `ep2-cf20d27f86a34b92a271432f4da627c2` and
  `ep2-4727c8a1c65a4c5084a102f0b197b464`; each used registry schema 1.3 and
  audit schema 1.1;
- after new-only cutover, the old operator failed TLS and the new operator
  succeeded; the gateway certificate could not access the operator Harness, and
  the operator certificate could not access the Agent backend;
- the runner completed a real scripted `eo_gym.crop`, idempotent retry and
  artifact metadata/content verification. Artifact
  `art-cc44d5f526f7279aa41c9b44b6d00f768965193b96be4ca97ca8b1b1e1b1abbf`
  contained 267,560 bytes whose SHA-256 matched the ID;
- the runner could not connect directly to the Agent backend, operator Harness
  or provider. Eight operator-only routes were denied by the Agent backend;
- full-stack recreation retained the episode, state version 1 and artifact
  identity. The final negative probes left the DB, both registries and both
  audit files byte-identical;
- runner ingress, gateway backend and operator control traffic each negotiated
  TLS 1.3 with `TLS_AES_256_GCM_SHA384`.

Focused tests pass 38/38, the full Python suite passes 312/312 with 34 existing
environment-dependent skips, and renderer tests pass 4/4. Compileall, 17 JSON
files, the five-layer Compose configuration, staged-payload checks and credential
scans also pass. Acceptance containers, three networks and seven TLS-copy
volumes were removed; ignored reports and mode-0600 host keys remain.

This acceptance is a scripted crop interaction, not Qwen or other model
reasoning. External enrollment, OCSP/CRL, automated renewal/revocation,
distributed abuse control, backup/recovery, off-host/WORM audit anchoring and
public multitenant attack acceptance remain incomplete.

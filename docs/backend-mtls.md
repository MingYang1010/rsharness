# Gateway-to-Harness mTLS

This opt-in profile replaces the Agent gateway's internal plaintext backend
connection with mutually authenticated TLS. It is additive: frozen V1/V2
operator APIs and the legacy internal-HTTP smoke profile remain unchanged.

```text
model runner -- mTLS + bearer --> Agent ingress --> Agent gateway -- mTLS --> Harness
```

## Fail-closed client contract

Set `EO_AGENT_REQUIRE_BACKEND_MTLS=1` only with an HTTPS backend origin and
three reviewed files:

- `EO_AGENT_BACKEND_CA_FILE`: CA used to verify the Harness server;
- `EO_AGENT_BACKEND_CERT_FILE`: gateway client certificate;
- `EO_AGENT_BACKEND_KEY_FILE`: matching owner-private client key.

The gateway refuses startup if the flag is not `0` or `1`, the origin is
plaintext, any file is missing/symlinked/empty/over 1 MiB, or the private key is
group/world accessible. The TLS context uses server authentication, hostname
verification and TLS 1.2 or newer, loads the gateway certificate chain, ignores
system proxy variables and refuses redirects. TLS/hostname failures become the
existing bounded `upstream_unavailable` response; raw certificate errors are
not exposed to the Agent.

The Harness process itself listens on TLS and requires a client certificate.
Use a dedicated client CA that signs only reviewed backend clients. The current
server verifies that CA rather than pinning one leaf certificate, so sharing the
CA with unrelated clients would expand authority.

## Compose profile and secret separation

`compose.backend-mtls.yaml` expects two external Docker volumes:

| Volume variable | Exact contents |
| --- | --- |
| `EO_BACKEND_HARNESS_TLS_VOLUME` | `harness.crt`, mode-0400 `harness.key`, `gateway-client-ca.crt` |
| `EO_BACKEND_GATEWAY_TLS_VOLUME` | `harness-server-ca.crt`, `gateway.crt`, mode-0400 `gateway.key` |

Do not mount either CA private key, the host TLS directory, runner keys or the
opposite service's private key. Neither service publishes a host port. The
Harness server certificate must contain `DNS:harness`; changing the Compose
service name requires a new reviewed certificate rather than disabling hostname
checking.

Validate the merged profile before starting it:

```sh
export EO_BACKEND_HARNESS_TLS_VOLUME=<reviewed-harness-server-volume>
export EO_BACKEND_GATEWAY_TLS_VOLUME=<reviewed-gateway-client-volume>
docker --context rootless compose \
  -f compose.eo-gym-smoke.yaml \
  -f compose.agent-smoke.yaml \
  -f compose.agent-mtls.yaml \
  -f compose.backend-mtls.yaml \
  --profile agent --profile agent-mtls config --quiet
```

The existing `issue-agent` helper is intentionally not given the gateway's
private key. For the current immutable benchmark workflow, either issue the
reviewed episode before switching Harness to this profile or add the separate
[operator mTLS profile](operator-mtls.md). The operator profile uses a distinct
Harness, client CA, certificate/key volume and policy pin; it never reuses the
gateway identity.

## A800 acceptance

The ignored evidence directory is
`runtime/backend-mtls-20260920-01`. The acceptance extended the existing
certificate-bound WorldCover session
`ep2-91587e85c72048d38cf234b0a3f022d0` and retained the same state version
before and after complete Harness/gateway/ingress recreation.

Observed results:

- the complete runner-to-Harness chain returned 200 with the approved runner
  certificate, bearer token and gateway backend certificate;
- direct Harness probes without a client certificate or with a certificate from
  an untrusted CA failed the TLS handshake;
- the approved backend client rejected the correct server certificate under a
  wrong DNS identity, and plaintext HTTP to the TLS port failed;
- the accepted backend negotiated TLS 1.3 with
  `TLS_AES_256_GCM_SHA384`; Harness server certificate SHA-256 was
  `3a85d366c3c2d22fa58dee414d65478d9c39ab104c0e118b3ee5adb9e813f231`
  and gateway client certificate SHA-256 was
  `c672ff57315ca8476b138f016ec45102ba82321fa553bce957984560633b6ea9`;
- Harness and gateway TLS volumes contained only their three allowlisted files.
  Private keys were mode 0400; services were read-only, capability-free and had
  no host ports. The runner network contained only the Agent ingress;
- negative probes and service recreation left registry, audit and episode DB
  hashes unchanged.

The three dedicated containers, three internal networks and three Docker volumes
containing the minimum TLS copies were removed after acceptance. The 0600 host
test material and hash reports remain under the ignored runtime directory.

Focused control/gateway tests pass 28/28, the full Python suite passes 307/307
with 34 existing environment-dependent skips, renderer tests pass 4/4, and
compileall, 12 config JSON files and the four-layer Compose config pass.

## Remaining security work

The test CAs are short-lived operator material, not external identity
infrastructure. A separate operator identity and manual old/new CA overlap plus
new-only cutover are accepted in [operator-mtls.md](operator-mtls.md). External
CA/IdP enrollment, OCSP/CRL, automated renewal/revocation, distributed abuse
control, backup/recovery, off-host/WORM audit anchoring and public multitenant
attack acceptance remain required before a production claim.

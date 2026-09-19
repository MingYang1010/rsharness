# Certificate-bound Agent mTLS ingress

This opt-in ingress requires both a valid client certificate and the existing
episode bearer token. It upgrades an internal governed session from an
operator-supplied subject string to a certificate-bound identity. It does not
change frozen V1/V2 operator APIs. The separate
[operator mTLS profile](operator-mtls.md) keeps those APIs off the Agent backend.

```text
model runner -- mTLS + bearer --> Nginx ingress -- private header --> Agent gateway -- optional mTLS --> Harness
```

## Identity contract

- `AgentIssuancePolicy@1.1.0` adds one unique SHA-256 DER certificate pin for
  every allowed subject. A mismatched or missing pin fails issuance. The original
  policy schema 1.0 remains the explicitly non-certificate internal mode.
- Certificate-bound sessions use `AgentCredentialRegistry@1.2.0`. Every record
  contains the subject certificate SHA-256 in addition to issuer, subject and
  policy pins. Schemas 1.0, 1.1 and 1.2 cannot be mixed in one non-empty registry.
- The issuer and lifecycle manager require
  `--subject-certificate-sha256`. Rotation/revocation must use the certificate
  pin already stored in the session and still present in the pinned policy.
- The gateway enables certificate checking only with
  `EO_AGENT_TRUSTED_MTLS_HEADER=1`. A certificate-bound session fails with 503 if
  that trusted ingress mode is absent. Missing, malformed or mismatched identity
  at request time fails before query/body/backend access.

## Ingress boundary

`config/nginx-agent-mtls.conf` requires a client certificate signed by the
mounted client CA and permits TLS 1.2/1.3 only. Nginx always overwrites
`X-EO-Client-Cert` with its own `$ssl_client_escaped_cert`. The gateway decodes
that bounded PEM value, converts it to DER with the standard TLS library, hashes
it with SHA-256 and compares it in constant time to the registry pin.

The header is trustworthy only when all of the following remain true:

1. the Agent gateway has no published port and its ingress-side network contains
   only the gateway and Nginx;
2. the model runner is attached only to the separate `agent-client` network;
3. no other service can write the trusted header to the gateway;
4. Nginx verifies the client CA and overwrites, rather than appends, the header.

The pinned runtime is
`nginxinc/nginx-unprivileged:1.28.0-alpine3.21@sha256:a6bd0e0995ab4723fb65068665f8016decb67dc6a6a32eddc415c7d1229cada6`
for linux/amd64. It runs as image user 101 with a read-only filesystem, all
capabilities dropped, no-new-privileges, bounded CPU/memory/PIDs and no host port
by default. TLS server material is supplied through a reviewed external Docker
volume containing only `server.crt`, `server.key` and `client-ca.crt`. Never mount
the CA private key, another runner's key or the whole operator TLS directory into
the ingress or model runner.

## Compose use

Create and populate the server-only volume outside Git, then set its exact name:

```sh
export EO_MTLS_TLS_VOLUME=<reviewed-server-only-volume>
docker --context rootless compose \
  -f compose.eo-gym-smoke.yaml \
  -f compose.agent-smoke.yaml \
  -f compose.agent-mtls.yaml \
  --profile agent --profile agent-mtls config --quiet
```

The overlay enables trusted mTLS-header mode on `agent-gateway`, adds the pinned
ingress and creates an internal `agent-client` network. The existing legacy
`agent-check` service is not an mTLS runner and must not be started as the model
client for this profile. A real runner receives only its own token, its own client
certificate/key and the public client CA; it does not join `agent-front` or the
backend network.

## A800 acceptance

The accepted evidence remains outside Git at
`runtime/agent-mtls-20260920-01`. It used a two-day test CA, a server certificate
for `agent-mtls-ingress`, two distinct client certificates and the real immutable
`worldcover-grounded-vqa@1.1.0` task. The certificate-bound registry is schema
1.2, episode `ep2-91587e85c72048d38cf234b0a3f022d0`, policy SHA-256
`b21282668d4b52e92135dc6553cd87636096a43f178ef152c77f2f7a017aa79a`
and audit head
`8dd7a51d5b5bb0d296fdef2b7547b880561e8e40b920d7574e60c7ead96cd360`.

Observed results:

- approved certificate plus the bound token returned the same episode before and
  after complete Harness/gateway/ingress recreation;
- no client certificate was rejected by Nginx before the gateway;
- another certificate signed by the same CA returned
  `client_identity_mismatch`, even when the caller supplied a forged trusted
  header;
- the approved certificate with a wrong token returned `unauthorized`;
- the runner network contained only Nginx and could not resolve `agent-gateway`;
- TLS negotiated TLS 1.3 with `TLS_AES_256_GCM_SHA384`; the verified server
  certificate SHA-256 was
  `92377e891b7db158cbb06f4fd88b3065fc591fbdc59a5f5582685f76884f7d72`;
- registry, audit and DB hashes were unchanged after negative probes. Registry,
  audit, bearer token and host private keys were mode 0600; plaintext token was
  absent from registry and audit.

Focused tests pass 27/27; the full Python suite passes 306/306 with 34 existing
environment-dependent skips. Nginx runtime startup, Compose config and renderer
4/4 pass. Earlier failed ingress starts are retained as diagnostics: a `--network
none` syntax check could not resolve the upstream, a host-UID container could not
read mode-0600 bind-mounted keys, and read-only Nginx required all temp paths
under `/tmp`. The accepted run uses a server-only volume owned by image user 101;
host key permissions were never widened.

## Remaining security work

The separate [backend mTLS profile](backend-mtls.md) now closes the optional
gateway-to-Harness plaintext hop. The [operator mTLS profile](operator-mtls.md)
adds a separate certificate-bound operator Harness and verifies a manual CA
overlap/new-only cutover. These milestones do not provide an external CA/IdP
enrollment service, OCSP/CRL policy, automated renewal/revocation, distributed
rate limiting, denial-of-service protection, backup/recovery, off-host audit
anchoring or public multitenant attack acceptance. Those remain required before
a production or public-deployment claim.

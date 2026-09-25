#!/usr/bin/env python3
"""Trusted operator issuance; never run inside the Agent sandbox."""
import argparse
import hashlib
import json
import os
import secrets
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.agent_credentials import (MAX_REGISTRY_BYTES, AgentCredentialRegistry,
                                   AgentSessionCredential, load_agent_registry,
                                   utc_now)
from app.control_plane import (MAX_AUDIT_BYTES, ControlEventInput,
                               append_control_event, authorize_issuance,
                               certificate_file_sha256,
                               load_issuance_policy)
from app.agent_gateway import (AgentBinding, MAX_JSON, build_backend_ssl_context,
                               build_binding)
from app.core.schemas import TaskManifest, V2EpisodeState
from app.core.tools.catalog import _is_public_image
from app.core.storage.quota import StorageQuota


def write_private(path: Path, content: bytes):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def replace_private(path: Path, content: bytes):
    descriptor, temporary_name = tempfile.mkstemp(prefix=".registry-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def operator_backend_verify(backend: str, ca_file: Path | None,
                            certificate_file: Path | None,
                            key_file: Path | None):
    url = urlsplit(backend)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in {"", "/"}
    ):
        raise SystemExit("backend must be an operator-configured HTTP origin")
    values = (ca_file, certificate_file, key_file)
    if any(value is not None for value in values) and not all(
        value is not None for value in values
    ):
        raise SystemExit("operator backend mTLS requires CA, certificate and key")
    if not any(value is not None for value in values):
        return True, None
    if url.scheme != "https":
        raise SystemExit("operator backend mTLS requires an HTTPS origin")
    try:
        return (
            build_backend_ssl_context(
                str(ca_file),
                str(certificate_file),
                str(key_file),
            ),
            certificate_file_sha256(certificate_file),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None


def publish_binding(runtime_root: Path, registry_path: Path, binding: AgentBinding,
                    ttl_seconds: int, governance: dict | None = None):
    quota = StorageQuota(runtime_root)
    with quota.hold(registry_path.parent, MAX_REGISTRY_BYTES + 64 * 1024,
                    "agent-credential-registry"):
        if registry_path.parent.is_symlink():
            raise SystemExit("registry directory must not be a symlink")
        registry_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if registry_path.exists():
            registry = load_agent_registry(registry_path)
        else:
            registry = AgentCredentialRegistry()
        if governance is not None and registry.sessions:
            expected_schema = (
                "1.3.0"
                if governance["actor_certificate_sha256"] is not None
                else "1.2.0"
                if governance["subject_certificate_sha256"] is not None
                else "1.1.0"
            )
            if registry.schema_version != expected_schema:
                raise SystemExit(
                    "legacy credential registry requires explicit reconciliation"
                )
        for item in registry.sessions:
            if item.binding.episode_id != binding.episode_id:
                continue
            if item.binding != binding:
                raise SystemExit("episode already has a different credential; use explicit rotation")
            if governance is not None and (
                item.issuer_id,
                item.subject_id,
                item.issuance_policy_id,
                item.issuance_policy_sha256,
                item.subject_certificate_sha256,
                item.issuer_certificate_sha256,
            ) != (
                governance["actor_id"],
                governance["subject_id"],
                governance["policy"].policy_id,
                governance["policy_sha256"],
                governance["subject_certificate_sha256"],
                governance["actor_certificate_sha256"],
            ):
                raise SystemExit("existing episode credential has different governance")
            return
        if governance is not None:
            current = utc_now()
            active = sum(
                item.status == "active"
                and item.expires_at > current
                and item.subject_id == governance["subject_id"]
                for item in registry.sessions
            )
            try:
                authorize_issuance(
                    governance["policy"],
                    actor_id=governance["actor_id"],
                    subject_id=governance["subject_id"],
                    task_id=binding.task.task_id,
                    task_version=binding.task.task_version,
                    ttl_seconds=ttl_seconds,
                    active_subject_sessions=active,
                    actor_certificate_sha256=governance[
                        "actor_certificate_sha256"
                    ],
                    subject_certificate_sha256=governance[
                        "subject_certificate_sha256"
                    ],
                    now=current,
                )
            except ValueError as error:
                raise SystemExit(str(error)) from None
        issued_at = utc_now()
        session = AgentSessionCredential(binding=binding, issued_at=issued_at,
                                         expires_at=issued_at + timedelta(seconds=ttl_seconds),
                                         issuer_id=governance["actor_id"] if governance else None,
                                         subject_id=governance["subject_id"] if governance else None,
                                         issuance_policy_id=(governance["policy"].policy_id
                                                             if governance else None),
                                         issuance_policy_sha256=(governance["policy_sha256"]
                                                                 if governance else None),
                                         subject_certificate_sha256=(
                                             governance["subject_certificate_sha256"]
                                             if governance else None
                                         ),
                                         issuer_certificate_sha256=(
                                             governance["actor_certificate_sha256"]
                                             if governance else None
                                         ))
        schema_version = (
            "1.3.0"
            if governance and governance["actor_certificate_sha256"] is not None
            else "1.2.0"
            if governance and governance["subject_certificate_sha256"] is not None
            else "1.1.0" if governance else registry.schema_version
        )
        updated = AgentCredentialRegistry(schema_version=schema_version,
                                          sessions=[*registry.sessions, session])
        content = updated.model_dump_json(indent=2).encode()
        if len(content) > MAX_REGISTRY_BYTES:
            raise SystemExit("credential registry exceeds limit")
        replace_private(registry_path, content)


def _governance(args, task_ref, registry_path: Path | None,
                *, actor_certificate_sha256: str | None = None,
                ignore_episode_id: str | None = None) -> dict | None:
    values = (
        args.issuance_policy,
        args.issuance_policy_sha256,
        args.actor_id,
        args.subject_id,
        args.audit_log,
    )
    requested = (
        *values,
        args.subject_certificate_sha256,
        actor_certificate_sha256,
    )
    if not any(value is not None for value in requested):
        return None
    if not all(value is not None for value in values) or registry_path is None:
        raise SystemExit("governed issuance requires policy checksum, identities, audit log and registry")
    runtime_root = (ROOT / "runtime").resolve()
    audit_path = args.audit_log.resolve()
    if (
        args.audit_log.is_symlink()
        or not audit_path.is_relative_to(runtime_root)
        or audit_path.parent == runtime_root
        or audit_path == registry_path
    ):
        raise SystemExit("control-plane audit requires a separate runtime directory")
    try:
        policy, policy_sha256 = load_issuance_policy(
            args.issuance_policy,
            args.issuance_policy_sha256,
        )
        if registry_path.exists():
            registry = load_agent_registry(registry_path)
            expected_schema = (
                "1.3.0"
                if actor_certificate_sha256 is not None
                else "1.2.0"
                if args.subject_certificate_sha256 is not None
                else "1.1.0"
            )
            if registry.sessions and registry.schema_version != expected_schema:
                raise ValueError("legacy registry requires reconciliation")
        else:
            registry = AgentCredentialRegistry(
                schema_version=(
                    "1.3.0"
                    if actor_certificate_sha256 is not None
                    else "1.2.0"
                    if args.subject_certificate_sha256 is not None
                    else "1.1.0"
                )
            )
        current = utc_now()
        active = sum(
            item.status == "active"
            and item.expires_at > current
            and item.subject_id == args.subject_id
            and item.binding.episode_id != ignore_episode_id
            for item in registry.sessions
        )
        authorize_issuance(
            policy,
            actor_id=args.actor_id,
            subject_id=args.subject_id,
            task_id=task_ref.task_id,
            task_version=task_ref.task_version,
            ttl_seconds=args.ttl_seconds,
            active_subject_sessions=active,
            actor_certificate_sha256=actor_certificate_sha256,
            subject_certificate_sha256=args.subject_certificate_sha256,
            now=current,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None
    return {
        "policy": policy,
        "policy_sha256": policy_sha256,
        "actor_id": args.actor_id,
        "actor_certificate_sha256": actor_certificate_sha256,
        "subject_id": args.subject_id,
        "subject_certificate_sha256": args.subject_certificate_sha256,
        "audit_path": audit_path,
    }


def _audit(governance: dict, event_type: str, *, operation_id: str, task_id: str,
           task_version: str, task_manifest_hash: str,
           episode_id: str | None = None, generation: int | None = None) -> None:
    runtime_root = (ROOT / "runtime").resolve()
    audit_path = governance["audit_path"]
    with StorageQuota(runtime_root).hold(
        audit_path.parent,
        MAX_AUDIT_BYTES + 64 * 1024,
        "agent-control-plane-audit",
    ):
        if audit_path.parent.is_symlink():
            raise SystemExit("audit directory must not be a symlink")
        audit_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        append_control_event(
            audit_path,
            ControlEventInput(
                event_type=event_type,
                operation_id=operation_id,
                actor_id=governance["actor_id"],
                subject_id=governance["subject_id"],
                policy_id=governance["policy"].policy_id,
                policy_sha256=governance["policy_sha256"],
                task_id=task_id,
                task_version=task_version,
                task_manifest_hash=task_manifest_hash,
                episode_id=episode_id,
                generation=generation,
                subject_certificate_sha256=governance[
                    "subject_certificate_sha256"
                ],
                actor_certificate_sha256=governance[
                    "actor_certificate_sha256"
                ],
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="http://harness:8000")
    parser.add_argument("--backend-ca-file", type=Path)
    parser.add_argument("--backend-certificate-file", type=Path)
    parser.add_argument("--backend-key-file", type=Path)
    parser.add_argument("--registry", type=Path,
                        help="shared hash-only credential registry under project runtime")
    parser.add_argument("--ttl-seconds", type=int, default=3600)
    parser.add_argument("--issuance-policy", type=Path)
    parser.add_argument("--issuance-policy-sha256")
    parser.add_argument("--actor-id")
    parser.add_argument("--subject-id")
    parser.add_argument("--subject-certificate-sha256")
    parser.add_argument("--audit-log", type=Path)
    parser.add_argument("--reviewed-public-task", action="store_true", required=True,
                        help="operator confirms prompt/schema/input IDs/descriptive fields are Agent-visible")
    args = parser.parse_args()
    backend_verify, actor_certificate_sha256 = operator_backend_verify(
        args.backend,
        args.backend_ca_file,
        args.backend_certificate_file,
        args.backend_key_file,
    )
    output = args.output.resolve()
    if args.output.is_symlink() or not output.is_relative_to((ROOT / "runtime").resolve()):
        raise SystemExit("session directory must be in project runtime")
    if not 60 <= args.ttl_seconds <= 7 * 24 * 60 * 60:
        raise SystemExit("credential TTL must be between 60 seconds and 7 days")
    registry_path = args.registry.resolve() if args.registry is not None else None
    if registry_path is not None:
        if args.registry.is_symlink() or not registry_path.is_relative_to((ROOT / "runtime").resolve()):
            raise SystemExit("credential registry must be in project runtime")
        if registry_path.parent == (ROOT / "runtime").resolve():
            raise SystemExit("credential registry requires a dedicated runtime directory")
    if args.job.stat().st_size > 1024 * 1024:
        raise SystemExit("job exceeds limit")
    job = json.loads(args.job.read_text())
    from app.core.schemas import TaskRef
    task_ref = TaskRef.model_validate(job["task_ref"])
    if output.exists():
        # Never repeat an ambiguous backend reset. Preserve pending receipt/token.
        if not (output / "binding.json").is_file() or not (output / "agent-token").is_file():
            raise SystemExit("incomplete issuance; operator must reconcile pending reset, do not automatically retry")
        if ((output / "binding.json").is_symlink() or (output / "agent-token").is_symlink()
                or (output / "binding.json").stat().st_size > 256 * 1024
                or (output / "agent-token").stat().st_size > 128):
            raise SystemExit("existing credential/binding is not a bounded regular file")
        binding = AgentBinding.model_validate_json((output / "binding.json").read_bytes())
        if (binding.task.task_id, binding.task.task_version) != (task_ref.task_id, task_ref.task_version):
            raise SystemExit("existing binding belongs to a different task; preserve it")
        if hashlib.sha256((output / "agent-token").read_bytes().strip()).hexdigest() != binding.token_sha256:
            raise SystemExit("existing credential/binding mismatch")
        governance = _governance(
            args,
            task_ref,
            registry_path,
            actor_certificate_sha256=actor_certificate_sha256,
            ignore_episode_id=binding.episode_id,
        )
        if registry_path is not None:
            publish_binding(
                (ROOT / "runtime").resolve(),
                registry_path,
                binding,
                args.ttl_seconds,
                governance,
            )
        if governance is not None:
            operation_id = "op-" + secrets.token_hex(16)
            registry = load_agent_registry(registry_path)
            session = next(
                item
                for item in registry.sessions
                if item.binding.episode_id == binding.episode_id
            )
            _audit(
                governance,
                "binding_reused",
                operation_id=operation_id,
                task_id=binding.task.task_id,
                task_version=binding.task.task_version,
                task_manifest_hash=binding.task_manifest_hash,
                episode_id=binding.episode_id,
                generation=session.generation,
            )
        print(json.dumps({"status": "existing_binding_preserved", "episode_id": binding.episode_id}))
        return
    with httpx.Client(base_url=args.backend, timeout=40, trust_env=False,
                      follow_redirects=False, verify=backend_verify) as client:
        def request(method, path, body=None):
            with client.stream(method, path, json=body) as response:
                if response.status_code not in {200, 201}:
                    raise RuntimeError("operator request failed; inspect backend privately")
                content = bytearray()
                for chunk in response.iter_bytes():
                    if len(content) + len(chunk) > MAX_JSON:
                        raise RuntimeError("operator response exceeds bound")
                    content.extend(chunk)
            return json.loads(content)["data"]
        ref = job["task_ref"]
        # Validate identifiers before interpolating the trusted task reference.
        manifest = TaskManifest.model_validate(request("GET", f"/v2/tasks/{task_ref.task_id}/versions/{task_ref.task_version}")["manifest"])
        if any(not _is_public_image(a) for a in manifest.assets if a.asset_id in manifest.task.inputs):
            raise SystemExit("task contains private/non-image inputs; no episode created")
        capabilities = request("GET", "/v2/capabilities")
        governance = _governance(
            args,
            task_ref,
            registry_path,
            actor_certificate_sha256=actor_certificate_sha256,
        )
        if governance is not None:
            operation_id = "op-" + secrets.token_hex(16)
            _audit(
                governance,
                "issuance_started",
                operation_id=operation_id,
                task_id=task_ref.task_id,
                task_version=task_ref.task_version,
                task_manifest_hash=manifest.task_manifest_hash,
            )
        with StorageQuota(ROOT / "runtime").hold(output, 2 * 1024 * 1024, "agent-session-issuance"):
            output.mkdir(mode=0o700)
            token = secrets.token_hex(32)
            write_private(output / "agent-token", (token + "\n").encode())
            write_private(output / "reset-pending.json", json.dumps({"task_ref": ref,
                "task_manifest_hash": manifest.task_manifest_hash, "public_task_reviewed": True,
                "warning": "if binding is absent, reset outcome is unknown; no automatic retry"}).encode())
            reset = request("POST", "/v2/reset", {"task_ref": ref, "seed": job.get("seed", 42)})
            binding = build_binding(manifest, V2EpisodeState.model_validate(reset["state"]), capabilities,
                                    hashlib.sha256(token.encode()).hexdigest())
            content = binding.model_dump_json(indent=2).encode()
            if len(content) > 256 * 1024:
                raise RuntimeError("public binding exceeds gateway limit")
            write_private(output / "binding.json", content)
            if registry_path is not None:
                publish_binding(
                    (ROOT / "runtime").resolve(),
                    registry_path,
                    binding,
                    args.ttl_seconds,
                    governance,
                )
            if governance is not None:
                _audit(
                    governance,
                    "issuance_completed",
                    operation_id=operation_id,
                    task_id=binding.task.task_id,
                    task_version=binding.task.task_version,
                    task_manifest_hash=binding.task_manifest_hash,
                    episode_id=binding.episode_id,
                    generation=1,
                )
            print(json.dumps({"status": "issued", "episode_id": binding.episode_id,
                              "task_manifest_hash": binding.task_manifest_hash}))


if __name__ == "__main__":
    main()

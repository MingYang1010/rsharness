#!/usr/bin/env python3
"""Trusted operator revocation and rotation for Agent gateway credentials."""
import argparse
import hashlib
import os
import secrets
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.agent_credentials import (MAX_REGISTRY_BYTES, AgentCredentialRegistry,
                                   load_agent_registry, utc_now)
from app.control_plane import (MAX_AUDIT_BYTES, ControlEventInput,
                               append_control_event, authorize_management,
                               certificate_file_sha256,
                               load_issuance_policy)
from app.v2.storage.quota import StorageQuota


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


def write_new_token(runtime_root: Path, path: Path, token: str):
    if path.is_symlink() or not path.is_relative_to(runtime_root) or path.parent == runtime_root:
        raise SystemExit("token output must use a dedicated directory in project runtime")
    with StorageQuota(runtime_root).hold(path.parent, 64 * 1024, "agent-token-rotation"):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((token + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())


def audit_event(runtime_root: Path, audit_path: Path, governance: dict,
                event_type: str, session, *, completed: bool):
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
                operation_id=governance["operation_id"],
                actor_id=governance["actor_id"],
                subject_id=session.subject_id,
                policy_id=session.issuance_policy_id,
                policy_sha256=session.issuance_policy_sha256,
                task_id=session.binding.task.task_id,
                task_version=session.binding.task.task_version,
                task_manifest_hash=session.binding.task_manifest_hash,
                episode_id=session.binding.episode_id if completed else None,
                generation=session.generation if completed else None,
                subject_certificate_sha256=session.subject_certificate_sha256,
                actor_certificate_sha256=governance[
                    "actor_certificate_sha256"
                ],
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--episode-id", required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--revoke", action="store_true")
    operation.add_argument("--rotate", action="store_true")
    parser.add_argument("--token-output", type=Path)
    parser.add_argument("--ttl-seconds", type=int, default=3600)
    parser.add_argument("--issuance-policy", type=Path)
    parser.add_argument("--issuance-policy-sha256")
    parser.add_argument("--actor-id")
    parser.add_argument("--subject-id")
    parser.add_argument("--subject-certificate-sha256")
    parser.add_argument("--actor-certificate-file", type=Path)
    parser.add_argument("--audit-log", type=Path)
    args = parser.parse_args()

    runtime_root = (ROOT / "runtime").resolve()
    registry_path = args.registry.resolve()
    if (args.registry.is_symlink() or not registry_path.is_relative_to(runtime_root)
            or registry_path.parent == runtime_root):
        raise SystemExit("credential registry must be in a dedicated project runtime directory")
    if not 60 <= args.ttl_seconds <= 7 * 24 * 60 * 60:
        raise SystemExit("credential TTL must be between 60 seconds and 7 days")
    if args.rotate != (args.token_output is not None):
        raise SystemExit("rotation requires --token-output; revocation forbids it")

    token_path = args.token_output.resolve() if args.token_output is not None else None
    replacement = None
    governance = None
    try:
        actor_certificate_sha256 = (
            certificate_file_sha256(args.actor_certificate_file)
            if args.actor_certificate_file is not None
            else None
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None
    with StorageQuota(runtime_root).hold(registry_path.parent,
                                         MAX_REGISTRY_BYTES + 64 * 1024,
                                         "agent-credential-registry"):
        registry = load_agent_registry(registry_path)
        matches = [index for index, item in enumerate(registry.sessions)
                   if item.binding.episode_id == args.episode_id]
        if len(matches) != 1:
            raise SystemExit("episode credential not found")
        index = matches[0]
        current = registry.sessions[index]
        if args.revoke and current.status == "revoked":
            print('{"status":"already_revoked"}')
            return
        governed_arguments = (
            args.issuance_policy,
            args.issuance_policy_sha256,
            args.actor_id,
            args.subject_id,
            args.audit_log,
        )
        if current.issuer_id is not None:
            if not all(value is not None for value in governed_arguments):
                raise SystemExit("governed credential management requires policy, identities and audit log")
            audit_path = args.audit_log.resolve()
            if (
                args.audit_log.is_symlink()
                or not audit_path.is_relative_to(runtime_root)
                or audit_path.parent == runtime_root
                or audit_path == registry_path
                or audit_path == token_path
                or args.subject_id != current.subject_id
                or args.subject_certificate_sha256
                != current.subject_certificate_sha256
                or actor_certificate_sha256
                != current.issuer_certificate_sha256
            ):
                raise SystemExit("governed management scope or subject is invalid")
            try:
                policy, policy_sha256 = load_issuance_policy(
                    args.issuance_policy,
                    args.issuance_policy_sha256,
                )
                if (
                    policy.policy_id != current.issuance_policy_id
                    or policy_sha256 != current.issuance_policy_sha256
                ):
                    raise ValueError("credential issuance policy pin changed")
                authorize_management(
                    policy,
                    actor_id=args.actor_id,
                    subject_id=args.subject_id,
                    task_id=current.binding.task.task_id,
                    task_version=current.binding.task.task_version,
                    ttl_seconds=args.ttl_seconds if args.rotate else None,
                    rotate=args.rotate,
                    actor_certificate_sha256=actor_certificate_sha256,
                    subject_certificate_sha256=args.subject_certificate_sha256,
                )
            except ValueError as error:
                raise SystemExit(str(error)) from None
            governance = {
                "actor_id": args.actor_id,
                "actor_certificate_sha256": actor_certificate_sha256,
                "audit_path": audit_path,
                "operation_id": "op-" + secrets.token_hex(16),
            }
            audit_event(
                runtime_root,
                audit_path,
                governance,
                "rotation_started" if args.rotate else "revocation_started",
                current,
                completed=False,
            )
        elif any(
            value is not None
            for value in (
                *governed_arguments,
                args.subject_certificate_sha256,
                args.actor_certificate_file,
            )
        ):
            raise SystemExit("legacy credential cannot use governed management arguments")
        token = secrets.token_hex(32) if args.rotate else None
        if token is not None:
            write_new_token(runtime_root, token_path, token)
        timestamp = utc_now()
        if args.revoke:
            replacement = current.model_copy(update={"status": "revoked", "revoked_at": timestamp})
            status = "revoked"
        else:
            binding = current.binding.model_copy(update={
                "token_sha256": hashlib.sha256(token.encode()).hexdigest()})
            replacement = current.model_copy(update={
                "binding": binding, "issued_at": timestamp,
                "expires_at": timestamp + timedelta(seconds=args.ttl_seconds),
                "status": "active", "revoked_at": None,
                "generation": current.generation + 1})
            status = "rotated"
        sessions = list(registry.sessions)
        sessions[index] = replacement
        updated = AgentCredentialRegistry(schema_version=registry.schema_version,
                                          sessions=sessions)
        content = updated.model_dump_json(indent=2).encode()
        if len(content) > MAX_REGISTRY_BYTES:
            raise SystemExit("credential registry exceeds limit")
        replace_private(registry_path, content)
    if governance is not None:
        audit_event(
            runtime_root,
            governance["audit_path"],
            governance,
            "rotation_completed" if args.rotate else "revocation_completed",
            replacement,
            completed=True,
        )
    print('{{"status":"{}","episode_id":"{}","generation":{}}}'.format(
        status, replacement.binding.episode_id, replacement.generation))


if __name__ == "__main__":
    main()

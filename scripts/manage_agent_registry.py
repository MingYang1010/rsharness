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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--episode-id", required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--revoke", action="store_true")
    operation.add_argument("--rotate", action="store_true")
    parser.add_argument("--token-output", type=Path)
    parser.add_argument("--ttl-seconds", type=int, default=3600)
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

    token = secrets.token_hex(32) if args.rotate else None
    token_path = args.token_output.resolve() if args.token_output is not None else None
    if token is not None:
        write_new_token(runtime_root, token_path, token)

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
        timestamp = utc_now()
        if args.revoke:
            if current.status == "revoked":
                print('{"status":"already_revoked"}')
                return
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
    print('{{"status":"{}","episode_id":"{}","generation":{}}}'.format(
        status, replacement.binding.episode_id, replacement.generation))


if __name__ == "__main__":
    main()

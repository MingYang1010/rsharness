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

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.agent_credentials import (MAX_REGISTRY_BYTES, AgentCredentialRegistry,
                                   AgentSessionCredential, load_agent_registry,
                                   utc_now)
from app.agent_gateway import AgentBinding, MAX_JSON, build_binding
from app.v2.schemas import TaskManifest, V2EpisodeState
from app.v2.tools.catalog import _is_public_image
from app.v2.storage.quota import StorageQuota


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


def publish_binding(runtime_root: Path, registry_path: Path, binding: AgentBinding,
                    ttl_seconds: int):
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
        for item in registry.sessions:
            if item.binding.episode_id != binding.episode_id:
                continue
            if item.binding != binding:
                raise SystemExit("episode already has a different credential; use explicit rotation")
            return
        issued_at = utc_now()
        session = AgentSessionCredential(binding=binding, issued_at=issued_at,
                                         expires_at=issued_at + timedelta(seconds=ttl_seconds))
        updated = AgentCredentialRegistry(schema_version=registry.schema_version,
                                          sessions=[*registry.sessions, session])
        content = updated.model_dump_json(indent=2).encode()
        if len(content) > MAX_REGISTRY_BYTES:
            raise SystemExit("credential registry exceeds limit")
        replace_private(registry_path, content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="http://harness:8000")
    parser.add_argument("--registry", type=Path,
                        help="shared hash-only credential registry under project runtime")
    parser.add_argument("--ttl-seconds", type=int, default=3600)
    parser.add_argument("--reviewed-public-task", action="store_true", required=True,
                        help="operator confirms prompt/schema/input IDs/descriptive fields are Agent-visible")
    args = parser.parse_args()
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
    from app.v2.schemas import TaskRef
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
        if registry_path is not None:
            publish_binding((ROOT / "runtime").resolve(), registry_path, binding, args.ttl_seconds)
        print(json.dumps({"status": "existing_binding_preserved", "episode_id": binding.episode_id}))
        return
    with httpx.Client(base_url=args.backend, timeout=40, trust_env=False, follow_redirects=False) as client:
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
                publish_binding((ROOT / "runtime").resolve(), registry_path, binding, args.ttl_seconds)
            print(json.dumps({"status": "issued", "episode_id": binding.episode_id,
                              "task_manifest_hash": binding.task_manifest_hash}))


if __name__ == "__main__":
    main()

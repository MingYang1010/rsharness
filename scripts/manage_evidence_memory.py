#!/usr/bin/env python3
"""Publish, invalidate or inspect governed cross-task geographic evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))

from app.control_plane import certificate_file_sha256
from app.core.evidence_memory import (
    MAX_STORE_BYTES,
    EvidenceMemoryStore,
    build_evidence_memory_record,
    load_evidence_memory_policy,
    load_source_evidence,
)
from app.core.storage.quota import StorageQuota


def runtime_path(value: Path, label: str) -> Path:
    runtime_root = (ROOT / "runtime").resolve()
    resolved = value.resolve()
    if value.is_symlink() or not resolved.is_relative_to(runtime_root) or resolved.parent == runtime_root:
        raise SystemExit(f"{label} must be in a dedicated project runtime directory")
    return resolved


def actor_pin(certificate_file: Path) -> str:
    try:
        return certificate_file_sha256(certificate_file)
    except ValueError as error:
        raise SystemExit(str(error)) from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--policy-sha256", required=True)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    publish = subparsers.add_parser("publish")
    publish.add_argument("--actor-id", required=True)
    publish.add_argument("--actor-certificate-file", required=True, type=Path)
    publish.add_argument("--episode-database", required=True, type=Path)
    publish.add_argument("--tasks", type=Path, default=ROOT / "tasks")
    publish.add_argument("--episode-id", required=True)
    publish.add_argument("--evidence-id", required=True)
    publish.add_argument("--object-type", required=True)
    publish.add_argument("--public-summary", required=True)
    publish.add_argument("--ttl-seconds", required=True, type=int)

    invalidate = subparsers.add_parser("invalidate")
    invalidate.add_argument("--actor-id", required=True)
    invalidate.add_argument("--actor-certificate-file", required=True, type=Path)
    invalidate.add_argument("--memory-id", required=True)
    invalidate.add_argument("--reason", required=True)
    invalidate.add_argument("--replacement-memory-id")

    subparsers.add_parser("snapshot")
    args = parser.parse_args()

    database = runtime_path(args.database, "evidence memory database")
    if args.operation != "publish" and not database.exists():
        raise SystemExit("evidence memory database does not exist")
    try:
        policy, policy_sha256 = load_evidence_memory_policy(
            args.policy, args.policy_sha256
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None
    runtime_root = (ROOT / "runtime").resolve()
    with StorageQuota(runtime_root).hold(
        database.parent,
        MAX_STORE_BYTES + 128 * 1024,
        "cross-task-evidence-memory",
    ):
        store = EvidenceMemoryStore(database)
        try:
            if args.operation == "publish":
                episode_database = runtime_path(
                    args.episode_database, "source episode database"
                )
                manifest, state, evaluation, evidence, source = load_source_evidence(
                    episode_database,
                    args.tasks.resolve(),
                    args.episode_id,
                    args.evidence_id,
                )
                record = build_evidence_memory_record(
                    policy=policy,
                    policy_sha256=policy_sha256,
                    manifest=manifest,
                    state=state,
                    evaluation=evaluation,
                    evidence=evidence,
                    source=source,
                    object_type=args.object_type,
                    public_summary=args.public_summary,
                    ttl_seconds=args.ttl_seconds,
                )
                snapshot = store.publish(
                    policy,
                    policy_sha256,
                    record,
                    actor_id=args.actor_id,
                    actor_certificate_sha256=actor_pin(
                        args.actor_certificate_file
                    ),
                )
                output = {
                    "status": "published",
                    "memory_id": record.memory_id,
                    "scope_id": snapshot.scope_id,
                    "snapshot_sequence": snapshot.sequence,
                    "snapshot_sha256": snapshot.snapshot_sha256,
                }
            elif args.operation == "invalidate":
                snapshot = store.invalidate(
                    policy,
                    policy_sha256,
                    args.memory_id,
                    args.reason,
                    actor_id=args.actor_id,
                    actor_certificate_sha256=actor_pin(
                        args.actor_certificate_file
                    ),
                    replacement_memory_id=args.replacement_memory_id,
                )
                output = {
                    "status": "invalidated",
                    "memory_id": args.memory_id,
                    "scope_id": snapshot.scope_id,
                    "snapshot_sequence": snapshot.sequence,
                    "snapshot_sha256": snapshot.snapshot_sha256,
                }
            else:
                snapshot = store.snapshot(policy.scope_id)
                output = {
                    "status": "verified",
                    "scope_id": snapshot.scope_id,
                    "snapshot_sequence": snapshot.sequence,
                    "snapshot_sha256": snapshot.snapshot_sha256,
                    "record_count": len(snapshot.records),
                    "active_record_count": len(snapshot.active_records),
                }
        except (ValidationError, ValueError, KeyError) as error:
            raise SystemExit(str(error)) from None
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()

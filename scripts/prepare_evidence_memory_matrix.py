#!/usr/bin/env python3
"""Freeze controlled multi-record evidence-memory benchmark cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))

from app.core.evidence_memory import (
    EvidenceMemoryRecord,
    EvidenceMemoryStore,
    load_evidence_memory_policy,
    validate_record_identity,
)
from app.core.events import canonical_json, sha256_json

CASES = {
    "correct-only": ["correct"],
    "conflict": ["correct", "conflicting"],
    "neighbor": ["correct", "neighbor"],
    "expired-only": ["expired"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    content = (canonical_json(value) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)


def _derive(record: EvidenceMemoryRecord, **changes: Any) -> EvidenceMemoryRecord:
    body = record.model_dump(mode="json")
    source = dict(body["source"])
    if "bbox" in changes:
        source["selector"] = {**source["selector"], "bbox": changes["bbox"]}
    body["source"] = source
    body.update(changes)
    body["provenance_sha256"] = sha256_json(source)
    body.pop("memory_id", None)
    provisional = EvidenceMemoryRecord.model_validate({**body, "memory_id": "mem-" + "0" * 64})
    identity = provisional.model_dump(mode="json", exclude={"memory_id", "provenance_sha256"})
    provenance = sha256_json(provisional.source.model_dump(mode="json"))
    identity["provenance_sha256"] = provenance
    value = EvidenceMemoryRecord.model_validate({
        **body,
        "provenance_sha256": provenance,
        "memory_id": "mem-" + sha256_json(identity),
    })
    validate_record_identity(value)
    return value


def _load_base(path: Path) -> EvidenceMemoryRecord:
    store = EvidenceMemoryStore(path)
    # Sequence 1 predates the acceptance invalidation and preserves the reviewed source record.
    snapshot = store.snapshot("worldcover-evidence-memory-acceptance", 1)
    if len(snapshot.active_records) != 1:
        raise ValueError("source snapshot must contain exactly one active reviewed record")
    return snapshot.active_records[0]


def _publish_records(
    output: Path,
    source_memory: Path,
    policy_path: Path,
    policy_sha256: str,
    actor_id: str,
    certificate_sha256: str,
) -> tuple[EvidenceMemoryStore, dict[str, EvidenceMemoryRecord]]:
    policy, _ = load_evidence_memory_policy(policy_path, policy_sha256)
    base = _load_base(source_memory)
    correct = _derive(base)
    conflict = _derive(
        base,
        public_summary="Review conflict: cropland was dominant in the fixed WorldCover AOI.",
    )
    neighbor_bbox = {"west": 121.56, "south": 31.2, "east": 121.66, "north": 31.3}
    neighbor = _derive(
        base,
        bbox=neighbor_bbox,
        public_summary="Built-up was dominant in a neighboring, non-overlapping WorldCover AOI.",
    )
    expired = _derive(
        base,
        available_at=base.available_at,
        expires_at="2026-09-20T01:20:13.000000Z",
        public_summary="Expired built-up evidence that must not be returned.",
    )
    records = {
        "correct": correct,
        "conflicting": conflict,
        "neighbor": neighbor,
        "expired": expired,
    }
    memory_path = output / "memory" / "events.sqlite3"
    store = EvidenceMemoryStore(memory_path)
    for record in records.values():
        store.publish(
            policy,
            policy_sha256,
            record,
            actor_id=actor_id,
            actor_certificate_sha256=certificate_sha256,
            now=record.available_at,
        )
    return store, records


def _task_files(source: Path, case: str, binding: dict, expected_memory_id: str) -> dict[str, dict]:
    values = {}
    for treatment, directory in (("with-memory", "with-memory"), ("without-memory", "without-memory")):
        task = json.loads((source / directory / "task.json").read_text())
        scenario = json.loads((source / directory / "scenario.json").read_text())
        evaluator = json.loads((source / directory / "evaluator.json").read_text())
        assets = json.loads((source / directory / "assets.json").read_text())
        task["metadata"]["benchmark_case"] = case
        if treatment == "with-memory":
            task["metadata"]["evidence_memory"] = binding
        else:
            task["metadata"].pop("evidence_memory", None)
        evaluator["config"]["expected_memory_id"] = expected_memory_id
        values[directory] = {
            "task.json": task,
            "scenario.json": scenario,
            "evaluator.json": evaluator,
            "assets.json": assets,
        }
    return values


def prepare(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise ValueError("matrix output already exists")
    policy_sha256 = _sha256(args.policy)
    store, records = _publish_records(
        args.output,
        args.source_memory,
        args.policy,
        policy_sha256,
        args.actor_id,
        args.actor_certificate_sha256,
    )
    snapshot = store.snapshot("worldcover-evidence-memory-acceptance")
    source_binding = json.loads(
        (args.source_tasks / "with-memory" / "task.json").read_text()
    )["metadata"]["evidence_memory"]
    binding = {
        **source_binding,
        "snapshot_sequence": snapshot.sequence,
        "snapshot_sha256": snapshot.snapshot_sha256,
    }
    variants = {}
    for case, names in CASES.items():
        directory = args.output / "tasks" / case
        values = _task_files(
            args.source_tasks,
            case,
            binding,
            records["correct"].memory_id,
        )
        for treatment, files in values.items():
            target = directory / treatment
            for name, value in files.items():
                _write(target / name, value)
        variants[case] = {
            "records": names,
            "task_directory": str(directory),
            "snapshot_sequence": snapshot.sequence,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "memory_ids": {name: records[name].memory_id for name in names},
        }
    report = {
        "schema_version": "evidence-memory-matrix-v1",
        "cases": variants,
        "record_count": len(records),
        "snapshot_sequence": snapshot.sequence,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "policy_sha256": policy_sha256,
    }
    _write(args.output / "matrix-manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-memory", required=True, type=Path)
    parser.add_argument("--source-tasks", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--actor-id", default="trusted-memory-curator")
    parser.add_argument("--actor-certificate-sha256", required=True)
    args = parser.parse_args()
    print(canonical_json(prepare(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

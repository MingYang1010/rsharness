#!/usr/bin/env python3
"""Freeze controlled multi-record evidence-memory benchmark cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
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

TASK_IDS = {
    "correct-only": "evidence-memory-correct-only",
    "conflict": "evidence-memory-conflict",
    "neighbor": "evidence-memory-neighbor",
    "expired-only": "evidence-memory-expired-only",
}
MAX_WALL_TIME_MS = 120000


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


def _one_second_after(value: str) -> str:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00")) + timedelta(seconds=1)
    return timestamp.isoformat().replace("+00:00", "Z")


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


def _load_base(path: Path, scope_id: str) -> EvidenceMemoryRecord:
    store = EvidenceMemoryStore(path)
    # Sequence 1 predates the acceptance invalidation and preserves the reviewed source record.
    snapshot = store.snapshot(scope_id, 1)
    if len(snapshot.active_records) != 1:
        raise ValueError("source snapshot must contain exactly one active reviewed record")
    return snapshot.active_records[0]


def _matrix_policy(
    output: Path, policy_path: Path
) -> tuple[dict, Path, str]:
    policy = json.loads(policy_path.read_text())
    granted = {
        (grant["task_id"], version)
        for grant in policy["reader_grants"]
        for version in grant["task_versions"]
    }
    for task_id in TASK_IDS.values():
        granted.add((task_id, "1.0.0"))
    policy["reader_grants"] = [
        {"task_id": task_id, "task_versions": [version]}
        for task_id, version in sorted(granted)
    ]
    target = output / "policy" / "evidence-memory-policy.json"
    _write(target, policy)
    return policy, target, _sha256(target)


def _publish_case_records(
    output: Path,
    source_memory: Path,
    policy_path: Path,
    source_policy_sha256: str,
    matrix_policy_sha256: str,
    scope_id: str,
    actor_id: str,
    certificate_sha256: str,
) -> dict[str, tuple[EvidenceMemoryStore, dict[str, EvidenceMemoryRecord]]]:
    policy, _ = load_evidence_memory_policy(policy_path, source_policy_sha256)
    base = _load_base(source_memory, scope_id)
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
        expires_at=_one_second_after(base.available_at),
        public_summary="Expired built-up evidence that must not be returned.",
    )
    records = {
        "correct": correct,
        "conflicting": conflict,
        "neighbor": neighbor,
        "expired": expired,
    }
    derived = {name: None for name in records}
    cases = {}
    for case, names in CASES.items():
        store = EvidenceMemoryStore(
            output / "matrix" / "cases" / case / "memory" / "events.sqlite3"
        )
        for name in names:
            record = derived[name] or _derive(
                records[name].model_copy(
                    update={"policy_sha256": matrix_policy_sha256}
                )
            )
            derived[name] = record
            store.publish(
                policy,
                matrix_policy_sha256,
                record,
                actor_id=actor_id,
                actor_certificate_sha256=certificate_sha256,
                now=record.available_at,
            )
        cases[case] = (
            store,
            {name: derived[name] for name in CASES[case]},
        )
    return cases


def _task_files(
    source: Path,
    case: str,
    task_id: str,
    binding: dict,
    expected_memory_id: str,
    expected_snapshot_sha256: str,
) -> dict[str, dict]:
    values = {}
    for treatment, directory in (("with-memory", "with-memory"), ("without-memory", "without-memory")):
        task = json.loads((source / directory / "task.json").read_text())
        scenario = json.loads((source / directory / "scenario.json").read_text())
        evaluator = json.loads((source / directory / "evaluator.json").read_text())
        assets = json.loads((source / directory / "assets.json").read_text())
        task["task_id"] = task_id
        task["budget"]["max_wall_time_ms"] = MAX_WALL_TIME_MS
        task["metadata"]["benchmark_case"] = case
        if treatment == "with-memory":
            task["metadata"]["evidence_memory"] = binding
        else:
            task["metadata"].pop("evidence_memory", None)
        evaluator["config"]["expected_memory_id"] = expected_memory_id
        evaluator["config"]["expected_snapshot_sha256"] = expected_snapshot_sha256
        evaluator["config"]["efficiency"]["wall_time_soft_limit_ms"] = MAX_WALL_TIME_MS
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
    source_binding = json.loads(
        (args.source_tasks / "with-memory" / "task.json").read_text()
    )["metadata"]["evidence_memory"]
    source_policy_sha256 = _sha256(args.policy)
    _, matrix_policy_path, policy_sha256 = _matrix_policy(
        args.output,
        args.policy,
    )
    cases = _publish_case_records(
        args.output,
        args.source_memory,
        args.policy,
        source_policy_sha256,
        policy_sha256,
        source_binding["scope_id"],
        args.actor_id,
        args.actor_certificate_sha256,
    )
    variants = {}
    for case, (store, records) in cases.items():
        snapshot = store.snapshot(source_binding["scope_id"])
        binding = {
            **source_binding,
            "policy_sha256": policy_sha256,
            "snapshot_sequence": snapshot.sequence,
            "snapshot_sha256": snapshot.snapshot_sha256,
        }
        task_id = TASK_IDS[case]
        values = _task_files(
            args.source_tasks,
            case,
            task_id,
            binding,
            records["correct"].memory_id
            if "correct" in CASES[case]
            else records[CASES[case][0]].memory_id,
            snapshot.snapshot_sha256,
        )
        for treatment, files in values.items():
            directory = args.output / "tasks" / f"{case}-{treatment}"
            for name, value in files.items():
                _write(directory / name, value)
            job = {
                "sample_id": f"{case}-{treatment}",
                "seed": 42,
                "task_ref": {
                    "task_id": task_id,
                    "task_version": files["task.json"]["task_version"],
                },
            }
            _write(args.output / "benchmark-agent" / f"{case}-{treatment}.json", job)
        variants[case] = {
            "records": CASES[case],
            "memory_store": str(
                args.output / "matrix" / "cases" / case / "memory" / "events.sqlite3"
            ),
            "task_directories": [
                str(args.output / "tasks" / f"{case}-{treatment}")
                for treatment in ("with-memory", "without-memory")
            ],
            "jobs": {
                treatment: str(
                    args.output / "benchmark-agent" / f"{case}-{treatment}.json"
                )
                for treatment in ("with-memory", "without-memory")
            },
            "snapshot_sequence": snapshot.sequence,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "memory_ids": {name: records[name].memory_id for name in CASES[case]},
        }
    report = {
        "schema_version": "evidence-memory-matrix-v1",
        "policy_path": str(matrix_policy_path),
        "cases": variants,
        "record_types": 4,
        "task_count": 8,
        "job_count": 8,
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

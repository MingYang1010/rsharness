#!/usr/bin/env python3
"""Freeze matched with-memory and without-memory task packs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))

from app.v2.capabilities import TaskRegistry
from app.v2.evidence_memory import (
    EvidenceMemoryBinding,
    EvidenceMemoryStore,
    MemorySearchArguments,
    load_evidence_memory_policy,
)
from app.v2.events import canonical_json, sha256_json
from app.v2.schemas import (
    BudgetSpec,
    Identifier,
    NonEmptyText,
    SemanticVersion,
    TaskManifest,
    TaskRef,
    UtcTimestamp,
    V2RequestModel,
)


DEFAULT_CONFIG = ROOT / "config" / "evidence-memory-benchmark-v1.json"
MAX_CONFIG_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
METRICS = [
    "task.accuracy",
    "evidence.memory_faithfulness",
    "process.efficiency",
]
WEIGHTS = {
    "task.accuracy": 0.6,
    "evidence.memory_faithfulness": 0.3,
    "process.efficiency": 0.1,
}


class BenchmarkConfig(V2RequestModel):
    schema_version: Literal["1.0.0"]
    benchmark_id: Identifier
    task_id: Identifier
    with_memory_version: SemanticVersion
    without_memory_version: SemanticVersion
    source_task_ref: TaskRef
    policy_id: Identifier
    scope_id: Identifier
    as_of: UtcTimestamp
    prompt: NonEmptyText
    expected_label: Identifier
    labels: list[Identifier] = Field(min_length=2, max_length=64)
    query: MemorySearchArguments
    budget: BudgetSpec

    @model_validator(mode="after")
    def valid_pair(self) -> "BenchmarkConfig":
        if self.with_memory_version == self.without_memory_version:
            raise ValueError("benchmark task versions must be distinct")
        if self.expected_label not in self.labels or len(self.labels) != len(set(self.labels)):
            raise ValueError("benchmark labels must be unique and contain the expected label")
        if self.budget.max_tool_calls < 1 or self.budget.max_input_bytes < 1:
            raise ValueError("benchmark budget must permit one bounded memory query")
        return self


BenchmarkConfig.model_rebuild(_types_namespace=globals())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> tuple[BenchmarkConfig, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("benchmark config must be a regular file")
    details = path.stat()
    if details.st_size <= 0 or details.st_size > MAX_CONFIG_BYTES or details.st_mode & 0o002:
        raise ValueError("benchmark config is unbounded or world-writable")
    content = path.read_bytes()
    return BenchmarkConfig.model_validate_json(content), hashlib.sha256(content).hexdigest()


def _write_json(path: Path, value: object) -> None:
    content = (canonical_json(value) + "\n").encode("utf-8")
    if len(content) > MAX_OUTPUT_BYTES:
        raise ValueError("generated benchmark metadata exceeds the allowed size")
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)


def _input_assets(registry: TaskRegistry, reference: TaskRef) -> list[dict]:
    manifest = registry.get(reference.task_id, reference.task_version)
    inputs = set(manifest.task.inputs)
    assets = [
        asset.model_dump(mode="json")
        for asset in manifest.assets
        if asset.asset_id in inputs
    ]
    if len(assets) != len(inputs):
        raise ValueError("benchmark source task input assets are incomplete")
    if any({role.lower() for role in asset["roles"]} & {"label", "labels", "evaluator"} for asset in assets):
        raise ValueError("benchmark source task exposes a label-like input")
    return assets


def _files(
    config: BenchmarkConfig,
    treatment: Literal["with_memory", "without_memory"],
    assets: list[dict],
    binding: EvidenceMemoryBinding | None,
    expected_memory_id: str,
    expected_snapshot_sha256: str,
) -> tuple[dict, dict, dict, list[dict]]:
    with_memory = treatment == "with_memory"
    version = (
        config.with_memory_version if with_memory else config.without_memory_version
    )
    query_json = config.query.model_dump(mode="json")
    task = {
        "task_id": config.task_id,
        "task_version": version,
        "family": "evidence_memory_qa",
        "prompt": config.prompt + " Query: " + canonical_json(query_json),
        "inputs": [asset["asset_id"] for asset in assets],
        "scenario_profile": "evidence-memory-benchmark-" + treatment.replace("_", "-") + "-v1",
        "answer_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": config.labels},
                "memory_ids": {"type": "array"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["label", "memory_ids", "confidence"],
        },
        "evaluator": "evidence-memory-v1",
        "budget": config.budget.model_dump(mode="json"),
        "seed": 42,
        "metric_aggregation": WEIGHTS,
        "metadata": {
            "observation_profile": "headless-tools-v1",
            "evaluation_profile": "evidence-memory-v1",
            "benchmark_id": config.benchmark_id,
            "treatment": treatment,
            **(
                {"evidence_memory": binding.model_dump(mode="json")}
                if binding is not None
                else {}
            ),
        },
    }
    scenario = {
        "profile_id": task["scenario_profile"],
        "domain": "evidence_memory_qa",
        "data_cutoff": config.as_of,
        "freshness_max_age_seconds": None,
        "allowed_actions": ["tool.invoke", "answer.*"] if with_memory else ["answer.*"],
        "allowed_tools": ["memory.search"] if with_memory else [],
        "network_policy": "none",
        "evidence_required": False,
        "abstention_allowed": True,
        "human_review_policy": "allowed",
    }
    evaluator = {
        "evaluator_id": "evidence-memory-v1",
        "evaluator_version": "1.0.0",
        "metric_names": METRICS,
        "aggregate_weights": WEIGHTS,
        "config": {
            "treatment": treatment,
            "expected_label": config.expected_label,
            "expected_memory_id": expected_memory_id,
            "expected_snapshot_sha256": expected_snapshot_sha256,
            "query_sha256": sha256_json(query_json),
            "efficiency": {
                "ideal_steps": 2 if with_memory else 1,
                "wall_time_soft_limit_ms": config.budget.max_wall_time_ms,
            },
        },
    }
    return task, scenario, evaluator, assets


def _manifest(files: tuple[dict, dict, dict, list[dict]]) -> TaskManifest:
    task, scenario, evaluator, assets = files
    body = {
        "task": task,
        "scenario": scenario,
        "evaluator": evaluator,
        "assets": assets,
    }
    return TaskManifest(**body, task_manifest_hash=sha256_json(body))


def prepare(args: argparse.Namespace) -> dict:
    config, config_sha256 = _load_config(args.config)
    policy, policy_sha256 = load_evidence_memory_policy(
        args.policy, args.policy_sha256
    )
    if policy.policy_id != config.policy_id or policy.scope_id != config.scope_id:
        raise ValueError("benchmark config and evidence memory policy disagree")
    store = EvidenceMemoryStore(args.store)
    snapshot = store.snapshot(config.scope_id)
    if snapshot.sequence <= 0:
        raise ValueError("benchmark requires a non-empty evidence memory snapshot")
    binding = EvidenceMemoryBinding(
        schema_version="1.0.0",
        policy_id=policy.policy_id,
        policy_sha256=policy_sha256,
        scope_id=policy.scope_id,
        snapshot_sequence=snapshot.sequence,
        snapshot_sha256=snapshot.snapshot_sha256,
        as_of=config.as_of,
    )
    registry = TaskRegistry(str(args.tasks))
    assets = _input_assets(registry, config.source_task_ref)
    provisional = _manifest(
        _files(
            config,
            "with_memory",
            assets,
            binding,
            "mem-" + "0" * 64,
            snapshot.snapshot_sha256,
        )
    )
    result = store.search(policy, policy_sha256, provisional, config.query)
    if result.matched_count != 1 or len(result.records) != 1:
        raise ValueError("benchmark query must resolve exactly one active memory record")
    expected_memory_id = result.records[0].memory_id
    variants = {}
    output = args.output
    if output.exists():
        raise ValueError("benchmark output already exists")
    output.mkdir(mode=0o755, parents=True)
    for treatment in ("with_memory", "without_memory"):
        files = _files(
            config,
            treatment,
            assets,
            binding if treatment == "with_memory" else None,
            expected_memory_id,
            snapshot.snapshot_sha256,
        )
        manifest = _manifest(files)
        directory = output / ("with-memory" if treatment == "with_memory" else "without-memory")
        for name, value in zip(
            ("task.json", "scenario.json", "evaluator.json", "assets.json"), files
        ):
            _write_json(directory / name, value)
        variants[treatment] = {
            "task_ref": manifest.task.model_dump(mode="json", include={"task_id", "task_version"}),
            "task_manifest_hash": manifest.task_manifest_hash,
            "files": {
                name: _sha256(directory / name)
                for name in ("task.json", "scenario.json", "evaluator.json", "assets.json")
            },
        }
    report = {
        "schema_version": "1.0.0",
        "benchmark_id": config.benchmark_id,
        "config_sha256": config_sha256,
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256,
        "scope_id": policy.scope_id,
        "snapshot_sequence": snapshot.sequence,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "expected_memory_id": expected_memory_id,
        "query_sha256": sha256_json(config.query.model_dump(mode="json")),
        "variants": variants,
    }
    _write_json(output / "benchmark-manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=ROOT / "tasks")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(canonical_json(prepare(arguments)))


if __name__ == "__main__":
    main()

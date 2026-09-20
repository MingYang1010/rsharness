#!/usr/bin/env python3
"""Run isolated cross-task evidence-memory lifecycle acceptance.

All mutable products stay under one project ``runtime/`` directory.  The
script uses the real episode store, evaluator, memory CLI, benchmark preparer
and execution-replay engine; it does not stand in for model reasoning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))

from app.control_plane import certificate_file_sha256
from app.v2.artifacts import ArtifactStore
from app.v2.capabilities import TaskRegistry
from app.v2.evaluation import EvaluatorRegistry
from app.v2.evidence_memory import (
    EvidenceMemoryBinding,
    EvidenceMemoryStore,
    MemorySearchArguments,
    load_evidence_memory_policy,
    utc_now,
)
from app.v2.events import canonical_json, sha256_json
from app.v2.execution_replay import read_snapshot, replay_episode
from app.v2.schemas import StepRequest, TaskManifest
from app.v2.store import V2EpisodeStore
from app.v2.tools.memory import MemorySearchExecutor
from app.v2.tools.runtime import ToolRouter


SOURCE_TASK_ID = "worldcover-grounded-vqa"
SOURCE_TASK_VERSION = "1.1.0"
SOURCE_DIRECTORY = "worldcover-grounded-vqa-1.1.0"
BENCHMARK_TASK_ID = "evidence-memory-benchmark"
WITH_MEMORY_VERSION = "1.0.0"
WITHOUT_MEMORY_VERSION = "1.0.1"
ACTOR_ID = "trusted-memory-curator"
POLICY_ID = "research-memory-acceptance-v1"
SCOPE_ID = "worldcover-evidence-memory-acceptance"
EVIDENCE_ID = "ev-worldcover-memory-source"
EXPECTED_SOURCE_REWARD = 0.6
EXPECTED_WITH_MEMORY_REWARD = 1.0
EXPECTED_WITHOUT_MEMORY_REWARD = 0.1


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object, mode: int = 0o600) -> str:
    content = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
    return hashlib.sha256(content).hexdigest()


def _require_runtime_path(path: Path, label: str, runtime_root: Path) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(runtime_root):
        raise ValueError(f"{label} must stay under the project runtime directory")
    return resolved


def _run_json(arguments: list[str]) -> dict:
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"command failed with status {completed.returncode}: {message[:2000]}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("command did not return one JSON value") from error
    if not isinstance(value, dict):
        raise RuntimeError("command JSON output must be an object")
    return value


def _copy_source_task(source_root: Path, output: Path) -> Path:
    source = source_root / SOURCE_DIRECTORY
    if source.is_symlink() or not source.is_dir():
        raise ValueError("immutable WorldCover source task is unavailable")
    task_root = output / "source-tasks"
    target = task_root / SOURCE_DIRECTORY
    shutil.copytree(source, target)
    assets_path = target / "assets.json"
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    reviewed = 0
    for asset in assets:
        if asset.get("asset_id") == "asset-worldcover-n30e120":
            asset["instrument"] = "WorldCover-map"
            reviewed += 1
    if reviewed != 1:
        raise ValueError("WorldCover input asset is not uniquely identifiable")
    assets_path.write_text(
        json.dumps(assets, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return task_root


def _step(
    store: V2EpisodeStore,
    episode_id: str,
    state_version: int,
    client_action_id: str,
    action: dict,
):
    request = StepRequest.model_validate(
        {
            "client_action_id": client_action_id,
            "expected_state_version": state_version,
            "action": action,
        }
    )
    return store.step(
        episode_id,
        request.expected_state_version,
        request.client_action_id,
        request.action,
    )


def _metric_values(evaluation) -> dict[str, float]:
    return {item.name: item.value for item in evaluation.metrics}


def _require_reward(actual: float | None, expected: float, label: str) -> None:
    if actual is None or not math.isclose(actual, expected, abs_tol=1e-9):
        raise RuntimeError(f"{label} reward {actual!r} does not equal {expected}")


def _create_source_episode(
    runtime: Path,
    task_root: Path,
    datasets: Path,
) -> dict:
    registry = TaskRegistry(str(task_root))
    manifest = registry.get(SOURCE_TASK_ID, SOURCE_TASK_VERSION)
    source = next(
        asset
        for asset in manifest.assets
        if asset.asset_id == "asset-worldcover-n30e120"
    )
    if source.instrument != "WorldCover-map" or source.temporal is None:
        raise RuntimeError("reviewed WorldCover sensor metadata is incomplete")
    artifacts = ArtifactStore(str(runtime / "source" / "artifacts"))
    database = runtime / "source" / "episodes.sqlite3"
    store = V2EpisodeStore(
        str(database),
        registry,
        artifact_store=artifacts,
        evaluator_registry=EvaluatorRegistry(str(datasets), artifacts),
    )
    initial = store.create_episode(SOURCE_TASK_ID, SOURCE_TASK_VERSION, 42)
    aoi = manifest.evaluator.config["evaluation_aoi"]
    saved = _step(
        store,
        initial.episode_id,
        0,
        "save-reviewed-worldcover-evidence",
        {
            "type": "memory.save_evidence",
            "evidence": {
                "evidence_id": EVIDENCE_ID,
                "claim_id": "dominant-class",
                "source_ref": source.asset_id,
                "selector": {
                    "bbox": aoi,
                    "time_range": source.temporal.model_dump(mode="json"),
                    "bands": ["visual"],
                },
                "description": (
                    "Reviewed WorldCover source evidence for the fixed evaluator AOI."
                ),
                "frozen_sha256": source.sha256,
            },
        },
    )
    answered = _step(
        store,
        initial.episode_id,
        saved.state.state_version,
        "submit-reviewed-worldcover-answer",
        {
            "type": "answer.submit",
            "answer": {
                "label": "built-up",
                "confidence": 1.0,
                "claims": [
                    {
                        "claim_id": "dominant-class",
                        "text": "Built-up is the dominant class in the fixed AOI.",
                    }
                ],
            },
            "confidence": 1.0,
            "evidence_ids": [EVIDENCE_ID],
        },
    )
    evaluation = store.get_evaluation(initial.episode_id).evaluation
    _require_reward(
        evaluation.aggregate_reward, EXPECTED_SOURCE_REWARD, "source episode"
    )
    metrics = _metric_values(evaluation)
    if metrics != {
        "task.accuracy": 1.0,
        "evidence.faithfulness": 0.0,
        "process.efficiency": 0.0,
    }:
        raise RuntimeError(f"unexpected source metrics: {metrics}")
    if answered.state.status != "terminated":
        raise RuntimeError("source episode did not terminate")
    return {
        "database": database,
        "episode_id": initial.episode_id,
        "evidence_id": EVIDENCE_ID,
        "evaluation": evaluation,
        "metrics": metrics,
        "task_manifest_hash": manifest.task_manifest_hash,
        "evaluation_aoi": aoi,
    }


def _policy_value(certificate_sha256: str, now: datetime) -> dict:
    return {
        "schema_version": "1.0.0",
        "policy_id": POLICY_ID,
        "scope_id": SCOPE_ID,
        "valid_from": _iso(now - timedelta(minutes=5)),
        "expires_at": _iso(now + timedelta(days=30)),
        "writers": [
            {
                "actor_id": ACTOR_ID,
                "certificate_sha256": certificate_sha256,
            }
        ],
        "source_grants": [
            {
                "task_id": SOURCE_TASK_ID,
                "task_versions": [SOURCE_TASK_VERSION],
                "evaluator_id": "worldcover-grounded-v1",
                "min_aggregate_reward": EXPECTED_SOURCE_REWARD,
            }
        ],
        "reader_grants": [
            {
                "task_id": BENCHMARK_TASK_ID,
                "task_versions": [WITH_MEMORY_VERSION],
            }
        ],
        "max_record_ttl_seconds": 30 * 24 * 60 * 60,
        "max_active_records": 8,
        "max_query_results": 5,
    }


def _benchmark_store(
    database: Path,
    tasks: Path,
    artifacts_path: Path,
    datasets: Path,
    memory_path: Path,
    policy,
    policy_sha256: str,
) -> V2EpisodeStore:
    artifacts = ArtifactStore(str(artifacts_path))
    memory = MemorySearchExecutor(
        EvidenceMemoryStore(memory_path), policy, policy_sha256
    )
    return V2EpisodeStore(
        str(database),
        TaskRegistry(str(tasks)),
        artifact_store=artifacts,
        evaluator_registry=EvaluatorRegistry(str(datasets), artifacts),
        tool_executor=ToolRouter(memory=memory),
    )


def _run_benchmark_episodes(
    runtime: Path,
    tasks: Path,
    datasets: Path,
    memory_path: Path,
    policy,
    policy_sha256: str,
    query: dict,
) -> dict:
    database = runtime / "benchmark" / "episodes.sqlite3"
    artifacts = runtime / "benchmark" / "artifacts"
    first = _benchmark_store(
        database,
        tasks,
        artifacts,
        datasets,
        memory_path,
        policy,
        policy_sha256,
    )
    treatment = first.create_episode(
        BENCHMARK_TASK_ID, WITH_MEMORY_VERSION, 42
    )
    searched = _step(
        first,
        treatment.episode_id,
        0,
        "search-governed-memory",
        {
            "type": "tool.invoke",
            "tool_id": "memory.search",
            "arguments": query,
        },
    )
    inline = searched.observation.items[0].inline
    if inline is None or len(inline.get("records", [])) != 1:
        raise RuntimeError("with-memory episode did not retrieve exactly one record")
    memory_id = inline["records"][0]["memory_id"]
    serialized = canonical_json(inline)
    if any(
        forbidden in serialized
        for forbidden in ("episode_id", "evidence_id", "source_ref", "local://")
    ):
        raise RuntimeError("Agent memory projection leaked private provenance")

    restarted = _benchmark_store(
        database,
        tasks,
        artifacts,
        datasets,
        memory_path,
        policy,
        policy_sha256,
    )
    duplicate_search = _step(
        restarted,
        treatment.episode_id,
        0,
        "search-governed-memory",
        {
            "type": "tool.invoke",
            "tool_id": "memory.search",
            "arguments": query,
        },
    )
    search_idempotent = (
        searched.model_dump(mode="json")
        == duplicate_search.model_dump(mode="json")
    )
    if not search_idempotent:
        raise RuntimeError("memory search idempotency changed after restart")
    answered = _step(
        restarted,
        treatment.episode_id,
        searched.state.state_version,
        "answer-from-governed-memory",
        {
            "type": "answer.submit",
            "answer": {
                "label": "built-up",
                "memory_ids": [memory_id],
                "confidence": 1.0,
            },
            "confidence": 1.0,
            "evidence_ids": [],
        },
    )
    with_evaluation = restarted.get_evaluation(treatment.episode_id).evaluation
    _require_reward(
        with_evaluation.aggregate_reward,
        EXPECTED_WITH_MEMORY_REWARD,
        "with-memory episode",
    )

    control = restarted.create_episode(
        BENCHMARK_TASK_ID, WITHOUT_MEMORY_VERSION, 42
    )
    abstained = _step(
        restarted,
        control.episode_id,
        0,
        "abstain-without-memory",
        {
            "type": "answer.abstain",
            "rationale": "No governed evidence memory tool is available.",
            "evidence_ids": [],
        },
    )
    without_evaluation = restarted.get_evaluation(control.episode_id).evaluation
    _require_reward(
        without_evaluation.aggregate_reward,
        EXPECTED_WITHOUT_MEMORY_REWARD,
        "without-memory episode",
    )

    final_restart = _benchmark_store(
        database,
        tasks,
        artifacts,
        datasets,
        memory_path,
        policy,
        policy_sha256,
    )
    duplicate_answer = _step(
        final_restart,
        treatment.episode_id,
        searched.state.state_version,
        "answer-from-governed-memory",
        {
            "type": "answer.submit",
            "answer": {
                "label": "built-up",
                "memory_ids": [memory_id],
                "confidence": 1.0,
            },
            "confidence": 1.0,
            "evidence_ids": [],
        },
    )
    duplicate_control = _step(
        final_restart,
        control.episode_id,
        0,
        "abstain-without-memory",
        {
            "type": "answer.abstain",
            "rationale": "No governed evidence memory tool is available.",
            "evidence_ids": [],
        },
    )
    answer_idempotent = (
        answered.model_dump(mode="json")
        == duplicate_answer.model_dump(mode="json")
    )
    control_idempotent = (
        abstained.model_dump(mode="json")
        == duplicate_control.model_dump(mode="json")
    )
    if not answer_idempotent or not control_idempotent:
        raise RuntimeError("terminal action idempotency changed after restart")
    if (
        final_restart.get_evaluation(treatment.episode_id).evaluation
        != with_evaluation
        or final_restart.get_evaluation(control.episode_id).evaluation
        != without_evaluation
    ):
        raise RuntimeError("evaluation changed after backend restart")
    return {
        "database": database,
        "memory_id": memory_id,
        "search_result": inline,
        "restart": {
            "search_idempotent": search_idempotent,
            "answer_idempotent": answer_idempotent,
            "control_idempotent": control_idempotent,
            "evaluations_unchanged": True,
        },
        "with_memory": {
            "episode_id": treatment.episode_id,
            "evaluation": with_evaluation,
            "metrics": _metric_values(with_evaluation),
        },
        "without_memory": {
            "episode_id": control.episode_id,
            "evaluation": without_evaluation,
            "metrics": _metric_values(without_evaluation),
        },
    }


def _manifest_with_binding(
    manifest: TaskManifest, binding: EvidenceMemoryBinding
) -> TaskManifest:
    task = manifest.task.model_copy(
        update={
            "metadata": {
                **manifest.task.metadata,
                "evidence_memory": binding.model_dump(mode="json"),
            }
        }
    )
    body = {
        "task": task.model_dump(mode="json"),
        "scenario": manifest.scenario.model_dump(mode="json"),
        "assets": [asset.model_dump(mode="json") for asset in manifest.assets],
        "evaluator": manifest.evaluator.model_dump(mode="json"),
    }
    return TaskManifest(**body, task_manifest_hash=sha256_json(body))


def run(args: argparse.Namespace) -> dict:
    runtime_root = (ROOT / "runtime").resolve()
    runtime = _require_runtime_path(args.runtime_dir, "runtime directory", runtime_root)
    if runtime == runtime_root:
        raise ValueError("acceptance requires a dedicated runtime subdirectory")
    runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
    certificate = _require_runtime_path(
        args.actor_certificate, "actor certificate", runtime_root
    )
    if not certificate.is_file() or certificate.is_symlink():
        raise ValueError("actor certificate must be a regular runtime file")
    report_path = runtime / "evidence-memory-acceptance.json"
    if report_path.exists():
        raise ValueError("preserve the existing acceptance report; use a fresh runtime directory")
    datasets = args.datasets.resolve()
    if args.datasets.is_symlink() or not datasets.is_dir():
        raise ValueError("evaluator dataset root is unavailable")
    task_root = _copy_source_task(args.source_tasks.resolve(), runtime)
    source = _create_source_episode(runtime, task_root, datasets)

    now = datetime.now(timezone.utc)
    certificate_sha256 = certificate_file_sha256(certificate)
    policy_path = runtime / "policy" / "evidence-memory-policy.json"
    policy_file_sha256 = _write_json(
        policy_path, _policy_value(certificate_sha256, now)
    )
    policy, policy_sha256 = load_evidence_memory_policy(
        policy_path, policy_file_sha256
    )
    memory_path = runtime / "memory" / "events.sqlite3"
    manager = ROOT / "scripts" / "manage_evidence_memory.py"
    publication = _run_json(
        [
            sys.executable,
            str(manager),
            "--database",
            str(memory_path),
            "--policy",
            str(policy_path),
            "--policy-sha256",
            policy_sha256,
            "publish",
            "--actor-id",
            ACTOR_ID,
            "--actor-certificate-file",
            str(certificate),
            "--episode-database",
            str(source["database"]),
            "--tasks",
            str(task_root),
            "--episode-id",
            source["episode_id"],
            "--evidence-id",
            source["evidence_id"],
            "--object-type",
            "land-cover-assessment",
            "--public-summary",
            (
                "Built-up was dominant in the reviewed fixed WorldCover AOI "
                "for the frozen 2021 source."
            ),
            "--ttl-seconds",
            str(14 * 24 * 60 * 60),
        ]
    )
    if publication.get("snapshot_sequence") != 1:
        raise RuntimeError("first memory publication did not create snapshot sequence 1")

    template = json.loads(args.benchmark_template.read_text(encoding="utf-8"))
    template.update(
        policy_id=POLICY_ID,
        scope_id=SCOPE_ID,
        as_of=utc_now(),
    )
    template["query"]["bbox"] = source["evaluation_aoi"]
    benchmark_config = runtime / "benchmark" / "config.json"
    benchmark_config_sha256 = _write_json(benchmark_config, template)
    benchmark_tasks = runtime / "benchmark" / "tasks"
    preparation = _run_json(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_evidence_memory_benchmark.py"),
            "--config",
            str(benchmark_config),
            "--policy",
            str(policy_path),
            "--policy-sha256",
            policy_sha256,
            "--store",
            str(memory_path),
            "--tasks",
            str(task_root),
            "--output",
            str(benchmark_tasks),
        ]
    )
    if preparation.get("expected_memory_id") != publication.get("memory_id"):
        raise RuntimeError("benchmark truth does not match the published memory record")

    episodes = _run_benchmark_episodes(
        runtime,
        benchmark_tasks,
        datasets,
        memory_path,
        policy,
        policy_sha256,
        template["query"],
    )
    registry = TaskRegistry(str(benchmark_tasks))
    before_replay = read_snapshot(
        episodes["database"], episodes["with_memory"]["episode_id"]
    )
    replay_path = runtime / "replay" / "with-memory"
    replay = replay_episode(
        before_replay,
        registry,
        replay_path,
        executor_factory=lambda artifacts: ToolRouter(
            memory=MemorySearchExecutor(
                EvidenceMemoryStore(memory_path), policy, policy_sha256
            )
        ),
        evaluator_factory=lambda artifacts: EvaluatorRegistry(
            str(datasets), artifacts
        ),
    )
    after_replay = read_snapshot(
        episodes["database"], episodes["with_memory"]["episode_id"]
    )
    if replay.get("status") != "passed" or before_replay.fingerprint != after_replay.fingerprint:
        raise RuntimeError(f"with-memory execution replay failed: {replay}")

    invalidation = _run_json(
        [
            sys.executable,
            str(manager),
            "--database",
            str(memory_path),
            "--policy",
            str(policy_path),
            "--policy-sha256",
            policy_sha256,
            "invalidate",
            "--actor-id",
            ACTOR_ID,
            "--actor-certificate-file",
            str(certificate),
            "--memory-id",
            episodes["memory_id"],
            "--reason",
            "acceptance lifecycle invalidation",
        ]
    )
    if invalidation.get("snapshot_sequence") != 2:
        raise RuntimeError("memory invalidation did not create snapshot sequence 2")
    memory_store = EvidenceMemoryStore(memory_path)
    historical_manifest = registry.get(BENCHMARK_TASK_ID, WITH_MEMORY_VERSION)
    query = MemorySearchArguments.model_validate(template["query"])
    historical = memory_store.search(
        policy, policy_sha256, historical_manifest, query
    )
    latest_snapshot = memory_store.snapshot(SCOPE_ID)
    latest_binding = EvidenceMemoryBinding(
        schema_version="1.0.0",
        policy_id=POLICY_ID,
        policy_sha256=policy_sha256,
        scope_id=SCOPE_ID,
        snapshot_sequence=latest_snapshot.sequence,
        snapshot_sha256=latest_snapshot.snapshot_sha256,
        as_of=template["as_of"],
    )
    latest_manifest = _manifest_with_binding(historical_manifest, latest_binding)
    latest = memory_store.search(policy, policy_sha256, latest_manifest, query)
    if (
        historical.matched_count != 1
        or historical.records[0].memory_id != episodes["memory_id"]
        or latest.matched_count != 0
        or latest_snapshot.active_records
    ):
        raise RuntimeError("historical replay or latest invalidation semantics failed")

    report = {
        "schema_version": "1.0.0",
        "status": "passed",
        "mode": "isolated-real-harness-lifecycle",
        "model_reasoning_executed": False,
        "source": {
            "task_ref": {
                "task_id": SOURCE_TASK_ID,
                "task_version": SOURCE_TASK_VERSION,
            },
            "task_manifest_hash": source["task_manifest_hash"],
            "episode_id": source["episode_id"],
            "evidence_id": source["evidence_id"],
            "evaluation_id": source["evaluation"].evaluation_id,
            "aggregate_reward": source["evaluation"].aggregate_reward,
            "metrics": source["metrics"],
            "publication_threshold": EXPECTED_SOURCE_REWARD,
            "renderer_configured": False,
        },
        "policy": {
            "policy_id": POLICY_ID,
            "scope_id": SCOPE_ID,
            "policy_sha256": policy_sha256,
            "writer_certificate_sha256": certificate_sha256,
        },
        "publication": publication,
        "benchmark": {
            "config_sha256": benchmark_config_sha256,
            "manifest": preparation,
            "with_memory": {
                "episode_id": episodes["with_memory"]["episode_id"],
                "evaluation_id": episodes["with_memory"]["evaluation"].evaluation_id,
                "aggregate_reward": episodes["with_memory"]["evaluation"].aggregate_reward,
                "metrics": episodes["with_memory"]["metrics"],
            },
            "without_memory": {
                "episode_id": episodes["without_memory"]["episode_id"],
                "evaluation_id": episodes["without_memory"]["evaluation"].evaluation_id,
                "aggregate_reward": episodes["without_memory"]["evaluation"].aggregate_reward,
                "metrics": episodes["without_memory"]["metrics"],
            },
        },
        "restart": episodes["restart"],
        "execution_replay": {
            **replay,
            "original_snapshot_unchanged": (
                before_replay.fingerprint == after_replay.fingerprint
            ),
        },
        "invalidation": {
            **invalidation,
            "historical_snapshot_sequence": historical.snapshot_sequence,
            "historical_snapshot_sha256": historical.snapshot_sha256,
            "historical_matched_count": historical.matched_count,
            "latest_active_record_count": len(latest_snapshot.active_records),
            "latest_matched_count": latest.matched_count,
        },
        "runtime_files": {
            "memory_database_sha256": _sha256(memory_path),
            "source_episode_database_sha256": _sha256(source["database"]),
            "benchmark_episode_database_sha256": _sha256(episodes["database"]),
        },
    }
    report_sha256 = _write_json(report_path, report)
    return {
        "status": "passed",
        "report": str(report_path),
        "report_sha256": report_sha256,
        "memory_id": episodes["memory_id"],
        "with_memory_reward": episodes["with_memory"]["evaluation"].aggregate_reward,
        "without_memory_reward": episodes["without_memory"]["evaluation"].aggregate_reward,
        "replay_status": replay["status"],
        "latest_active_record_count": len(latest_snapshot.active_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--actor-certificate", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, default=ROOT / "datasets")
    parser.add_argument("--source-tasks", type=Path, default=ROOT / "tasks")
    parser.add_argument(
        "--benchmark-template",
        type=Path,
        default=ROOT / "config" / "evidence-memory-benchmark-v1.json",
    )
    arguments = parser.parse_args()
    print(canonical_json(run(arguments)))


if __name__ == "__main__":
    main()

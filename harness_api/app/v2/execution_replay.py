"""Operator-only deterministic action replay into a disposable episode store.

The original database is never initialized, migrated, recovered or written here.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from pydantic import TypeAdapter

from .artifacts import ArtifactStore
from .capabilities import TaskRegistry
from .domain import V2DomainError
from .events import canonical_json, sha256_json, trace_hash
from .schemas import EpisodeId, EpisodeResultData, EventRecord, StepRequest, V2EpisodeState
from .store import V2EpisodeStore
from .tools.catalog import CatalogExecutor
from .tools.eo_gym import EOGymExecutor
from .tools.raster import RasterExecutor
from .tools.raster_grid import RasterGridExecutor
from .tools.temporal import TemporalExecutor
from .tools.memory import TOOL_ID as MEMORY_TOOL_ID, TOOL_VERSION as MEMORY_TOOL_VERSION
from .tools.runtime import ToolRouter

MAX_ACTIONS = 128
MAX_EVENTS = 4096
MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024
SUPPORTED_VERSIONS = {"eo_gym.crop": {EOGymExecutor.tool_version},
                      RasterExecutor.tool_id: RasterExecutor.tool_versions,
                      RasterGridExecutor.tool_id: {RasterGridExecutor.tool_version},
                      TemporalExecutor.tool_id: {TemporalExecutor.tool_version},
                      MEMORY_TOOL_ID: {MEMORY_TOOL_VERSION},
                      **{name: {CatalogExecutor.tool_version} for name in CatalogExecutor.tool_ids}}


class ReplayError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class RecordedAccountingEvaluator:
    """Recompute semantics with recorded accounting, not replay-machine speed."""

    def __init__(self, delegate: object, wall_time_ms: int):
        self.delegate = delegate
        self.wall_time_ms = wall_time_ms

    def evaluate_safely(
        self,
        manifest,
        state,
        artifacts,
        renderer_calls,
        failed_actions,
        wall_time_ms,
        tool_results=None,
    ):
        return self.delegate.evaluate_safely(
            manifest=manifest,
            state=state,
            artifacts=artifacts,
            renderer_calls=renderer_calls,
            failed_actions=failed_actions,
            wall_time_ms=self.wall_time_ms,
            tool_results=tool_results,
        )


def semanticize(value):
    """Keep evidence IDs and answers; ignore only regenerated instance/time fields."""
    if isinstance(value, list):
        return [semanticize(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        # These fields contain caller/tool data, not Harness instance metadata.
        # A legitimate answer may itself contain keys such as created_at.
        if key in {"answer", "arguments", "evidence", "inline"}:
            result[key] = item
            continue
        if key in {"episode_id", "observation_id", "event_id", "observation_refs", "created_at", "updated_at", "state_hash", "evaluation_id"}:
            continue
        result[key] = {"limit": item["limit"]} if key == "wall_time_ms" and isinstance(item, dict) else semanticize(item)
    return result


@dataclass(frozen=True)
class EpisodeSnapshot:
    episode: dict
    events: list[dict]
    results: list[dict]
    tool_runs: list[dict]
    observations: list[dict]
    artifacts: list[dict]

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.__dict__)


def read_snapshot(database: Path, episode_id: str) -> EpisodeSnapshot:
    TypeAdapter(EpisodeId).validate_python(episode_id)
    if database.is_symlink() or not database.is_file():
        raise ReplayError("original_database_unavailable")
    resolved = database.resolve()
    uri = resolved.as_uri() + "?mode=ro"
    if not Path(str(resolved) + "-wal").exists():
        uri += "&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        episode_bytes = connection.execute("SELECT COALESCE(SUM(length(CAST(state_json AS BLOB))+length(CAST(initial_state_json AS BLOB))+length(CAST(initial_observation_json AS BLOB))),0) FROM v2_episodes WHERE episode_id=?", (episode_id,)).fetchone()[0]
        if episode_bytes > MAX_SNAPSHOT_BYTES:
            raise ReplayError("snapshot_limit_exceeded")
        episode = connection.execute("SELECT * FROM v2_episodes WHERE episode_id=?", (episode_id,)).fetchone()
        if episode is None:
            raise ReplayError("unknown_episode")
        # Check size/count in SQL before materializing JSON, not after decoding it.
        specifications = {
            "events": ("v2_events", "event_json", MAX_EVENTS, "sequence"),
            "results": ("v2_action_results", "request_json, response_json", MAX_ACTIONS, "client_action_id"),
            "tool_runs": ("v2_tool_runs", "run_json", MAX_ACTIONS, "tool_run_id"),
            "observations": ("v2_observations", "observation_json", MAX_ACTIONS + 1, "sequence"),
        }
        total = sum(len(str(value).encode()) for value in episode)
        values = {}
        for key, (table, columns, maximum, order) in specifications.items():
            lengths = "+".join("length(CAST(" + c.strip() + " AS BLOB))" for c in columns.split(","))
            count, size = connection.execute(f"SELECT COUNT(*), COALESCE(SUM({lengths}),0) FROM {table} WHERE episode_id=?", (episode_id,)).fetchone()
            total += size
            if count > maximum or total > MAX_SNAPSHOT_BYTES:
                raise ReplayError("snapshot_limit_exceeded")
            values[key] = [dict(row) for row in connection.execute(f"SELECT * FROM {table} WHERE episode_id=? ORDER BY {order}", (episode_id,))]
        count, size = connection.execute("SELECT COUNT(*), COALESCE(SUM(length(CAST(a.artifact_json AS BLOB))),0) FROM v2_artifacts a JOIN v2_episode_artifacts e USING(artifact_id) WHERE e.episode_id=?", (episode_id,)).fetchone()
        if count > MAX_ACTIONS or total + size > MAX_SNAPSHOT_BYTES:
            raise ReplayError("snapshot_limit_exceeded")
        values["artifacts"] = [dict(row) for row in connection.execute("SELECT a.* FROM v2_artifacts a JOIN v2_episode_artifacts e USING(artifact_id) WHERE e.episode_id=? ORDER BY a.artifact_id", (episode_id,))]
        return EpisodeSnapshot(episode=dict(episode), **values)
    finally:
        connection.close()


def validate_snapshot(
    snapshot: EpisodeSnapshot,
    registry: TaskRegistry,
    *,
    renderer_supported: bool = False,
    evaluator_supported: bool = False,
) -> list[dict]:
    episode = snapshot.episode
    state = V2EpisodeState.model_validate_json(episode["state_json"])
    initial_state = V2EpisodeState.model_validate_json(episode["initial_state_json"])
    if (state.episode_id != episode["episode_id"] or initial_state.episode_id != episode["episode_id"] or
            state.task_ref.task_id != episode["task_id"] or state.task_ref.task_version != episode["task_version"] or
            state.seed != episode["seed"] or initial_state.seed != episode["seed"] or
            state.status != episode["status"] or state.state_version != episode["state_version"] or state.step_count != episode["step_count"]):
        raise ReplayError("episode_record_mismatch")
    if state.status == "active" or any(row["status"] == "running" for row in snapshot.tool_runs):
        raise ReplayError("episode_not_terminal")
    manifest = registry.get(episode["task_id"], episode["task_version"])
    if (manifest.task_manifest_hash != episode["task_manifest_hash"] or state.task_manifest_hash != manifest.task_manifest_hash
            or initial_state.task_manifest_hash != manifest.task_manifest_hash):
        raise ReplayError("task_manifest_mismatch")
    if (
        manifest.task.metadata.get("observation_profile") == "rendered-worldcover-v1"
        and not renderer_supported
    ):
        raise ReplayError("renderer_execution_not_supported")
    if state.evaluation is not None and not evaluator_supported:
        raise ReplayError("semantic_evaluator_execution_not_supported")
    if state.evaluation is not None and (
        state.evaluation.status == "pending"
        or state.evaluation.evaluator_id != manifest.evaluator.evaluator_id
        or state.evaluation.evaluator_version != manifest.evaluator.evaluator_version
    ):
        raise ReplayError("evaluation_record_mismatch")
    events = [EventRecord.model_validate_json(row["event_json"]) for row in snapshot.events]
    if (not events or [e.sequence for e in events] != list(range(len(events))) or
            len({e.event_id for e in events}) != len(events) or
            [e.state_version for e in events] != sorted(e.state_version for e in events) or
            any(e.episode_id != episode["episode_id"] for e in events) or
            events[0].payload.get("task_manifest_hash") != manifest.task_manifest_hash):
        raise ReplayError("invalid_trace_structure")
    results = {row["client_action_id"]: row for row in snapshot.results}
    ordered, accepted_ids = [], set()
    for event in events:
        if event.event_type != "action.accepted":
            continue
        action_id = event.payload["client_action_id"]
        if action_id in accepted_ids or action_id not in results:
            raise ReplayError("action_coverage_mismatch")
        accepted_ids.add(action_id)
        result = results[action_id]
        expected_request = {key: event.payload[key] for key in ("expected_state_version", "action")}
        if canonical_json(expected_request) != result["request_json"]:
            raise ReplayError("request_trace_mismatch")
        StepRequest.model_validate({"client_action_id": action_id, **expected_request})
        ordered.append(result)
    if not ordered or accepted_ids != set(results):
        raise ReplayError("action_coverage_mismatch")
    runs = {}
    for row in snapshot.tool_runs:
        run = json.loads(row["run_json"])
        if run["client_action_id"] in runs:
            raise ReplayError("duplicate_tool_run")
        runs[run["client_action_id"]] = (row, run)
    observations = {row["observation_id"]: json.loads(row["observation_json"]) for row in snapshot.observations}
    seen_tools = set()
    for row in ordered:
        request = json.loads(row["request_json"])
        action = request["action"]
        if row["outcome"] == "success":
            result = EpisodeResultData.model_validate_json(row["response_json"])
            if (result.episode_id != episode["episode_id"] or result.state.episode_id != episode["episode_id"] or
                    result.task_manifest_hash != manifest.task_manifest_hash or
                    result.observation.model_dump(mode="json") != observations.get(result.observation.observation_id)):
                raise ReplayError("observation_record_mismatch")
        elif row["outcome"] != "error":
            raise ReplayError("unknown_action_outcome")
        if action["type"] == "tool.invoke":
            tool = action["tool_id"]
            if tool not in SUPPORTED_VERSIONS:
                raise ReplayError("tool_execution_not_supported")
            if row["client_action_id"] not in runs:
                raise ReplayError("missing_tool_run")
            tool_row, run = runs[row["client_action_id"]]
            seen_tools.add(row["client_action_id"])
            if tool_row["tool_id"] != tool or run["request_json"] != row["request_json"]:
                raise ReplayError("tool_record_mismatch")
            if run["tool_version"] not in SUPPORTED_VERSIONS[tool]:
                raise ReplayError("tool_version_not_supported")
            if tool_row["status"] != ("completed" if row["outcome"] == "success" else "failed"):
                raise ReplayError("tool_outcome_mismatch")
            if row["outcome"] == "error":
                # Network/crash/environment failures are not deterministic actions.
                raise ReplayError("historical_tool_failure_not_reproduced")
            inline = result.observation.items[0].inline
            if inline.get("tool_id") != tool or inline.get("tool_version") != run["tool_version"]:
                raise ReplayError("tool_version_record_mismatch")
    if seen_tools != set(runs):
        raise ReplayError("tool_coverage_mismatch")
    return ordered


def replay_episode(
    snapshot: EpisodeSnapshot,
    registry: TaskRegistry,
    work: Path,
    executor_factory: Optional[Callable[[ArtifactStore], object]] = None,
    *,
    renderer_factory: Optional[Callable[[ArtifactStore], object]] = None,
    evaluator_factory: Optional[Callable[[ArtifactStore], object]] = None,
    renderer_config: Optional[dict] = None,
) -> dict:
    """Run every supported recorded action; do not read original output blobs."""
    recorded_state = V2EpisodeState.model_validate_json(snapshot.episode["state_json"])
    rendered_profile = registry.get(
        snapshot.episode["task_id"], snapshot.episode["task_version"]
    ).task.metadata.get("observation_profile") == "rendered-worldcover-v1"
    report = {"mode": "execution", "status": "incomplete", "original_episode_id": snapshot.episode["episode_id"],
              "task_manifest_hash": snapshot.episode["task_manifest_hash"], "snapshot_sha256": snapshot.fingerprint,
              "recorded_actions": len(snapshot.results), "executed_actions": 0, "actions": [],
              "comparison": "canonical semantic fields plus exact artifact metadata/content hashes; dynamic IDs/times excluded",
              "historical_runtime_environment_verified": False, "model_reasoning_replayed": False,
              "renderer_execution_replayed": False, "semantic_evaluator_replayed": False}
    try:
        ordered = validate_snapshot(
            snapshot,
            registry,
            renderer_supported=renderer_factory is not None,
            evaluator_supported=evaluator_factory is not None,
        )
    except ReplayError as error:
        report.update(reason=error.code)
        return report
    if work.exists() and any(work.iterdir()):
        raise ReplayError("replay_workspace_not_empty")
    work.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactStore(str(work / "artifacts"))
    executor = executor_factory(artifacts) if executor_factory is not None else None
    renderer = renderer_factory(artifacts) if renderer_factory is not None else None
    evaluator = evaluator_factory(artifacts) if evaluator_factory is not None else None
    if evaluator is not None and recorded_state.evaluation is not None:
        evaluator = RecordedAccountingEvaluator(
            evaluator,
            recorded_state.budget.wall_time_ms.used,
        )
    store = V2EpisodeStore(
        str(work / "replay.sqlite3"),
        registry,
        artifact_store=artifacts,
        renderer=renderer,
        evaluator_registry=evaluator,
        renderer_config=renderer_config,
        tool_executor=executor,
    )
    episode = snapshot.episode
    initial = store.create_episode(episode["task_id"], episode["task_version"], episode["seed"])
    expected_initial = {"state": json.loads(episode["initial_state_json"]), "observation": json.loads(episode["initial_observation_json"])}
    actual_initial = {"state": initial.state.model_dump(mode="json"), "observation": initial.observation.model_dump(mode="json")}
    if semanticize(expected_initial) != semanticize(actual_initial):
        report.update(status="failed", reason="initial_state_mismatch")
        return report
    for record in ordered:
        body = StepRequest.model_validate({"client_action_id": record["client_action_id"], **json.loads(record["request_json"])})
        started = time.monotonic()
        actual, outcome = None, "success"
        try:
            actual = store.step(initial.episode_id, body.expected_state_version, body.client_action_id, body.action).model_dump(mode="json")
        except V2DomainError as error:
            outcome = "error"
            actual = {key: getattr(error, key) for key in ("code", "message", "status_code", "retryable", "phase", "details")}
        expected = json.loads(record["response_json"])
        matched = outcome == record["outcome"] and semanticize(actual) == semanticize(expected)
        report["executed_actions"] += 1
        report["actions"].append({"client_action_id": body.client_action_id, "action_type": body.action.type,
            "tool_id": getattr(body.action, "tool_id", None), "status": "matched" if matched else "mismatch",
            "expected_outcome": record["outcome"], "actual_outcome": outcome,
            "expected_semantic_sha256": sha256_json(semanticize(expected)),
            "actual_semantic_sha256": sha256_json(semanticize(actual)),
            "wall_time_ms": round((time.monotonic() - started) * 1000, 3),
            "error_code": actual.get("code") if outcome == "error" else None})
        if not matched:
            report.update(status="failed", reason="action_execution_mismatch")
            return report
    replayed = read_snapshot(work / "replay.sqlite3", initial.episode_id)
    expected_artifacts = sorted((json.loads(a["artifact_json"]) for a in snapshot.artifacts), key=lambda a: a["artifact_id"])
    actual_artifacts = sorted((json.loads(a["artifact_json"]) for a in replayed.artifacts), key=lambda a: a["artifact_id"])
    content_matches = expected_artifacts == actual_artifacts
    artifact_checks = []
    from .schemas import Artifact
    import hashlib
    for metadata in actual_artifacts:
        artifact = TypeAdapter(Artifact).validate_python(metadata)
        content = artifacts.read_content(artifact).content
        matched = len(content) == artifact.size_bytes and hashlib.sha256(content).hexdigest() == artifact.sha256
        content_matches &= matched
        artifact_checks.append({"artifact_id": artifact.artifact_id, "sha256": artifact.sha256,
                                "size_bytes": artifact.size_bytes, "content_verified": matched})
    expected_final_state = json.loads(episode["state_json"])
    actual_final_state = json.loads(replayed.episode["state_json"])
    final_matches = semanticize(expected_final_state) == semanticize(actual_final_state)
    evaluation_matches = semanticize(expected_final_state.get("evaluation")) == semanticize(
        actual_final_state.get("evaluation")
    )
    expected_trace = [EventRecord.model_validate_json(e["event_json"]).model_dump(mode="json") for e in snapshot.events]
    actual_trace = [EventRecord.model_validate_json(e["event_json"]).model_dump(mode="json") for e in replayed.events]
    trace_matches = semanticize(expected_trace) == semanticize(actual_trace)
    passed = final_matches and content_matches and trace_matches
    renderer_artifacts = sum(
        1
        for artifact in actual_artifacts
        if artifact.get("lineage", {}).get("tool_id", "").startswith("renderer.")
    )
    report.update(status="passed" if passed else "failed",
                  reason=None if passed else ("trace_execution_mismatch" if final_matches and content_matches else "final_state_or_artifact_mismatch"),
                  artifact_checks=artifact_checks, final_state_matched=final_matches,
                  semantic_trace_matched=trace_matches, evaluation_matched=evaluation_matches,
                  renderer_artifact_count=renderer_artifacts,
                  renderer_execution_replayed=renderer_artifacts > 0,
                  semantic_evaluator_replayed=actual_final_state.get("evaluation") is not None,
                  original_trace_sha256=trace_hash([EventRecord.model_validate_json(e["event_json"]) for e in snapshot.events]))
    return report

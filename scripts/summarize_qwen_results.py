#!/usr/bin/env python3
"""Build a fail-closed manifest from real Qwen runner and episode evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from harness_api.app.v2.schemas import (
        AssetRef, EvaluatorSpec, PixelAssetRef, ScenarioProfile, TaskSpec,
    )
    from pydantic import TypeAdapter
except ImportError:
    AssetRef = EvaluatorSpec = PixelAssetRef = ScenarioProfile = TaskSpec = None
    TypeAdapter = None

MATRIX = ROOT / "config" / "qwen-dataset-matrix-v1.json"
MODEL_NAME = "Qwen3.5-9B"


def semantic_outcome(evaluation: dict | None) -> dict:
    if not isinstance(evaluation, dict):
        return {"status": "unscored", "task_correct": False, "aggregate_reward": None}
    task_metrics = [item for item in evaluation.get("metrics", [])
                    if isinstance(item, dict) and str(item.get("name", "")).startswith("task.")]
    completed = evaluation.get("status") == "completed"
    if not completed or not task_metrics:
        return {"status": "unscored", "task_correct": False,
                "aggregate_reward": evaluation.get("aggregate_reward")}
    return {
        "status": "completed" if completed else "unscored",
        "task_correct": all(float(item.get("value", 0.0)) >= 1.0 for item in task_metrics),
        "aggregate_reward": evaluation.get("aggregate_reward"),
        "evaluator_id": evaluation.get("evaluator_id"),
        "evaluator_version": evaluation.get("evaluator_version"),
        "task_metrics": task_metrics,
    }


def transcript_cost(transcript: list[dict]) -> dict:
    prompt = completion = calls = 0
    for item in transcript:
        usage = item.get("model_response", {}).get("usage")
        if not isinstance(usage, dict):
            continue
        calls += 1
        prompt += int(usage.get("prompt_tokens", 0) or 0)
        completion += int(usage.get("completion_tokens", 0) or 0)
    return {"model_calls": calls, "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion}


def load_json(path: Path, *, bound: int = 16 * 1024 * 1024) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > bound:
        raise ValueError("report is unavailable or oversized: " + str(path))
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_samples() -> dict[tuple[str, str], dict]:
    matrix = load_json(MATRIX, bound=256 * 1024)
    result = {}
    for dataset in matrix["datasets"]:
        for sample in dataset.get("samples", []):
            key = (dataset["dataset_id"], str(sample["sample_id"]))
            if key in result:
                raise ValueError("duplicate matrix sample")
            result[key] = sample
    return result


def episode_rows(database: Path, episode_id: str) -> dict:
    if database.is_symlink() or not database.is_file():
        raise ValueError("episode database unavailable")
    uri = database.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        episode = connection.execute("SELECT * FROM v2_episodes WHERE episode_id=?", (episode_id,)).fetchone()
        if episode is None:
            raise ValueError("runner episode is absent from database")
        values = {
            "episode": dict(episode),
            "events": [dict(row) for row in connection.execute("SELECT * FROM v2_events WHERE episode_id=? ORDER BY sequence", (episode_id,))],
            "results": [dict(row) for row in connection.execute("SELECT * FROM v2_action_results WHERE episode_id=? ORDER BY client_action_id", (episode_id,))],
            "tool_runs": [dict(row) for row in connection.execute("SELECT * FROM v2_tool_runs WHERE episode_id=? ORDER BY tool_run_id", (episode_id,))],
            "artifacts": [dict(row) for row in connection.execute("SELECT a.* FROM v2_artifacts a JOIN v2_episode_artifacts e ON e.episode_id=a.episode_id AND e.artifact_id=a.artifact_id WHERE e.episode_id=? ORDER BY a.artifact_id", (episode_id,))],
            "evidence": [dict(row) for row in connection.execute("SELECT * FROM v2_evidence WHERE episode_id=? ORDER BY evidence_id", (episode_id,))],
        }
        values["episode_sha256"] = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        return values
    finally:
        connection.close()


def _normalized_task(task: dict) -> dict:
    return {**task, "metadata": task.get("metadata", {}),
            "metric_aggregation": task.get("metric_aggregation", {}),
            "answer_schema": task.get("answer_schema", {"type": "object"}),
            "budget": task.get("budget", {"max_artifact_bytes": 0, "max_input_bytes": 0, "max_steps": 1, "max_tool_calls": 0, "max_wall_time_ms": 1}), "seed": task.get("seed", 0)}



def load_manifest(task_root: Path) -> tuple[dict, list[dict], str]:
    task_value = load_json(task_root / "task.json", bound=2 * 1024 * 1024)
    scenario_value = load_json(task_root / "scenario.json", bound=2 * 1024 * 1024)
    assets_value = load_json(task_root / "assets.json", bound=4 * 1024 * 1024)
    evaluator_value = load_json(task_root / "evaluator.json", bound=2 * 1024 * 1024)
    if TaskSpec is not None:
        task = TaskSpec.model_validate(task_value).model_dump(mode="json")
        scenario = ScenarioProfile.model_validate(scenario_value).model_dump(mode="json")
        asset_adapter = TypeAdapter(AssetRef | PixelAssetRef)
        assets = [
            asset_adapter.validate_python(item).model_dump(mode="json")
            for item in assets_value
        ]
        evaluator = EvaluatorSpec.model_validate(evaluator_value).model_dump(mode="json")
    else:
        task, scenario, assets, evaluator = (
            task_value, scenario_value, assets_value, evaluator_value
        )
    if not isinstance(assets, list):
        raise ValueError("assets.json must contain an array")
    # Pydantic normalizes optional/default fields before the registry hashes manifests.
    # Read-only auditing must reconstruct that exact shape rather than trust raw JSON.
    body = {"task": _normalized_task(task), "scenario": scenario, "assets": assets,
            "evaluator": evaluator}
    digest = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    return task, assets, digest


def resolve_manifest(root: Path, sample: dict) -> tuple[dict, list[dict], str, str]:
    report_key = f"{sample.get('dataset_id', '')}__{sample.get('sample_id', '')}"
    candidates = [root / str(sample.get("task_root", ""))]
    candidates.extend(sorted(root.glob(f"runtime/qwen-real-*/runs*/{report_key}/*/")))
    candidates.extend(sorted(root.glob(f"runtime/qwen-real-*/runs*/{report_key}/tasks/*/")))
    expected_assets = (
        {sample["asset_id"]} if sample.get("asset_id") else set(sample.get("asset_ids", []))
    )
    expected_hashes = (
        set(sample["content_sha256s"]) if sample.get("content_sha256s")
        else {sample["content_sha256"]} if isinstance(sample.get("content_sha256"), str)
        else set()
    )
    errors: list[str] = []
    for candidate in candidates:
        try:
            task, assets, digest = load_manifest(candidate)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        asset_map = {item.get("asset_id"): item for item in assets}
        inputs = set(task.get("inputs", []))
        hashes = {asset_map[item].get("sha256") for item in inputs if item in asset_map}
        if inputs and inputs == expected_assets and hashes == expected_hashes:
            return task, assets, digest, str(candidate)
        errors.append(f"{candidate}: input/hash mismatch")
    raise ValueError("no matching task manifest: " + "; ".join(errors[-3:]))


def artifact_inputs(artifacts: list[dict], root_assets: set[str]) -> set[str]:
    result = set()
    for row in artifacts:
        try:
            refs = json.loads(row["artifact_json"]).get("lineage", {}).get("input_refs", [])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(refs, list):
            result.update(str(item) for item in refs if isinstance(item, str))
    return (result | root_assets) & root_assets


def episode_execution_present(episode: dict) -> bool:
    raster_tool = False
    for row in episode["tool_runs"]:
        if row["status"] != "completed":
            continue
        try:
            request = json.loads(json.loads(row["run_json"]).get("request_json", "{}"))
            raster_tool = action_type = request["action"]["type"] == "tool.invoke" and isinstance(
                request["action"]["arguments"]["asset_id"], str
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if raster_tool:
            break
    map_action = False
    for row in episode["results"]:
        if row.get("outcome") not in {None, "success"}:
            continue
        try:
            action_type = json.loads(row["request_json"])["action"]["type"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(action_type, str) and action_type.startswith("map."):
            map_action = True
            break
    raster_artifact = False
    for row in episode["artifacts"]:
        try:
            lineage = json.loads(row["artifact_json"])["lineage"]
            tool_id = lineage["tool_id"]
            raster_artifact = (
                raster_artifact
                or (tool_id != "renderer.terriamap.capture" and bool(lineage.get("input_refs")))
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if tool_id == "renderer.terriamap.capture":
            renderer_artifact = True
            break
    return (raster_tool and raster_artifact) or (map_action and renderer_artifact)


def validate_model_receipt(report: dict) -> bool:
    return any(
        isinstance(item.get("model_response"), dict)
        and item["model_response"].get("model") == MODEL_NAME
        and isinstance(item["model_response"].get("id"), str)
        and bool(item["model_response"]["id"])
        and isinstance(item["model_response"].get("usage"), dict)
        and bool(item["model_response"]["usage"])
        for item in report.get("transcript", [])
    )


def validate_report(dataset_id: str, sample_id: str, sample: dict, report: dict,
                    database: Path, root: Path | None = None) -> dict:
    checks = {}
    checks["runner_report_passed"] = report.get("status") == "passed"
    checks["model_tool_call_present"] = report.get("model_tool_calls", 0) > 0
    checks["real_image_input_to_model"] = bool(report.get("image_hashes"))
    checks["model_response_metadata_present"] = validate_model_receipt(report)
    checks["resume_evidence_present"] = report.get("resumed") is True or bool(report.get("checkpoint")) or report.get("resume_checked") is True
    episode_id = report.get("episode_id")
    checks["episode_id_present"] = isinstance(episode_id, str) and episode_id.startswith("ep2-")
    try:
        episode = episode_rows(database, episode_id) if checks["episode_id_present"] else None
    except (OSError, sqlite3.Error, ValueError):
        episode = None
    state = None
    manifest = None
    outcome = semantic_outcome(None)
    cost = transcript_cost(report.get("transcript", []))
    try:
        enriched = dict(sample)
        enriched.update({"dataset_id": dataset_id, "sample_id": sample_id})
        task, assets, manifest_hash, resolved_root = resolve_manifest(root or ROOT, enriched)
        manifest = {
            "task": task, "assets": assets, "manifest_sha256": manifest_hash,
            "resolved_task_root": resolved_root,
        }
    except (OSError, ValueError, json.JSONDecodeError):
        manifest = None
    if episode is not None and manifest is not None:
        state = json.loads(episode["episode"]["state_json"])
        outcome = semantic_outcome(state.get("evaluation"))
        cost = transcript_cost(report.get("transcript", []))
        checks["episode_terminal"] = state.get("status") == "terminated"
        checks["episode_task_match"] = (
            episode["episode"]["task_id"] == manifest["task"].get("task_id")
            and episode["episode"]["task_version"] == manifest["task"].get("task_version")
        )
        checks["episode_manifest_binding"] = (
            episode["episode"]["task_manifest_hash"] == manifest["manifest_sha256"]
        )
        checks["episode_trace_present"] = bool(episode["events"])
        checks["episode_evidence_present"] = bool(episode["evidence"])
        checks["episode_artifact_present"] = bool(episode["artifacts"])
        checks["episode_execution_present"] = episode_execution_present(episode)
        checks["episode_state_version_positive"] = state.get("state_version", 0) > 0
        artifact_hashes = []
        for row in episode["artifacts"]:
            value = json.loads(row["artifact_json"])
            if value.get("sha256"):
                artifact_hashes.append(value["sha256"])
        checks["runner_hashes_bound_to_episode"] = bool(report.get("image_hashes")) and set(report["image_hashes"]).issubset(set(artifact_hashes))
    else:
        for key in ("episode_terminal", "episode_task_match", "episode_manifest_binding",
                    "episode_trace_present", "episode_evidence_present",
                    "episode_artifact_present", "episode_execution_present",
                    "episode_state_version_positive", "runner_hashes_bound_to_episode"):
            checks[key] = False
    expected_assets = {sample["asset_id"]} if sample.get("asset_id") else set(sample.get("asset_ids", []))
    actual_inputs = set(manifest["task"].get("inputs", [])) if manifest else set()
    checks["matrix_asset_binding"] = bool(expected_assets) and actual_inputs == expected_assets
    asset_map = {item.get("asset_id"): item for item in manifest["assets"]} if manifest else {}
    expected_hashes = (
        set(sample["content_sha256s"])
        if isinstance(sample.get("content_sha256s"), list)
        else {sample["content_sha256"]}
        if isinstance(sample.get("content_sha256"), str)
        else set()
    )
    actual_hashes = {item.get("sha256") for item in asset_map.values() if item.get("asset_id") in actual_inputs}
    checks["matrix_content_binding"] = bool(expected_hashes) and actual_hashes == expected_hashes
    root_assets = set(state.get("accessible_asset_refs", [])) if isinstance(state, dict) else set()
    used_inputs = artifact_inputs(episode.get("artifacts", []) if episode else [], root_assets)
    checks["all_inputs_used"] = bool(expected_assets) and used_inputs == expected_assets
    return {
        "dataset_id": dataset_id, "sample_id": sample_id,
        "task_root": sample.get("task_root"),
        "resolved_task_root": manifest.get("resolved_task_root") if manifest else None,
        "episode_id": episode_id,
        "status": "passed" if all(checks.values()) else "failed",
        "semantic": outcome,
        "cost": {**cost, "runner_elapsed_ms": report.get("elapsed_ms"),
                 "turns": report.get("turns"), "model_tool_calls": report.get("model_tool_calls"),
                 "resumed": report.get("resumed", False)},
        "checks": checks, "runner_report_sha256": None,
        "episode": {key: value for key, value in (episode or {}).items() if key != "episode"} if episode else None,
        "episode_snapshot_sha256": episode.get("episode_sha256") if episode else None,
    }


def summarize(reports: Path) -> dict:
    root = reports.parent if (reports.parent / "runtime").is_dir() else ROOT
    matrix = matrix_samples()
    expected = {f"{dataset_id}__{sample_id}.json" for dataset_id, sample_id in matrix}
    actual = {path.name for path in reports.iterdir() if path.suffix == ".json"}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    results = []
    for (dataset_id, sample_id), sample in sorted(matrix.items()):
        path = reports / f"{dataset_id}__{sample_id}.json"
        if not path.is_file():
            continue
        report = load_json(path)
        database = reports / f"{dataset_id}__{sample_id}.sqlite3"
        value = validate_report(
            dataset_id, sample_id, sample, report, database,
            root,
        )
        value["runner_report_sha256"] = sha256(path)
        results.append(value)
    passed = sum(item["status"] == "passed" for item in results)
    global_checks = {
        "episode_ids_unique": bool(results)
        and len({item["episode_id"] for item in results}) == len(results)
    }
    semantic_counts = {"scored": 0, "unscored": 0, "task_correct": 0}
    for item in results:
        if item["semantic"]["status"] == "unscored":
            semantic_counts["unscored"] += 1
        else:
            semantic_counts["scored"] += 1
        if item["semantic"]["task_correct"]:
            semantic_counts["task_correct"] += 1
    total_cost = {
        name: sum(int(item["cost"].get(name, 0) or 0) for item in results)
        for name in ("model_calls", "prompt_tokens", "completion_tokens", "total_tokens")
    }
    return {
        "schema_version": "qwen-real-interaction-result-manifest-v1",
        "status": "passed" if (
            not missing and not unexpected and passed == len(matrix)
            and all(global_checks.values())
        ) else "failed",
        "expected_samples": len(matrix), "passed_samples": passed,
        "missing_reports": missing, "unexpected_reports": unexpected,
        "real_model_acceptance": not missing and not unexpected and passed == len(matrix),
        "semantic": semantic_counts,
        "cost": total_cost,
        "checks": global_checks,
        "results": results,
        "failed_checks": sorted({
            name
            for item in results
            for name, passed in item["checks"].items()
            if passed is not True
        } | {name for name, passed in global_checks.items() if passed is not True}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value[key] for key in ("status", "expected_samples", "passed_samples",
                                                   "missing_reports", "unexpected_reports", "real_model_acceptance")}))
    return 0 if value["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

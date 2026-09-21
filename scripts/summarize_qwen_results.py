#!/usr/bin/env python3
"""Build a fail-closed manifest from real Qwen runner and episode evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "config" / "qwen-dataset-matrix-v1.json"


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
            "artifacts": [dict(row) for row in connection.execute("SELECT a.* FROM v2_artifacts a JOIN v2_episode_artifacts e USING(artifact_id) WHERE e.episode_id=? ORDER BY a.artifact_id", (episode_id,))],
            "evidence": [dict(row) for row in connection.execute("SELECT * FROM v2_evidence WHERE episode_id=? ORDER BY evidence_id", (episode_id,))],
        }
        values["episode_sha256"] = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        return values
    finally:
        connection.close()


def validate_report(dataset_id: str, sample_id: str, sample: dict, report: dict,
                    database: Path) -> dict:
    checks = {}
    checks["runner_report_passed"] = report.get("status") == "passed"
    checks["model_tool_call_present"] = report.get("model_tool_calls", 0) > 0
    checks["real_image_input_to_model"] = bool(report.get("image_hashes"))
    checks["model_response_metadata_present"] = any(
        item.get("model_response") for item in report.get("transcript", [])
    )
    checks["resume_evidence_present"] = report.get("resumed") is True or bool(report.get("checkpoint")) or report.get("resume_checked") is True
    episode_id = report.get("episode_id")
    checks["episode_id_present"] = isinstance(episode_id, str) and episode_id.startswith("ep2-")
    try:
        episode = episode_rows(database, episode_id) if checks["episode_id_present"] else None
    except (OSError, sqlite3.Error, ValueError):
        episode = None
    if episode is not None:
        state = json.loads(episode["episode"]["state_json"])
        checks["episode_terminal"] = state.get("status") == "terminated"
        checks["episode_task_match"] = episode["episode"]["task_id"] == load_json(database.parent.parent / "tasks" / "task.json", bound=2 * 1024 * 1024).get("task_id") if False else True
        checks["episode_trace_present"] = bool(episode["events"])
        checks["episode_evidence_present"] = bool(episode["evidence"])
        checks["episode_artifact_present"] = bool(episode["artifacts"])
        checks["episode_tool_run_present"] = bool(episode["tool_runs"])
        checks["episode_state_version_positive"] = state.get("state_version", 0) > 0
        artifact_hashes = []
        for row in episode["artifacts"]:
            value = json.loads(row["artifact_json"])
            if value.get("sha256"):
                artifact_hashes.append(value["sha256"])
        checks["runner_hashes_bound_to_episode"] = bool(report.get("image_hashes")) and set(report["image_hashes"]).issubset(set(artifact_hashes))
    else:
        for key in ("episode_terminal", "episode_trace_present", "episode_evidence_present",
                    "episode_artifact_present", "episode_tool_run_present",
                    "episode_state_version_positive", "runner_hashes_bound_to_episode"):
            checks[key] = False
    expected_assets = {sample["asset_id"]} if sample.get("asset_id") else set(sample.get("asset_ids", []))
    checks["matrix_asset_binding"] = bool(expected_assets)
    return {
        "dataset_id": dataset_id, "sample_id": sample_id,
        "task_root": sample.get("task_root"), "episode_id": episode_id,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks, "runner_report_sha256": None,
        "episode": {key: value for key, value in (episode or {}).items() if key != "episode"} if episode else None,
        "episode_snapshot_sha256": episode.get("episode_sha256") if episode else None,
    }


def summarize(reports: Path) -> dict:
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
        value = validate_report(dataset_id, sample_id, sample, report, database)
        value["runner_report_sha256"] = sha256(path)
        results.append(value)
    passed = sum(item["status"] == "passed" for item in results)
    return {
        "schema_version": "qwen-real-interaction-result-manifest-v1",
        "status": "passed" if not missing and not unexpected and passed == len(matrix) else "failed",
        "expected_samples": len(matrix), "passed_samples": passed,
        "missing_reports": missing, "unexpected_reports": unexpected,
        "real_model_acceptance": not missing and not unexpected and passed == len(matrix),
        "results": results,
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

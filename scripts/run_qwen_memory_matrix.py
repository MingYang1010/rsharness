#!/usr/bin/env python3
"""Run the controlled governed-evidence-memory matrix through Qwen."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_qwen_agent.py"
ISSUER_PATH = ROOT / "scripts" / "issue_agent_session.py"
CASES = ("correct-only", "conflict", "neighbor", "expired-only")
TREATMENTS = ("with-memory", "without-memory")


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "qwen_matrix_agent_runner", RUNNER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load Qwen runner")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _json(path: Path) -> Any:
    if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("report is unbounded or invalid: " + str(path))
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _episode_state(runtime: Path, episode_id: str) -> dict:
    database = runtime / "matrix-run" / "state" / "episodes.sqlite3"
    uri = "file:" + str(database) + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(
            "SELECT state_json FROM v2_episodes WHERE episode_id=?", (episode_id,)
        ).fetchone()
    if row is None:
        raise RuntimeError("episode absent from authoritative store")
    return json.loads(row[0])


def _terminal_state(runtime: Path, report: dict) -> dict:
    if not isinstance(report.get("episode_id"), str):
        return report.setdefault("terminal_state", {})
    state = _episode_state(runtime, report["episode_id"])
    terminal = report.setdefault(
        "terminal_state",
        {
            "episode_id": report["episode_id"],
            "status": state.get("status"),
            "final_answer": state.get("final_answer"),
        },
    )
    terminal["status"] = state.get("status")
    terminal["final_answer"] = state.get("final_answer")
    terminal["evaluation"] = state.get("evaluation")
    return terminal


def _retrieval(runtime: Path, report: dict) -> dict:
    episode_id = report.get("episode_id")
    if not isinstance(episode_id, str):
        return {"count": None, "memory_ids": []}
    database = runtime / "matrix-run" / "state" / "episodes.sqlite3"
    uri = "file:" + str(database) + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT ar.response_json FROM v2_action_results ar "
            "JOIN v2_tool_runs tr ON tr.episode_id=ar.episode_id "
            "AND tr.client_action_id=ar.client_action_id "
            "WHERE ar.episode_id=? AND tr.tool_id='memory.search' "
            "ORDER BY ar.created_at",
            (episode_id,),
        ).fetchall()
    identifiers: list[str] = []
    for (raw,) in rows:
        value = json.loads(raw)
        inline = value.get("observation", {}).get("items", [])
        for item in inline:
            result = item.get("inline") or {}
            if "matched_count" not in result:
                continue
            identifiers.extend(
                record.get("memory_id")
                for record in result.get("records", [])
                if isinstance(record.get("memory_id"), str)
            )
    return {"count": len(identifiers), "memory_ids": identifiers}


def _issue(job: Path, output: Path, registry: Path, backend: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "harness_api")
    completed = subprocess.run(
        [
            sys.executable,
            str(ISSUER_PATH),
            "--job",
            str(job),
            "--output",
            str(output),
            "--registry",
            str(registry),
            "--backend",
            backend,
            "--ttl-seconds",
            "21600",
            "--reviewed-public-task",
        ],
        cwd=str(ROOT),
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError("credential issuance failed for " + job.name)


def _metric(evaluation: dict, name: str) -> Any:
    for item in evaluation.get("metrics", []):
        if item.get("name") == name:
            return item.get("value")
    return None


def _actions(report: dict) -> list[str]:
    return [
        item["name"]
        for item in report.get("transcript", [])
        if isinstance(item.get("name"), str)
    ]


def _summary(reports: dict[tuple[str, str], dict], runtime: Path) -> dict:
    rows = []
    for (case, treatment), report in sorted(reports.items()):
        terminal = report.get("terminal_state", {})
        evaluation = terminal.get("evaluation", {})
        answer = terminal.get("final_answer") or {}
        submitted = answer.get("answer") or {}
        row = {
            "case": case,
            "treatment": treatment,
            "episode_id": report.get("episode_id"),
            "runner_status": report.get("status"),
            "runner_reason": report.get("reason"),
            "final_outcome": answer.get("outcome"),
            "submitted_label": submitted.get("label"),
            "cited_memory_ids": submitted.get("memory_ids", []),
            "retrieval": _retrieval(runtime, report),
            "aggregate_reward": evaluation.get("aggregate_reward"),
            "task_accuracy": _metric(evaluation, "task.accuracy"),
            "memory_faithfulness": _metric(
                evaluation, "evidence.memory_faithfulness"
            ),
            "process_efficiency": _metric(evaluation, "process.efficiency"),
            "model_calls": report.get("cost", {}).get("model_calls", 0),
            "prompt_tokens": report.get("cost", {}).get("prompt_tokens", 0),
            "completion_tokens": report.get("cost", {}).get("completion_tokens", 0),
            "total_tokens": report.get("cost", {}).get("total_tokens", 0),
            "elapsed_ms": report.get("elapsed_ms", 0),
            "actions": _actions(report),
        }
        rows.append(row)
    return {
        "schema_version": "qwen-evidence-memory-matrix-v1",
        "episodes": rows,
    }


def _job_paths(runtime: Path) -> dict[tuple[str, str], Path]:
    values = {}
    for case in CASES:
        for treatment in TREATMENTS:
            path = runtime / "benchmark-agent" / f"{case}-{treatment}.json"
            values[(case, treatment)] = path
    return values


def run_matrix(args: argparse.Namespace) -> dict:
    runtime = args.runtime.resolve()
    if args.runtime.is_symlink() or not runtime.is_relative_to(ROOT / "runtime"):
        raise ValueError("runtime must be under the project runtime directory")
    matrix = _json(runtime / "matrix-manifest.json")
    if matrix.get("task_count") != len(CASES) * len(TREATMENTS):
        raise ValueError("matrix manifest does not contain the expected eight jobs")
    jobs = _job_paths(runtime)
    if any(not job.is_file() for job in jobs.values()):
        raise ValueError("one or more matrix job files are unavailable")
    report_root = runtime / "matrix-run" / "reports"
    checkpoint_root = runtime / "matrix-run" / "checkpoints"
    registry = runtime / "matrix-run" / "credentials" / "registry.json"
    manifest_path = runtime / "matrix-run" / "matrix-manifest.json"
    if manifest_path.exists():
        raise ValueError("matrix run manifest already exists; choose a fresh runtime")
    reports: dict[tuple[str, str], dict] = {}
    runner = _load_runner()
    for key, job in jobs.items():
        credential = runtime / "matrix-run" / "credentials" / ("-".join(key))
        _issue(job, credential, registry, args.backend)
        client = runner.model_client(args.openai_base_url)
        try:
            report = runner.run(
                args.gateway,
                (credential / "agent-token").read_text().strip(),
                client,
                args.max_turns,
                checkpoint_root / ("-".join(key) + ".json"),
            )
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                close()
        _terminal_state(runtime, report)
        _write(report_root / ("-".join(key) + ".json"), report)
        reports[key] = report
        _write(
            manifest_path,
            {
                "schema_version": "qwen-evidence-memory-matrix-run-v1",
                "runtime": str(runtime),
                "completed": ["-".join(key) for key in reports],
                "summary": _summary(reports, runtime),
            },
        )
        print(
            json.dumps(
                {
                    "case": key[0],
                    "treatment": key[1],
                    "status": report.get("status"),
                    "episode_id": report.get("episode_id"),
                }
            ),
            flush=True,
        )
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--openai-base-url", required=True)
    parser.add_argument("--max-turns", type=int, default=3)
    args = parser.parse_args()
    try:
        run_matrix(args)
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": type(error).__name__,
                    "message": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

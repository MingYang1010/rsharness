#!/usr/bin/env python3
"""Run a WHU development package through scoped Qwen credentials, fail-open never."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_qwen_agent.py"
ISSUER_PATH = ROOT / "scripts" / "issue_agent_session.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load module: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bounded_json(path: Path, bound: int = 16 * 1024 * 1024) -> Any:
    if path.is_symlink() or path.stat().st_size > bound:
        raise ValueError("JSON file is missing, a symlink, or oversized: " + str(path))
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _safe_name(value: str) -> str:
    if not value or any(
        not (character.isalnum() or character in {"-", "_"})
        for character in value
    ):
        raise ValueError("unsafe sample id: " + value)
    return value


def _sample_id(job: dict) -> str:
    return _safe_name(str(job.get("sample_id", "")))


def _ensure_credential(
    *,
    job_path: Path,
    output: Path,
    registry: Path,
    backend: str,
    ttl_seconds: int,
) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    token_file = output / "agent-token"
    command = [
        sys.executable,
        str(ISSUER_PATH),
        "--job", str(job_path),
        "--output", str(output),
        "--backend", backend,
        "--registry", str(registry),
        "--ttl-seconds", str(ttl_seconds),
        "--reviewed-public-task",
    ]
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "harness_api")
    completed = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError("credential issuance failed for " + _sample_id(_bounded_json(job_path)))
    token = token_file.read_text().strip()
    if not token:
        raise RuntimeError("credential token unavailable after issuance")
    return token


def _summarize(jobs: list[dict], reports: dict[str, dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for sample, report in reports.items():
        job = next(item for item in jobs if _sample_id(item) == sample)
        truth = job.get("truth", {})
        key = str(truth.get("change_class", "unknown")) + "/" + str(truth.get("change_direction", "unknown"))
        groups.setdefault(key, []).append(report)
    group_rows = []
    for key, items in sorted(groups.items()):
        expected_abstained = 0
        unnecessary = 0
        false_confident = 0
        for report in items:
            transcript_names = [item.get("name") for item in report.get("transcript", []) if item.get("name")]
            if "answer.abstain" in transcript_names:
                expected_abstained += 1
            diagnostics = report.get("terminal_state", {}).get("evaluation", {}).get("diagnostics", {})
            unnecessary += bool(diagnostics.get("unnecessary_abstention"))
            false_confident += bool(diagnostics.get("false_confidence"))
        group_rows.append({
            "truth": key,
            "episodes": len(items),
            "passed": sum(report.get("status") == "passed" for report in items),
            "abstained": expected_abstained,
            "unnecessary_abstention": unnecessary,
            "false_confidence": false_confident,
        })
    return {
        "schema_version": "qwen-whu-batch-v1",
        "total": len(reports),
        "passed": sum(report.get("status") == "passed" for report in reports.values()),
        "failed": sum(report.get("status") != "passed" for report in reports.values()),
        "abstained": sum(
            any(item.get("name") == "answer.abstain" for item in report.get("transcript", []))
            for report in reports.values()
        ),
        "total_tokens": sum(report.get("cost", {}).get("total_tokens", 0) for report in reports.values()),
        "elapsed_ms": sum(report.get("elapsed_ms", 0) for report in reports.values()),
        "groups": group_rows,
    }


def batch_run(args: argparse.Namespace) -> dict:
    runtime = args.runtime.resolve()
    runtime_boundary = getattr(args, "runtime_boundary", ROOT / "runtime")
    if args.runtime.is_symlink() or not runtime.is_relative_to(runtime_boundary):
        raise ValueError("runtime must be under the project runtime directory")
    jobs_value = _bounded_json(runtime / "job.json")
    jobs = jobs_value.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("runtime job.json contains no jobs")
    selected = list(jobs)
    if args.samples:
        wanted = {item for item in args.samples.split(",") if item}
        selected = [job for job in jobs if _sample_id(job) in wanted]
        if len(selected) != len(wanted):
            raise ValueError("unknown or duplicate sample requested")
    manifest_path = args.manifest
    manifest = json.loads(manifest_path.read_text()) if (
        manifest_path and manifest_path.is_file()
    ) else {
        "schema_version": "qwen-whu-batch-v1",
        "runtime": str(runtime),
        "openai_base_url": args.openai_base_url,
        "jobs": [_sample_id(job) for job in selected],
    }
    reports: dict[str, dict] = {}
    report_root = runtime / "reports"
    checkpoint_root = runtime / "qwen-checkpoints"
    agent_root = runtime / "agent-qwen"
    for sample in manifest["jobs"]:
        job = next(item for item in selected if _sample_id(item) == sample)
        report_path = report_root / ("qwen-" + sample + ".json")
        if report_path.is_file():
            report = _bounded_json(report_path)
        else:
            token = _ensure_credential(
                job_path=runtime / "jobs" / (sample + ".json"),
                output=agent_root / sample,
                registry=args.registry,
                backend=args.backend,
                ttl_seconds=args.ttl_seconds,
            )
            client = args.model_client_factory(args.openai_base_url)
            try:
                report = args.run_episode(
                    args.gateway, token, client, args.max_turns,
                    checkpoint_root / (sample + ".json"),
                )
            finally:
                close = getattr(client, "close", None)
                if close is not None:
                    close()
            _write_json(report_path, report)
        reports[sample] = report
        _write_json(manifest_path, {**manifest, "completed": sorted(reports), "summary": _summarize(selected, reports)})
        print(json.dumps({"sample": sample, "status": report.get("status"), "reason": report.get("reason")}), flush=True)
    return {"manifest": manifest_path, "reports": reports, "summary": _summarize(selected, reports)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--backend", default="http://127.0.0.1:18080")
    parser.add_argument("--openai-base-url", required=True)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--samples")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--ttl-seconds", type=int, default=3600)
    args = parser.parse_args()
    runner = _load_module("qwen_agent_runner_batch", RUNNER_PATH)
    args.model_client_factory = runner.model_client
    args.run_episode = runner.run
    try:
        result = batch_run(args)
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error": type(error).__name__, "message": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

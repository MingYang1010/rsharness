#!/usr/bin/env python3
"""Run matched governed-evidence-memory Qwen episodes and preserve full cost."""
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
sys.path.insert(0, str(ROOT / "harness_api"))
RUNNER_PATH = ROOT / "scripts" / "run_qwen_agent.py"
ISSUER_PATH = ROOT / "scripts" / "issue_agent_session.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("qwen_memory_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Qwen runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> Any:
    if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("report is unbounded or invalid: " + str(path))
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _episode_state(runtime: Path, episode_id: str) -> dict:
    database = runtime / "pair-run" / "state" / "episodes.sqlite3"
    uri = "file:" + str(database) + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(
            "SELECT state_json FROM v2_episodes WHERE episode_id=?",
            (episode_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError("episode absent from authoritative store")
    return json.loads(row[0])


def _terminal_state(runtime: Path, report: dict) -> dict:
    state = _episode_state(runtime, report["episode_id"])
    terminal = report.setdefault("terminal_state", {
        "episode_id": report["episode_id"],
        "status": state.get("status"),
        "final_answer": state.get("final_answer"),
    })
    terminal["status"] = state.get("status")
    terminal["final_answer"] = state.get("final_answer")
    terminal["evaluation"] = state.get("evaluation")
    return terminal


def _issue(job: Path, output: Path, registry: Path, backend: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "harness_api")
    completed = subprocess.run(
        [sys.executable, str(ISSUER_PATH), "--job", str(job), "--output", str(output),
         "--registry", str(registry), "--backend", backend, "--ttl-seconds", "21600",
         "--reviewed-public-task"],
        cwd=str(ROOT), env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError("credential issuance failed")


def _summary(reports: dict[str, dict]) -> dict:
    rows=[]
    for name, report in reports.items():
        evaluation=report.get("terminal_state",{}).get("evaluation",{})
        rows.append({
            "condition":name,
            "episode_id":report.get("episode_id"),
            "status":report.get("status"),
            "outcome":report.get("terminal_state",{}).get("final_answer",{}).get("outcome"),
            "aggregate_reward":evaluation.get("aggregate_reward"),
            "metrics":{item["name"]:item["value"] for item in evaluation.get("metrics",[])},
            "model_calls":report.get("cost",{}).get("model_calls",0),
            "total_tokens":report.get("cost",{}).get("total_tokens",0),
            "elapsed_ms":report.get("elapsed_ms",0),
        })
    return {"schema_version":"qwen-evidence-memory-pair-v1","episodes":rows}


def run_pair(args: argparse.Namespace) -> dict:
    runtime=args.runtime.resolve()
    if args.runtime.is_symlink() or not runtime.is_relative_to(ROOT / "runtime"):
        raise ValueError("runtime must be under the project runtime directory")
    jobs={
        "with_memory":runtime/"benchmark-agent"/"with-memory-job.json",
        "without_memory":runtime/"benchmark-agent"/"without-memory-job.json",
    }
    if any(not job.is_file() for job in jobs.values()):
        raise ValueError("matched benchmark job files are unavailable")
    report_root=runtime/"pair-run"/"reports"
    checkpoint_root=runtime/"pair-run"/"checkpoints"
    registry=runtime/"pair-run"/"credentials"/"registry.json"
    manifest_path=runtime/"pair-run"/"pair-manifest.json"
    if manifest_path.exists():
        raise ValueError("pair manifest already exists; preserve it and choose a fresh runtime")
    reports={}
    runner=_load_runner()
    for condition,job in jobs.items():
        credential=runtime/"pair-run"/("agent-"+condition)
        _issue(job,credential,registry,args.backend)
        client=runner.model_client(args.openai_base_url)
        try:
            report=runner.run(
                args.gateway,(credential/"agent-token").read_text().strip(),client,
                args.max_turns,checkpoint_root/(condition+".json"),
            )
        finally:
            close=getattr(client,"close",None)
            if close is not None: close()
        _terminal_state(runtime,report)
        _write(report_root/(condition+".json"),report)
        reports[condition]=report
        _write(manifest_path,{"schema_version":"qwen-evidence-memory-pair-v1",
                              "runtime":str(runtime),"completed":sorted(reports),
                              "summary":_summary(reports)})
        print(json.dumps({"condition":condition,"status":report.get("status"),"episode_id":report.get("episode_id")}),flush=True)
    return reports


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime",required=True,type=Path)
    parser.add_argument("--gateway",required=True)
    parser.add_argument("--backend",required=True)
    parser.add_argument("--openai-base-url",required=True)
    parser.add_argument("--max-turns",type=int,default=3)
    args=parser.parse_args()
    try:
        run_pair(args)
        return 0
    except Exception as error:
        print(json.dumps({"status":"failed","error":type(error).__name__,"message":str(error)}),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())

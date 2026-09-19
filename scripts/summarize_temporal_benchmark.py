#!/usr/bin/env python3
"""Summarize immutable temporal benchmark acceptance reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2**20:
        raise ValueError("required bounded report is unavailable: %s" % path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("report is not an object: %s" % path)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output exists; preserve it and choose a fresh path")
    pack = _load(args.pack / "pack.json")
    correct = []
    counterfactual = []
    for run in pack["runs"]:
        case_root = args.pack / run["case_id"]
        gateway = _load(case_root / "reports" / "gateway.json")
        accepted = _load(case_root / "reports" / "accepted.json")
        cached = _load(case_root / "reports" / "cached.json")
        replay = _load(case_root / "reports" / "replay" / "execution.json")
        passed = bool(
            gateway.get("status") == "passed"
            and accepted.get("status") == "passed"
            and cached.get("status") == "passed"
            and replay.get("status") == "passed"
            and replay.get("semantic_evaluator_replayed") is True
            and replay.get("evaluation_matched") is True
            and replay.get("original_snapshot_unchanged") is True
            and replay.get("task_snapshot_unchanged") is True
            and replay.get("original_artifact_bytes_read") is False
        )
        record = {
            "case_id": run["case_id"],
            "passed": passed,
            "aggregate_reward": accepted.get("aggregate_reward"),
            "false_confidence": accepted.get("false_confidence"),
            "episode_id": accepted.get("episode_id"),
        }
        if run["counterfactual_false_confidence"]:
            counterfactual.append(record)
        else:
            correct.append(record)
    rejection = [
        item
        for item in correct
        if item["case_id"] != "valid-pair"
    ]
    result = {
        "status": "passed"
        if (
            len(correct) == 4
            and all(item["passed"] for item in correct)
            and len(rejection) == 3
            and all(item["false_confidence"] is False for item in rejection)
            and len(counterfactual) == 1
            and counterfactual[0]["passed"]
            and counterfactual[0]["false_confidence"] is True
        )
        else "failed",
        "correct_run_success": {
            "passed": sum(item["passed"] for item in correct),
            "total": len(correct),
        },
        "correct_rejection_false_confidence": {
            "count": sum(item["false_confidence"] is True for item in rejection),
            "total": len(rejection),
        },
        "counterfactual_detection": {
            "passed": sum(
                item["passed"] and item["false_confidence"] is True
                for item in counterfactual
            ),
            "total": len(counterfactual),
        },
        "runs": correct + counterfactual,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

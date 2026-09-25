#!/usr/bin/env python3
"""Verify hidden-label scoring after a scoped Agent gateway run."""
from __future__ import annotations

import argparse
import json
import time

from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--gateway-report", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("report exists; preserve it and choose a fresh path")
    job = json.loads(args.job.read_text())
    gateway = json.loads(args.gateway_report.read_text())
    admission = json.loads(Path("/smoke/admission.json").read_text())
    provider = json.loads(Path("/smoke/inputs.json").read_text())
    assert gateway["status"] == "passed"
    assert gateway["sample_id"] == job["sample_id"]
    assert gateway["oracle_answer_injected"] is True
    assert admission["labels_provider_mounted"] is False
    assert admission["redistribution_allowed"] is False
    assert set(provider) == {
        item["before_asset_id"]
        for item in json.loads(Path("/smoke/job.json").read_text())["jobs"]
    } | {
        item["after_asset_id"]
        for item in json.loads(Path("/smoke/job.json").read_text())["jobs"]
    }
    with httpx.Client(
        base_url="http://harness:8000",
        timeout=45,
        trust_env=False,
    ) as client:
        for attempt in range(30):
            try:
                client.get("/healthz").raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 29:
                    raise
                time.sleep(1)

        def request(method: str, path: str):
            response = client.request(method, path)
            response.raise_for_status()
            return response.json()["data"]

        task_ref = job["task_ref"]
        manifest = request(
            "GET",
            (
                "/v2/tasks/"
                + task_ref["task_id"]
                + "/versions/"
                + task_ref["task_version"]
            ),
        )["manifest"]
        assert manifest["task"]["inputs"] == [
            job["before_asset_id"],
            job["after_asset_id"],
        ]
        labels = [
            asset
            for asset in manifest["assets"]
            if "evaluator" in asset["roles"]
        ]
        assert len(labels) == 2
        assert all(
            "building_label" in asset["roles"]
            and asset["asset_id"] not in manifest["task"]["inputs"]
            and asset["asset_id"] not in provider
            for asset in labels
        )
        episode_id = gateway["episode_id"]
        state = request(
            "GET", "/v2/episodes/" + episode_id + "/state"
        )["state"]
        evaluation = state["evaluation"]
        assert evaluation["status"] == "completed"
        assert evaluation["evaluator_id"] == "whu-building-change-v1"
        assert evaluation["evaluator_version"] == "1.0.0"
        metric_values = {
            metric["name"]: metric["value"]
            for metric in evaluation["metrics"]
        }
        expected_outcome = job.get("expected_outcome", "submitted")
        expected_metrics = {
            "task.change_class_accuracy": 1.0 if expected_outcome == "submitted" else 0.0,
            "task.direction_accuracy": 1.0 if expected_outcome == "submitted" else 0.0,
            "task.changed_fraction_score": 1.0 if expected_outcome == "submitted" else 0.0,
            "answer.abstention_correctness": 1.0,
            "evidence.faithfulness": 1.0 if expected_outcome == "submitted" else 0.0,
            "process.efficiency": (
                1.0
                if expected_outcome == "submitted"
                else 0.0
            ),
        }
        assert metric_values.keys() == expected_metrics.keys()
        for name, expected in expected_metrics.items():
            if name == "process.efficiency":
                assert metric_values[name] >= 0.75, name
            else:
                assert metric_values[name] == expected, name
        assert evaluation["aggregate_reward"] >= (
            0.95 if expected_outcome == "submitted" else 0.1
        )
        if expected_outcome == "abstained":
            assert evaluation["diagnostics"]["false_confidence"] is False
            assert evaluation["diagnostics"]["unnecessary_abstention"] is False
            assert evaluation["diagnostics"]["insufficient_input_asset_ids"] == [
                job["after_asset_id"]
            ]
        truth = evaluation["diagnostics"]["truth"]
        for key, expected in job["truth"].items():
            assert truth[key] == expected
        stored = request(
            "GET", "/v2/episodes/" + episode_id + "/evaluation"
        )["evaluation"]
        assert stored == evaluation
        structural = request(
            "POST", "/v2/episodes/" + episode_id + "/replay"
        )
        assert structural["status"] == "passed"
        assert all(check["passed"] for check in structural["checks"])
        trace = request(
            "GET", "/v2/episodes/" + episode_id + "/trace"
        )
        report = {
            "status": "passed",
            "sample_id": job["sample_id"],
            "episode_id": episode_id,
            "task_manifest_hash": gateway["task_manifest_hash"],
            "metric_values": metric_values,
            "aggregate_reward": evaluation["aggregate_reward"],
            "truth": truth,
            "structural_replay": structural,
            "trace_hash": trace["trace_hash"],
            "trace_events": len(trace["events"]),
            "hidden_label_assets": [asset["asset_id"] for asset in labels],
            "hidden_labels_absent_from_provider": True,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "status",
                        "sample_id",
                        "episode_id",
                        "aggregate_reward",
                        "trace_events",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()

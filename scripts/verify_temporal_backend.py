#!/usr/bin/env python3
"""Verify temporal scoring and structural replay from the operator network."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import httpx


EXPECTED_METRICS = {
    "temporal.validity": 1.0,
    "spatial.coverage": 1.0,
    "answer.abstention_correctness": 1.0,
    "evidence.faithfulness": 1.0,
    "process.efficiency": 1.0,
}


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
    provider = json.loads(Path("/smoke/native-inputs.json").read_text())
    assert gateway["status"] == "passed"
    assert gateway["case_id"] == job["case_id"]
    assert gateway["oracle_answer_injected"] is True
    assert gateway["gateway_hidden_evaluation_absent"] is True
    assert admission["imagery_committed_to_git"] is False
    assert admission["license_scope"] == "local-research"
    assert admission["not_redistribution_authorization"] is True
    assert len(provider) == 6

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
        assert manifest["task_manifest_hash"] == job["task_manifest_hash"]
        assert set(manifest["task"]["inputs"]) == set(provider)
        assert len(manifest["task"]["inputs"]) == 6
        assert manifest["scenario"]["allowed_tools"] == [
            "temporal.select_align"
        ]
        evaluator = manifest["evaluator"]
        assert evaluator["evaluator_id"] == "temporal-selection-v1"
        assert evaluator["evaluator_version"] == "1.0.0"
        for key, expected in job["truth"].items():
            assert evaluator["config"][key] == expected

        episode_id = gateway["episode_id"]
        state = request(
            "GET", "/v2/episodes/" + episode_id + "/state"
        )["state"]
        evaluation = state["evaluation"]
        assert evaluation["status"] == "completed"
        assert evaluation["evaluator_id"] == "temporal-selection-v1"
        assert evaluation["evaluator_version"] == "1.0.0"
        metric_values = {
            metric["name"]: metric["value"]
            for metric in evaluation["metrics"]
        }
        expected_metrics = dict(EXPECTED_METRICS)
        counterfactual = job["counterfactual_false_confidence"]
        if counterfactual:
            expected_metrics["answer.abstention_correctness"] = 0.0
            expected_metrics["evidence.faithfulness"] = 0.0
        assert metric_values == expected_metrics
        assert math.isclose(
            evaluation["aggregate_reward"],
            0.6 if counterfactual else 1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert evaluation["diagnostics"]["false_confidence"] is counterfactual
        assert evaluation["diagnostics"]["unnecessary_abstention"] is False
        assert evaluation["diagnostics"]["expected_selection_reason"] == job[
            "truth"
        ]["expected_selection_reason"]
        stored = request(
            "GET", "/v2/episodes/" + episode_id + "/evaluation"
        )["evaluation"]
        assert stored == evaluation
        structural = request(
            "POST", "/v2/episodes/" + episode_id + "/replay"
        )
        assert structural["status"] == "passed"
        assert all(check["passed"] for check in structural["checks"])
        trace = request("GET", "/v2/episodes/" + episode_id + "/trace")
        report = {
            "status": "passed",
            "scope": "operator-only hidden temporal evaluation",
            "case_id": job["case_id"],
            "episode_id": episode_id,
            "task_manifest_hash": gateway["task_manifest_hash"],
            "metric_values": metric_values,
            "aggregate_reward": evaluation["aggregate_reward"],
            "false_confidence": evaluation["diagnostics"]["false_confidence"],
            "counterfactual_false_confidence": counterfactual,
            "structural_replay": structural,
            "trace_hash": trace["trace_hash"],
            "trace_events": len(trace["events"]),
            "hidden_evaluator_absent_from_agent": True,
            "native_input_count": len(provider),
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
                        "case_id",
                        "episode_id",
                        "aggregate_reward",
                        "false_confidence",
                        "trace_events",
                    )
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

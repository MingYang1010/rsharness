#!/usr/bin/env python3
"""Run one temporal benchmark oracle through the scoped Agent gateway."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--token", default=Path("/run/agent-token"), type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--cached-from", type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("report exists; preserve it and choose a fresh path")
    job = json.loads(args.job.read_text())
    token = args.token.read_text().strip()
    with httpx.Client(
        base_url="http://agent-gateway:8083",
        timeout=45,
        trust_env=False,
        headers={"Authorization": "Bearer " + token},
    ) as client:
        for attempt in range(30):
            try:
                client.get("/healthz").raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 29:
                    raise
                time.sleep(1)

        def request(method: str, path: str, body=None):
            response = client.request(method, path, json=body)
            response.raise_for_status()
            return response.json()

        if args.cached_from is not None:
            original = json.loads(args.cached_from.read_text())
            for action in original["actions"]:
                assert request("POST", "/agent/step", action["request"]) == action["response"]
            artifact = original.get("artifact")
            if artifact is not None:
                content = client.get(
                    "/agent/artifacts/" + artifact["artifact_id"] + "/content"
                )
                content.raise_for_status()
                assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
            report = {
                "status": "passed",
                "scope": "cached gateway actions after service recreation",
                "case_id": job["case_id"],
                "episode_id": original["episode_id"],
                "cached_actions": len(original["actions"]),
                "artifact_rechecked": artifact is not None,
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
            print(json.dumps(report, sort_keys=True))
            return

        denied = []
        for host, port in (
            ("harness", 8000),
            ("provider", 8081),
            ("raster", 8084),
            ("storage", 8082),
        ):
            try:
                with socket.create_connection((host, port), timeout=2):
                    raise AssertionError("Agent reached a private service")
            except OSError:
                denied.append(host)
        if os.environ.get("EO_TEST_BACKEND_IP"):
            try:
                with socket.create_connection(
                    (os.environ["EO_TEST_BACKEND_IP"], 8000), timeout=2
                ):
                    raise AssertionError("Agent reached the backend host")
            except OSError:
                denied.append("backend-ip")
        for private_path in (
            "/v2/capabilities",
            "/v2/tasks/private/versions/1.0.0",
            "/docs",
            "/agent/reset",
        ):
            assert client.get(private_path).status_code == 404
        assert client.get(
            "/agent/session",
            headers={"Authorization": "Bearer invalid"},
        ).status_code == 401

        session = request("GET", "/agent/session")
        assert session["task"]["task_id"] == job["task_ref"]["task_id"]
        assert session["task"]["allowed_tools"] == ["temporal.select_align"]
        assert len(session["task"]["input_asset_refs"]) == 6
        assert "evaluation" not in session["state"]
        assert "expected_selection_reason" not in json.dumps(session)
        state = session["state"]
        actions = []

        def step(action: dict) -> dict:
            nonlocal state
            body = {
                "client_action_id": "temporal-%s-%d" % (job["case_id"], len(actions)),
                "expected_state_version": state["state_version"],
                "action": action,
            }
            response = request("POST", "/agent/step", body)
            assert request("POST", "/agent/step", body) == response
            actions.append({"request": body, "response": response})
            state = response["state"]
            return response

        result = step(
            {
                "type": "tool.invoke",
                "tool_id": "temporal.select_align",
                "arguments": job["arguments"],
            }
        )
        tool = result["observation"]["items"][0]["inline"]
        assert tool["selection"]["status"] == job["truth"]["expected_selection_status"]
        assert tool["selection"]["reason"] == job["truth"]["expected_selection_reason"]
        artifact = None
        evidence_ids = []
        if tool["selection"]["status"] == "selected":
            assert result["observation"]["items"][1]["type"] == "temporal_stack"
            artifact_id = result["observation"]["items"][1]["artifact_ref"]
            artifact = request("GET", "/agent/artifacts/" + artifact_id)["artifact"]
            assert artifact["lineage"]["input_refs"] == job["truth"]["expected_input_asset_ids"]
            content = client.get("/agent/artifacts/" + artifact_id + "/content")
            content.raise_for_status()
            assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
            evidence_id = "ev-temporal-" + job["case_id"]
            step(
                {
                    "type": "memory.save_evidence",
                    "evidence": {
                        "evidence_id": evidence_id,
                        "claim_id": "temporal-selection",
                        "source_ref": artifact_id,
                        "selector": {
                            "bbox": artifact["spatial"]["bbox"],
                            "time_range": artifact["temporal"],
                        },
                        "description": "Full aligned two-date temporal stack.",
                        "frozen_sha256": artifact["sha256"],
                    },
                }
            )
            evidence_ids.append(evidence_id)
        else:
            assert len(result["observation"]["items"]) == 1

        if job["expected_oracle_outcome"] == "submitted":
            final = step(
                {
                    "type": "answer.submit",
                    "answer": {
                        "label": "valid_pair",
                        "claims": [
                            {
                                "claim_id": "temporal-selection",
                                "text": "The reviewed temporal pair is valid.",
                            }
                        ],
                    },
                    "confidence": 1.0,
                    "evidence_ids": evidence_ids,
                }
            )
        else:
            final = step(
                {
                    "type": "answer.abstain",
                    "rationale": "Temporal selection rejected: %s."
                    % job["truth"]["expected_selection_reason"],
                    "evidence_ids": [],
                }
            )
        assert final["terminated"]
        assert "evaluation" not in final["state"]
        report = {
            "status": "passed",
            "scope": "operator oracle through scoped Agent gateway; no model reasoning",
            "case_id": job["case_id"],
            "episode_id": state["episode_id"],
            "task_manifest_hash": state["task_manifest_hash"],
            "actions": actions,
            "action_count": len(actions),
            "artifact": artifact,
            "observed_selection": tool["selection"],
            "oracle_outcome": job["expected_oracle_outcome"],
            "counterfactual_false_confidence": job[
                "counterfactual_false_confidence"
            ],
            "private_connections_denied": denied,
            "gateway_hidden_evaluation_absent": True,
            "oracle_answer_injected": True,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "case_id": report["case_id"],
                    "episode_id": report["episode_id"],
                    "action_count": report["action_count"],
                    "artifact": artifact is not None,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

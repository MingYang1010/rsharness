#!/usr/bin/env python3
"""Run one WHU task through the scoped Agent gateway only."""
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

        denied = []
        for host, port in (
            ("harness", 8000),
            ("provider", 8081),
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
        assert session["task"]["input_asset_refs"] == [
            job["before_asset_id"],
            job["after_asset_id"],
        ]
        assert session["task"]["allowed_tools"] == [
            "catalog.search", "catalog.inspect_asset", "eo_gym.crop"
        ]
        assert "evaluation" not in session["state"]
        serialized = json.dumps(session)
        assert "before-label" not in serialized
        assert "after-label" not in serialized
        state = session["state"]
        actions = []
        artifacts = []
        def step(action: dict) -> dict:
            nonlocal state
            body = {
                "client_action_id": (
                    "whu-" + job["sample_id"] + "-action-" + str(len(actions))
                ),
                "expected_state_version": state["state_version"],
                "action": action,
            }
            response = request("POST", "/agent/step", body)
            assert request("POST", "/agent/step", body) == response
            actions.append({"request": body, "response": response})
            state = response["state"]
            return response

        catalog = request(
            "POST",
            "/agent/step",
            {
                "client_action_id": (
                    "whu-" + job["sample_id"] + "-action-catalog-" + str(len(actions))
                ),
                "expected_state_version": state["state_version"],
                "action": {
                    "type": "tool.invoke",
                    "tool_id": "catalog.search",
                    "arguments": {"limit": 20},
                },
            },
        )
        actions.append({"request": "catalog-search", "response": catalog})
        state = catalog["state"]
        expected_outcome = job.get("expected_outcome", "submitted")
        if expected_outcome == "abstained":
            final = step(
                {
                    "type": "answer.abstain",
                    "rationale": (
                        "Public input coverage is below the task policy; "
                        "the temporal comparison is not answerable."
                    ),
                    "evidence_ids": [],
                }
            )
            assert final["terminated"]
            assert "evaluation" not in final["state"]
            report = {
                "status": "passed",
                "scope": "operator oracle through scoped Agent gateway; no model reasoning",
                "sample_id": job["sample_id"],
                "expected_outcome": expected_outcome,
                "episode_id": state["episode_id"],
                "task_manifest_hash": state["task_manifest_hash"],
                "actions": len(actions) + 1,
                "catalog_observation": catalog["observation"],
                "private_connections_denied": denied,
                "gateway_hidden_labels_absent": True,
                "oracle_answer_injected": True,
                "gateway_hidden_evaluation_absent": True,
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x") as stream:
                json.dump(report, stream, indent=2)
                stream.write(chr(10))
            print(
                json.dumps(
                    {
                        key: report[key]
                        for key in (
                            "status", "sample_id", "expected_outcome",
                            "episode_id", "actions",
                        )
                    }
                )
            )
            return


        for period, asset_id in (
            ("before", job["before_asset_id"]),
            ("after", job["after_asset_id"]),
        ):
            result = step(
                {
                    "type": "tool.invoke",
                    "tool_id": "eo_gym.crop",
                    "arguments": {
                        "asset_id": asset_id,
                        "aoi": [0.0, 0.0, 1.0, 1.0],
                    },
                }
            )
            artifact_id = result["observation"]["items"][1]["artifact_ref"]
            metadata = request(
                "GET", "/agent/artifacts/" + artifact_id
            )["artifact"]
            assert metadata["pixel"] == {
                "coordinate_system": "pixel",
                "width": job["width"],
                "height": job["height"],
                "channels": 3,
            }
            content = client.get(
                "/agent/artifacts/" + artifact_id + "/content"
            )
            content.raise_for_status()
            assert (
                hashlib.sha256(content.content).hexdigest()
                == metadata["sha256"]
            )
            artifacts.append(
                {
                    "period": period,
                    "input_asset_id": asset_id,
                    "artifact": metadata,
                }
            )

        evidence_ids = []
        for record in artifacts:
            artifact = record["artifact"]
            evidence_id = (
                "ev-whu-" + job["sample_id"] + "-" + record["period"]
            )
            step(
                {
                    "type": "memory.save_evidence",
                    "evidence": {
                        "evidence_id": evidence_id,
                        "claim_id": "claim-building-change",
                        "source_ref": artifact["artifact_id"],
                        "selector": {
                            "pixel_window": [
                                0,
                                0,
                                job["width"],
                                job["height"],
                            ]
                        },
                        "description": (
                            "Full "
                            + record["period"]
                            + " image crop for temporal comparison."
                        ),
                        "frozen_sha256": artifact["sha256"],
                    },
                }
            )
            evidence_ids.append(evidence_id)

        truth = job["truth"]
        final = step(
            {
                "type": "answer.submit",
                "answer": {
                    "change_class": truth["change_class"],
                    "change_direction": truth["change_direction"],
                    "changed_fraction": truth["changed_fraction"],
                    "claims": [
                        {
                            "claim_id": "claim-building-change",
                            "text": (
                                "Building change is supported by both "
                                "full-image temporal crops."
                            ),
                        }
                    ],
                },
                "confidence": 1.0,
                "evidence_ids": evidence_ids,
            }
        )
        assert final["terminated"]
        assert "evaluation" not in final["state"]
        report = {
            "status": "passed",
            "scope": "operator oracle through scoped Agent gateway; no model reasoning",
            "sample_id": job["sample_id"],
            "episode_id": state["episode_id"],
            "task_manifest_hash": state["task_manifest_hash"],
            "actions": len(actions),
            "artifacts": artifacts,
            "private_connections_denied": denied,
            "gateway_hidden_labels_absent": True,
            "oracle_answer_injected": True,
            "gateway_hidden_evaluation_absent": True,
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
                        "actions",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()

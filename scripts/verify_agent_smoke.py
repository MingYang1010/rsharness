#!/usr/bin/env python3
"""Scripted Agent acceptance; this container has no backend network or task files."""
import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    token = Path("/run/agent-token").read_text().strip()
    with httpx.Client(base_url="http://agent-gateway:8083", timeout=45, trust_env=False,
                      headers={"Authorization": "Bearer " + token}) as client:
        for attempt in range(30):
            try:
                client.get("/healthz").raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 29:
                    raise
                time.sleep(1)

        def request(method, path, body=None):
            response = client.request(method, path, json=body)
            response.raise_for_status()
            return response.json()

        denied_network = []
        for host, port in [("harness", 8000), ("provider", 8081), ("storage", 8082)]:
            try:
                with socket.create_connection((host, port), timeout=2):
                    raise AssertionError("Agent reached private backend")
            except OSError:
                denied_network.append(host)
        if os.environ.get("EO_TEST_BACKEND_IP"):
            try:
                with socket.create_connection((os.environ["EO_TEST_BACKEND_IP"], 8000), timeout=2):
                    raise AssertionError("Agent reached private backend IP")
            except OSError:
                denied_network.append("backend-ip")
        assert client.get("/agent/session", headers={"Authorization": "Bearer invalid"}).status_code == 401
        for path in ("/v2/capabilities", "/v2/tasks/hidden/versions/1.0.0", "/docs", "/agent/reset"):
            assert client.get(path).status_code == 404
        assert client.get("/agent/state?episode_id=other").status_code == 422
        checkpoint_path = Path("/reports/agent-checkpoint.json")
        if args.resume:
            checkpoint = json.loads(checkpoint_path.read_text())
            for record in checkpoint["actions"]:
                assert request("POST", "/agent/step", record["request"]) == record["response"]
            assert request("GET", "/agent/state")["state"] == checkpoint["state"]
            for artifact in checkpoint["artifacts"]:
                content = client.get("/agent/artifacts/" + artifact["artifact_id"] + "/content")
                content.raise_for_status()
                assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
            report = {"status": "passed", "episode_id": checkpoint["state"]["episode_id"],
                      "cached_actions": len(checkpoint["actions"]), "images": len(checkpoint["artifacts"]),
                      "private_connections_denied": denied_network, "scope": "scripted scoped Agent, not Qwen or semantic scoring"}
            Path("/reports/agent-resume.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report))
            return
        session = request("GET", "/agent/session")
        state = session["state"]
        assert "evaluation" not in state and "provenance" not in session["observation"]
        assert "catalog.search" in session["tool_schemas"]
        actions, artifacts = [], []

        def step(action):
            nonlocal state
            body = {"client_action_id": "agent-smoke-" + str(len(actions)),
                    "expected_state_version": state["state_version"], "action": action}
            response = request("POST", "/agent/step", body)
            assert request("POST", "/agent/step", body) == response
            actions.append({"request": body, "response": response})
            state = response["state"]
            return response

        def tool(name, arguments):
            return step({"type": "tool.invoke", "tool_id": name, "arguments": arguments})

        assets, offset = [], 0
        while offset is not None:
            result = tool("catalog.search", {"limit": 2, "offset": offset})["observation"]["items"][0]["inline"]
            assets.extend(result["assets"])
            offset = result["next_offset"]
        assert len(assets) == 3
        assert [a["asset_id"] for a in assets] == session["task"]["input_asset_refs"]
        for asset in assets:
            assert not {"uri", "source", "roles"}.intersection(asset)
            inspected = tool("catalog.inspect_asset", {"asset_id": asset["asset_id"]})["observation"]["items"][0]["inline"]["asset"]
            assert inspected == asset
            crop = tool("eo_gym.crop", {"asset_id": inspected["asset_id"], "aoi": [0.25, 0.25, 0.75, 0.75]})
            ref = crop["observation"]["items"][1]["artifact_ref"]
            metadata = request("GET", "/agent/artifacts/" + ref)["artifact"]
            assert "uri" not in metadata
            content = client.get("/agent/artifacts/" + ref + "/content")
            content.raise_for_status()
            assert hashlib.sha256(content.content).hexdigest() == metadata["sha256"]
            artifacts.append(metadata)
        evidence_ids = []
        for index, artifact in enumerate(artifacts):
            evidence_id = f"ev-agent-{index}"
            step({"type": "memory.save_evidence", "evidence": {"evidence_id": evidence_id, "claim_id": "claim-crops",
                "source_ref": artifact["artifact_id"], "selector": {"pixel_window": [0, 0, artifact["pixel"]["width"], artifact["pixel"]["height"]]},
                "description": "Reviewed central crop", "frozen_sha256": artifact["sha256"]}})
            evidence_ids.append(evidence_id)
        final = step({"type": "answer.submit", "answer": {"label": "crop-dimensions", "confidence": 1.0,
            "claims": [{"claim_id": "claim-crops", "text": "Three central crops are supported by frozen artifacts."}]},
            "confidence": 1.0, "evidence_ids": evidence_ids})
        assert final["terminated"] and "evaluation" not in final["state"]
        checkpoint = {"state": state, "actions": actions, "artifacts": artifacts, "private_connections_denied": denied_network}
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n")
        print(json.dumps({"status": "checkpoint_saved", "episode_id": state["episode_id"],
                          "actions": len(actions), "images": len(artifacts), "private_connections_denied": denied_network}))


if __name__ == "__main__":
    main()

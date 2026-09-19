#!/usr/bin/env python3
"""Agent-only real-image acceptance: same pixel bytes, distinct crop derivations."""
import argparse
import hashlib
import json
import socket
from pathlib import Path

import httpx


def write_report(path, report):
    content = json.dumps(report, indent=2).encode()
    if len(content) > 1024 * 1024:
        raise ValueError("report exceeds bound")
    with path.open("xb") as stream:
        stream.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    token = Path("/run/agent-token").read_text().strip()
    for host, port in (("harness", 8000), ("provider", 8081), ("storage", 8082)):
        try:
            with socket.create_connection((host, port), timeout=2):
                raise AssertionError("Agent reached private backend")
        except OSError:
            pass
    with httpx.Client(base_url="http://agent-gateway:8083", timeout=45, trust_env=False,
                      headers={"Authorization": "Bearer " + token}) as client:
        def request(method, path, body=None):
            response = client.request(method, path, json=body)
            response.raise_for_status()
            return response.json()

        checkpoint_path = Path("/reports/derivation-checkpoint.json")
        if args.resume:
            checkpoint = json.loads(checkpoint_path.read_text())
            for record in checkpoint["actions"]:
                assert request("POST", "/agent/step", record["request"]) == record["response"]
            assert request("GET", "/agent/state")["state"] == checkpoint["state"]
            for artifact in checkpoint["artifacts"]:
                assert request("GET", "/agent/artifacts/" + artifact["artifact_id"])["artifact"] == artifact
                content = client.get("/agent/artifacts/" + artifact["artifact_id"] + "/content")
                content.raise_for_status()
                assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
            report = {"status": "passed", "episode_id": checkpoint["state"]["episode_id"],
                      "cached_actions": len(checkpoint["actions"]), "derivations": 2, "content_objects": 1}
            write_report(Path("/reports/derivation-resume.json"), report)
            print(json.dumps(report))
            return
        if checkpoint_path.exists():
            raise SystemExit("preserve existing checkpoint; use --resume")
        session = request("GET", "/agent/session")
        assert session["task"]["artifact_identity"] == "derivation-sha256-v1"
        state, actions, artifacts, crops = session["state"], [], [], []

        def step(action):
            nonlocal state
            body = {"client_action_id": "derivation-" + str(len(actions)),
                    "expected_state_version": state["state_version"], "action": action}
            response = request("POST", "/agent/step", body)
            assert request("POST", "/agent/step", body) == response
            state = response["state"]
            actions.append({"request": body, "response": response})
            return response

        def tool(name, arguments):
            return step({"type": "tool.invoke", "tool_id": name, "arguments": arguments})

        found = tool("catalog.search", {})["observation"]["items"][0]["inline"]["assets"]
        assert len(found) == 1
        asset = tool("catalog.inspect_asset", {"asset_id": found[0]["asset_id"]})["observation"]["items"][0]["inline"]["asset"]
        # At the permitted decode sizes these distinct normalized windows round
        # to identical pixel windows. Provider still executes each crop worker.
        for aoi in ([.25, .25, .75, .75], [.25000001, .25, .75, .75]):
            crop = tool("eo_gym.crop", {"asset_id": asset["asset_id"], "aoi": aoi})
            crops.append(crop["observation"]["items"][0]["inline"])
            ref = crop["observation"]["items"][1]["artifact_ref"]
            artifact = request("GET", "/agent/artifacts/" + ref)["artifact"]
            content = client.get("/agent/artifacts/" + ref + "/content")
            content.raise_for_status()
            assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
            assert ref[4:] != artifact["sha256"]
            assert "uri" not in artifact
            artifacts.append(artifact)
        assert crops[0]["bbox_px"] == crops[1]["bbox_px"]
        assert crops[0]["aoi_norm"] != crops[1]["aoi_norm"]
        assert artifacts[0]["sha256"] == artifacts[1]["sha256"]
        assert artifacts[0]["artifact_id"] != artifacts[1]["artifact_id"]
        assert artifacts[0]["lineage"]["parameters_hash"] != artifacts[1]["lineage"]["parameters_hash"]
        evidence_ids = []
        for index, artifact in enumerate(artifacts):
            evidence_id = f"ev-derivation-{index}"
            step({"type": "memory.save_evidence", "evidence": {"evidence_id": evidence_id, "claim_id": "claim-crops",
                "source_ref": artifact["artifact_id"], "selector": {"pixel_window": [0, 0, artifact["pixel"]["width"], artifact["pixel"]["height"]]},
                "description": "Distinct crop derivation with verified bytes", "frozen_sha256": artifact["sha256"]}})
            evidence_ids.append(evidence_id)
        final = step({"type": "answer.submit", "answer": {"label": "same-pixels-distinct-derivations", "confidence": 1.0,
            "claims": [{"claim_id": "claim-crops", "text": "Two requested crop derivations have identical pixel contents."}]},
            "confidence": 1.0, "evidence_ids": evidence_ids})
        assert final["terminated"]
        assert [e["source_ref"] for e in state["evidence_refs"]] == [a["artifact_id"] for a in artifacts]
        write_report(checkpoint_path, {"state": state, "actions": actions, "artifacts": artifacts,
                     "task_manifest_hash": state["task_manifest_hash"], "scope": "scripted interaction, not Qwen or semantic scoring"})
        print(json.dumps({"status": "checkpoint_saved", "episode_id": state["episode_id"], "actions": len(actions),
                          "derivations": 2, "content_objects": 1, "content_sha256": artifacts[0]["sha256"]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""HTTP catalog -> inspect -> EO-Gym crops; cached queries survive recreation."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    job = json.loads(Path("/smoke/job.json").read_text())
    report_path = Path("/reports/catalog-checkpoint.json")
    with httpx.Client(base_url="http://harness:8000", timeout=40, trust_env=False) as client:
        for attempt in range(30):
            try:
                client.get("/healthz").raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 29:
                    raise
                time.sleep(1)

        def request(method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()["data"]

        if args.resume:
            report = json.loads(report_path.read_text())
            assert report["job"] == job
            episode = report["episode_id"]
            for record in report["actions"]:
                assert request("POST", f"/v2/episodes/{episode}/step", json=record["request"]) == record["response"]
            assert request("GET", f"/v2/episodes/{episode}/state")["state"] == report["state"]
            for artifact in report["artifacts"]:
                content = client.get(f"/v2/artifacts/{artifact['artifact_id']}/content", params={"episode_id": episode})
                content.raise_for_status()
                assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
            result = {"status": "passed", "episode_id": episode, "cached_actions": len(report["actions"]),
                      "images": len(report["artifacts"]), "scope": "scripted HTTP and restart, not Qwen or semantic scoring"}
            Path("/reports/catalog-resume.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result))
            return
        capabilities = request("GET", "/v2/capabilities")
        assert {"catalog.search", "catalog.inspect_asset", "eo_gym.crop"}.issubset(capabilities["tools"])
        reset = request("POST", "/v2/reset", json={"task_ref": job["task_ref"], "seed": job["seed"]})
        episode, state = reset["episode_id"], reset["state"]
        actions, artifacts = [], []

        def step(action):
            nonlocal state
            body = {"client_action_id": "catalog-smoke-" + str(len(actions)),
                    "expected_state_version": state["state_version"], "action": action}
            response = request("POST", f"/v2/episodes/{episode}/step", json=body)
            assert request("POST", f"/v2/episodes/{episode}/step", json=body) == response
            state = response["state"]
            actions.append({"request": body, "response": response})
            return response

        def tool(name, arguments):
            return step({"type": "tool.invoke", "tool_id": name, "arguments": arguments})

        discovered, offset = [], 0
        while offset is not None:
            response = tool("catalog.search", {"limit": 2, "offset": offset})
            assert len(response["observation"]["items"]) == 1
            result = response["observation"]["items"][0]["inline"]
            discovered.extend(result["assets"])
            offset = result["next_offset"]
        assert [a["asset_id"] for a in discovered] == [a["asset_id"] for a in job["expected"]]
        for asset in discovered:
            assert not {"uri", "source", "roles", "geometry"}.intersection(asset)
            inspected = tool("catalog.inspect_asset", {"asset_id": asset["asset_id"]})["observation"]["items"][0]["inline"]["asset"]
            assert inspected == asset and asset["spatial"] is None
            crop = tool("eo_gym.crop", {"asset_id": inspected["asset_id"], "aoi": [0.25, 0.25, 0.75, 0.75]})
            ref = crop["observation"]["items"][1]["artifact_ref"]
            artifact = request("GET", f"/v2/artifacts/{ref}", params={"episode_id": episode})["artifact"]
            expected_width = round(inspected["pixel"]["width"] * .75) - round(inspected["pixel"]["width"] * .25)
            expected_height = round(inspected["pixel"]["height"] * .75) - round(inspected["pixel"]["height"] * .25)
            assert (artifact["pixel"]["width"], artifact["pixel"]["height"]) == (expected_width, expected_height)
            artifacts.append(artifact)
        evidence_ids = []
        for index, artifact in enumerate(artifacts):
            evidence_id = f"ev-catalog-{index}"
            step({"type": "memory.save_evidence", "evidence": {"evidence_id": evidence_id,
                "claim_id": "claim-crops", "source_ref": artifact["artifact_id"],
                "selector": {"pixel_window": [0, 0, artifact["pixel"]["width"], artifact["pixel"]["height"]]},
                "description": "Central crop with verified dimensions", "frozen_sha256": artifact["sha256"]}})
            evidence_ids.append(evidence_id)
        step({"type": "answer.submit", "answer": {"label": "crop-dimensions", "confidence": 1.0,
            "claims": [{"claim_id": "claim-crops", "text": "Each inspected image has a verified central crop."}]},
            "confidence": 1.0, "evidence_ids": evidence_ids})
        report = {"job": job, "episode_id": episode, "actions": actions, "artifacts": artifacts, "state": state}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": "checkpoint_saved", "episode_id": episode, "actions": len(actions), "images": len(artifacts)}))


if __name__ == "__main__":
    main()

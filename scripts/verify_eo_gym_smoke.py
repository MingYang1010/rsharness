#!/usr/bin/env python3
"""Exercise live Harness -> EO-Gym -> evidence -> submission over HTTP."""
import hashlib
import argparse
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-check", action="store_true")
    options = parser.parse_args()
    job = json.loads(Path("/smoke/job.json").read_text())
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
        if options.resume_check:
            report = json.loads(Path("/reports/live-smoke.json").read_text())
            episode = report["episode_id"]
            state = request("GET", f"/v2/episodes/{episode}/state")
            assert state["state"] == report["final_state"]
            replay = request("POST", f"/v2/episodes/{episode}/replay")
            assert replay["status"] == "passed" and replay["trace_hash"] == report["replay"]["trace_hash"]
            artifact = "art-" + report["artifact_sha256"]
            content = client.get(f"/v2/artifacts/{artifact}/content", params={"episode_id": episode})
            content.raise_for_status()
            assert hashlib.sha256(content.content).hexdigest() == report["artifact_sha256"]
            result = {"status": "passed", "episode_id": episode, "checks": ["state_equal", "trace_hash_equal", "artifact_hash_equal"]}
            Path("/reports/resume-check.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result))
            return
        capabilities = request("GET", "/v2/capabilities")
        assert "eo_gym.crop" in capabilities["tools"]
        reset = request("POST", "/v2/reset", json={"task_ref": job["task_ref"], "seed": job["seed"]})
        if job.get("coordinate_system") == "pixel":
            assert reset["state"]["map"] is None
            assert reset["observation"]["primary_type"] == "asset_metadata"
            task_ref = job["task_ref"]
            manifest = request("GET", f"/v2/tasks/{task_ref['task_id']}/versions/{task_ref['task_version']}")["manifest"]
            asset = next(a for a in manifest["assets"] if a["asset_id"] == job["asset_id"])
            assert asset["spatial"] is None and asset["pixel"]["coordinate_system"] == "pixel"
            assert (asset["pixel"]["width"], asset["pixel"]["height"]) == (job["source_width"], job["source_height"])
        episode = reset["episode_id"]
        path = f"/v2/episodes/{episode}/step"
        action = {"client_action_id": "smoke-crop", "expected_state_version": 0,
            "action": {"type": "tool.invoke", "tool_id": "eo_gym.crop", "arguments": {"asset_id": job["asset_id"], "aoi": job["aoi"]}}}
        crop = request("POST", path, json=action)
        assert request("POST", path, json=action) == crop
        result = crop["observation"]["items"][0]["inline"]
        expected_width = round(job["source_width"] * .75) - round(job["source_width"] * .25)
        expected_height = round(job["source_height"] * .75) - round(job["source_height"] * .25)
        assert (result["width"], result["height"]) == (expected_width, expected_height)
        artifact = crop["observation"]["items"][1]["artifact_ref"]
        content = client.get(f"/v2/artifacts/{artifact}/content", params={"episode_id": episode})
        content.raise_for_status()
        assert hashlib.sha256(content.content).hexdigest() == artifact[4:]
        evidence = {"evidence_id": "ev-smoke-crop", "claim_id": "claim-crop-size", "source_ref": artifact,
            "selector": {"pixel_window": [0, 0, expected_width, expected_height]},
            "description": f"Crop dimensions: {expected_width} by {expected_height}", "frozen_sha256": artifact[4:]}
        request("POST", path, json={"client_action_id": "smoke-evidence", "expected_state_version": 1,
            "action": {"type": "memory.save_evidence", "evidence": evidence}})
        answer = request("POST", path, json={"client_action_id": "smoke-answer", "expected_state_version": 2,
            "action": {"type": "answer.submit", "answer": {"label": "crop-size", "confidence": 1.0,
                "claims": [{"claim_id": "claim-crop-size", "text": evidence["description"]}]},
                "confidence": 1.0, "evidence_ids": ["ev-smoke-crop"]}})
        assert answer["terminated"]
        replay = request("POST", f"/v2/episodes/{episode}/replay")
        assert replay["status"] == "passed" and all(check["passed"] for check in replay["checks"])
        trace = request("GET", f"/v2/episodes/{episode}/trace")
        report = {"status": "passed", "scope": "real-image scripted interaction, not Qwen or semantic evaluation",
                  "job": job, "episode_id": episode, "artifact_sha256": artifact[4:],
                  "crop_width": expected_width, "crop_height": expected_height, "final_state": answer["state"],
                  "replay": replay, "trace": trace}
        Path("/reports/live-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("status", "scope", "episode_id", "artifact_sha256", "crop_width", "crop_height")}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Agent-only categorical-grid acceptance; no tasks, inputs or backend mount."""
import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

import httpx
import numpy as np
from rasterio.io import MemoryFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    token = Path("/run/agent-token").read_text().strip()
    denied = []
    targets = [("harness", 8000), ("provider", 8081), ("raster", 8084), ("storage", 8082)]
    if os.environ.get("EO_TEST_BACKEND_IP"):
        targets.append((os.environ["EO_TEST_BACKEND_IP"], 8000))
    for host, port in targets:
        try:
            with socket.create_connection((host, port), timeout=2):
                raise AssertionError("private connection succeeded")
        except OSError:
            denied.append("backend-ip" if host == os.environ.get("EO_TEST_BACKEND_IP") else host)
    checkpoint = Path("/reports/grid-checkpoint.json")
    with httpx.Client(base_url="http://agent-gateway:8083", timeout=45, trust_env=False,
                      headers={"Authorization": "Bearer " + token}) as client:
        def request(method, path, body=None):
            response = client.request(method, path, json=body)
            response.raise_for_status()
            return response.json()

        if args.resume:
            saved = json.loads(checkpoint.read_text())
            for action in saved["actions"]:
                assert request("POST", "/agent/step", action["request"]) == action["response"]
            assert request("GET", "/agent/state")["state"] == saved["state"]
            for artifact in saved["artifacts"]:
                content = client.get("/agent/artifacts/" + artifact["artifact_id"] + "/content")
                content.raise_for_status()
                assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
            report = {"status": "passed", "episode_id": saved["state"]["episode_id"],
                      "cached_actions": len(saved["actions"]), "categorical_rasters": 3,
                      "denied": denied}
            Path("/reports/grid-resume.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report))
            return

        session = request("GET", "/agent/session")
        state, actions, artifacts, results = session["state"], [], [], []
        assert "raster.resample" in session["tool_schemas"]

        def step(action):
            nonlocal state
            body = {"client_action_id": "grid-" + str(len(actions)),
                    "expected_state_version": state["state_version"], "action": action}
            response = request("POST", "/agent/step", body)
            assert request("POST", "/agent/step", body) == response
            state = response["state"]
            actions.append({"request": body, "response": response})
            return response

        def tool(name, arguments):
            return step({"type": "tool.invoke", "tool_id": name, "arguments": arguments})

        listing = tool("catalog.search", {"limit": 20})["observation"]["items"][0]["inline"]["assets"]
        assert len(listing) == 6
        pairs = {}
        for asset in listing:
            inspected = tool("catalog.inspect_asset", {"asset_id": asset["asset_id"]})["observation"]["items"][0]["inline"]["asset"]
            assert inspected == asset and "uri" not in inspected and "roles" not in inspected
            pairs.setdefault(asset["temporal"]["start"], {})[asset["bands"][0]] = asset
        assert len(pairs) == 3
        evidence_ids = []
        for index, (acquired, pair) in enumerate(sorted(pairs.items())):
            response = tool("raster.resample", {"source_asset_id": pair["scl"]["asset_id"],
                            "reference_asset_id": pair["red"]["asset_id"], "method": "nearest"})
            result = response["observation"]["items"][0]["inline"]
            assert result["acquired"] == acquired and result["method"] == "nearest"
            assert result["cloud_mask_applied"] is False and len(result["class_counts"]) == 12
            ref = response["observation"]["items"][1]["artifact_ref"]
            artifact = request("GET", "/agent/artifacts/" + ref)["artifact"]
            content = client.get("/agent/artifacts/" + ref + "/content")
            content.raise_for_status()
            assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
            with MemoryFile(content.content) as memory, memory.open(driver="GTiff") as image:
                assert image.crs.to_string() == result["crs"] and list(image.transform)[:6] == result["transform"]
                assert image.dtypes == ("uint8",) and image.nodata == 255
                pixels, valid = image.read(1), image.dataset_mask() > 0
                assert int(valid.sum()) == result["valid_pixels"]
                assert np.all(pixels[~valid] == 255) and np.all((pixels[valid] >= 1) & (pixels[valid] <= 11))
                assert np.bincount(pixels[valid], minlength=12).tolist() == result["class_counts"]
            artifacts.append(artifact)
            results.append(result)
            evidence_id = "ev-grid-" + str(index)
            evidence_ids.append(evidence_id)
            step({"type": "memory.save_evidence", "evidence": {"evidence_id": evidence_id,
                  "claim_id": "scl-grid", "source_ref": ref,
                  "selector": {"bbox": artifact["spatial"]["bbox"], "time_range": artifact["temporal"]},
                  "frozen_sha256": artifact["sha256"],
                  "description": "Nearest-neighbor SCL alignment; cloud mask not applied"}})
        final = step({"type": "answer.submit", "answer": {"label": "scl-grid", "confidence": 1.,
                      "claims": [{"claim_id": "scl-grid",
                                  "valid_fractions": [r["valid_fraction"] for r in results],
                                  "cloud_mask_applied": False}]},
                      "confidence": 1., "evidence_ids": evidence_ids})
        assert final["terminated"] and "evaluation" not in final["state"]
        checkpoint.write_text(json.dumps({"actions": actions, "state": state, "artifacts": artifacts,
                                          "results": results, "denied": denied}, indent=2))
        print(json.dumps({"status": "passed", "episode_id": state["episode_id"],
                          "actions": len(actions), "rasters": 3,
                          "valid_fractions": [r["valid_fraction"] for r in results], "denied": denied}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Agent-only real artifact-chain acceptance; no task/input/backend mount."""
import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

import httpx
import numpy as np
from rasterio.io import MemoryFile

POLICY = "sentinel-2-scl-cloud-shadow-cirrus-snow-saturation-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    token = Path("/run/agent-token").read_text().strip()
    denied = []
    targets = [("harness", 8000), ("provider", 8081), ("raster", 8084),
               ("storage", 8082)]
    if os.environ.get("EO_TEST_BACKEND_IP"):
        targets.append((os.environ["EO_TEST_BACKEND_IP"], 8000))
    for host, port in targets:
        try:
            with socket.create_connection((host, port), timeout=2):
                raise AssertionError("private connection succeeded")
        except OSError:
            denied.append("backend-ip" if host == os.environ.get("EO_TEST_BACKEND_IP") else host)

    checkpoint = Path("/reports/cloud-checkpoint.json")
    with httpx.Client(base_url="http://agent-gateway:8083", timeout=45,
                      trust_env=False,
                      headers={"Authorization": "Bearer " + token}) as client:
        def request(method, path, body=None):
            response = client.request(method, path, json=body)
            response.raise_for_status()
            return response.json()

        if arguments.resume:
            saved = json.loads(checkpoint.read_text())
            for action in saved["actions"]:
                assert request("POST", "/agent/step", action["request"]) == action["response"]
            assert request("GET", "/agent/state")["state"] == saved["state"]
            for artifact in saved["artifacts"]:
                content = client.get("/agent/artifacts/" + artifact["artifact_id"] + "/content")
                content.raise_for_status()
                assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
            report = {"status": "passed", "episode_id": saved["state"]["episode_id"],
                      "cached_actions": len(saved["actions"]), "artifacts": 6,
                      "cloud_mask_policy": POLICY, "denied": denied}
            Path("/reports/cloud-resume.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report))
            return

        session = request("GET", "/agent/session")
        state, actions, artifacts, results = session["state"], [], [], []
        assert {"raster.resample", "raster.band_math"}.issubset(session["tool_schemas"])

        def step(action):
            nonlocal state
            body = {"client_action_id": "cloud-" + str(len(actions)),
                    "expected_state_version": state["state_version"], "action": action}
            response = request("POST", "/agent/step", body)
            assert request("POST", "/agent/step", body) == response
            state = response["state"]
            actions.append({"request": body, "response": response})
            return response

        def tool(name, tool_arguments):
            return step({"type": "tool.invoke", "tool_id": name,
                         "arguments": tool_arguments})

        listing = tool("catalog.search", {"limit": 20})["observation"]["items"][0]["inline"]["assets"]
        assert len(listing) == 9
        triples = {}
        for asset in listing:
            inspected = tool("catalog.inspect_asset", {"asset_id": asset["asset_id"]})[
                "observation"]["items"][0]["inline"]["asset"]
            assert inspected == asset and "uri" not in inspected and "roles" not in inspected
            triples.setdefault(asset["temporal"]["start"], {})[asset["bands"][0]] = asset
        assert len(triples) == 3 and all(set(triple) == {"scl", "red", "nir"}
                                         for triple in triples.values())

        evidence_ids = []
        for index, (acquired, triple) in enumerate(sorted(triples.items())):
            grid_response = tool("raster.resample", {
                "source_asset_id": triple["scl"]["asset_id"],
                "reference_asset_id": triple["red"]["asset_id"], "method": "nearest"})
            grid_result = grid_response["observation"]["items"][0]["inline"]
            assert (grid_result["acquired"] == acquired
                    and grid_result["cloud_mask_applied"] is False)
            mask_id = grid_response["observation"]["items"][1]["artifact_ref"]
            mask_artifact = request("GET", "/agent/artifacts/" + mask_id)["artifact"]
            mask_content = client.get("/agent/artifacts/" + mask_id + "/content")
            mask_content.raise_for_status()
            assert hashlib.sha256(mask_content.content).hexdigest() == mask_artifact["sha256"]

            ndvi_response = tool("raster.band_math", {"operation": "ndvi",
                "red_asset_id": triple["red"]["asset_id"],
                "nir_asset_id": triple["nir"]["asset_id"],
                "mask_artifact_id": mask_id, "cloud_policy": POLICY})
            ndvi_result = ndvi_response["observation"]["items"][0]["inline"]
            assert (ndvi_result["tool_version"] == "1.1.0"
                    and ndvi_result["acquired"] == acquired
                    and ndvi_result["mask_artifact_id"] == mask_id
                    and ndvi_result["mask_sha256"] == mask_artifact["sha256"]
                    and ndvi_result["cloud_mask_applied"] is True
                    and ndvi_result["cloud_policy_version"] == POLICY
                    and ndvi_result["mask_valid_pixels"] ==
                        ndvi_result["clear_mask_pixels"] + ndvi_result["cloud_excluded_pixels"]
                    and ndvi_result["valid_pixels"] <= ndvi_result["clear_mask_pixels"])
            ndvi_id = ndvi_response["observation"]["items"][1]["artifact_ref"]
            ndvi_artifact = request("GET", "/agent/artifacts/" + ndvi_id)["artifact"]
            ndvi_content = client.get("/agent/artifacts/" + ndvi_id + "/content")
            ndvi_content.raise_for_status()
            assert hashlib.sha256(ndvi_content.content).hexdigest() == ndvi_artifact["sha256"]
            with MemoryFile(ndvi_content.content) as memory, memory.open(driver="GTiff") as image:
                values, valid = image.read(1), image.dataset_mask() > 0
                assert (image.dtypes == ("float32",) and image.nodata == -9999.
                        and int(valid.sum()) == ndvi_result["valid_pixels"]
                        and np.all(values[~valid] == -9999.)
                        and np.all((values[valid] >= -1) & (values[valid] <= 1)))
            artifacts.extend([mask_artifact, ndvi_artifact])
            results.append({"acquired": acquired, "grid": grid_result, "ndvi": ndvi_result,
                            "mask_artifact": mask_artifact,
                            "ndvi_artifact": ndvi_artifact})
            evidence_id = "ev-cloud-" + str(index)
            evidence_ids.append(evidence_id)
            step({"type": "memory.save_evidence", "evidence": {
                "evidence_id": evidence_id, "claim_id": "cloud-masked-ndvi",
                "source_ref": ndvi_id,
                "selector": {"bbox": ndvi_artifact["spatial"]["bbox"],
                             "time_range": ndvi_artifact["temporal"]},
                "frozen_sha256": ndvi_artifact["sha256"],
                "description": "NDVI filtered by fixed versioned SCL policy; not cloud ground truth"}})

        final = step({"type": "answer.submit", "answer": {
            "label": "cloud-masked-ndvi", "confidence": 1.,
            "claims": [{"claim_id": "cloud-masked-ndvi", "cloud_policy": POLICY,
                "means": [result["ndvi"]["mean"] for result in results],
                "valid_pixels": [result["ndvi"]["valid_pixels"] for result in results],
                "cloud_excluded_pixels": [result["ndvi"]["cloud_excluded_pixels"]
                                           for result in results]}]},
            "confidence": 1., "evidence_ids": evidence_ids})
        assert final["terminated"] and "evaluation" not in final["state"]
        checkpoint.write_text(json.dumps({"actions": actions, "state": state,
            "artifacts": artifacts, "results": results, "denied": denied}, indent=2))
        print(json.dumps({"status": "passed", "episode_id": state["episode_id"],
            "actions": len(actions), "grid_rasters": 3, "masked_ndvi_rasters": 3,
            "means": [result["ndvi"]["mean"] for result in results], "denied": denied}))


if __name__ == "__main__":
    main()

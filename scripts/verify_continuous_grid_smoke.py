#!/usr/bin/env python3
"""Agent-only acceptance for one continuous B11-to-B08 alignment."""
import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

import httpx
import numpy as np
from rasterio.io import MemoryFile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    token = Path("/run/agent-token").read_text().strip()
    denied = []
    targets = [("harness", 8000), ("provider", 8081),
               ("raster", 8084), ("storage", 8082)]
    if os.environ.get("EO_TEST_BACKEND_IP"):
        targets.append((os.environ["EO_TEST_BACKEND_IP"], 8000))
    for host, port in targets:
        try:
            with socket.create_connection((host, port), timeout=2):
                raise AssertionError("private connection succeeded")
        except OSError:
            denied.append("backend-ip" if host == os.environ.get(
                "EO_TEST_BACKEND_IP") else host)
    checkpoint = Path("/reports/continuous-grid-checkpoint.json")
    with httpx.Client(
            base_url="http://agent-gateway:8083", timeout=45, trust_env=False,
            headers={"Authorization": "Bearer " + token}) as client:
        def request(method, path, body=None):
            response = client.request(method, path, json=body)
            response.raise_for_status()
            return response.json()

        if arguments.resume:
            saved = json.loads(checkpoint.read_text())
            for action in saved["actions"]:
                assert request("POST", "/agent/step",
                               action["request"]) == action["response"]
            assert request("GET", "/agent/state")["state"] == saved["state"]
            for artifact in saved["artifacts"]:
                content = client.get(
                    "/agent/artifacts/" + artifact["artifact_id"] + "/content")
                content.raise_for_status()
                assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
            report = {"status": "passed",
                      "episode_id": saved["state"]["episode_id"],
                      "cached_actions": len(saved["actions"]),
                      "continuous_rasters": 1, "denied": denied}
            Path("/reports/continuous-grid-resume.json").write_text(
                json.dumps(report, indent=2))
            print(json.dumps(report))
            return

        session = request("GET", "/agent/session")
        state, actions, artifacts = session["state"], [], []
        assert "raster.resample" in session["tool_schemas"]

        def step(action):
            nonlocal state
            body = {"client_action_id": "continuous-grid-" + str(len(actions)),
                    "expected_state_version": state["state_version"],
                    "action": action}
            response = request("POST", "/agent/step", body)
            assert request("POST", "/agent/step", body) == response
            state = response["state"]
            actions.append({"request": body, "response": response})
            return response

        def tool(name, arguments):
            return step({"type": "tool.invoke", "tool_id": name,
                         "arguments": arguments})

        listing = tool("catalog.search", {"limit": 10})[
            "observation"]["items"][0]["inline"]["assets"]
        assert len(listing) == 2
        pair = {}
        for asset in listing:
            inspected = tool("catalog.inspect_asset", {
                "asset_id": asset["asset_id"]})[
                    "observation"]["items"][0]["inline"]["asset"]
            assert inspected == asset and "uri" not in inspected and "roles" not in inspected
            pair[asset["bands"][0]] = asset
        assert set(pair) == {"swir16", "nir"}
        response = tool("raster.resample", {
            "source_asset_id": pair["swir16"]["asset_id"],
            "reference_asset_id": pair["nir"]["asset_id"],
            "method": "bilinear"})
        result = response["observation"]["items"][0]["inline"]
        assert (result["tool_version"] == "1.1.0"
                and result["operation"] == "continuous-to-reference-grid"
                and result["method"] == "bilinear"
                and result["source_band"] == "swir16"
                and result["dtype"] == "float32"
                and result["nodata"] == -9999.
                and result["source_scale"] == .0001
                and result["source_offset"] == -.1)
        ref = response["observation"]["items"][1]["artifact_ref"]
        artifact = request("GET", "/agent/artifacts/" + ref)["artifact"]
        content = client.get("/agent/artifacts/" + ref + "/content")
        content.raise_for_status()
        assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
        with MemoryFile(content.content) as memory, memory.open(driver="GTiff") as image:
            assert (image.crs.to_string() == result["crs"]
                    and list(image.transform)[:6] == result["transform"]
                    and image.dtypes == ("float32",) and image.nodata == -9999.)
            pixels, valid = image.read(1), image.dataset_mask() > 0
            assert int(valid.sum()) == result["valid_pixels"]
            assert result["valid_fraction"] == int(valid.sum()) / valid.size
            assert np.isfinite(pixels).all() and np.all(pixels[~valid] == -9999.)
            selected = pixels[valid].astype("float64")
            assert (float(selected.min()) == result["minimum"]
                    and float(selected.max()) == result["maximum"]
                    and float(selected.mean()) == result["mean"])
        artifacts.append(artifact)
        step({"type": "memory.save_evidence", "evidence": {
            "evidence_id": "ev-continuous-grid", "claim_id": "continuous-grid",
            "source_ref": ref,
            "selector": {"bbox": artifact["spatial"]["bbox"],
                         "time_range": artifact["temporal"]},
            "frozen_sha256": artifact["sha256"],
            "description": ("B11 physical reflectance aligned bilinearly to "
                            "the B08 grid; B08 values and mask ignored")}})
        final = step({"type": "answer.submit",
                      "answer": {"label": "continuous-grid", "confidence": 1.,
                                 "claims": [{"claim_id": "continuous-grid",
                                             "valid_fraction": result["valid_fraction"],
                                             "mean": result["mean"]}]},
                      "confidence": 1.,
                      "evidence_ids": ["ev-continuous-grid"]})
        assert final["terminated"] and "evaluation" not in final["state"]
        checkpoint.write_text(json.dumps({
            "actions": actions, "state": state, "artifacts": artifacts,
            "results": [result], "denied": denied}, indent=2))
        print(json.dumps({
            "status": "passed", "episode_id": state["episode_id"],
            "actions": len(actions), "rasters": 1,
            "valid_fraction": result["valid_fraction"],
            "mean": result["mean"], "denied": denied}))


if __name__ == "__main__":
    main()

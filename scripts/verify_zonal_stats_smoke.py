#!/usr/bin/env python3
"""Agent-only acceptance for one continuous-raster zonal-statistics episode."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

import httpx

ZONE_ID = "zone-central-pixel-centres"


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
    checkpoint = Path("/reports/zonal-stats-checkpoint.json")
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
                      "continuous_rasters": len(saved["artifacts"]),
                      "zonal_outputs": 1, "denied": denied}
            Path("/reports/zonal-stats-resume.json").write_text(
                json.dumps(report, indent=2))
            print(json.dumps(report))
            return

        session = request("GET", "/agent/session")
        state, actions, artifacts = session["state"], [], []
        assert {"raster.resample", "raster.zonal_stats"}.issubset(
            session["tool_schemas"])

        def step(action):
            nonlocal state
            body = {"client_action_id": "zonal-stats-" + str(len(actions)),
                    "expected_state_version": state["state_version"],
                    "action": action}
            response = request("POST", "/agent/step", body)
            assert request("POST", "/agent/step", body) == response
            state = response["state"]
            actions.append({"request": body, "response": response})
            return response

        def tool(name, tool_arguments):
            return step({"type": "tool.invoke", "tool_id": name,
                         "arguments": tool_arguments})

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
        grid_response = tool("raster.resample", {
            "source_asset_id": pair["swir16"]["asset_id"],
            "reference_asset_id": pair["nir"]["asset_id"],
            "method": "bilinear"})
        grid_result = grid_response["observation"]["items"][0]["inline"]
        assert (grid_result["tool_version"] == "1.1.0"
                and grid_result["operation"] == "continuous-to-reference-grid")
        ref = grid_response["observation"]["items"][1]["artifact_ref"]
        artifact = request("GET", "/agent/artifacts/" + ref)["artifact"]
        content = client.get("/agent/artifacts/" + ref + "/content")
        content.raise_for_status()
        assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
        artifacts.append(artifact)

        zonal_response = tool("raster.zonal_stats", {
            "raster_artifact_id": ref, "zone_id": ZONE_ID})
        assert len(zonal_response["observation"]["items"]) == 1
        zonal = zonal_response["observation"]["items"][0]["inline"]
        assert (zonal["tool_version"] == "1.0.0"
                and zonal["operation"] == "zonal-statistics"
                and zonal["source_artifact_id"] == ref
                and zonal["source_sha256"] == artifact["sha256"]
                and zonal["source_lineage_parameters_hash"]
                == artifact["lineage"]["parameters_hash"]
                and zonal["zone_id"] == ZONE_ID
                and zonal["inclusion_policy"]
                == "pixel-centre-in-polygon-boundary-inclusive"
                and zonal["validity_policy"]
                == "source-mask-and-finite-and-not-nodata"
                and zonal["zone_pixels"] > 0
                and zonal["valid_pixels"] + zonal["invalid_pixels"]
                == zonal["zone_pixels"])
        bbox = dict(zip(("west", "south", "east", "north"),
                        zonal["zone_bbox_wgs84"]))
        step({"type": "memory.save_evidence", "evidence": {
            "evidence_id": "ev-zonal-stats", "claim_id": "zonal-stats",
            "source_ref": ref,
            "selector": {"bbox": bbox, "time_range": artifact["temporal"]},
            "frozen_sha256": artifact["sha256"],
            "description": ("Pinned polygon zonal statistics over B11 physical "
                            "reflectance aligned to the B08 grid")}})
        final = step({"type": "answer.submit",
                      "answer": {"label": "zonal-stats", "confidence": 1.,
                                 "claims": [{"claim_id": "zonal-stats",
                                             "zone_pixels": zonal["zone_pixels"],
                                             "valid_pixels": zonal["valid_pixels"],
                                             "valid_fraction": zonal["valid_fraction"],
                                             "minimum": zonal["minimum"],
                                             "maximum": zonal["maximum"],
                                             "mean": zonal["mean"]}]},
                      "confidence": 1., "evidence_ids": ["ev-zonal-stats"]})
        assert final["terminated"] and "evaluation" not in final["state"]
        checkpoint.write_text(json.dumps({
            "actions": actions, "state": state, "artifacts": artifacts,
            "grid_result": grid_result, "zonal_result": zonal,
            "denied": denied}, indent=2))
        print(json.dumps({
            "status": "passed", "episode_id": state["episode_id"],
            "actions": len(actions), "rasters": len(artifacts),
            "zone_pixels": zonal["zone_pixels"],
            "valid_pixels": zonal["valid_pixels"],
            "mean": zonal["mean"], "denied": denied}))


if __name__ == "__main__":
    main()

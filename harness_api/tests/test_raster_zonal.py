import copy
import gc
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import httpx
import numpy as np
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds

from app.agent_gateway import build_binding, create_app as gateway
from app.main import create_app
from app.raster_bridge import RasterBridge, create_app as provider_app
from app.v2.artifact_identity import with_derivation_identity
from app.v2.artifacts import ArtifactStore
from app.v2.capabilities import TaskRegistry
from app.v2.domain import V2DomainError
from app.v2.execution_replay import read_snapshot, replay_episode
from app.v2.raster_grid import (CONTINUOUS_VERSION, GridArguments,
                                TOOL_ID as GRID_TOOL)
from app.v2.raster_zonal import (INCLUSION_POLICY, TOOL_ID, VERSION,
                                 ZoneSpec, ZonalArguments, ZonalRequest,
                                 ZonalSource, compute_zonal_stats)
from app.v2.schemas import ToolInvokeAction, V2EpisodeState
from app.v2.tools.raster_grid import RasterGridExecutor
from app.v2.tools.raster_zonal import RasterZonalExecutor
from app.v2.tools.runtime import ToolRouter
import test_raster_grid as grid_fixtures
from v2.test_tool_execution import make_tool_tasks

ROOT = Path(__file__).resolve().parents[2]


class RasterZonalTests(unittest.TestCase):
    compute = grid_fixtures.ContinuousGridTests.compute

    def setUp(self):
        grid_fixtures.ContinuousGridTests.setUp(self)
        self.content, self.grid_result = self.compute()
        self.tasks = make_tool_tasks(self.root)
        directory = self.tasks / "crop-smoke"
        task = json.loads((directory / "task.json").read_text())
        task.update(inputs=[self.source.asset_id, self.reference.asset_id])
        zone_bounds = (660010., 3549970., 660030., 3549990.)
        bbox = list(transform_bounds(
            self.reference.crs, "EPSG:4326", *zone_bounds, densify_pts=21))
        self.zone = ZoneSpec(
            zone_id="zone-central",
            crs=self.reference.crs,
            coordinates=[[660010., 3549990.], [660030., 3549990.],
                         [660030., 3549970.], [660010., 3549970.],
                         [660010., 3549990.]],
            bbox_wgs84=bbox,
            inclusion_policy=INCLUSION_POLICY,
        )
        task["metadata"].update(
            artifact_identity="derivation-sha256-v1",
            grid_inputs={profile.asset_id: profile.model_dump(mode="json")
                         for profile in (self.source, self.reference)},
            zonal_inputs={self.zone.zone_id: self.zone.model_dump(mode="json")},
        )
        (directory / "task.json").write_text(json.dumps(task))
        scenario = json.loads((directory / "scenario.json").read_text())
        scenario["allowed_tools"] = [GRID_TOOL, TOOL_ID]
        (directory / "scenario.json").write_text(json.dumps(scenario))
        template = json.loads((directory / "assets.json").read_text())[0]
        west, south, east, north = self.grid_result.bbox_wgs84
        assets = []
        for profile, path in ((self.source, self.sp),
                              (self.reference, self.rp)):
            asset = copy.deepcopy(template)
            asset.update(
                asset_id=profile.asset_id,
                sha256=profile.sha256,
                size_bytes=path.stat().st_size,
                roles=["input_image", "reflectance"],
                bands=[profile.band],
                temporal={"start": profile.acquired, "end": profile.acquired},
                spatial={"crs": "EPSG:4326",
                         "bbox": dict(west=west, south=south,
                                      east=east, north=north),
                         "gsd_meters": profile.transform[0],
                         "shape": [profile.height, profile.width, 1]},
            )
            assets.append(asset)
        (directory / "assets.json").write_text(json.dumps(assets))
        self.registry = TaskRegistry(self.tasks)
        self.manifest = self.registry.get("crop-smoke", "1.0.0")
        self.artifacts = ArtifactStore(str(self.root / "artifacts"))
        self.grid_calls = 0
        self.zonal_calls = 0
        self.bad_zonal = None

        def grid_handle(request):
            self.grid_calls += 1
            return httpx.Response(
                200,
                content=self.content,
                headers={
                    "X-Raster-Grid-Version": CONTINUOUS_VERSION,
                    "X-Raster-Grid-Metadata": self.grid_result.model_dump_json(),
                    "X-Content-SHA256": hashlib.sha256(self.content).hexdigest(),
                },
            )

        def zonal_handle(request):
            self.zonal_calls += 1
            if self.bad_zonal == "offline":
                raise httpx.ConnectError("offline")
            parsed = ZonalRequest.model_validate_json(
                request.headers["X-Raster-Zonal-Request"])
            path = self.root / ("zonal-source-" + str(self.zonal_calls) + ".tif")
            path.write_bytes(request.content)
            result = compute_zonal_stats(path, parsed)
            payload = result.model_dump(mode="json")
            if self.bad_zonal == "identity":
                payload["zone_id"] = "zone-other"
            return httpx.Response(
                200,
                json=payload,
                headers={"X-Raster-Zonal-Version": VERSION},
            )

        self.grid_transport = httpx.MockTransport(grid_handle)
        self.zonal_transport = httpx.MockTransport(zonal_handle)
        self.grid = RasterGridExecutor(
            "http://raster", self.artifacts, self.grid_transport)
        self.zonal = RasterZonalExecutor(
            "http://raster", self.artifacts, self.zonal_transport)
        self.grid_action = ToolInvokeAction(
            type="tool.invoke", tool_id=GRID_TOOL,
            arguments={"source_asset_id": self.source.asset_id,
                       "reference_asset_id": self.reference.asset_id,
                       "method": "bilinear"})

    def artifact(self):
        output = self.grid.plan(
            self.grid_action, self.manifest,
            self.manifest.task.inputs).invoke()
        return with_derivation_identity(output.artifact)

    def action(self, artifact_id):
        return ToolInvokeAction(
            type="tool.invoke", tool_id=TOOL_ID,
            arguments={"raster_artifact_id": artifact_id,
                       "zone_id": self.zone.zone_id})

    def backend(self):
        return create_app(
            database_path=str(self.root / "state.db"),
            v2_tasks_path=str(self.tasks),
            v2_artifacts_path=str(self.artifacts.root),
            v2_tool_executor=ToolRouter(grid=self.grid, zonal=self.zonal),
        )

    def test_contract_computes_explicit_pixel_centre_zone(self):
        artifact = self.artifact()
        prepared = self.zonal.plan(
            self.action(artifact.artifact_id), self.manifest,
            self.manifest.task.inputs, {artifact.artifact_id: artifact})
        self.assertTrue(prepared.metadata_only)
        self.assertEqual(prepared.input_bytes, artifact.size_bytes)
        self.assertEqual(prepared.max_output_bytes, 0)
        output = prepared.invoke()
        result = output.metadata
        self.assertIsNone(output.artifact)
        self.assertEqual(result["zone_pixels"], 4)
        self.assertEqual(result["valid_pixels"], 4)
        with MemoryFile(self.content) as memory, memory.open() as image:
            selected = image.read(1)[1:3, 1:3].astype("float64")
        self.assertEqual(result["minimum"], float(selected.min()))
        self.assertEqual(result["maximum"], float(selected.max()))
        self.assertEqual(result["mean"], float(selected.mean()))

    def test_zone_schema_rejects_open_self_intersecting_and_wrong_crs(self):
        base = self.zone.model_dump(mode="json")
        changes = [
            {"coordinates": base["coordinates"][:-1]},
            {"coordinates": [[0., 0.], [2., 2.], [0., 2.], [2., 0.], [0., 0.]]},
            {"crs": "EPSG:4326"},
            {"bbox_wgs84": [1., 2., 1., 3.]},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                ZoneSpec.model_validate({**base, **change})

    def test_policy_rejects_foreign_artifact_zone_and_lineage(self):
        artifact = self.artifact()
        action = self.action(artifact.artifact_id)
        with self.assertRaises(V2DomainError):
            self.zonal.plan(action, self.manifest,
                            self.manifest.task.inputs, {})
        wrong = artifact.model_copy(deep=True)
        wrong.lineage.tool_version = "1.0.0"
        with self.assertRaises(V2DomainError):
            self.zonal.plan(action, self.manifest,
                            self.manifest.task.inputs,
                            {artifact.artifact_id: wrong})
        with self.assertRaises(V2DomainError):
            self.zonal.plan(
                ToolInvokeAction(type="tool.invoke", tool_id=TOOL_ID,
                                 arguments={"raster_artifact_id": artifact.artifact_id,
                                            "zone_id": "zone-unknown"}),
                self.manifest, self.manifest.task.inputs,
                {artifact.artifact_id: artifact})
        broken = self.manifest.model_copy(deep=True)
        broken.task.metadata["zonal_inputs"][self.zone.zone_id][
            "bbox_wgs84"][0] += .0001
        with self.assertRaises(V2DomainError):
            self.zonal.plan(action, broken, broken.task.inputs,
                            {artifact.artifact_id: artifact})

    def test_provider_identity_and_offline_fail_closed(self):
        artifact = self.artifact()
        prepared = self.zonal.plan(
            self.action(artifact.artifact_id), self.manifest,
            self.manifest.task.inputs, {artifact.artifact_id: artifact})
        for mode in ("identity", "offline"):
            self.bad_zonal = mode
            with self.subTest(mode=mode), self.assertRaises(V2DomainError):
                prepared.invoke()

    def test_real_subprocess_provider_and_missing_worker(self):
        artifact = self.artifact()
        request = ZonalRequest(
            arguments=ZonalArguments.model_validate(
                self.action(artifact.artifact_id).arguments),
            source=ZonalSource(
                artifact_id=artifact.artifact_id,
                sha256=artifact.sha256,
                size_bytes=artifact.size_bytes,
                crs=self.reference.crs,
                transform=self.reference.transform,
                width=self.reference.width,
                height=self.reference.height,
                dtype="float32",
                nodata=-9999.,
                lineage_parameters_hash=artifact.lineage.parameters_hash,
            ),
            zone=self.zone,
        )
        bridge = RasterBridge(
            self.root, {}, ROOT / "scripts/raster_worker.py",
            zonal_worker=ROOT / "scripts/raster_zonal_worker.py")
        with TestClient(provider_app(bridge)) as client:
            response = client.post(
                "/zonal-stats",
                content=self.content,
                headers={"X-Raster-Zonal-Request": request.model_dump_json()})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["X-Raster-Zonal-Version"], VERSION)
            self.assertEqual(response.json()["zone_pixels"], 4)
        with TestClient(provider_app(RasterBridge(
                self.root, {}, ROOT / "scripts/raster_worker.py"))) as client:
            response = client.post(
                "/zonal-stats",
                content=self.content,
                headers={"X-Raster-Zonal-Request": request.model_dump_json()})
            self.assertEqual(response.status_code, 503)

    def test_gateway_restart_and_execution_replay(self):
        backend = self.backend()
        with TestClient(backend) as operator:
            initial = operator.post("/v2/reset", json={"task_ref": {
                "task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
            episode = initial["episode_id"]
            binding = build_binding(
                self.manifest,
                V2EpisodeState.model_validate(initial["state"]),
                operator.get("/v2/capabilities").json()["data"],
                hashlib.sha256(b"4" * 64).hexdigest(),
            )
            with TestClient(
                    gateway(binding, "http://operator",
                            httpx.ASGITransport(app=backend)),
                    headers={"Authorization": "Bearer " + "4" * 64}) as client:
                grid_request = {
                    "client_action_id": "grid",
                    "expected_state_version": 0,
                    "action": self.grid_action.model_dump(),
                }
                grid = client.post("/agent/step", json=grid_request)
                self.assertEqual(grid.status_code, 200, grid.text)
                ref = grid.json()["observation"]["items"][1]["artifact_ref"]
                zonal_request = {
                    "client_action_id": "zonal",
                    "expected_state_version": 1,
                    "action": self.action(ref).model_dump(),
                }
                zonal = client.post("/agent/step", json=zonal_request)
                self.assertEqual(zonal.status_code, 200, zonal.text)
                self.assertEqual(
                    zonal.json()["observation"]["items"][0]["inline"]["zone_pixels"], 4)
                self.assertEqual(client.post(
                    "/agent/step", json=zonal_request).json(), zonal.json())
                evidence = {"type": "memory.save_evidence", "evidence": {
                    "evidence_id": "ev-zone", "claim_id": "zone",
                    "source_ref": ref,
                    "selector": {"bbox": dict(zip(
                        ("west", "south", "east", "north"),
                        self.zone.bbox_wgs84))},
                    "description": "Pinned polygon zonal statistics",
                    "frozen_sha256": grid.json()["observation"]["items"][0][
                        "inline"]["input_sha256"][0],
                }}
                artifact = client.get("/agent/artifacts/" + ref).json()["artifact"]
                evidence["evidence"]["frozen_sha256"] = artifact["sha256"]
                self.assertEqual(client.post("/agent/step", json={
                    "client_action_id": "evidence",
                    "expected_state_version": 2,
                    "action": evidence}).status_code, 200)
                submit = client.post("/agent/step", json={
                    "client_action_id": "submit",
                    "expected_state_version": 3,
                    "action": {"type": "answer.submit",
                               "answer": {"label": "zone", "confidence": 1.,
                                          "claims": []},
                               "confidence": 1.,
                               "evidence_ids": ["ev-zone"]}})
                self.assertEqual(submit.status_code, 200, submit.text)
            with TestClient(self.backend()) as restarted:
                cached = restarted.post(
                    f"/v2/episodes/{episode}/step", json=zonal_request)
                self.assertEqual(cached.status_code, 200, cached.text)
        self.assertEqual(self.grid_calls, 1)
        self.assertEqual(self.zonal_calls, 1)
        gc.collect()
        before = hashlib.sha256((self.root / "state.db").read_bytes()).hexdigest()
        report = replay_episode(
            read_snapshot(self.root / "state.db", episode),
            self.registry,
            self.root / "replay",
            lambda artifacts: ToolRouter(
                grid=RasterGridExecutor(
                    "http://raster", artifacts, self.grid_transport),
                zonal=RasterZonalExecutor(
                    "http://raster", artifacts, self.zonal_transport),
            ),
        )
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(self.grid_calls, 2)
        self.assertEqual(self.zonal_calls, 2)
        self.assertEqual(hashlib.sha256(
            (self.root / "state.db").read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()

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
from app.v2.raster_math import (NDMI_FORMULA, NDMI_INVALID_POLICY,
                                NDMI_VERSION, AlignedSWIRInput,
                                BandMathArguments, MAX_OUTPUT, NDMIResult,
                                TOOL_ID as MATH_TOOL, compute_ndmi,
                                validate_ndmi)
from app.v2.raster_zonal import (INCLUSION_POLICY,
                                 NDMI_VERSION as NDMI_ZONAL_VERSION,
                                 TOOL_ID as ZONAL_TOOL, ZoneSpec,
                                 ZonalArguments, ZonalRequest,
                                 compute_zonal_stats)
from app.v2.schemas import ToolInvokeAction, V2EpisodeState
from app.v2.tools.raster import RasterExecutor
from app.v2.tools.raster_grid import RasterGridExecutor
from app.v2.tools.raster_zonal import RasterZonalExecutor
from app.v2.tools.runtime import ToolRouter
import test_raster_grid as grid_fixtures
from v2.test_tool_execution import make_tool_tasks


ROOT = Path(__file__).resolve().parents[2]


class RasterNDMITests(unittest.TestCase):
    compute_grid = grid_fixtures.ContinuousGridTests.compute

    def setUp(self):
        grid_fixtures.ContinuousGridTests.setUp(self)
        self.aligned_content, self.grid_result = self.compute_grid()
        self.tasks = make_tool_tasks(self.root)
        directory = self.tasks / "crop-smoke"
        task = json.loads((directory / "task.json").read_text())
        task.update(inputs=[self.source.asset_id, self.reference.asset_id])
        bounds = (660010., 3549970., 660030., 3549990.)
        bbox = list(transform_bounds(
            self.reference.crs, "EPSG:4326", *bounds, densify_pts=21))
        self.zone = ZoneSpec(
            zone_id="zone-ndmi",
            crs=self.reference.crs,
            coordinates=[[660010., 3549990.], [660030., 3549990.],
                         [660030., 3549970.], [660010., 3549970.],
                         [660010., 3549990.]],
            bbox_wgs84=bbox,
            inclusion_policy=INCLUSION_POLICY,
        )
        task["metadata"].update(
            artifact_identity="derivation-sha256-v1",
            band_math_formula=NDMI_FORMULA,
            grid_inputs={profile.asset_id: profile.model_dump(mode="json")
                         for profile in (self.source, self.reference)},
            zonal_inputs={self.zone.zone_id: self.zone.model_dump(mode="json")},
        )
        (directory / "task.json").write_text(json.dumps(task))
        scenario = json.loads((directory / "scenario.json").read_text())
        scenario["allowed_tools"] = [GRID_TOOL, MATH_TOOL, ZONAL_TOOL]
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
        self.calls = {"grid": 0, "ndmi": 0, "zonal": 0}
        self.bad_ndmi = None

        def handle(request):
            if request.url.path == "/resample-grid":
                self.calls["grid"] += 1
                return httpx.Response(200, content=self.aligned_content, headers={
                    "X-Raster-Grid-Version": CONTINUOUS_VERSION,
                    "X-Raster-Grid-Metadata": self.grid_result.model_dump_json(),
                    "X-Content-SHA256": hashlib.sha256(
                        self.aligned_content).hexdigest(),
                })
            if request.url.path == "/ndmi-band-math":
                self.calls["ndmi"] += 1
                if self.bad_ndmi == "offline":
                    raise httpx.ConnectError("offline")
                arguments = BandMathArguments.model_validate_json(
                    request.headers["X-Raster-Arguments"])
                swir = AlignedSWIRInput.model_validate_json(
                    request.headers["X-Raster-NDMI-Input"])
                path = self.root / ("aligned-" + str(self.calls["ndmi"]) + ".tif")
                path.write_bytes(request.content)
                content, result = compute_ndmi(
                    self.rp, path, arguments, self.reference, swir)
                metadata = result.model_dump(mode="json")
                if self.bad_ndmi == "identity":
                    metadata["swir_artifact_sha256"] = "f" * 64
                return httpx.Response(200, content=content, headers={
                    "X-Raster-Version": NDMI_VERSION,
                    "X-Raster-Metadata": json.dumps(metadata),
                    "X-Content-SHA256": hashlib.sha256(content).hexdigest(),
                })
            if request.url.path == "/zonal-stats":
                self.calls["zonal"] += 1
                parsed = ZonalRequest.model_validate_json(
                    request.headers["X-Raster-Zonal-Request"])
                path = self.root / ("zonal-" + str(self.calls["zonal"]) + ".tif")
                path.write_bytes(request.content)
                result = compute_zonal_stats(path, parsed)
                return httpx.Response(200, json=result.model_dump(mode="json"),
                                      headers={"X-Raster-Zonal-Version":
                                               NDMI_ZONAL_VERSION})
            raise AssertionError("unexpected provider path")

        self.transport = httpx.MockTransport(handle)
        self.grid = RasterGridExecutor(
            "http://raster", self.artifacts, self.transport)
        self.math = RasterExecutor(
            "http://raster", self.artifacts, self.transport)
        self.zonal = RasterZonalExecutor(
            "http://raster", self.artifacts, self.transport)
        self.grid_action = ToolInvokeAction(
            type="tool.invoke", tool_id=GRID_TOOL,
            arguments={"source_asset_id": self.source.asset_id,
                       "reference_asset_id": self.reference.asset_id,
                       "method": "bilinear"})

    def backend(self):
        return create_app(
            database_path=str(self.root / "state.db"),
            v2_tasks_path=str(self.tasks),
            v2_artifacts_path=str(self.artifacts.root),
            v2_tool_executor=ToolRouter(
                raster=self.math, grid=self.grid, zonal=self.zonal),
        )

    def aligned_artifact(self):
        output = self.grid.plan(
            self.grid_action, self.manifest,
            self.manifest.task.inputs).invoke()
        return with_derivation_identity(output.artifact)

    def math_action(self, artifact_id):
        return ToolInvokeAction(
            type="tool.invoke", tool_id=MATH_TOOL,
            arguments={"operation": "ndmi",
                       "nir_asset_id": self.reference.asset_id,
                       "swir_artifact_id": artifact_id})

    def ndmi_artifact(self, aligned):
        output = self.math.plan(
            self.math_action(aligned.artifact_id), self.manifest,
            self.manifest.task.inputs,
            {aligned.artifact_id: aligned}).invoke()
        return output, with_derivation_identity(output.artifact)

    def zonal_action(self, artifact_id):
        return ToolInvokeAction(
            type="tool.invoke", tool_id=ZONAL_TOOL,
            arguments={"raster_artifact_id": artifact_id,
                       "zone_id": self.zone.zone_id})

    def test_fixed_ndmi_pixels_and_explicit_zonal_version(self):
        aligned = self.aligned_artifact()
        prepared = self.math.plan(
            self.math_action(aligned.artifact_id), self.manifest,
            self.manifest.task.inputs, {aligned.artifact_id: aligned})
        self.assertEqual(prepared.tool_version, NDMI_VERSION)
        self.assertEqual(prepared.input_bytes,
                         aligned.size_bytes + self.rp.stat().st_size)
        output = prepared.invoke()
        derived = with_derivation_identity(output.artifact)
        result = NDMIResult.model_validate(output.metadata)
        content = self.artifacts.read_content(output.artifact).content
        validate_ndmi(content, result)
        with MemoryFile(self.rp.read_bytes()) as memory, memory.open() as nir_image, \
                MemoryFile(self.aligned_content) as aligned_memory, \
                aligned_memory.open() as swir_image, \
                MemoryFile(content) as output_memory, output_memory.open() as image:
            nir = (nir_image.read(1).astype("float64") * nir_image.scales[0]
                   + nir_image.offsets[0])
            swir = swir_image.read(1).astype("float64")
            valid = ((nir_image.read_masks(1) > 0)
                     & (swir_image.read_masks(1) > 0)
                     & np.isfinite(nir) & np.isfinite(swir)
                     & (nir >= 0) & (swir >= 0) & (nir + swir > 1e-6))
            expected = np.full(nir.shape, -9999., dtype="float32")
            expected[valid] = ((nir[valid] - swir[valid])
                               / (nir[valid] + swir[valid])).astype("float32")
            self.assertTrue(np.array_equal(image.read_masks(1) > 0, valid))
            self.assertTrue(np.array_equal(image.read(1), expected))
        zonal = self.zonal.plan(
            self.zonal_action(derived.artifact_id), self.manifest,
            self.manifest.task.inputs,
            {aligned.artifact_id: aligned, derived.artifact_id: derived})
        self.assertEqual(zonal.tool_version, NDMI_ZONAL_VERSION)
        self.assertTrue(zonal.metadata_only)
        self.assertEqual(zonal.invoke().metadata["source_artifact_id"],
                         derived.artifact_id)

    def test_arguments_refuse_expression_constants_paths_and_mixed_modes(self):
        good = {"operation": "ndmi", "nir_asset_id": "asset-nir",
                "swir_artifact_id": "art-" + "a" * 64}
        parsed = BandMathArguments.model_validate(good)
        self.assertFalse(parsed.masked)
        self.assertNotIn("formula", BandMathArguments.model_json_schema()["properties"])
        for change in ({"formula": "(a-b)/(a+b)"}, {"constant": 1},
                       {"path": "/tmp/input.tif"}, {"red_asset_id": "asset-red"},
                       {"mask_artifact_id": "art-" + "b" * 64},
                       {"operation": "ndvi"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                BandMathArguments.model_validate({**good, **change})

    def test_policy_rejects_cross_episode_parent_and_lineage_tamper(self):
        aligned = self.aligned_artifact()
        action = self.math_action(aligned.artifact_id)
        with self.assertRaises(V2DomainError):
            self.math.plan(action, self.manifest,
                           self.manifest.task.inputs, {})
        broken_task = self.manifest.model_copy(deep=True)
        broken_task.task.metadata.pop("band_math_formula")
        with self.assertRaises(V2DomainError):
            self.math.plan(action, broken_task, broken_task.task.inputs,
                           {aligned.artifact_id: aligned})
        broken = aligned.model_copy(deep=True)
        broken.lineage.parameters_hash = "f" * 64
        with self.assertRaises(V2DomainError):
            self.math.plan(action, self.manifest,
                           self.manifest.task.inputs,
                           {aligned.artifact_id: broken})
        _, derived = self.ndmi_artifact(aligned)
        with self.assertRaises(V2DomainError):
            self.zonal.plan(
                self.zonal_action(derived.artifact_id), self.manifest,
                self.manifest.task.inputs,
                {derived.artifact_id: derived})

    def test_provider_identity_and_offline_fail_closed(self):
        aligned = self.aligned_artifact()
        prepared = self.math.plan(
            self.math_action(aligned.artifact_id), self.manifest,
            self.manifest.task.inputs, {aligned.artifact_id: aligned})
        for mode in ("identity", "offline"):
            self.bad_ndmi = mode
            with self.subTest(mode=mode), self.assertRaises(V2DomainError):
                prepared.invoke()

    def test_real_subprocess_provider_and_missing_worker(self):
        aligned = self.aligned_artifact()
        source, reference = self.source, self.reference
        lineage = RasterZonalExecutor.grid_lineage(source, reference)
        swir = AlignedSWIRInput(
            artifact_id=aligned.artifact_id, sha256=aligned.sha256,
            size_bytes=aligned.size_bytes, source_asset_id=source.asset_id,
            source_sha256=source.sha256, reference_asset_id=reference.asset_id,
            reference_sha256=reference.sha256, acquired=reference.acquired,
            crs=reference.crs, transform=reference.transform,
            width=reference.width, height=reference.height, dtype="float32",
            nodata=-9999., lineage_parameters_hash=lineage)
        args = BandMathArguments(
            operation="ndmi", nir_asset_id=reference.asset_id,
            swir_artifact_id=aligned.artifact_id)
        manifest = {reference.asset_id: {
            "filename": self.rp.name,
            "native": reference.model_dump(mode="json")}}
        bridge = RasterBridge(
            self.root, manifest, ROOT / "scripts/raster_worker.py",
            ndmi_worker=ROOT / "scripts/raster_ndmi_worker.py")
        headers = {"X-Raster-Arguments": args.model_dump_json(exclude_none=True),
                   "X-Raster-NDMI-Input": swir.model_dump_json(),
                   "Content-Type": "image/tiff"}
        with TestClient(provider_app(bridge)) as client:
            response = client.post(
                "/ndmi-band-math", content=self.aligned_content,
                headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["X-Raster-Version"], NDMI_VERSION)
            validate_ndmi(response.content, NDMIResult.model_validate_json(
                response.headers["X-Raster-Metadata"]))
        with TestClient(provider_app(RasterBridge(
                self.root, manifest, ROOT / "scripts/raster_worker.py"))) as client:
            self.assertEqual(client.post(
                "/ndmi-band-math", content=self.aligned_content,
                headers=headers).status_code, 503)

    def test_ndmi_budget_refusal_precedes_provider_call(self):
        path = self.tasks / "crop-smoke/task.json"
        task = json.loads(path.read_text())
        task["budget"]["max_artifact_bytes"] = (
            len(self.aligned_content) + MAX_OUTPUT - 1)
        path.write_text(json.dumps(task))
        with TestClient(self.backend()) as operator:
            initial = operator.post("/v2/reset", json={"task_ref": {
                "task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
            episode = initial["episode_id"]
            grid = operator.post(f"/v2/episodes/{episode}/step", json={
                "client_action_id": "grid", "expected_state_version": 0,
                "action": self.grid_action.model_dump()})
            self.assertEqual(grid.status_code, 200, grid.text)
            ref = grid.json()["data"]["observation"]["items"][1]["artifact_ref"]
            response = operator.post(f"/v2/episodes/{episode}/step", json={
                "client_action_id": "ndmi", "expected_state_version": 1,
                "action": self.math_action(ref).model_dump()})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(response.json()["error"]["code"],
                             "tool_budget_exceeded")
            self.assertEqual(self.calls["ndmi"], 0)

    def test_gateway_restart_cross_episode_and_execution_replay(self):
        backend = self.backend()
        with TestClient(backend) as operator:
            initial = operator.post("/v2/reset", json={"task_ref": {
                "task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
            episode = initial["episode_id"]
            binding = build_binding(
                self.manifest, V2EpisodeState.model_validate(initial["state"]),
                operator.get("/v2/capabilities").json()["data"],
                hashlib.sha256(b"8" * 64).hexdigest())
            with TestClient(
                    gateway(binding, "http://operator",
                            httpx.ASGITransport(app=backend)),
                    headers={"Authorization": "Bearer " + "8" * 64}) as client:
                state_version = 0

                def step(action, name):
                    nonlocal state_version
                    body = {"client_action_id": name,
                            "expected_state_version": state_version,
                            "action": action.model_dump()}
                    response = client.post("/agent/step", json=body)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(client.post("/agent/step", json=body).json(),
                                     response.json())
                    state_version = response.json()["state"]["state_version"]
                    return body, response.json()

                _, grid = step(self.grid_action, "grid")
                aligned_ref = grid["observation"]["items"][1]["artifact_ref"]
                ndmi_body, ndmi = step(self.math_action(aligned_ref), "ndmi")
                ndmi_ref = ndmi["observation"]["items"][1]["artifact_ref"]
                inline = ndmi["observation"]["items"][0]["inline"]
                self.assertEqual(inline["formula_id"], NDMI_FORMULA)
                self.assertEqual(inline["invalid_policy"], NDMI_INVALID_POLICY)
                _, zonal = step(self.zonal_action(ndmi_ref), "zonal")
                self.assertEqual(zonal["observation"]["items"][0]["inline"]
                                 ["tool_version"], NDMI_ZONAL_VERSION)
                artifact = client.get(
                    "/agent/artifacts/" + ndmi_ref).json()["artifact"]
                evidence = {"type": "memory.save_evidence", "evidence": {
                    "evidence_id": "ev-ndmi", "claim_id": "ndmi",
                    "source_ref": ndmi_ref,
                    "selector": {"bbox": dict(zip(
                        ("west", "south", "east", "north"),
                        self.zone.bbox_wgs84))},
                    "description": "Fixed NDMI over pinned zone",
                    "frozen_sha256": artifact["sha256"]}}
                response = client.post("/agent/step", json={
                    "client_action_id": "evidence",
                    "expected_state_version": state_version,
                    "action": evidence})
                self.assertEqual(response.status_code, 200, response.text)
                state_version = response.json()["state"]["state_version"]
                response = client.post("/agent/step", json={
                    "client_action_id": "submit",
                    "expected_state_version": state_version,
                    "action": {"type": "answer.submit",
                               "answer": {"label": "ndmi", "confidence": 1.,
                                          "claims": []},
                               "confidence": 1.,
                               "evidence_ids": ["ev-ndmi"]}})
                self.assertEqual(response.status_code, 200, response.text)
                foreign = operator.post("/v2/reset", json={"task_ref": {
                    "task_id": "crop-smoke",
                    "task_version": "1.0.0"}}).json()["data"]["episode_id"]
                self.assertIn(operator.get(
                    "/v2/artifacts/" + ndmi_ref,
                    params={"episode_id": foreign}).status_code, (403, 404))
            with TestClient(self.backend()) as restarted:
                cached = restarted.post(
                    f"/v2/episodes/{episode}/step", json=ndmi_body)
                self.assertEqual(cached.status_code, 200, cached.text)
        self.assertEqual(self.calls, {"grid": 1, "ndmi": 1, "zonal": 1})
        gc.collect()
        before = hashlib.sha256((self.root / "state.db").read_bytes()).hexdigest()
        report = replay_episode(
            read_snapshot(self.root / "state.db", episode), self.registry,
            self.root / "replay",
            lambda artifacts: ToolRouter(
                raster=RasterExecutor(
                    "http://raster", artifacts, self.transport),
                grid=RasterGridExecutor(
                    "http://raster", artifacts, self.transport),
                zonal=RasterZonalExecutor(
                    "http://raster", artifacts, self.transport)))
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(self.calls, {"grid": 2, "ndmi": 2, "zonal": 2})
        self.assertEqual(hashlib.sha256(
            (self.root / "state.db").read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()

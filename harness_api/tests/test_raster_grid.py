import copy
import gc
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import httpx
import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import transform as transform_coordinates

from v2.test_tool_execution import make_tool_tasks
from app.main import create_app
from app.agent_gateway import build_binding, create_app as gateway
from app.raster_bridge import RasterBridge, create_app as provider_app
from app.v2.artifacts import ArtifactStore
from app.v2.capabilities import TaskRegistry
from app.v2.domain import V2DomainError
from app.v2.execution_replay import read_snapshot, replay_episode
from app.v2.raster_math import MAX_INPUT, MAX_OUTPUT, NativeBand
from app.v2.raster_grid import (GridArguments, GridResult, NativeSCL, NODATA,
                                TOOL_ID, VERSION, checked_grids, compute_grid, validate_grid)
from app.v2.schemas import ToolInvokeAction, V2EpisodeState
from app.v2.tools.raster_grid import RasterGridExecutor
from app.v2.tools.runtime import ToolRouter

ROOT = Path(__file__).resolve().parents[2]


def fixture(path, values, band, transform=None, mask=None, crs="EPSG:32650"):
    transform = transform if transform is not None else from_origin(660000, 3550000, 20 if band == "scl" else 10, 20 if band == "scl" else 10)
    dtype = "uint8" if band == "scl" else "uint16"
    scale, offset = (1., 0.) if band == "scl" else (.0001, -.1)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
                           count=1, dtype=dtype, crs=crs, transform=transform, nodata=0) as image:
            image.write(values.astype(dtype), 1)
            image.scales = (scale,)
            image.offsets = (offset,)
            if mask is not None:
                image.write_mask(mask)
    cls = NativeSCL if band == "scl" else NativeBand
    return cls(asset_id="asset-" + band, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               item_id="scene-one", band=band, acquired="2024-04-05T00:00:00Z", crs=crs,
               transform=list(transform)[:6], width=values.shape[1], height=values.shape[0],
               dtype=dtype, scale=scale, offset=offset, nodata=0.)


def independent_reference(source_path, target_path):
    """Pixel-centre inverse mapping; deliberately no rasterio reproject/resample."""
    with rasterio.open(source_path) as source, rasterio.open(target_path) as target:
        values, source_mask = source.read(1), source.read_masks(1) > 0
        rr, cc = np.indices((target.height, target.width))
        xx, yy = target.transform * (cc.flatten() + .5, rr.flatten() + .5)
        if source.crs != target.crs:
            xx, yy = transform_coordinates(target.crs, source.crs, xx.tolist(), yy.tolist())
        sx, sy = ~source.transform * (np.asarray(xx), np.asarray(yy))
        ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
        inside = (ix >= 0) & (ix < source.width) & (iy >= 0) & (iy < source.height)
        expected = np.full(target.width * target.height, 255, dtype="uint8")
        indices = np.flatnonzero(inside)
        valid = source_mask[iy[indices], ix[indices]] & (values[iy[indices], ix[indices]] != 0)
        indices = indices[valid]
        expected[indices] = values[iy[indices], ix[indices]]
        return expected.reshape(target.height, target.width)


class GridTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sp, self.rp = self.root / "scl.tif", self.root / "red.tif"
        self.source = fixture(self.sp, np.array([[4, 5], [6, 9]]), "scl")
        self.reference = fixture(self.rp, np.full((4, 4), 2000), "red")

    def compute(self):
        return compute_grid(self.sp, self.rp, self.source, self.reference)

    def assert_reference(self):
        content, result = self.compute()
        with MemoryFile(content) as memory, memory.open() as image:
            expected = independent_reference(self.sp, self.rp)
            np.testing.assert_array_equal(image.read(1), expected)
            np.testing.assert_array_equal(image.read_masks(1) > 0, expected != 255)
            self.assertEqual(list(image.transform)[:6], self.reference.transform)
        validate_grid(content, result)
        return content, result

    def test_nearest_upsample_preserves_classes_and_cloud(self):
        content, result = self.assert_reference()
        with MemoryFile(content) as memory, memory.open() as image:
            np.testing.assert_array_equal(image.read(1), [[4, 4, 5, 5], [4, 4, 5, 5], [6, 6, 9, 9], [6, 6, 9, 9]])
        self.assertEqual(result.class_counts[9], 4)
        self.assertEqual(result.valid_fraction, 1.)
        self.assertFalse(result.cloud_mask_applied)

    def test_shifted_grid_marks_outside_source_invalid(self):
        self.reference = fixture(self.rp, np.full((5, 6), 2000), "red", from_origin(659990, 3550010, 10, 10))
        _, result = self.assert_reference()
        self.assertEqual(result.valid_pixels, 16)
        self.assertEqual(result.total_pixels, 30)

    def test_downsample_is_nearest_not_class_average(self):
        self.source = fixture(self.sp, np.array([[1, 4, 2, 5], [6, 7, 8, 9], [4, 5, 6, 7], [8, 9, 10, 11]]), "scl")
        self.reference = fixture(self.rp, np.ones((2, 2)), "red", from_origin(660001, 3549999, 40, 40))
        self.assert_reference()

    def test_zero_class_and_explicit_mask_both_invalid(self):
        self.source = fixture(self.sp, np.array([[0, 4], [5, 6]]), "scl", mask=np.array([[255, 0], [255, 255]], dtype="uint8"))
        _, result = self.assert_reference()
        self.assertEqual(result.valid_pixels, 8)
        self.assertEqual(result.class_counts[0], 0)

    def test_reference_mask_and_values_are_not_science_mask(self):
        self.reference = fixture(self.rp, np.zeros((4, 4)), "red", mask=np.zeros((4, 4), dtype="uint8"))
        _, result = self.assert_reference()
        self.assertEqual(result.valid_pixels, 16)

    def test_all_invalid_is_explicit_zero_coverage(self):
        self.source = fixture(self.sp, np.zeros((2, 2)), "scl")
        _, result = self.assert_reference()
        self.assertEqual(result.class_counts, [0] * 12)
        self.assertEqual(result.valid_fraction, 0.)

    def test_disjoint_grid_has_zero_coverage(self):
        self.reference = fixture(self.rp, np.ones((4, 4)), "red", from_origin(670000, 3550000, 10, 10))
        _, result = self.assert_reference()
        self.assertEqual(result.valid_pixels, 0)

    def test_cross_crs_reprojection_matches_inverse_mapping(self):
        x, y = transform_coordinates("EPSG:32650", "EPSG:32651", [660000.], [3550000.])
        self.reference = fixture(self.rp, np.ones((5, 5)), "red", from_origin(x[0] + 1, y[0] - 1, 9, 9), crs="EPSG:32651")
        _, result = self.assert_reference()
        self.assertGreater(result.valid_pixels, 0)

    def test_dates_scene_and_alias_rejected(self):
        for change in ({"acquired": "2024-04-15T00:00:00Z"}, {"item_id": "other"}, {"asset_id": self.source.asset_id}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                checked_grids(self.source, self.reference.model_copy(update=change))

    def test_unreviewed_class_codes_rejected_even_masked(self):
        for value in (12, 255):
            self.source = fixture(self.sp, np.full((2, 2), value), "scl", mask=np.zeros((2, 2), dtype="uint8"))
            with self.assertRaises(ValueError):
                self.compute()

    def test_source_and_reference_hash_profile_tamper_rejected(self):
        for position in (0, 1):
            for change in ({"sha256": "f" * 64}, {"width": 3}, {"crs": "EPSG:32651"}):
                profiles = [self.source, self.reference]
                profiles[position] = profiles[position].model_copy(update=change)
                with self.subTest(position=position, change=change), self.assertRaises(ValueError):
                    compute_grid(self.sp, self.rp, *profiles)

    def test_schema_rejects_invalid_grid_scaling_and_huge_shape(self):
        for change in ({"scale": .01}, {"offset": -.1}, {"dtype": "float32"}, {"nodata": None},
                       {"transform": [0, 0, 0, 0, -20, 0]}, {"transform": [20, 1, 0, 0, -20, 0]},
                       {"transform": [20, 0, float("inf"), 0, -20, 0]}, {"crs": "EPSG:32661"}, {"width": 1025}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                compute_grid(self.sp, self.rp, self.source.model_copy(update=change), self.reference)

    def test_paths_non_tiff_and_input_bounds_rejected(self):
        link = self.root / "alias.tif"
        link.symlink_to(self.sp)
        with self.assertRaises(ValueError):
            compute_grid(link, self.rp, self.source, self.reference)
        self.sp.write_bytes(b"<VRTDataset/>")
        with self.assertRaises(ValueError):
            self.compute()
        with self.sp.open("wb") as stream:
            stream.truncate(MAX_INPUT + 1)
        with self.assertRaises(ValueError):
            self.compute()

    def test_arguments_reject_bilinear_urls_expressions_and_alias(self):
        args = {"source_asset_id": "asset-scl", "reference_asset_id": "asset-red", "method": "nearest"}
        GridArguments.model_validate(args)
        for change in ({"method": "bilinear"}, {"source_asset_id": "https://private"},
                       {"expression": "eval"}, {"reference_asset_id": "asset-scl"}):
            with self.assertRaises(ValueError):
                GridArguments.model_validate({**args, **change})

    def test_result_grid_histogram_coverage_and_payload_tamper_rejected(self):
        content, result = self.compute()
        for change in ({"valid_fraction": .5}, {"class_counts": [0] * 12}, {"crs": "EPSG:32651"},
                       {"width": 3}, {"nodata": 0}, {"cloud_mask_applied": True},
                       {"input_asset_ids": ["asset-scl", "asset-scl"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_grid(content, result.model_copy(update=change))
        with self.assertRaises(ValueError):
            validate_grid(b"<VRTDataset/>", result)

    def test_repeated_computation_has_same_bytes_and_result(self):
        self.assertEqual(self.compute(), self.compute())

    def test_real_subprocess_provider_and_missing_worker(self):
        manifest = {profile.asset_id: {"filename": path.name, "native": profile.model_dump(mode="json")}
                    for path, profile in ((self.sp, self.source), (self.rp, self.reference))}
        args = GridArguments(source_asset_id=self.source.asset_id,
                             reference_asset_id=self.reference.asset_id, method="nearest")
        bridge = RasterBridge(self.root, manifest, ROOT / "scripts/raster_worker.py",
                              ROOT / "scripts/raster_grid_worker.py")
        content, result = bridge.execute_grid(args)
        validate_grid(content, result)
        with TestClient(provider_app(bridge)) as client:
            response = client.post("/resample-grid", json=args.model_dump())
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["X-Raster-Grid-Version"], VERSION)
            self.assertEqual(hashlib.sha256(response.content).hexdigest(), response.headers["X-Content-SHA256"])
        unavailable = RasterBridge(self.root, manifest, ROOT / "scripts/raster_worker.py")
        with TestClient(provider_app(unavailable)) as client:
            self.assertEqual(client.post("/resample-grid", json=args.model_dump()).status_code, 503)


class GridHarnessTests(unittest.TestCase):
    compute = GridTests.compute

    def setUp(self):
        GridTests.setUp(self)
        self.content, self.result = self.compute()
        self.tasks = make_tool_tasks(self.root)
        directory = self.tasks / "crop-smoke"
        task = json.loads((directory / "task.json").read_text())
        task.update(inputs=[self.source.asset_id, self.reference.asset_id])
        task["metadata"].update(artifact_identity="derivation-sha256-v1",
                                grid_inputs={profile.asset_id: profile.model_dump(mode="json")
                                             for profile in (self.source, self.reference)})
        (directory / "task.json").write_text(json.dumps(task))
        scenario = json.loads((directory / "scenario.json").read_text())
        scenario["allowed_tools"] = [TOOL_ID, "catalog.search", "catalog.inspect_asset"]
        (directory / "scenario.json").write_text(json.dumps(scenario))
        template = json.loads((directory / "assets.json").read_text())[0]
        west, south, east, north = self.result.bbox_wgs84
        assets = []
        for profile, path, roles in ((self.source, self.sp, ["input_image", "scene_classification"]),
                                     (self.reference, self.rp, ["input_image", "reflectance"])):
            asset = copy.deepcopy(template)
            asset.update(asset_id=profile.asset_id, sha256=profile.sha256, size_bytes=path.stat().st_size,
                         roles=roles, bands=[profile.band],
                         temporal={"start": profile.acquired, "end": profile.acquired},
                         spatial={"crs": "EPSG:4326",
                                  "bbox": dict(west=west, south=south, east=east, north=north),
                                  "gsd_meters": profile.transform[0],
                                  "shape": [profile.height, profile.width, 1]})
            assets.append(asset)
        (directory / "assets.json").write_text(json.dumps(assets))
        self.registry = TaskRegistry(self.tasks)
        self.manifest = self.registry.get("crop-smoke", "1.0.0")
        self.calls, self.bad = 0, None

        def handle(request):
            self.calls += 1
            if self.bad == "timeout":
                raise httpx.ReadTimeout("fixture")
            data = self.content if self.bad != "bytes" else b"wrong"
            metadata = self.result.model_dump(mode="json")
            if self.bad == "source":
                metadata["input_sha256"] = ["f" * 64] * 2
            return httpx.Response(200, content=data, headers={
                "X-Raster-Grid-Version": VERSION,
                "X-Raster-Grid-Metadata": json.dumps(metadata),
                "X-Content-SHA256": hashlib.sha256(self.content).hexdigest()})

        self.transport = httpx.MockTransport(handle)
        self.artifacts = ArtifactStore(str(self.root / "artifacts"))
        self.executor = RasterGridExecutor("http://raster", self.artifacts, self.transport)
        self.action = ToolInvokeAction(type="tool.invoke", tool_id=TOOL_ID,
            arguments={"source_asset_id": self.source.asset_id,
                       "reference_asset_id": self.reference.asset_id, "method": "nearest"})

    def backend(self):
        return create_app(database_path=str(self.root / "state.db"),
            v2_tasks_path=str(self.tasks), v2_artifacts_path=str(self.artifacts.root),
            v2_tool_executor=ToolRouter(grid=self.executor))

    def test_scope_roles_profiles_and_output_budget(self):
        with self.assertRaises(V2DomainError):
            self.executor.plan(self.action, self.manifest, [self.source.asset_id])
        broken = self.manifest.model_copy(deep=True)
        broken.assets[0].roles = ["label"]
        with self.assertRaises(V2DomainError):
            self.executor.plan(self.action, broken, broken.task.inputs)
        prepared = self.executor.plan(self.action, self.manifest, self.manifest.task.inputs)
        self.assertEqual(prepared.input_bytes, sum(a.size_bytes for a in self.manifest.assets))
        self.assertEqual(prepared.max_output_bytes, MAX_OUTPUT)
        self.assertEqual(self.calls, 0)
        task_path = self.tasks / "crop-smoke/task.json"
        task = json.loads(task_path.read_text())
        task["budget"]["max_artifact_bytes"] = MAX_OUTPUT - 1
        task_path.write_text(json.dumps(task))
        with TestClient(self.backend()) as operator:
            initial = operator.post("/v2/reset", json={"task_ref": {
                "task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
            response = operator.post(f"/v2/episodes/{initial['episode_id']}/step", json={
                "client_action_id": "over-budget", "expected_state_version": 0,
                "action": self.action.model_dump()})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(response.json()["error"]["code"], "tool_budget_exceeded")
            self.assertEqual(self.calls, 0)

    def test_provider_tamper_and_timeout_fail_without_artifact(self):
        for bad in ("bytes", "source", "timeout"):
            self.bad = bad
            with self.assertRaises(V2DomainError):
                self.executor.plan(self.action, self.manifest, self.manifest.task.inputs).invoke()
        self.assertFalse(self.artifacts.root.exists())

    def test_gateway_evidence_restart_and_execution_replay(self):
        backend = self.backend()
        with TestClient(backend) as operator:
            initial = operator.post("/v2/reset", json={"task_ref": {
                "task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
            episode = initial["episode_id"]
            binding = build_binding(self.manifest, V2EpisodeState.model_validate(initial["state"]),
                operator.get("/v2/capabilities").json()["data"], hashlib.sha256(b"2" * 64).hexdigest())
            with TestClient(gateway(binding, "http://operator", httpx.ASGITransport(app=backend)),
                            headers={"Authorization": "Bearer " + "2" * 64}) as client:
                request = {"client_action_id": "grid-1", "expected_state_version": 0,
                           "action": self.action.model_dump()}
                wrong = copy.deepcopy(request)
                wrong["action"]["arguments"]["reference_asset_id"] = "asset-other"
                self.assertEqual(client.post("/agent/step", json=wrong).status_code, 403)
                response = client.post("/agent/step", json=request)
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                self.assertEqual(client.post("/agent/step", json=request).json(), data)
                self.assertEqual(self.calls, 1)
                inline = data["observation"]["items"][0]["inline"]
                self.assertEqual(inline["method"], "nearest")
                self.assertEqual(inline["class_counts"], self.result.class_counts)
                ref = data["observation"]["items"][1]["artifact_ref"]
                artifact = client.get("/agent/artifacts/" + ref).json()["artifact"]
                content = client.get("/agent/artifacts/" + ref + "/content")
                self.assertEqual(content.status_code, 200, content.text[:100])
                self.assertEqual(content.content, self.content)
                evidence = {"type": "memory.save_evidence", "evidence": {
                    "evidence_id": "ev-grid", "claim_id": "grid", "source_ref": ref,
                    "selector": {"pixel_window": [0, 0, 4, 4]},
                    "description": "Nearest SCL categorical grid", "frozen_sha256": artifact["sha256"]}}
                response = client.post("/agent/step", json={"client_action_id": "evidence",
                    "expected_state_version": 1, "action": evidence})
                self.assertEqual(response.status_code, 200, response.text)
                response = client.post("/agent/step", json={"client_action_id": "submit",
                    "expected_state_version": 2, "action": {"type": "answer.submit",
                    "answer": {"label": "grid", "confidence": 1., "claims": []},
                    "confidence": 1., "evidence_ids": ["ev-grid"]}})
                self.assertEqual(response.status_code, 200, response.text)
            with TestClient(self.backend()) as restarted:
                cached = restarted.post(f"/v2/episodes/{episode}/step", json=request).json()["data"]
                self.assertEqual(cached["state"]["state_version"], 1)
            self.assertEqual(self.calls, 1)
        gc.collect()
        before = hashlib.sha256((self.root / "state.db").read_bytes()).hexdigest()
        report = replay_episode(read_snapshot(self.root / "state.db", episode), self.registry,
            self.root / "replay", lambda artifacts: ToolRouter(grid=RasterGridExecutor(
                "http://raster", artifacts, self.transport)))
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(self.calls, 2)
        self.assertEqual(hashlib.sha256((self.root / "state.db").read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()

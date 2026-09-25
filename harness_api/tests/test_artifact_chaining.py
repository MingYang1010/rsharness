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

from v2.test_tool_execution import make_tool_tasks
from app.main import create_app
from app.raster_bridge import RasterBridge, create_app as provider_app
from app.core.artifacts import ArtifactStore
from app.core.capabilities import TaskRegistry
from app.core.execution_replay import read_snapshot, replay_episode
from app.core.raster_grid import (GridArguments, NativeSCL, TOOL_ID as GRID_TOOL,
                                VERSION as GRID_VERSION, compute_grid)
from app.core.raster_math import (BandMathArguments, CLOUD_POLICY, MASKED_VERSION,
                                MaskArtifactInput, NativeBand, NODATA,
                                TOOL_ID as RASTER_TOOL, compute_masked_ndvi,
                                validate_masked_ndvi)
from app.core.schemas import ToolInvokeAction
from app.core.tools.raster import RasterExecutor
from app.core.tools.raster_grid import RasterGridExecutor
from app.core.tools.runtime import ToolRouter

ROOT = Path(__file__).resolve().parents[2]


def native(path: Path, band: str, values, *, dtype="uint16"):
    transform = from_origin(660000, 3550000, 10, 10)
    scale, offset, nodata = ((1., 0., 0.) if band == "scl" else (.0001, -.1, 0.))
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
                           count=1, dtype=dtype, crs="EPSG:32650", transform=transform,
                           nodata=nodata) as image:
            image.write(values.astype(dtype), 1)
            image.scales = (scale,)
            image.offsets = (offset,)
    cls = NativeSCL if band == "scl" else NativeBand
    return cls(asset_id="asset-" + band, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               item_id="scene-one", band=band, acquired="2024-04-05T00:00:00Z",
               crs="EPSG:32650", transform=list(transform)[:6], width=values.shape[1],
               height=values.shape[0], dtype=dtype, scale=scale, offset=offset, nodata=nodata)


class MaskedKernelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.red_path, self.nir_path, self.scl_path = [self.root / name for name in
                                                        ("red.tif", "nir.tif", "scl.tif")]
        self.red = native(self.red_path, "red", np.array([[2000, 1000, 500], [0, 2000, 3000]]))
        self.nir = native(self.nir_path, "nir", np.array([[4000, 1000, 4000], [4000, 2000, 2000]]))
        self.scl = native(self.scl_path, "scl", np.array([[4, 8, 9], [2, 6, 11]]), dtype="uint8")
        self.mask_content, self.grid_result = compute_grid(self.scl_path, self.red_path,
                                                            self.scl, self.red)
        self.mask_path = self.root / "mask.tif"
        self.mask_path.write_bytes(self.mask_content)
        self.mask = MaskArtifactInput(artifact_id="art-" + "a" * 64,
            sha256=hashlib.sha256(self.mask_content).hexdigest(), size_bytes=len(self.mask_content),
            cloud_policy=CLOUD_POLICY)

    def test_versioned_policy_excludes_cloud_shadow_cirrus_snow(self):
        content, result = compute_masked_ndvi(self.red_path, self.nir_path, self.mask_path,
                                               self.red, self.nir, self.mask)
        validate_masked_ndvi(content, result)
        self.assertEqual(result.mask_valid_pixels, 6)
        self.assertEqual(result.cloud_excluded_pixels, 3)
        self.assertEqual(result.clear_mask_pixels, 3)
        self.assertEqual(result.valid_pixels, 2)
        self.assertTrue(result.cloud_mask_applied)
        with MemoryFile(content) as memory, memory.open() as image:
            values = image.read(1)
            self.assertAlmostEqual(float(values[0, 0]), .5)
            self.assertEqual(values[0, 1], NODATA)
            self.assertEqual(values[1, 2], NODATA)

    def test_mask_hash_grid_and_policy_fail_closed(self):
        variants = [self.mask.model_copy(update={"sha256": "f" * 64}),
                    self.mask.model_copy(update={"cloud_policy": "bad"})]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                compute_masked_ndvi(self.red_path, self.nir_path, self.mask_path,
                                    self.red, self.nir, value)
        shifted = self.red.model_copy(update={"transform": [10., 0., 660010., 0., -10., 3550000.]})
        with self.assertRaises(ValueError):
            compute_masked_ndvi(self.red_path, self.nir_path, self.mask_path, shifted, self.nir, self.mask)

    def test_arguments_require_paired_mask_and_fixed_policy(self):
        base = {"operation": "ndvi", "red_asset_id": "asset-red", "nir_asset_id": "asset-nir"}
        BandMathArguments.model_validate(base)
        good = {**base, "mask_artifact_id": self.mask.artifact_id, "cloud_policy": CLOUD_POLICY}
        self.assertTrue(BandMathArguments.model_validate(good).masked)
        for change in ({"mask_artifact_id": self.mask.artifact_id},
                       {"cloud_policy": CLOUD_POLICY},
                       {"mask_artifact_id": self.mask.artifact_id, "cloud_policy": "cloud-v2"}):
            with self.assertRaises(ValueError):
                BandMathArguments.model_validate({**base, **change})

    def test_provider_accepts_only_bounded_binary_mask(self):
        manifest = {profile.asset_id: {"filename": path.name, "native": profile.model_dump(mode="json")}
                    for path, profile in ((self.red_path, self.red), (self.nir_path, self.nir))}
        bridge = RasterBridge(self.root, manifest, ROOT / "scripts/raster_worker.py")
        args = BandMathArguments(operation="ndvi", red_asset_id=self.red.asset_id,
            nir_asset_id=self.nir.asset_id, mask_artifact_id=self.mask.artifact_id,
            cloud_policy=CLOUD_POLICY)
        with TestClient(provider_app(bridge)) as client:
            headers = {"X-Raster-Arguments": args.model_dump_json(exclude_none=True),
                       "X-Raster-Masked-Request": self.mask.model_dump_json(),
                       "Content-Type": "image/tiff"}
            response = client.post("/masked-band-math", content=self.mask_content, headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["X-Raster-Version"], MASKED_VERSION)
            broken = self.mask.model_copy(update={"sha256": "f" * 64})
            headers["X-Raster-Masked-Request"] = broken.model_dump_json()
            self.assertEqual(client.post("/masked-band-math", content=self.mask_content,
                                         headers=headers).status_code, 422)


class EpisodeArtifactChainingTests(MaskedKernelTests):
    def setUp(self):
        super().setUp()
        self.masked_content, self.masked_result = compute_masked_ndvi(
            self.red_path, self.nir_path, self.mask_path, self.red, self.nir, self.mask)
        self.tasks = make_tool_tasks(self.root)
        directory = self.tasks / "crop-smoke"
        task = json.loads((directory / "task.json").read_text())
        task.update(inputs=[self.scl.asset_id, self.red.asset_id, self.nir.asset_id])
        task["metadata"].update(artifact_identity="derivation-sha256-v1",
            cloud_mask_policy=CLOUD_POLICY,
            grid_inputs={profile.asset_id: profile.model_dump(mode="json")
                         for profile in (self.scl, self.red, self.nir)},
            raster_inputs={profile.asset_id: profile.model_dump(mode="json")
                           for profile in (self.red, self.nir)})
        (directory / "task.json").write_text(json.dumps(task))
        scenario = json.loads((directory / "scenario.json").read_text())
        scenario["allowed_tools"] = [GRID_TOOL, RASTER_TOOL]
        (directory / "scenario.json").write_text(json.dumps(scenario))
        template = json.loads((directory / "assets.json").read_text())[0]
        west, south, east, north = self.grid_result.bbox_wgs84
        assets = []
        for profile, path, roles in ((self.scl, self.scl_path, ["input_image", "scene_classification"]),
                                     (self.red, self.red_path, ["input_image", "reflectance"]),
                                     (self.nir, self.nir_path, ["input_image", "reflectance"])):
            asset = copy.deepcopy(template)
            asset.update(asset_id=profile.asset_id, sha256=profile.sha256,
                         size_bytes=path.stat().st_size, roles=roles, bands=[profile.band],
                         temporal={"start": profile.acquired, "end": profile.acquired},
                         spatial={"crs": "EPSG:4326",
                                  "bbox": {"west": west, "south": south, "east": east, "north": north},
                                  "gsd_meters": profile.transform[0],
                                  "shape": [profile.height, profile.width, 1]})
            assets.append(asset)
        (directory / "assets.json").write_text(json.dumps(assets))
        self.registry = TaskRegistry(self.tasks)
        self.artifacts = ArtifactStore(str(self.root / "artifacts"))
        self.grid_calls = 0
        self.masked_calls = 0
        self.offline = False

        def handle(request):
            if request.url.path == "/resample-grid":
                self.grid_calls += 1
                return httpx.Response(200, content=self.mask_content, headers={
                    "X-Raster-Grid-Version": GRID_VERSION,
                    "X-Raster-Grid-Metadata": self.grid_result.model_dump_json(),
                    "X-Content-SHA256": hashlib.sha256(self.mask_content).hexdigest()})
            if request.url.path == "/masked-band-math":
                self.masked_calls += 1
                if self.offline:
                    raise httpx.ConnectError("offline", request=request)
                supplied = MaskArtifactInput.model_validate_json(request.headers["X-Raster-Masked-Request"])
                args = BandMathArguments.model_validate_json(request.headers["X-Raster-Arguments"])
                self.assertEqual(request.content, self.mask_content)
                self.assertEqual(args.mask_artifact_id, supplied.artifact_id)
                provider_mask = self.root / "provider-mask.tif"
                provider_mask.write_bytes(request.content)
                content, result = compute_masked_ndvi(self.red_path, self.nir_path,
                    provider_mask, self.red, self.nir, supplied)
                return httpx.Response(200, content=content, headers={
                    "X-Raster-Version": MASKED_VERSION,
                    "X-Raster-Metadata": result.model_dump_json(),
                    "X-Content-SHA256": hashlib.sha256(content).hexdigest()})
            return httpx.Response(404)

        self.transport = httpx.MockTransport(handle)

    def backend(self, artifacts=None):
        artifacts = artifacts or self.artifacts
        return create_app(database_path=str(self.root / "state.db"),
            v2_tasks_path=str(self.tasks), v2_artifacts_path=str(artifacts.root),
            v2_tool_executor=ToolRouter(
                raster=RasterExecutor("http://raster", artifacts, self.transport),
                grid=RasterGridExecutor("http://raster", artifacts, self.transport)))

    @staticmethod
    def grid_action():
        return ToolInvokeAction(type="tool.invoke", tool_id=GRID_TOOL,
            arguments={"source_asset_id": "asset-scl", "reference_asset_id": "asset-red",
                       "method": "nearest"})

    @staticmethod
    def masked_action(mask_id):
        return ToolInvokeAction(type="tool.invoke", tool_id=RASTER_TOOL,
            arguments={"operation": "ndvi", "red_asset_id": "asset-red",
                       "nir_asset_id": "asset-nir", "mask_artifact_id": mask_id,
                       "cloud_policy": CLOUD_POLICY})

    def create_mask(self, client, episode):
        response = client.post(f"/v2/episodes/{episode}/step", json={
            "client_action_id": "grid", "expected_state_version": 0,
            "action": self.grid_action().model_dump()})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["data"]["observation"]["items"][1]["artifact_ref"]

    def test_same_episode_chain_restart_cross_episode_and_replay(self):
        backend = self.backend()
        with TestClient(backend) as client:
            initial = client.post("/v2/reset", json={"task_ref": {
                "task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
            episode = initial["episode_id"]
            mask_id = self.create_mask(client, episode)
            request = {"client_action_id": "masked", "expected_state_version": 1,
                       "action": self.masked_action(mask_id).model_dump()}
            response = client.post(f"/v2/episodes/{episode}/step", json=request)
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()["data"]
            self.assertEqual(data["observation"]["items"][0]["inline"]["tool_version"], MASKED_VERSION)
            output_id = data["observation"]["items"][1]["artifact_ref"]
            output = client.get(f"/v2/artifacts/{output_id}", params={"episode_id": episode}).json()["data"]["artifact"]
            self.assertEqual(output["lineage"]["input_refs"], ["asset-red", "asset-nir", mask_id])
            self.assertEqual(client.post(f"/v2/episodes/{episode}/step", json=request).json()["data"], data)
            other = client.post("/v2/reset", json={"task_ref": {
                "task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]["episode_id"]
            denied = client.post(f"/v2/episodes/{other}/step", json={
                "client_action_id": "foreign", "expected_state_version": 0,
                "action": self.masked_action(mask_id).model_dump()})
            self.assertEqual(denied.status_code, 403, denied.text)
            finished = client.post(f"/v2/episodes/{episode}/step", json={
                "client_action_id": "stop", "expected_state_version": 2,
                "action": {"type": "answer.abstain", "rationale": "test", "evidence_ids": []}})
            self.assertEqual(finished.status_code, 200, finished.text)
        self.assertEqual((self.grid_calls, self.masked_calls), (1, 1))
        with TestClient(self.backend()) as restarted:
            cached = restarted.post(f"/v2/episodes/{episode}/step", json=request)
            self.assertEqual(cached.status_code, 200, cached.text)
        self.assertEqual((self.grid_calls, self.masked_calls), (1, 1))
        gc.collect()
        before = hashlib.sha256((self.root / "state.db").read_bytes()).hexdigest()
        report = replay_episode(read_snapshot(self.root / "state.db", episode), self.registry,
            self.root / "replay", lambda artifacts: ToolRouter(
                raster=RasterExecutor("http://raster", artifacts, self.transport),
                grid=RasterGridExecutor("http://raster", artifacts, self.transport)))
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual((self.grid_calls, self.masked_calls), (2, 2))
        self.assertEqual(hashlib.sha256((self.root / "state.db").read_bytes()).hexdigest(), before)

    def test_tampered_content_and_provider_offline_fail_closed(self):
        with TestClient(self.backend()) as client:
            initial = client.post("/v2/reset", json={"task_ref": {
                "task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
            episode = initial["episode_id"]
            mask_id = self.create_mask(client, episode)
            artifact = client.get(f"/v2/artifacts/{mask_id}", params={"episode_id": episode}).json()["data"]["artifact"]
            path = self.artifacts.content_path(artifact["sha256"])
            original = path.read_bytes()
            path.write_bytes(b"x" * len(original))
            response = client.post(f"/v2/episodes/{episode}/step", json={
                "client_action_id": "tampered", "expected_state_version": 1,
                "action": self.masked_action(mask_id).model_dump()})
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(response.json()["error"]["code"], "artifact_content_corrupt")
            self.assertEqual(self.masked_calls, 0)
            path.write_bytes(original)
            self.offline = True
            response = client.post(f"/v2/episodes/{episode}/step", json={
                "client_action_id": "offline", "expected_state_version": 2,
                "action": self.masked_action(mask_id).model_dump()})
            self.assertEqual(response.status_code, 502, response.text)
            self.assertEqual(response.json()["error"]["code"], "tool_unavailable")

    def test_combined_input_budget_is_reserved_before_artifact_read(self):
        task_path = self.tasks / "crop-smoke/task.json"
        task = json.loads(task_path.read_text())
        grid_bytes = self.scl_path.stat().st_size + self.red_path.stat().st_size
        masked_bytes = self.red_path.stat().st_size + self.nir_path.stat().st_size + len(self.mask_content)
        task["budget"]["max_input_bytes"] = grid_bytes + masked_bytes - 1
        task_path.write_text(json.dumps(task))
        self.registry = TaskRegistry(self.tasks)
        with TestClient(self.backend()) as client:
            initial = client.post("/v2/reset", json={"task_ref": {
                "task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
            episode = initial["episode_id"]
            mask_id = self.create_mask(client, episode)
            response = client.post(f"/v2/episodes/{episode}/step", json={
                "client_action_id": "budget", "expected_state_version": 1,
                "action": self.masked_action(mask_id).model_dump()})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(response.json()["error"]["code"], "tool_budget_exceeded")
            self.assertEqual(self.masked_calls, 0)


if __name__ == "__main__":
    unittest.main()

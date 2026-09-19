import copy
import gc
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import httpx
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from app.raster_bridge import RasterBridge, create_app
from app.agent_gateway import build_binding, create_app as gateway
from app.main import create_app as harness_app
from app.v2.artifacts import ArtifactStore
from app.v2.capabilities import TaskRegistry
from app.v2.execution_replay import read_snapshot, replay_episode
from app.v2.raster_grid import NativeSCL
from app.v2.raster_math import CLOUD_POLICY, NativeBand
from app.v2.schemas import ToolInvokeAction, V2EpisodeState
from app.v2.temporal import (
    STACK_BANDS,
    STACK_NODATA,
    TemporalAlignRequest,
    checked_temporal_inputs,
    compute_temporal_stack,
    validate_temporal_stack,
)
from app.v2.tools.runtime import ToolRouter
from app.v2.tools.temporal import TemporalExecutor
from v2.test_tool_execution import make_tool_tasks


ROOT = Path(__file__).resolve().parents[2]


def fixture(path, values, band, item, acquired, transform=None, mask=None):
    transform = transform or from_origin(
        669060, 3542980, 20 if band == "scl" else 10, 20 if band == "scl" else 10
    )
    dtype = "uint8" if band == "scl" else "uint16"
    scale, offset = (1.0, 0.0) if band == "scl" else (0.0001, -0.1)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=values.shape[1],
            height=values.shape[0],
            count=1,
            dtype=dtype,
            crs="EPSG:32650",
            transform=transform,
            nodata=0,
        ) as image:
            image.write(values.astype(dtype), 1)
            image.scales = (scale,)
            image.offsets = (offset,)
            if mask is not None:
                image.write_mask(mask)
    model = NativeSCL if band == "scl" else NativeBand
    return model(
        asset_id=f"asset-{item}-{band}",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        item_id=item,
        band=band,
        acquired=acquired,
        crs="EPSG:32650",
        transform=list(transform)[:6],
        width=values.shape[1],
        height=values.shape[0],
        dtype=dtype,
        scale=scale,
        offset=offset,
        nodata=0.0,
    )


class TemporalStackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = (
            self.root / "before-red.tif",
            self.root / "before-scl.tif",
            self.root / "after-red.tif",
            self.root / "after-scl.tif",
        )
        before = "2024-04-05T02:58:52.954000Z"
        after = "2024-05-10T02:58:56.417000Z"
        self.profiles = (
            fixture(self.paths[0], np.full((4, 4), 1000), "red", "before", before),
            fixture(
                self.paths[1], np.array([[4, 4], [4, 8]]), "scl", "before", before
            ),
            fixture(self.paths[2], np.full((4, 4), 2000), "red", "after", after),
            fixture(
                self.paths[3], np.array([[5, 5], [5, 5]]), "scl", "after", after
            ),
        )

    def compute(self):
        return compute_temporal_stack(self.paths, self.profiles)

    def test_stack_is_deterministic_and_preserves_fixed_band_semantics(self):
        content, result = self.compute()
        repeated, repeated_result = self.compute()
        self.assertEqual(content, repeated)
        self.assertEqual(result, repeated_result)
        self.assertEqual(result.band_order, STACK_BANDS)
        self.assertEqual(result.before_cloud_fraction, 0.25)
        self.assertEqual(result.after_cloud_fraction, 0.0)
        self.assertEqual(result.aligned_coverage_fraction, 1.0)
        with MemoryFile(content) as memory, memory.open() as image:
            self.assertEqual(list(image.descriptions), STACK_BANDS)
            np.testing.assert_array_equal(image.read(1), np.full((4, 4), 1000))
            np.testing.assert_array_equal(
                image.read(2),
                [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 8, 8], [4, 4, 8, 8]],
            )
            np.testing.assert_array_equal(image.read(3), np.full((4, 4), 2000))
        validate_temporal_stack(content, result)

    def test_cross_date_gap_is_explicit_nodata(self):
        mask = np.full((4, 4), 255, dtype="uint8")
        mask[0, 0] = 0
        self.profiles = (
            self.profiles[0],
            self.profiles[1],
            fixture(
                self.paths[2], np.full((4, 4), 2000), "red", "after",
                "2024-05-10T02:58:56.417000Z", mask=mask
            ),
            self.profiles[3],
        )
        content, result = self.compute()
        self.assertEqual(result.aligned_valid_pixels, 15)
        with MemoryFile(content) as memory, memory.open() as image:
            self.assertTrue(np.all(image.read()[:, 0, 0] == STACK_NODATA))
            self.assertEqual(image.dataset_mask()[0, 0], 0)

    def test_profile_payload_and_metadata_tamper_fail_closed(self):
        content, result = self.compute()
        damaged = bytearray(content)
        damaged[0] = 0
        with self.assertRaises(ValueError):
            validate_temporal_stack(bytes(damaged), result)
        with self.assertRaises(ValueError):
            validate_temporal_stack(content, result.model_copy(update={"width": 3}))

    def test_inputs_require_ordered_exact_reflectance_grid(self):
        with self.assertRaisesRegex(ValueError, "exact grid"):
            checked_temporal_inputs(
                self.profiles[0],
                self.profiles[1],
                self.profiles[2].model_copy(update={"width": 5}),
                self.profiles[3],
            )
        with self.assertRaisesRegex(ValueError, "precede"):
            checked_temporal_inputs(
                self.profiles[2], self.profiles[3], self.profiles[0], self.profiles[1]
            )

    def test_provider_subprocess_revalidates_inputs_and_output(self):
        manifest = {
            profile.asset_id: {
                "filename": path.name,
                "native": profile.model_dump(mode="json"),
            }
            for path, profile in zip(self.paths, self.profiles)
        }
        bridge = RasterBridge(
            self.root,
            manifest,
            ROOT / "scripts/raster_worker.py",
            temporal_worker=ROOT / "scripts/temporal_stack_worker.py",
        )
        request = TemporalAlignRequest(
            before_red_asset_id=self.profiles[0].asset_id,
            before_scl_asset_id=self.profiles[1].asset_id,
            after_red_asset_id=self.profiles[2].asset_id,
            after_scl_asset_id=self.profiles[3].asset_id,
            cloud_policy=CLOUD_POLICY,
        )
        with TestClient(create_app(bridge)) as client:
            health = client.get("/healthz")
            self.assertEqual(health.json()["temporal_tool_version"], "1.0.0")
            response = client.post("/temporal-align", json=request.model_dump(mode="json"))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                hashlib.sha256(response.content).hexdigest(),
                response.headers["X-Content-SHA256"],
            )
            self.assertEqual(response.headers["X-Temporal-Version"], "1.0.0")
            invalid = request.model_copy(
                update={"after_scl_asset_id": "asset-not-reviewed"}
            )
            self.assertEqual(
                client.post("/temporal-align", json=invalid.model_dump(mode="json")).status_code,
                422,
            )


class TemporalHarnessTests(unittest.TestCase):
    compute = TemporalStackTests.compute

    def setUp(self):
        TemporalStackTests.setUp(self)
        self.content, self.result = self.compute()
        self.tasks = make_tool_tasks(self.root)
        directory = self.tasks / "crop-smoke"
        task = json.loads((directory / "task.json").read_text())
        west, south, east, north = self.result.bbox_wgs84
        profiles = []
        for red, scl, cloud in (
            (self.profiles[0], self.profiles[1], self.result.before_cloud_fraction),
            (self.profiles[2], self.profiles[3], self.result.after_cloud_fraction),
        ):
            profiles.append({
                "item_id": red.item_id,
                "acquired": red.acquired,
                "platform": "sentinel-2",
                "instrument": "msi",
                "red": red.model_dump(mode="json"),
                "scl": scl.model_dump(mode="json"),
                "bbox_wgs84": [west, south, east, north],
                "cloud_fraction": cloud,
            })
        task.update(inputs=[profile.asset_id for profile in self.profiles])
        task["metadata"].update(
            artifact_identity="derivation-sha256-v1",
            cloud_mask_policy=CLOUD_POLICY,
            temporal_inputs=profiles,
        )
        task["budget"].update(
            max_input_bytes=512 * 1024 * 1024,
            max_artifact_bytes=128 * 1024 * 1024,
        )
        (directory / "task.json").write_text(json.dumps(task))
        scenario = json.loads((directory / "scenario.json").read_text())
        scenario["allowed_tools"] = ["temporal.select_align"]
        scenario["allowed_actions"] = ["tool.invoke", "memory.save_evidence", "answer.*"]
        (directory / "scenario.json").write_text(json.dumps(scenario))
        template = json.loads((directory / "assets.json").read_text())[0]
        assets = []
        for profile, path in zip(self.profiles, self.paths):
            asset = copy.deepcopy(template)
            cloud = self.result.before_cloud_fraction if profile.item_id == "before" else self.result.after_cloud_fraction
            asset.update(
                asset_id=profile.asset_id,
                uri="local://reviewed-temporal/" + path.name,
                media_type="image/tiff",
                roles=["input_image", "scene_classification" if profile.band == "scl" else "reflectance"],
                sha256=profile.sha256,
                size_bytes=path.stat().st_size,
                spatial={"crs": "EPSG:4326",
                         "bbox": {"west": west, "south": south, "east": east, "north": north},
                         "gsd_meters": profile.transform[0],
                         "shape": [profile.height, profile.width, 1]},
                temporal={"start": profile.acquired, "end": profile.acquired},
                platform="sentinel-2",
                instrument="msi",
                bands=[profile.band],
                quality={"cloud_cover_percent": cloud * 100.0, "nodata_fraction": 0.0},
                license="Contains modified Copernicus Sentinel data (2024)",
                source="reviewed temporal fixture",
            )
            assets.append(asset)
        (directory / "assets.json").write_text(json.dumps(assets))
        self.registry = TaskRegistry(self.tasks)
        self.manifest = self.registry.get("crop-smoke", "1.0.0")
        self.calls = 0

        def handle(request):
            self.calls += 1
            return httpx.Response(200, content=self.content, headers={
                "X-Temporal-Version": "1.0.0",
                "X-Temporal-Metadata": self.result.model_dump_json(),
                "X-Content-SHA256": hashlib.sha256(self.content).hexdigest(),
            })

        self.transport = httpx.MockTransport(handle)
        self.artifacts = ArtifactStore(str(self.root / "artifacts"))
        self.executor = TemporalExecutor("http://raster", self.artifacts, self.transport)
        self.arguments = {
            "operation": "select_align",
            "before": {"start": "2024-04-01T00:00:00Z", "end": "2024-04-10T23:59:59Z"},
            "after": {"start": "2024-05-01T00:00:00Z", "end": "2024-05-20T23:59:59Z"},
            "aoi": {"west": west, "south": south, "east": east, "north": north},
            "band": "red",
            "minimum_coverage_fraction": 0.95,
            "maximum_cloud_fraction": 0.3,
            "cloud_policy": CLOUD_POLICY,
        }
        self.action = ToolInvokeAction(
            type="tool.invoke", tool_id="temporal.select_align", arguments=self.arguments
        )

    def backend(self):
        return harness_app(
            database_path=str(self.root / "state.db"),
            v2_tasks_path=str(self.tasks),
            v2_artifacts_path=str(self.artifacts.root),
            v2_tool_executor=ToolRouter(temporal=self.executor),
        )

    def test_rejected_selection_is_typed_metadata_only_and_does_not_call_provider(self):
        action = self.action.model_copy(deep=True)
        action.arguments["maximum_cloud_fraction"] = 0.1
        prepared = self.executor.plan(action, self.manifest, self.manifest.task.inputs)
        self.assertTrue(prepared.metadata_only)
        output = prepared.invoke()
        self.assertIsNone(output.artifact)
        self.assertEqual(output.metadata["selection"]["reason"], "cloudy")
        self.assertEqual(self.calls, 0)
        with TestClient(self.backend()) as operator:
            initial = operator.post("/v2/reset", json={"task_ref": {
                "task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
            response = operator.post(f"/v2/episodes/{initial['episode_id']}/step", json={
                "client_action_id": "cloudy", "expected_state_version": 0,
                "action": action.model_dump(mode="json")})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(response.json()["data"]["observation"]["items"]), 1)
            self.assertEqual(self.calls, 0)

    def test_gateway_typed_artifact_restart_and_execution_replay(self):
        backend = self.backend()
        with TestClient(backend) as operator:
            initial = operator.post("/v2/reset", json={"task_ref": {
                "task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
            episode = initial["episode_id"]
            binding = build_binding(
                self.manifest,
                V2EpisodeState.model_validate(initial["state"]),
                operator.get("/v2/capabilities").json()["data"],
                hashlib.sha256(b"3" * 64).hexdigest(),
            )
            with TestClient(
                gateway(binding, "http://operator", httpx.ASGITransport(app=backend)),
                headers={"Authorization": "Bearer " + "3" * 64},
            ) as client:
                request = {"client_action_id": "temporal-1", "expected_state_version": 0,
                           "action": self.action.model_dump(mode="json")}
                response = client.post("/agent/step", json=request)
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                self.assertEqual(data["observation"]["items"][1]["type"], "temporal_stack")
                self.assertEqual(data["observation"]["items"][0]["inline"]["selection"]["status"], "selected")
                artifact_id = data["observation"]["items"][1]["artifact_ref"]
                artifact = client.get("/agent/artifacts/" + artifact_id)
                self.assertEqual(artifact.status_code, 200, artifact.text)
                metadata = artifact.json()["artifact"]
                self.assertEqual(metadata["temporal_stack"]["before"]["platform"], "sentinel-2")
                content = client.get("/agent/artifacts/" + artifact_id + "/content")
                self.assertEqual(content.status_code, 200, content.text[:100])
                self.assertEqual(content.content, self.content)
                evidence = {"type": "memory.save_evidence", "evidence": {
                    "evidence_id": "ev-temporal", "claim_id": "temporal", "source_ref": artifact_id,
                    "selector": {"pixel_window": [0, 0, 4, 4]},
                    "description": "Aligned two-date stack", "frozen_sha256": metadata["sha256"]}}
                self.assertEqual(client.post("/agent/step", json={"client_action_id": "evidence",
                    "expected_state_version": 1, "action": evidence}).status_code, 200)
                final = client.post("/agent/step", json={"client_action_id": "submit",
                    "expected_state_version": 2, "action": {"type": "answer.submit",
                    "answer": {"label": "aligned", "confidence": 1.0, "claims": []},
                    "confidence": 1.0, "evidence_ids": ["ev-temporal"]}})
                self.assertEqual(final.status_code, 200, final.text)
            self.assertEqual(self.calls, 1)
        with TestClient(self.backend()) as restarted:
            cached = restarted.post(f"/v2/episodes/{episode}/step", json=request)
            self.assertEqual(cached.status_code, 200, cached.text)
        self.assertEqual(self.calls, 1)
        gc.collect()
        before = hashlib.sha256((self.root / "state.db").read_bytes()).hexdigest()
        report = replay_episode(
            read_snapshot(self.root / "state.db", episode),
            self.registry,
            self.root / "replay",
            lambda artifacts: ToolRouter(
                temporal=TemporalExecutor("http://raster", artifacts, self.transport)
            ),
        )
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(self.calls, 2)
        self.assertTrue(report["artifact_checks"][0]["content_verified"])
        self.assertEqual(
            hashlib.sha256((self.root / "state.db").read_bytes()).hexdigest(), before
        )


if __name__ == "__main__":
    unittest.main()

"""Pixel-only tasks never acquire fabricated geography through the API."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from app.main import create_app
from app.core.artifacts import ArtifactStore
from app.core.capabilities import TaskRegistry
from app.core.evidence import validate_evidence
from app.core.schemas import EvidenceRef, PixelAssetRef, TaskAsset, TaskManifest
from .test_tool_execution import FakeExecutor, make_tool_tasks


def make_pixel_tasks(root: Path) -> Path:
    tasks = make_tool_tasks(root)
    path = tasks / "crop-smoke" / "assets.json"
    assets = json.loads(path.read_text())
    assets[0].update(spatial=None, temporal=None,
                     pixel={"coordinate_system": "pixel", "width": 16, "height": 12, "channels": 3},
                     source="synthetic pixel-only input", bands=["red", "green", "blue"])
    path.write_text(json.dumps(assets))
    return tasks


class PixelAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tasks = make_pixel_tasks(self.root)
        self.manifest = TaskRegistry(str(self.tasks)).get("crop-smoke", "1.0.0")
        self.asset = self.manifest.assets[0]
        self.artifacts = ArtifactStore(str(self.root / "artifacts"))
        self.executor = FakeExecutor(self.artifacts)

    def tearDown(self):
        self.temp.cleanup()

    def app(self):
        return create_app(database_path=str(self.root / "episode.db"), v2_tasks_path=str(self.tasks),
                          v2_artifacts_path=str(self.artifacts.root), v2_tool_executor=self.executor)

    def evidence(self, selector):
        return EvidenceRef(evidence_id="ev-pixel", claim_id="claim-pixel", source_ref=self.asset.asset_id,
                           selector=selector, description="Pixel-only image evidence", frozen_sha256=self.asset.sha256)

    def test_coordinate_variant_roundtrip(self):
        self.assertIsInstance(self.asset, PixelAssetRef)
        self.assertIsNone(self.asset.spatial)
        restored = TaskManifest.model_validate_json(self.manifest.model_dump_json())
        self.assertEqual(restored, self.manifest)
        self.assertEqual(restored.assets[0].pixel.width, 16)

    def test_reject_missing_dimensions_fake_geography_and_coerced_dimensions(self):
        value = self.asset.model_dump(mode="json")
        candidates = []
        missing = copy.deepcopy(value)
        del missing["pixel"]
        candidates.append(missing)
        geo = copy.deepcopy(value)
        geo["spatial"] = {"crs": "EPSG:4326", "bbox": {"west": 0, "south": 0, "east": 1, "north": 1}}
        candidates.append(geo)
        for bad in (0, -1, True, 16.0, "16"):
            changed = copy.deepcopy(value)
            changed["pixel"]["width"] = bad
            candidates.append(changed)
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValidationError):
                TypeAdapter(TaskAsset).validate_python(candidate)

    def test_pixel_task_must_be_headless_and_cannot_declare_map_actions(self):
        value = self.manifest.model_dump(mode="json")
        value["task"]["metadata"] = {}
        with self.assertRaisesRegex(ValidationError, "headless"):
            TaskManifest.model_validate(value)
        value = self.manifest.model_dump(mode="json")
        value["scenario"]["allowed_actions"].append("map.*")
        with self.assertRaisesRegex(ValidationError, "geographic map"):
            TaskManifest.model_validate(value)

    def test_duplicate_assets_and_inputs_are_rejected(self):
        for field in ("assets", "inputs"):
            value = self.manifest.model_dump(mode="json")
            items = value["assets"] if field == "assets" else value["task"]["inputs"]
            items.append(copy.deepcopy(items[0]))
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, "duplicate"):
                TaskManifest.model_validate(value)

    def test_pixel_evidence_bounds_and_coordinate_types(self):
        sources = {self.asset.asset_id: self.asset}
        validate_evidence(self.evidence({"pixel_window": [0, 0, 16, 12]}), sources)
        validate_evidence(self.evidence({"pixel_window": [15, 11, 1, 1]}), sources)
        for selector in (
            {"pixel_window": [15, 11, 2, 1]},
            {"pixel_window": [0, 0, 16, 13]},
            {"bbox": {"west": 0, "south": 0, "east": 1, "north": 1}},
            {"geometry": {"type": "Point", "coordinates": [0, 0]}},
            {"bands": ["imaginary-band"]},
        ):
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                validate_evidence(self.evidence(selector), sources)
        for window in ([0.0, 0, 1, 1], [True, 0, 1, 1], ["0", 0, 1, 1]):
            with self.subTest(window=window), self.assertRaises(ValidationError):
                self.evidence({"pixel_window": window})

    def test_reset_tool_evidence_answer_and_restart_without_map(self):
        with TestClient(self.app()) as client:
            reset = client.post("/v2/reset", json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}})
            self.assertEqual(reset.status_code, 201, reset.text)
            data = reset.json()["data"]
            self.assertIsNone(data["state"]["map"])
            self.assertEqual(data["observation"]["primary_type"], "asset_metadata")
            episode = data["episode_id"]
            url = f"/v2/episodes/{episode}/step"
            denied = client.post(url, json={"client_action_id": "geo", "expected_state_version": 0,
                                          "action": {"type": "map.zoom", "direction": "in", "factor": 2.0}})
            self.assertEqual(denied.status_code, 403, denied.text)
            self.assertEqual(denied.json()["error"]["code"], "coordinate_system_mismatch")
            crop = {"client_action_id": "crop", "expected_state_version": 0,
                    "action": {"type": "tool.invoke", "tool_id": "eo_gym.crop",
                               "arguments": {"asset_id": self.asset.asset_id, "aoi": [0, 0, .5, .5]}}}
            output = client.post(url, json=crop)
            self.assertEqual(output.status_code, 200, output.text)
            self.assertEqual(output.json()["data"], client.post(url, json=crop).json()["data"])
            self.assertEqual(self.executor.calls, 1)
            saved = client.post(url, json={"client_action_id": "evidence", "expected_state_version": 1,
                "action": {"type": "memory.save_evidence", "evidence": self.evidence({"pixel_window": [0, 0, 16, 12]}).model_dump(mode="json")}})
            self.assertEqual(saved.status_code, 200, saved.text)
            answer = client.post(url, json={"client_action_id": "answer", "expected_state_version": 2,
                "action": {"type": "answer.submit", "answer": {"label": "pixel", "confidence": 1,
                    "claims": []}, "evidence_ids": ["ev-pixel"]}})
            self.assertEqual(answer.status_code, 200, answer.text)
            self.assertTrue(answer.json()["data"]["terminated"])
            state = client.get(f"/v2/episodes/{episode}/state").json()["data"]
            replay = client.post(f"/v2/episodes/{episode}/replay").json()["data"]
            self.assertEqual(replay["status"], "passed")
        with TestClient(self.app()) as client:
            self.assertEqual(client.get(f"/v2/episodes/{episode}/state").json()["data"], state)
            self.assertEqual(client.post(url, json=crop).json()["data"], output.json()["data"])
            manifest = client.get("/v2/tasks/crop-smoke/versions/1.0.0").json()["data"]["manifest"]
            self.assertEqual(manifest["assets"][0]["pixel"]["coordinate_system"], "pixel")

    def test_primary_input_not_first_hidden_asset_and_mixed_inputs_have_no_map(self):
        from app.core.domain import create_initial_state
        from .helpers import TASKS_ROOT
        geo = TaskRegistry(str(TASKS_ROOT)).get("worldcover-grounded-vqa", "1.0.0").assets[0].model_copy(deep=True)
        geo.asset_id = "asset-hidden-geographic"
        value = self.manifest.model_dump(mode="json")
        value["assets"].insert(0, geo.model_dump(mode="json"))
        manifest = TaskManifest.model_validate(value)
        state, _ = create_initial_state("ep2-" + "1" * 32, manifest, 42)
        self.assertIsNone(state.map)
        self.assertNotIn(geo.asset_id, state.accessible_asset_refs)
        value["task"]["inputs"].insert(0, geo.asset_id)
        state, _ = create_initial_state("ep2-" + "2" * 32, TaskManifest.model_validate(value), 42)
        self.assertIsNone(state.map)

    def test_provider_must_agree_with_requested_aoi_and_pixel_contract(self):
        import hashlib
        import httpx
        from app.eo_gym_bridge import REVISION
        from app.core.domain import V2DomainError
        from app.core.schemas import ToolInvokeAction
        from app.core.tools.eo_gym import EOGymExecutor
        from .test_m2_runtime import PNG_BYTES
        manifest = self.manifest.model_copy(deep=True)
        manifest.assets[0].pixel.width = 2
        manifest.assets[0].pixel.height = 2
        action = ToolInvokeAction(type="tool.invoke", tool_id="eo_gym.crop",
                                  arguments={"asset_id": self.asset.asset_id, "aoi": [0, 0, .5, .5]})
        result = {"provider": "eo-gym", "upstream_revision": REVISION, "simulation": False,
                  "input_asset_id": self.asset.asset_id, "input_sha256": self.asset.sha256,
                  "sha256": hashlib.sha256(PNG_BYTES).hexdigest(), "size_bytes": len(PNG_BYTES),
                  "media_type": "image/png", "width": 1, "height": 1,
                  "bbox_px": [0, 0, 1, 1], "aoi_norm": [0, 0, .5, .5]}
        for change in ({}, {"bbox_px": [0, 0, 2, 1]}, {"aoi_norm": [0, 0, 1, 1]}, {"width": 2}):
            with self.subTest(change=change):
                def transport(request):
                    if request.method == "POST":
                        return httpx.Response(200, json={"output": {**result, **change}})
                    return httpx.Response(200, content=PNG_BYTES)
                executor = EOGymExecutor("http://provider", self.artifacts, transport=httpx.MockTransport(transport))
                if change:
                    with self.assertRaises(V2DomainError) as caught:
                        executor.invoke(action, manifest)
                    self.assertEqual(caught.exception.code, "invalid_tool_output")
                else:
                    self.assertEqual(executor.invoke(action, manifest).metadata["width"], 1)


if __name__ == "__main__":
    unittest.main()

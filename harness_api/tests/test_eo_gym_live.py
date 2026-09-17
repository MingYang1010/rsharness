"""Opt-in real upstream tests; no network or pretrained models required."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.eo_gym_bridge import CropBridge, create_app


@unittest.skipUnless(os.environ.get("EO_GYM_TEST_SOURCE"), "requires pinned upstream and CPU dependencies")
class RealEOGymTests(unittest.TestCase):
    def test_real_upstream_crop_pixels_hash_and_repeat(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            source_image = inputs / "source.png"
            image = Image.new("RGB", (16, 12))
            image.putdata([(x * 10, y * 12, (x + y) * 5) for y in range(12) for x in range(16)])
            image.save(source_image)
            digest = hashlib.sha256(source_image.read_bytes()).hexdigest()
            manifest = {"asset-fixture": {"role": "input_image", "filename": "source.png", "sha256": digest}}
            bridge = CropBridge(Path(os.environ["EO_GYM_TEST_SOURCE"]), inputs, root / "out", manifest,
                                Path(__file__).resolve().parents[2] / "scripts/eo_gym_crop_worker.py")
            client = TestClient(create_app(bridge))
            request = {"tool_name": "crop_optical_or_sar_image", "arguments": {"asset_id": "asset-fixture", "aoi": [0.25, 0.25, 0.75, 0.75]}}
            first = client.post("/execute", json=request)
            self.assertEqual(first.status_code, 200, first.text)
            result = first.json()["output"]
            self.assertEqual(result["bbox_px"], [4, 3, 12, 9])
            self.assertEqual((result["width"], result["height"]), (8, 6))
            self.assertFalse(result["simulation"])
            self.assertEqual(client.post("/execute", json=request).json(), first.json())
            content = client.get("/artifacts/" + result["sha256"])
            self.assertEqual(content.status_code, 200)
            self.assertEqual(hashlib.sha256(content.content).hexdigest(), result["sha256"])
            with Image.open(bridge.outputs / (result["sha256"] + ".png")) as actual:
                self.assertEqual(actual.tobytes(), image.crop((4, 3, 12, 9)).tobytes())
            self.assertEqual(client.get("/artifacts/not-a-hash").status_code, 404)
            denied = {"tool_name": request["tool_name"], "arguments": {**request["arguments"], "asset_id": "asset-hidden-label"}}
            self.assertEqual(client.post("/execute", json=denied).status_code, 403)

    def test_harness_calls_real_provider_and_saves_evidence(self):
        import httpx
        from PIL import Image
        from app.main import create_app as create_harness
        from app.v2.artifacts import ArtifactStore
        from app.v2.tools.eo_gym import EOGymExecutor
        from v2.test_tool_execution import make_tool_tasks
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            image = inputs / "fixture.png"
            Image.new("RGB", (16, 12), color=(100, 130, 180)).save(image)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            tasks = make_tool_tasks(root)
            asset_id = "asset-worldcover-n30e120"
            path = tasks / "crop-smoke" / "assets.json"
            assets = json.loads(path.read_text())
            assets[0].update(sha256=digest, source_snapshot_hash=digest, size_bytes=image.stat().st_size,
                             uri="local://fixture/fixture.png", source="synthetic georeferenced fixture, not WorldCover")
            path.write_text(json.dumps(assets))
            bridge = CropBridge(Path(os.environ["EO_GYM_TEST_SOURCE"]), inputs, root / "provider-out",
                {asset_id: {"role": "input_image", "filename": image.name, "sha256": digest}},
                Path(__file__).resolve().parents[2] / "scripts/eo_gym_crop_worker.py")
            with TestClient(create_app(bridge)) as provider:
                def transport(request):
                    result = provider.request(request.method, request.url.path, content=request.content,
                                              headers={"content-type": "application/json"})
                    return httpx.Response(result.status_code, content=result.content, headers=result.headers)
                artifacts = ArtifactStore(str(root / "artifacts"))
                executor = EOGymExecutor("http://provider", artifacts, transport=httpx.MockTransport(transport))
                app = create_harness(database_path=str(root / "episode.db"), v2_tasks_path=str(tasks),
                    v2_artifacts_path=str(artifacts.root), v2_tool_executor=executor)
                with TestClient(app) as client:
                    reset = client.post("/v2/reset", json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}, "seed": 42})
                    episode = reset.json()["data"]["episode_id"]
                    url = f"/v2/episodes/{episode}/step"
                    request = {"client_action_id": "crop", "expected_state_version": 0,
                        "action": {"type": "tool.invoke", "tool_id": "eo_gym.crop", "arguments": {"asset_id": asset_id, "aoi": [0.25, 0.25, 0.75, 0.75]}}}
                    response = client.post(url, json=request)
                    self.assertEqual(response.status_code, 200, response.text)
                    result = response.json()["data"]
                    self.assertEqual(client.post(url, json=request).json()["data"], result)
                    artifact_id = result["observation"]["items"][1]["artifact_ref"]
                    evidence = {"evidence_id": "ev-crop", "claim_id": "claim-size", "source_ref": artifact_id,
                        "selector": {"pixel_window": [0, 0, 8, 6]}, "description": "Upstream crop is 8 by 6 pixels", "frozen_sha256": artifact_id[4:]}
                    saved = client.post(url, json={"client_action_id": "evidence", "expected_state_version": 1,
                        "action": {"type": "memory.save_evidence", "evidence": evidence}})
                    self.assertEqual(saved.status_code, 200, saved.text)
                    answer = client.post(url, json={"client_action_id": "answer", "expected_state_version": 2,
                        "action": {"type": "answer.submit", "answer": {"label": "crop", "confidence": 1.0,
                        "claims": [{"claim_id": "claim-size", "text": "8 by 6 pixels"}]}, "confidence": 1.0, "evidence_ids": ["ev-crop"]}})
                    self.assertEqual(answer.status_code, 200, answer.text)
                    self.assertTrue(answer.json()["data"]["terminated"])
                    self.assertEqual(client.post(f"/v2/episodes/{episode}/replay").status_code, 200)
                    # This test verifies interaction, not semantic task accuracy.
                    self.assertIsNone(answer.json()["data"]["state"]["evaluation"])

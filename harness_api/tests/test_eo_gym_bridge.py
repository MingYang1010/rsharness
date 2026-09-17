import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.eo_gym_bridge import CropArguments, CropBridge, create_app, verify_upstream


class EOGymBridgeTests(unittest.TestCase):
    def test_request_rejects_remote_paths_and_simulation_tools(self):
        class FakeBridge:
            outputs = Path("/tmp/not-used-eo-bridge")
            def crop(self, request):
                return {"output": {"asset_id": request.arguments.asset_id}}
        client = TestClient(create_app(FakeBridge()))
        valid = {"tool_name": "crop_optical_or_sar_image", "arguments": {"asset_id": "asset-test", "aoi": [0, 0, 1, 1]}}
        self.assertEqual(client.post("/execute", json=valid).status_code, 200)
        self.assertEqual(client.post("/execute", json={**valid, "tool_name": "get_object_bbox_by_optical_image"}).status_code, 422)
        self.assertEqual(client.post("/execute", json={**valid, "image_path": "/secret"}).status_code, 422)
        valid["arguments"]["image_url"] = "https://example.com/a.jpg"
        self.assertEqual(client.post("/execute", json=valid).status_code, 422)

    def test_invalid_aoi_is_rejected(self):
        for aoi in [[0, 0, 0, 1], [0, 0, 2, 1], [0, 0, float("nan"), 1], [True, 0, 1, 1]]:
            with self.assertRaises(ValueError):
                CropArguments(asset_id="asset-a", aoi=aoi)

    def test_manifest_checks_content_and_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "input.png"
            image.write_bytes(b"sample")
            manifest = {"asset-a": {"role": "input_image", "filename": "input.png", "sha256": hashlib.sha256(b"sample").hexdigest()}}
            with patch("app.eo_gym_bridge.verify_upstream"):
                bridge = CropBridge(root, root, root / "outputs", manifest, root / "worker.py")
            self.assertEqual(bridge.resolve("asset-a")[0], image)
            with self.assertRaises(HTTPException):
                bridge.resolve("asset-unknown")
            manifest["asset-a"]["filename"] = "../input.png"
            with self.assertRaises(HTTPException):
                bridge.resolve("asset-a")
            manifest["asset-a"]["filename"] = "input.png"
            image.write_bytes(b"tampered")
            with self.assertRaises(HTTPException):
                bridge.resolve("asset-a")

    def test_source_receipt_is_not_trusted_without_pinned_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "acquisition.json").write_text(json.dumps({"revision": "fake", "files": []}))
            with self.assertRaises(ValueError):
                verify_upstream(source)

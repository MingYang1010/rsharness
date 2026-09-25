"""Decoded dimensions survive storage and constrain evidence after restart."""
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError
from rasterio.io import MemoryFile

from app.main import create_app
from app.core.artifacts import ArtifactStore, ArtifactStoreError
from app.core.evidence import validate_evidence
from app.core.schemas import Artifact, ArtifactLineage, EvidenceRef, PixelArtifactRef, PixelExtent
from app.core.tools.eo_gym import EOGymExecutor, ToolOutput
from .test_pixel_assets import make_pixel_tasks


def png_bytes(width=8, height=6):
    with MemoryFile() as memory:
        with memory.open(driver="PNG", width=width, height=height, count=3, dtype="uint8") as image:
            image.write(np.arange(3 * height * width, dtype=np.uint8).reshape(3, height, width))
        return memory.read()


class PixelExecutor(EOGymExecutor):
    max_output_bytes = 4096

    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = 0

    def invoke(self, action, manifest):
        _, asset = self.prepare(action, manifest)
        self.calls += 1
        artifact = self.artifacts.put_bytes(png_bytes(), kind="image", media_type="image/png",
            lineage=ArtifactLineage(tool_id=self.tool_id, tool_version=self.tool_version,
                input_refs=[asset.asset_id], parameters_hash=hashlib.sha256(b"test").hexdigest()),
            pixel=PixelExtent(coordinate_system="pixel", width=8, height=6, channels=3))
        return ToolOutput(artifact, {"width": 8, "height": 6}, asset.size_bytes)


class PixelArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ArtifactStore(str(self.root / "artifacts"))
        self.content = png_bytes()
        self.lineage = ArtifactLineage(tool_id="test.pixels", tool_version="1.0.0",
            input_refs=["asset-fixture"], parameters_hash="0" * 64)
        self.pixel = PixelExtent(coordinate_system="pixel", width=8, height=6, channels=3)

    def tearDown(self):
        self.temp.cleanup()

    def put(self, **kwargs):
        return self.store.put_bytes(kwargs.pop("content", self.content),
            kind=kwargs.pop("kind", "image"), media_type="image/png",
            lineage=self.lineage, pixel=kwargs.pop("pixel", self.pixel), **kwargs)

    def evidence(self, artifact, selector):
        return EvidenceRef(evidence_id="ev-artifact", claim_id="claim-pixel",
            source_ref=artifact.artifact_id, selector=selector,
            description="Verified image pixels", frozen_sha256=artifact.sha256)

    def test_typed_roundtrip_and_legacy_serialization_preserved(self):
        legacy = self.put(pixel=None)
        self.assertNotIn("pixel", legacy.model_dump(mode="json"))
        encoded = legacy.model_dump_json()
        self.assertEqual(TypeAdapter(Artifact).validate_json(encoded).model_dump_json(), encoded)
        typed = self.put()
        self.assertIsInstance(typed, PixelArtifactRef)
        self.assertEqual(typed.artifact_id, legacy.artifact_id)
        self.assertEqual(TypeAdapter(Artifact).validate_json(typed.model_dump_json()), typed)
        self.assertIsNone(typed.spatial)
        self.assertTrue(self.store.audit_exists(typed))

    def test_bad_claimed_dimensions_rejected_before_content_write(self):
        for change in ({"width": 7}, {"height": 8}, {"channels": 4}):
            with self.subTest(change=change), self.assertRaises(ArtifactStoreError):
                self.put(pixel=PixelExtent(**{**self.pixel.model_dump(), **change}))
        self.assertFalse(self.store.root.exists())

    def test_bad_payload_and_non_image_kind_rejected(self):
        for kwargs in ({"content": b"not PNG"}, {"content": self.content[:33]},
                       {"kind": "text"}, {"content": b"<VRTDataset/>"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ArtifactStoreError):
                self.put(**kwargs)
        self.assertFalse(self.store.root.exists())
        value = self.put().model_dump(mode="json")
        value["kind"] = "text"
        with self.assertRaises(ValidationError):
            TypeAdapter(Artifact).validate_python(value)

    def test_pixel_window_bounds_and_unknown_dimensions_fail_closed(self):
        artifact = self.put()
        sources = {artifact.artifact_id: artifact}
        validate_evidence(self.evidence(artifact, {"pixel_window": [0, 0, 8, 6]}), sources)
        validate_evidence(self.evidence(artifact, {"pixel_window": [7, 5, 1, 1]}), sources)
        for selector in ({"pixel_window": [7, 5, 2, 1]}, {"pixel_window": [0, 0, 8, 7]},
                         {"bbox": {"west": 0, "south": 0, "east": 1, "north": 1}},
                         {"geometry": {"type": "Point", "coordinates": [0, 0]}},
                         {"bands": ["nir"]}):
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                validate_evidence(self.evidence(artifact, selector), sources)
        legacy = self.put(pixel=None)
        with self.assertRaisesRegex(ValueError, "verified source dimensions"):
            validate_evidence(self.evidence(legacy, {"pixel_window": [0, 0, 1, 1]}), {legacy.artifact_id: legacy})

    def test_metadata_and_evidence_bounds_survive_restart_and_retry(self):
        tasks = make_pixel_tasks(self.root)
        executor = PixelExecutor(self.store)
        def app():
            return create_app(database_path=str(self.root / "episode.db"), v2_tasks_path=str(tasks),
                              v2_artifacts_path=str(self.store.root), v2_tool_executor=executor)
        with TestClient(app()) as client:
            reset = client.post("/v2/reset", json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}})
            episode = reset.json()["data"]["episode_id"]
            url = f"/v2/episodes/{episode}/step"
            crop = {"client_action_id": "crop", "expected_state_version": 0,
                "action": {"type": "tool.invoke", "tool_id": "eo_gym.crop", "arguments": {
                    "asset_id": "asset-worldcover-n30e120", "aoi": [0, 0, .5, .5]}}}
            result = client.post(url, json=crop)
            self.assertEqual(result.status_code, 200, result.text)
            artifact_id = result.json()["data"]["observation"]["items"][1]["artifact_ref"]
            metadata_url = f"/v2/artifacts/{artifact_id}?episode_id={episode}"
            self.assertEqual(client.get(f"/v2/artifacts/{artifact_id}").status_code, 422)
            self.assertEqual(client.get(f"/v2/artifacts/{artifact_id}/content").status_code, 422)
            self.assertEqual(client.get(f"/v2/artifacts/{artifact_id}?episode_id=invalid").status_code, 422)
            metadata = client.get(metadata_url).json()["data"]
            self.assertEqual(metadata["artifact"]["pixel"], self.pixel.model_dump())
            artifact = TypeAdapter(Artifact).validate_python(metadata["artifact"])
            invalid = {"client_action_id": "outside", "expected_state_version": 1,
                "action": {"type": "memory.save_evidence", "evidence": self.evidence(
                    artifact, {"pixel_window": [7, 5, 2, 1]}).model_dump(mode="json")}}
            rejected = client.post(url, json=invalid)
            self.assertEqual(rejected.status_code, 422, rejected.text)
            self.assertEqual(rejected.json()["error"]["code"], "invalid_evidence")
        with TestClient(app()) as client:
            self.assertEqual(client.get(metadata_url).json()["data"], metadata)
            self.assertEqual(client.post(url, json=crop).json()["data"], result.json()["data"])
            self.assertEqual(executor.calls, 1)
            self.assertEqual(client.post(url, json=invalid).json()["error"], rejected.json()["error"])
            fresh_invalid = {**invalid, "client_action_id": "outside-after-restart"}
            self.assertEqual(client.post(url, json=fresh_invalid).status_code, 422)
            valid = {"client_action_id": "valid", "expected_state_version": 1,
                "action": {"type": "memory.save_evidence", "evidence": self.evidence(
                    artifact, {"pixel_window": [0, 0, 8, 6]}).model_dump(mode="json")}}
            saved = client.post(url, json=valid)
            self.assertEqual(saved.status_code, 200, saved.text)
            self.assertEqual(client.post(f"/v2/episodes/{episode}/replay").json()["data"]["status"], "passed")
            other = client.post("/v2/reset", json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]["episode_id"]
            self.assertEqual(client.get(f"/v2/artifacts/{artifact_id}?episode_id={other}").status_code, 403)
            self.assertEqual(client.get(f"/v2/artifacts/{artifact_id}/content?episode_id={other}").status_code, 403)
            self.assertEqual(client.get(f"/v2/artifacts/{artifact_id}/content?episode_id={other}",
                                        headers={"Range": "bytes=0-7"}).status_code, 403)
            foreign = {**valid, "client_action_id": "foreign-evidence", "expected_state_version": 0}
            self.assertEqual(client.post(f"/v2/episodes/{other}/step", json=foreign).status_code, 422)
        with sqlite3.connect(self.root / "episode.db") as connection:
            stored = json.loads(connection.execute("SELECT artifact_json FROM v2_artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()[0])
            self.assertEqual(stored["pixel"], self.pixel.model_dump())


if __name__ == "__main__":
    unittest.main()

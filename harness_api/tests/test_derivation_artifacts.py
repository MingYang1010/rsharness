import copy
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from v2.test_tool_execution import FakeExecutor, make_tool_tasks
from v2.test_pixel_artifacts import png_bytes
from app.main import create_app
from app.agent_gateway import build_binding, create_app as create_gateway
from app.core.artifacts import ArtifactStore
from app.core.artifact_identity import DERIVATION_SCHEME, with_derivation_identity, validate_derivation_metadata
from app.core.capabilities import TaskRegistry
from app.core.events import sha256_json
from app.core.execution_replay import read_snapshot, replay_episode
from app.core.schemas import Artifact, ArtifactLineage, V2EpisodeState, StepRequest, PixelExtent, PixelArtifactRef
from app.core.tools.runtime import ToolOutput, ToolRouter


class ParameterExecutor(FakeExecutor):
    """Same real PNG fixture bytes, distinct lineage for each parameter/input."""
    def invoke(self, action, manifest):
        output = super().invoke(action, manifest)
        _, asset = self.prepare(action, manifest)
        artifact = output.artifact.model_copy(update={"lineage": ArtifactLineage(
            tool_id=self.tool_id, tool_version=self.tool_version, input_refs=[asset.asset_id],
            parameters_hash=sha256_json({"arguments": action.arguments, "input_sha256": asset.sha256}))})
        return ToolOutput(artifact, {"width": 1, "height": 1, "bbox_px": [0, 0, 1, 1],
            "aoi_norm": list(action.arguments["aoi"]), "input_asset_id": asset.asset_id,
            "input_sha256": asset.sha256, "upstream_revision": "f" * 40}, output.input_bytes)


class DerivationArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tasks = make_tool_tasks(self.root)
        directory = self.tasks / "crop-smoke"
        task = json.loads((directory / "task.json").read_text())
        task["metadata"]["artifact_identity"] = DERIVATION_SCHEME
        task["inputs"].append("asset-alias")
        (directory / "task.json").write_text(json.dumps(task))
        assets = json.loads((directory / "assets.json").read_text())
        assets.append({**copy.deepcopy(assets[0]), "asset_id": "asset-alias"})
        (directory / "assets.json").write_text(json.dumps(assets))
        self.registry = TaskRegistry(self.tasks)
        self.artifacts = ArtifactStore(str(self.root / "artifacts"))
        self.executor = ParameterExecutor(self.artifacts)
        self.backend = self.make_backend()
        self.operator = TestClient(self.backend)
        self.addCleanup(self.operator.close)
        initial = self.operator.post("/v2/reset", json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
        self.episode = initial["episode_id"]
        self.version = 0
        self.url = f"/v2/episodes/{self.episode}/step"
        self.binding = build_binding(self.registry.get("crop-smoke", "1.0.0"),
            V2EpisodeState.model_validate(initial["state"]), self.operator.get("/v2/capabilities").json()["data"],
            hashlib.sha256(b"1" * 64).hexdigest())

    def make_backend(self):
        return create_app(database_path=str(self.root / "state.db"), v2_tasks_path=str(self.tasks),
                          v2_artifacts_path=str(self.artifacts.root), v2_tool_executor=ToolRouter(self.executor))

    def crop(self, aoi=None, asset="asset-worldcover-n30e120"):
        request = {"client_action_id": "crop-" + str(self.version), "expected_state_version": self.version,
                   "action": {"type": "tool.invoke", "tool_id": "eo_gym.crop",
                              "arguments": {"asset_id": asset, "aoi": aoi or [0, 0, .5, .5]}}}
        response = self.operator.post(self.url, json=request)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.version = data["state"]["state_version"]
        artifact = self.operator.get("/v2/artifacts/" + data["observation"]["items"][1]["artifact_ref"],
                                     params={"episode_id": self.episode}).json()["data"]["artifact"]
        return request, data, artifact

    def count(self, table):
        with sqlite3.connect(self.root / "state.db") as connection:
            return connection.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]

    def test_distinct_parameters_share_bytes_not_provenance(self):
        _, _, first = self.crop()
        _, _, second = self.crop([.1, .1, .6, .6])
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["uri"], second["uri"])
        self.assertNotEqual(first["artifact_id"], second["artifact_id"])
        self.assertNotEqual(first["lineage"]["parameters_hash"], second["lineage"]["parameters_hash"])
        validate_derivation_metadata(first)
        validate_derivation_metadata(second)
        self.assertEqual(self.count("v2_artifacts"), 2)
        self.assertEqual(len(list((self.artifacts.root / "sha256").rglob("*"))), 2)  # shard + one blob
        stored = self.operator.get("/v2/artifacts/" + first["artifact_id"], params={"episode_id": self.episode})
        self.assertEqual(stored.json()["data"]["artifact"], first)

    def test_different_inputs_have_distinct_derivations(self):
        _, _, first = self.crop()
        _, _, second = self.crop(asset="asset-alias")
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertNotEqual(first["artifact_id"], second["artifact_id"])
        self.assertNotEqual(first["lineage"]["input_refs"], second["lineage"]["input_refs"])

    def test_same_derivation_is_stable_and_retries_do_not_execute(self):
        request, response, first = self.crop()
        self.assertEqual(self.operator.post(self.url, json=request).json()["data"], response)
        self.assertEqual(self.executor.calls, 1)
        _, _, second = self.crop()
        self.assertEqual(first, second)
        self.assertEqual(self.count("v2_artifacts"), 1)
        self.assertEqual(self.count("v2_observation_artifacts"), 2)

    def test_restart_preserves_independent_refs_and_cached_responses(self):
        request, response, first = self.crop()
        _, _, second = self.crop([.1, .1, .6, .6])
        with TestClient(self.make_backend()) as restarted:
            self.assertEqual(restarted.post(self.url, json=request).json()["data"], response)
            for artifact in (first, second):
                self.assertEqual(restarted.get("/v2/artifacts/" + artifact["artifact_id"], params={"episode_id": self.episode}).json()["data"]["artifact"], artifact)
        self.assertEqual(self.executor.calls, 2)

    def test_two_precise_evidence_refs_preserve_same_frozen_content_hash(self):
        artifacts = [self.crop()[2], self.crop([.1, .1, .6, .6])[2]]
        for index, artifact in enumerate(artifacts):
            body = {"client_action_id": f"evidence-{index}", "expected_state_version": self.version,
                "action": {"type": "memory.save_evidence", "evidence": {"evidence_id": f"ev-{index}",
                    "claim_id": "claim-test", "source_ref": artifact["artifact_id"], "selector": {"bands": ["visual"]},
                    "description": "Exact derivation", "frozen_sha256": artifact["sha256"]}}}
            response = self.operator.post(self.url, json=body)
            self.assertEqual(response.status_code, 200, response.text)
            self.version = response.json()["data"]["state"]["state_version"]
        refs = response.json()["data"]["state"]["evidence_refs"]
        self.assertEqual([e["source_ref"] for e in refs], [a["artifact_id"] for a in artifacts])

    def test_gateway_validates_derivation_metadata_and_content_independently(self):
        artifact = self.crop()[2]
        gateway = create_gateway(self.binding, "http://operator", httpx.ASGITransport(app=self.backend))
        with TestClient(gateway, headers={"Authorization": "Bearer " + "1" * 64}) as agent:
            result = agent.get("/agent/artifacts/" + artifact["artifact_id"])
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(agent.get("/agent/session").json()["task"]["artifact_identity"], DERIVATION_SCHEME)
            self.assertNotIn("uri", result.json()["artifact"])
            content = agent.get("/agent/artifacts/" + artifact["artifact_id"] + "/content")
            self.assertEqual(content.status_code, 200, content.text)
            self.assertEqual(hashlib.sha256(content.content).hexdigest(), artifact["sha256"])
            self.assertNotEqual(artifact["artifact_id"][4:], artifact["sha256"])

    def test_foreign_episode_cannot_read_or_cite_derivation(self):
        artifact = self.crop()[2]
        other = self.operator.post("/v2/reset", json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
        for suffix in ("", "/content"):
            response = self.operator.get("/v2/artifacts/" + artifact["artifact_id"] + suffix, params={"episode_id": other["episode_id"]})
            self.assertEqual(response.status_code, 403)
        response = self.operator.post(f"/v2/episodes/{other['episode_id']}/step", json={"client_action_id": "foreign", "expected_state_version": 0,
            "action": {"type": "memory.save_evidence", "evidence": {"evidence_id": "ev-foreign", "claim_id": "claim-test",
            "source_ref": artifact["artifact_id"], "selector": {"bands": ["visual"]}, "description": "foreign",
            "frozen_sha256": artifact["sha256"]}}})
        self.assertEqual(response.status_code, 422, response.text)

    def test_gateway_rejects_tampered_derivation_even_with_unchanged_bytes(self):
        artifact = self.crop()[2]
        tampered = copy.deepcopy(artifact)
        tampered["lineage"]["parameters_hash"] = "f" * 64
        with sqlite3.connect(self.root / "state.db") as connection:
            connection.execute("UPDATE v2_artifacts SET artifact_json=? WHERE artifact_id=?",
                               (json.dumps(tampered), artifact["artifact_id"]))
        gateway = create_gateway(self.binding, "http://operator", httpx.ASGITransport(app=self.backend))
        with TestClient(gateway, headers={"Authorization": "Bearer " + "1" * 64}) as agent:
            response = agent.get("/agent/artifacts/" + artifact["artifact_id"] + "/content")
            self.assertEqual(response.status_code, 502, response.text)
            self.assertEqual(response.json()["error"]["code"], "upstream_scope_mismatch")

    def test_pixel_dimensions_survive_identity_conversion_and_affect_hash(self):
        original = self.artifacts.put_bytes(png_bytes(), kind="image", media_type="image/png",
            lineage=ArtifactLineage(tool_id="test.pixels", tool_version="1.0.0", input_refs=["asset-input"], parameters_hash="a" * 64),
            pixel=PixelExtent(coordinate_system="pixel", width=8, height=6, channels=3))
        result = with_derivation_identity(original)
        self.assertIsInstance(result, PixelArtifactRef)
        self.assertEqual(result.pixel, original.pixel)
        value = result.model_dump(mode="json")
        value["pixel"]["width"] = 9
        with self.assertRaises(ValueError):
            validate_derivation_metadata(value)
        self.assertEqual(self.artifacts.read_content(result).content, self.artifacts.read_content(original).content)

    def test_tampered_lineage_checksum_size_uri_and_identity_rejected(self):
        artifact = self.crop()[2]
        variants = []
        lineage = copy.deepcopy(artifact)
        lineage["lineage"]["parameters_hash"] = "f" * 64
        variants.append(lineage)
        for key, value in (("sha256", "e" * 64), ("size_bytes", 3),
                           ("artifact_id", "art-" + "f" * 64), ("uri", "artifact://sha256/ff/" + "f" * 64)):
            variants.append({**artifact, key: value})
        for value in variants:
            with self.assertRaises(ValueError):
                validate_derivation_metadata(TypeAdapter(Artifact).validate_python(value).model_dump(mode="json"))

    def test_derivation_conversion_is_idempotent_and_legacy_unchanged(self):
        request, _, derived = self.crop()
        action = StepRequest.model_validate(request).action
        original = self.executor.invoke(action, self.registry.get("crop-smoke", "1.0.0")).artifact
        before = original.model_dump(mode="json")
        converted = with_derivation_identity(original)
        self.assertEqual(original.model_dump(mode="json"), before)
        self.assertNotIn("identity_scheme", before)
        self.assertEqual(original.artifact_id, "art-" + original.sha256)
        self.assertEqual(converted.model_dump(mode="json"), derived)
        self.assertEqual(with_derivation_identity(converted), converted)

    def test_unknown_or_rendered_policy_rejected(self):
        value = self.registry.get("crop-smoke", "1.0.0").model_dump(mode="json")
        for policy, profile in (("unknown", "headless-tools-v1"), (DERIVATION_SCHEME, "rendered-worldcover-v1")):
            trial = copy.deepcopy(value)
            trial["task"]["metadata"].update(artifact_identity=policy, observation_profile=profile)
            with self.assertRaises(ValidationError):
                type(self.registry.get("crop-smoke", "1.0.0")).model_validate(trial)

    def test_execution_replay_preserves_both_derivations(self):
        self.crop()
        self.crop([.1, .1, .6, .6])
        response = self.operator.post(self.url, json={"client_action_id": "stop", "expected_state_version": self.version,
            "action": {"type": "answer.abstain", "rationale": "test only", "evidence_ids": []}})
        self.assertEqual(response.status_code, 200, response.text)
        snapshot = read_snapshot(self.root / "state.db", self.episode)
        result = replay_episode(snapshot, self.registry, self.root / "replay", lambda artifacts: ToolRouter(ParameterExecutor(artifacts)))
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(len(result["artifact_checks"]), 2)
        self.assertTrue(result["semantic_trace_matched"])

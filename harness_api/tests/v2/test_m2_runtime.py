import base64
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds

from .helpers import PROJECT_ROOT, TASKS_ROOT

from app.main import create_app
from app.v2.artifacts import ArtifactStore
from app.v2.capabilities import TaskRegistry
from app.v2.domain import create_initial_state
from app.v2.evaluation import EvaluatorRegistry
from app.v2.renderer.base import RenderResult
from app.v2.schemas import (
    AnswerRecord,
    ArtifactLineage,
    EvidenceRef,
    EvidenceSelector,
    Metric,
    MetricResult,
    SpatialBoundingBox,
    SpatialExtent,
)


AOI = SpatialBoundingBox(west=121.45, south=31.2, east=121.55, north=31.3)
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "/x8AAusB9Wl2nQAAAABJRU5ErkJggg=="
)


class FakeRenderer:
    def __init__(self, artifact_store):
        self.artifact_store = artifact_store
        self.closed = []

    def health(self):
        return {"status": "ok", "versions": {"renderer": "test-1.0.0"}}

    def render(self, episode_id, map_state, semantic_state_hash):
        artifact = self.artifact_store.put_bytes(
            PNG_BYTES,
            kind="image",
            media_type="image/png",
            lineage=ArtifactLineage(
                tool_id="renderer.test.capture",
                tool_version="1.0.0",
                input_refs=[semantic_state_hash],
                parameters_hash=hashlib.sha256(b"test-renderer").hexdigest(),
            ),
            spatial=SpatialExtent(crs="EPSG:4326", bbox=map_state.bbox),
            temporal=map_state.active_time_range,
        )
        return RenderResult(
            artifact=artifact,
            pixel_stats={"sampled_unique_colors": 32},
            provenance={"versions": {"renderer": "test-1.0.0"}},
            readback={
                "consistent": True,
                "stable": True,
                "target_bbox": map_state.bbox.model_dump(mode="json"),
            },
        )

    def close_session(self, episode_id):
        self.closed.append(episode_id)


class FakeEvaluatorRegistry:
    def capability(self):
        return "available", {"evaluator_id": "test", "version": "1.1.0"}

    def evaluate_safely(
        self,
        manifest,
        state,
        artifacts,
        renderer_calls,
        failed_actions,
        wall_time_ms,
    ):
        metrics = [
            Metric(name="task.accuracy", value=1.0, weight=0.6),
            Metric(name="evidence.faithfulness", value=1.0, weight=0.3),
            Metric(name="process.efficiency", value=1.0, weight=0.1),
        ]
        return MetricResult(
            evaluation_id="eval-test-runtime",
            status="completed",
            metrics=metrics,
            aggregate_reward=1.0,
            evaluator_id=manifest.evaluator.evaluator_id,
            evaluator_version=manifest.evaluator.evaluator_version,
            diagnostics={
                "artifact_count": len(artifacts),
                "failed_actions": failed_actions,
                "renderer_calls": renderer_calls,
                "wall_time_ms": wall_time_ms,
            },
        )


class V2M2RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.database = self.root / "episodes.sqlite3"
        self.artifacts_root = self.root / "artifacts"
        self.artifact_store = ArtifactStore(str(self.artifacts_root))
        self.renderer = FakeRenderer(self.artifact_store)
        self.evaluator = FakeEvaluatorRegistry()
        self.client = TestClient(self._make_app(), raise_server_exceptions=False)

    def tearDown(self):
        self.client.close()
        self.tempdir.cleanup()

    def _make_app(self, renderer=None):
        return create_app(
            database_path=str(self.database),
            v2_tasks_path=str(TASKS_ROOT),
            v2_enabled=True,
            v2_artifacts_path=str(self.artifacts_root),
            v2_datasets_path=str(self.root / "datasets"),
            v2_renderer_config_path=str(
                PROJECT_ROOT / "config" / "v2" / "renderer.json"
            ),
            v2_renderer=renderer or self.renderer,
            v2_evaluator_registry=self.evaluator,
        )

    def _step(self, episode_id, action_id, version, action):
        return self.client.post(
            "/v2/episodes/%s/step" % episode_id,
            json={
                "client_action_id": action_id,
                "expected_state_version": version,
                "action": action,
            },
        )

    def _render_episode(self):
        reset = self.client.post(
            "/v2/reset",
            json={
                "task_ref": {
                    "task_id": "worldcover-grounded-vqa",
                    "task_version": "1.1.0",
                },
                "seed": 42,
            },
        )
        self.assertEqual(reset.status_code, 201, reset.text)
        episode_id = reset.json()["data"]["episode_id"]
        rendered = self._step(
            episode_id,
            "m2-set-view",
            0,
            {"type": "map.set_view", "bbox": AOI.model_dump(mode="json")},
        )
        self.assertEqual(rendered.status_code, 200, rendered.text)
        observation = rendered.json()["data"]["observation"]
        self.assertEqual(observation["primary_type"], "rendered_view")
        artifact_id = next(
            item["artifact_ref"]
            for item in observation["items"]
            if item["type"] == "rendered_view"
        )
        return episode_id, observation, artifact_id

    def test_render_artifact_evaluation_restart_and_replay(self):
        episode_id, observation, artifact_id = self._render_episode()
        observation_response = self.client.get(
            "/v2/episodes/%s/observations/%s"
            % (episode_id, observation["observation_id"])
        )
        self.assertEqual(observation_response.status_code, 200)
        self.assertEqual(
            observation_response.json()["data"]["observation"],
            observation,
        )

        artifact_response = self.client.get("/v2/artifacts/%s?episode_id=%s" % (artifact_id, episode_id))
        self.assertEqual(artifact_response.status_code, 200)
        artifact = artifact_response.json()["data"]["artifact"]
        content = self.client.get("/v2/artifacts/%s/content?episode_id=%s" % (artifact_id, episode_id))
        self.assertEqual(content.status_code, 200)
        self.assertEqual(content.content, PNG_BYTES)
        partial = self.client.get(
            "/v2/artifacts/%s/content?episode_id=%s" % (artifact_id, episode_id),
            headers={"Range": "bytes=0-7"},
        )
        self.assertEqual(partial.status_code, 206)
        self.assertEqual(partial.content, PNG_BYTES[:8])
        self.assertEqual(
            partial.headers["content-range"],
            "bytes 0-7/%s" % len(PNG_BYTES),
        )
        invalid_range = self.client.get(
            "/v2/artifacts/%s/content?episode_id=%s" % (artifact_id, episode_id),
            headers={"Range": "items=0-7"},
        )
        self.assertEqual(invalid_range.status_code, 416)
        self.assertEqual(invalid_range.json()["error"]["code"], "invalid_range")

        source_evidence = self._step(
            episode_id,
            "m2-source-evidence",
            1,
            {
                "type": "memory.save_evidence",
                "evidence": {
                    "evidence_id": "ev-m2-source",
                    "claim_id": "claim-dominant",
                    "source_ref": "asset-worldcover-n30e120",
                    "selector": {"bbox": AOI.model_dump(mode="json")},
                    "description": "Fixed AOI in the source WorldCover visual asset.",
                    "frozen_sha256": (
                        "9f376abaca38815c5c743126147aeffd1916bb1907ad98929d341d4e6c87381c"
                    ),
                },
            },
        )
        self.assertEqual(source_evidence.status_code, 200, source_evidence.text)
        rendered_evidence = self._step(
            episode_id,
            "m2-rendered-evidence",
            2,
            {
                "type": "memory.save_evidence",
                "evidence": {
                    "evidence_id": "ev-m2-rendered",
                    "claim_id": "claim-dominant",
                    "source_ref": artifact_id,
                    "selector": {"bbox": AOI.model_dump(mode="json")},
                    "description": "Deterministic rendered view of the fixed AOI.",
                    "frozen_sha256": artifact["sha256"],
                },
            },
        )
        self.assertEqual(rendered_evidence.status_code, 200, rendered_evidence.text)
        answer = self._step(
            episode_id,
            "m2-answer",
            3,
            {
                "type": "answer.submit",
                "answer": {
                    "label": "built-up",
                    "confidence": 0.95,
                    "claims": [
                        {"claim_id": "claim-dominant", "text": "Built-up dominates."}
                    ],
                },
                "confidence": 0.95,
                "evidence_ids": ["ev-m2-source", "ev-m2-rendered"],
            },
        )
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.json()["data"]["state"]["evaluation"]["status"], "completed")
        evaluation = self.client.get(
            "/v2/episodes/%s/evaluation" % episode_id
        )
        self.assertEqual(evaluation.status_code, 200, evaluation.text)
        self.assertEqual(len(evaluation.json()["data"]["evaluation"]["metrics"]), 3)
        self.assertIn(episode_id, self.renderer.closed)

        trace_before = self.client.get(
            "/v2/episodes/%s/trace" % episode_id
        ).json()["data"]
        self.assertIn(
            "artifact.created",
            [event["event_type"] for event in trace_before["events"]],
        )
        self.assertIn(
            "evaluation.completed",
            [event["event_type"] for event in trace_before["events"]],
        )
        state_before = self.client.get(
            "/v2/episodes/%s/state" % episode_id
        ).json()["data"]
        self.client.close()

        restarted_renderer = FakeRenderer(self.artifact_store)
        self.client = TestClient(
            self._make_app(renderer=restarted_renderer),
            raise_server_exceptions=False,
        )
        state_after = self.client.get(
            "/v2/episodes/%s/state" % episode_id
        ).json()["data"]
        self.assertEqual(state_after, state_before)
        self.assertEqual(
            self.client.get("/v2/artifacts/%s/content?episode_id=%s" % (artifact_id, episode_id)).content,
            PNG_BYTES,
        )
        self.assertEqual(
            self.client.get(
                "/v2/episodes/%s/evaluation" % episode_id
            ).json()["data"]["evaluation"],
            evaluation.json()["data"]["evaluation"],
        )
        replay = self.client.post("/v2/episodes/%s/replay" % episode_id)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json()["data"]["status"], "passed")
        trace_after = self.client.get(
            "/v2/episodes/%s/trace" % episode_id
        ).json()["data"]
        self.assertEqual(trace_after["trace_hash"], trace_before["trace_hash"])
        self.assertEqual(
            trace_after["semantic_trace_hash"],
            trace_before["semantic_trace_hash"],
        )

    def test_registered_artifact_missing_or_corrupt_returns_typed_error(self):
        episode_id, _, artifact_id = self._render_episode()
        artifact = self.client.get(
            "/v2/artifacts/%s?episode_id=%s" % (artifact_id, episode_id)
        ).json()["data"]["artifact"]
        path = self.artifact_store.content_path(artifact["sha256"])
        path.write_bytes(b"corrupt")
        corrupt = self.client.get("/v2/artifacts/%s/content?episode_id=%s" % (artifact_id, episode_id))
        self.assertEqual(corrupt.status_code, 503)
        self.assertEqual(corrupt.json()["error"]["code"], "artifact_content_corrupt")
        path.unlink()
        missing = self.client.get("/v2/artifacts/%s/content?episode_id=%s" % (artifact_id, episode_id))
        self.assertEqual(missing.status_code, 503)
        self.assertEqual(missing.json()["error"]["code"], "artifact_content_missing")


class WorldCoverEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset_root = self.root / "datasets"
        self.raster_path = self.dataset_root / "worldcover-test" / "canonical.tif"
        self.raster_path.parent.mkdir(parents=True)
        values = numpy.full((10, 10), 50, dtype=numpy.uint8)
        values[:2, :] = 10
        with rasterio.open(
            self.raster_path,
            "w",
            driver="GTiff",
            width=10,
            height=10,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_bounds(AOI.west, AOI.south, AOI.east, AOI.north, 10, 10),
            nodata=0,
        ) as dataset:
            dataset.write(values, 1)
        digest = hashlib.sha256(self.raster_path.read_bytes()).hexdigest()

        registry = TaskRegistry(str(TASKS_ROOT))
        self.manifest = registry.get(
            "worldcover-grounded-vqa", "1.1.0"
        ).model_copy(deep=True)
        canonical = next(
            asset
            for asset in self.manifest.assets
            if asset.asset_id == "asset-worldcover-n30e120-canonical"
        )
        canonical.uri = "local://dataset/worldcover-test/canonical.tif"
        canonical.sha256 = digest
        canonical.source_snapshot_hash = digest
        canonical.size_bytes = self.raster_path.stat().st_size
        canonical.spatial.bbox = AOI
        canonical.spatial.shape = [10, 10]
        self.manifest.evaluator.config["expected_pixel_count"] = 100
        self.manifest.evaluator.config["expected_distribution"] = {
            "10": 20,
            "50": 80,
        }

        self.artifact_store = ArtifactStore(str(self.root / "artifacts"))
        self.evaluator = EvaluatorRegistry(
            str(self.dataset_root),
            self.artifact_store,
        )
        self.state, _ = create_initial_state(
            "ep2-00000000000000000000000000000001",
            self.manifest,
            42,
            timestamp="2026-08-30T00:00:00Z",
        )
        artifact = self.artifact_store.put_bytes(
            PNG_BYTES,
            kind="image",
            media_type="image/png",
            lineage=ArtifactLineage(
                tool_id="renderer.test.capture",
                tool_version="1.0.0",
                input_refs=["0" * 64],
                parameters_hash=hashlib.sha256(b"test").hexdigest(),
            ),
            spatial=SpatialExtent(crs="EPSG:4326", bbox=AOI),
        )
        source_evidence = EvidenceRef(
            evidence_id="ev-evaluator-source",
            claim_id="claim-dominant",
            source_ref="asset-worldcover-n30e120",
            selector=EvidenceSelector(bbox=AOI),
            description="Canonical AOI source evidence.",
            frozen_sha256=(
                "9f376abaca38815c5c743126147aeffd1916bb1907ad98929d341d4e6c87381c"
            ),
        )
        rendered_evidence = EvidenceRef(
            evidence_id="ev-evaluator-rendered",
            claim_id="claim-dominant",
            source_ref=artifact.artifact_id,
            selector=EvidenceSelector(bbox=AOI),
            description="Rendered AOI evidence.",
            frozen_sha256=artifact.sha256,
        )
        self.artifacts = {artifact.artifact_id: artifact}
        self.state.evidence_refs = [source_evidence, rendered_evidence]
        self.state.final_answer = AnswerRecord(
            outcome="submitted",
            answer={"label": "built-up", "confidence": 0.95, "claims": []},
            confidence=0.95,
            evidence_ids=[source_evidence.evidence_id, rendered_evidence.evidence_id],
        )
        self.state.step_count = 4
        self.state.status = "terminated"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_real_raster_evaluator_returns_three_grounded_metrics(self):
        result = self.evaluator.evaluate_safely(
            manifest=self.manifest,
            state=self.state,
            artifacts=self.artifacts,
            renderer_calls=1,
            failed_actions=0,
            wall_time_ms=1000,
        )
        self.assertEqual(result.status, "completed", result.model_dump())
        self.assertEqual(
            {metric.name: metric.value for metric in result.metrics},
            {
                "task.accuracy": 1.0,
                "evidence.faithfulness": 1.0,
                "process.efficiency": 1.0,
            },
        )
        self.assertEqual(result.aggregate_reward, 1.0)
        self.assertEqual(
            result.diagnostics["class_distribution"],
            {"10": 20, "50": 80},
        )

    def test_evaluator_failure_keeps_answer_and_nullable_reward(self):
        canonical = next(
            asset
            for asset in self.manifest.assets
            if asset.asset_id == "asset-worldcover-n30e120-canonical"
        )
        canonical.sha256 = "0" * 64
        answer_before = self.state.final_answer.model_dump(mode="json")
        result = self.evaluator.evaluate_safely(
            manifest=self.manifest,
            state=self.state,
            artifacts=self.artifacts,
            renderer_calls=1,
            failed_actions=0,
            wall_time_ms=1000,
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics, [])
        self.assertIsNone(result.aggregate_reward)
        self.assertEqual(
            result.diagnostics["code"],
            "gold_asset_checksum_mismatch",
        )
        self.assertEqual(
            self.state.final_answer.model_dump(mode="json"),
            answer_before,
        )


if __name__ == "__main__":
    unittest.main()

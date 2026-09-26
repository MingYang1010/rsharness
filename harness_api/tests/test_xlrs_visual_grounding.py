import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy
from rasterio.io import MemoryFile

from app.core.artifacts import ArtifactStore
from app.core.domain import create_initial_state
from app.core.evaluation import EvaluatorRegistry
from app.core.schemas import (
    AnswerRecord,
    ArtifactLineage,
    BudgetSpec,
    EvaluatorSpec,
    EvidenceRef,
    EvidenceSelector,
    PixelAssetRef,
    PixelExtent,
    ScenarioProfile,
    TaskManifest,
    TaskSpec,
)


INPUT_ID = "asset-xlrs-grounding"
EXPECTED_BBOX = [0.2, 0.2, 0.4, 0.4]
WIDTH = 10
HEIGHT = 10


def _png() -> bytes:
    values = numpy.zeros((3, HEIGHT, WIDTH), dtype=numpy.uint8)
    values[:, 2:4, 2:4] = 255
    with MemoryFile() as memory:
        with memory.open(
            driver="PNG",
            width=WIDTH,
            height=HEIGHT,
            count=3,
            dtype="uint8",
        ) as dataset:
            dataset.write(values)
        return memory.read()


def _manifest() -> TaskManifest:
    weights = {
        "task.accuracy": 0.55,
        "evidence.faithfulness": 0.35,
        "process.efficiency": 0.1,
    }
    task = TaskSpec(
        task_id="xlrs-grounding-test",
        task_version="1.0.0",
        family="visual_grounding",
        prompt="Locate the described target.",
        inputs=[INPUT_ID],
        scenario_profile="xlrs-grounding-test-v1",
        answer_schema={
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                    "minItems": 4,
                    "maxItems": 4,
                }
            },
            "required": ["bbox"],
        },
        evaluator="xlrs-visual-grounding-v1",
        budget=BudgetSpec(
            max_steps=12,
            max_tool_calls=4,
            max_wall_time_ms=300000,
            max_input_bytes=1024,
            max_artifact_bytes=1024,
        ),
        seed=42,
        metric_aggregation=weights,
        metadata={
            "observation_profile": "headless-tools-v1",
            "artifact_identity": "derivation-sha256-v1",
        },
    )
    scenario = ScenarioProfile(
        profile_id="xlrs-grounding-test-v1",
        domain="visual_grounding",
        data_cutoff="2026-09-26T00:00:00Z",
        allowed_actions=["tool.invoke", "memory.save_evidence", "answer.*"],
        allowed_tools=["eo_gym.crop"],
        network_policy="none",
        evidence_required=True,
        abstention_allowed=True,
        human_review_policy="allowed",
    )
    evaluator = EvaluatorSpec(
        evaluator_id="xlrs-visual-grounding-v1",
        evaluator_version="1.0.0",
        metric_names=list(weights),
        aggregate_weights=weights,
        config={
            "input_asset_id": INPUT_ID,
            "expected_bbox": EXPECTED_BBOX,
            "minimum_iou": 0.2,
            "expected_outcome": "submitted",
            "public_question_sha256": hashlib.sha256(
                "Locate the described target.".encode()
            ).hexdigest(),
            "efficiency": {
                "ideal_steps": 3,
                "wall_time_soft_limit_ms": 30000,
            },
        },
    )
    asset = PixelAssetRef(
        asset_id=INPUT_ID,
        uri="local://approved-input/grounding.jpg",
        media_type="image/jpeg",
        roles=["input_image"],
        sha256=hashlib.sha256(b"source-image").hexdigest(),
        size_bytes=1000,
        spatial=None,
        pixel=PixelExtent(
            coordinate_system="pixel",
            width=WIDTH,
            height=HEIGHT,
            channels=3,
        ),
        license="existing-local-research-copy",
        source="XLRS visual grounding test",
    )
    return TaskManifest(
        task=task,
        scenario=scenario,
        assets=[asset],
        evaluator=evaluator,
        task_manifest_hash="0" * 64,
    )


class XLRSVisualGroundingEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.artifacts = ArtifactStore(str(self.root / "artifacts"))
        self.manifest = _manifest()

    def tearDown(self):
        self.tempdir.cleanup()

    def _episode(self, answer_value, evidence=True, crop_aoi=None):
        state, _ = create_initial_state(
            "ep2-" + "1" * 32,
            self.manifest,
            42,
            timestamp="2026-09-26T00:00:00Z",
        )
        content = _png()
        artifact = self.artifacts.put_bytes(
            content,
            kind="image",
            media_type="image/png",
            lineage=ArtifactLineage(
                tool_id="eo_gym.crop",
                tool_version="1.1.0",
                input_refs=[INPUT_ID],
                parameters_hash=hashlib.sha256(b"crop").hexdigest(),
            ),
            pixel=PixelExtent(
                coordinate_system="pixel",
                width=WIDTH,
                height=HEIGHT,
                channels=3,
            ),
        )
        evidence_ref = EvidenceRef(
            evidence_id="ev-crop",
            claim_id="claim-grounding",
            source_ref=artifact.artifact_id,
            selector=EvidenceSelector(pixel_window=[0, 0, WIDTH, HEIGHT]),
            description="Full-image crop.",
            frozen_sha256=artifact.sha256,
        )
        state.evidence_refs = [evidence_ref] if evidence else []
        state.final_answer = AnswerRecord(
            outcome="submitted",
            answer={"bbox": answer_value},
            confidence=1.0,
            evidence_ids=[evidence_ref.evidence_id] if evidence else [],
        )
        state.step_count = 3
        state.status = "terminated"
        tool_results = [
            {
                "tool_id": "eo_gym.crop",
                "tool_version": "1.1.0",
                "status": "completed",
                "artifact_id": artifact.artifact_id,
                "aoi_norm": (
                    [0.0, 0.0, 1.0, 1.0]
                    if crop_aoi is None
                    else crop_aoi
                ),
            }
        ]
        result = EvaluatorRegistry(
            str(self.root / "datasets"), self.artifacts
        ).evaluate_safely(
            self.manifest,
            state,
            {artifact.artifact_id: artifact},
            renderer_calls=0,
            failed_actions=0,
            wall_time_ms=1000,
            tool_results=tool_results,
        )
        return result, artifact

    def test_exact_partial_and_no_overlap_accuracy(self):
        cases = [
            ([0.2, 0.2, 0.4, 0.4], 1.0, 1.0),
            ([0.2, 0.2, 0.6, 0.6], 1.0, 0.25),
            ([0.6, 0.6, 0.8, 0.8], 0.0, 0.0),
        ]
        for answer, accuracy, iou in cases:
            with self.subTest(answer=answer):
                result, _ = self._episode(answer)
                self.assertEqual(result.status, "completed")
                metric = next(
                    item for item in result.metrics if item.name == "task.accuracy"
                )
                self.assertEqual(metric.value, accuracy)
                self.assertAlmostEqual(metric.diagnostics["iou"], iou)

    def test_malformed_answer_fails_closed_without_crashing(self):
        for answer in ([0.4, 0.4, 0.2, 0.2], [0, 1], "0.2,0.2,0.4,0.4"):
            with self.subTest(answer=answer):
                result, _ = self._episode(answer)
                self.assertEqual(result.status, "completed")
                accuracy = next(
                    item for item in result.metrics if item.name == "task.accuracy"
                )
                faithfulness = next(
                    item
                    for item in result.metrics
                    if item.name == "evidence.faithfulness"
                )
                self.assertEqual(accuracy.value, 0.0)
                self.assertFalse(accuracy.diagnostics["answer_bbox_valid"])
                self.assertEqual(faithfulness.value, 0.0)

    def test_missing_or_non_containing_crop_evidence_fails_faithfulness(self):
        for evidence, crop_aoi in ((False, None), (True, [0.8, 0.8, 1.0, 1.0])):
            with self.subTest(evidence=evidence, crop_aoi=crop_aoi):
                result, _ = self._episode(
                    EXPECTED_BBOX,
                    evidence=evidence,
                    crop_aoi=crop_aoi,
                )
                self.assertEqual(result.status, "completed")
                faithfulness = next(
                    item
                    for item in result.metrics
                    if item.name == "evidence.faithfulness"
                )
                expected = evidence and crop_aoi is None
                self.assertEqual(faithfulness.value, float(expected))


if __name__ == "__main__":
    unittest.main()

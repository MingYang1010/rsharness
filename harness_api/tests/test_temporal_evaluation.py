import hashlib
import tempfile
import unittest
from pathlib import Path

from app.core.artifacts import ArtifactStore
from app.core.domain import create_initial_state
from app.core.evaluation import EvaluatorRegistry
from app.core.raster_math import CLOUD_POLICY
from app.core.schemas import (
    AnswerAbstainAction,
    AnswerRecord,
    ArtifactLineage,
    AssetRef,
    AssetQuality,
    BudgetSpec,
    EvaluatorSpec,
    EvidenceRef,
    EvidenceSelector,
    ScenarioProfile,
    SpatialBoundingBox,
    SpatialExtent,
    TaskManifest,
    TaskSpec,
    TemporalExtent,
    TemporalStackDescriptor,
    TemporalStackMember,
    ToolInvokeAction,
)
from app.core.store import V2EpisodeStore
from app.core.temporal import (
    STACK_BANDS,
    TOOL_ID,
    VERSION,
    TemporalSelectedInput,
    TemporalSelectionResult,
    TemporalStackResult,
    TemporalToolResult,
)
from app.core.tools.runtime import PreparedTool, ToolOutput, ToolRouter


INPUT_IDS = ["before-red", "before-scl", "after-red", "after-scl"]
BEFORE_ITEM = "sentinel-before"
AFTER_ITEM = "sentinel-after"
BEFORE_TIME = "2024-04-05T03:00:00Z"
AFTER_TIME = "2024-04-15T03:00:00Z"
BBOX = SpatialBoundingBox(west=118.79, south=31.99, east=118.81, north=32.01)


def _asset(asset_id: str, band: str, timestamp: str) -> AssetRef:
    return AssetRef(
        asset_id=asset_id,
        uri="local://approved-input/%s.tif" % asset_id,
        media_type="image/tiff",
        roles=["input", "reflectance" if band == "red" else "scene_classification"],
        sha256=hashlib.sha256(asset_id.encode()).hexdigest(),
        size_bytes=1024,
        spatial=SpatialExtent(
            crs="EPSG:4326",
            bbox=BBOX,
            gsd_meters=10.0,
            shape=[4, 4, 1],
        ),
        temporal=TemporalExtent(start=timestamp, end=timestamp),
        platform="sentinel-2",
        instrument="msi",
        bands=[band],
        quality=AssetQuality(
            cloud_cover_percent=5.0 if timestamp == BEFORE_TIME else 8.0,
            coverage_fraction=1.0,
        ),
        license="Copernicus Sentinel Data Terms",
        source="reviewed-local-window",
    )


def _manifest(reason: str) -> TaskManifest:
    status = "selected" if reason == "selected" else "rejected"
    coverage = {
        "selected": "pass",
        "cloudy": "pass",
        "sensor_mismatch": "pass",
        "insufficient_coverage": "fail",
        "wrong_date": "not_evaluated",
    }[reason]
    metrics = {
        "temporal.validity": 0.3,
        "spatial.coverage": 0.2,
        "answer.abstention_correctness": 0.2,
        "evidence.faithfulness": 0.2,
        "process.efficiency": 0.1,
    }
    config = {
        "expected_selection_status": status,
        "expected_selection_reason": reason,
        "expected_coverage_decision": coverage,
        "minimum_aligned_coverage_fraction": 0.9,
        "efficiency": {
            "ideal_steps": 3 if status == "selected" else 2,
            "ideal_tool_calls": 1,
            "expected_renderer_calls": 0,
            "wall_time_soft_limit_ms": 10000,
        },
    }
    if status == "selected":
        config.update(
            expected_before_item_id=BEFORE_ITEM,
            expected_after_item_id=AFTER_ITEM,
            expected_input_asset_ids=INPUT_IDS,
        )
    assets = [
        _asset("before-red", "red", BEFORE_TIME),
        _asset("before-scl", "scl", BEFORE_TIME),
        _asset("after-red", "red", AFTER_TIME),
        _asset("after-scl", "scl", AFTER_TIME),
    ]
    return TaskManifest(
        task=TaskSpec(
            task_id="temporal-eval-%s" % reason,
            task_version="1.0.0",
            family="temporal_selection",
            prompt="Select a valid temporal pair or abstain.",
            inputs=INPUT_IDS,
            scenario_profile="temporal-selection-test-v1",
            answer_schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "claims": {"type": "array"},
                },
                "required": ["label", "claims"],
            },
            evaluator="temporal-selection-v1",
            budget=BudgetSpec(
                max_steps=6,
                max_tool_calls=2,
                max_wall_time_ms=60000,
                max_input_bytes=100000,
                max_artifact_bytes=100000,
            ),
            seed=42,
            metric_aggregation=metrics,
            metadata={
                "observation_profile": "headless-tools-v1",
                "artifact_identity": "derivation-sha256-v1",
                "evaluation_profile": "temporal-selection-v1",
            },
        ),
        scenario=ScenarioProfile(
            profile_id="temporal-selection-test-v1",
            domain="temporal_selection",
            data_cutoff="2026-09-20T00:00:00Z",
            allowed_actions=["tool.invoke", "memory.save_evidence", "answer.*"],
            allowed_tools=[TOOL_ID],
            network_policy="none",
            evidence_required=status == "selected",
            abstention_allowed=True,
            human_review_policy="allowed",
        ),
        assets=assets,
        evaluator=EvaluatorSpec(
            evaluator_id="temporal-selection-v1",
            evaluator_version="1.0.0",
            metric_names=list(metrics),
            aggregate_weights=metrics,
            config=config,
        ),
        task_manifest_hash="0" * 64,
    )


def _selection(reason: str) -> TemporalToolResult:
    common = {
        "operation": "select_align",
        "considered_item_ids": [BEFORE_ITEM, AFTER_ITEM],
        "minimum_coverage_fraction": 0.9,
        "maximum_cloud_fraction": 0.2,
        "cloud_policy": CLOUD_POLICY,
    }
    if reason != "selected":
        return TemporalToolResult(
            selection=TemporalSelectionResult(
                status="rejected",
                reason=reason,
                **common,
            )
        )
    before = TemporalSelectedInput(
        item_id=BEFORE_ITEM,
        acquired=BEFORE_TIME,
        platform="sentinel-2",
        instrument="msi",
        red_asset_id=INPUT_IDS[0],
        scl_asset_id=INPUT_IDS[1],
        coverage_fraction=1.0,
        cloud_fraction=0.05,
    )
    after = TemporalSelectedInput(
        item_id=AFTER_ITEM,
        acquired=AFTER_TIME,
        platform="sentinel-2",
        instrument="msi",
        red_asset_id=INPUT_IDS[2],
        scl_asset_id=INPUT_IDS[3],
        coverage_fraction=1.0,
        cloud_fraction=0.08,
    )
    stack = TemporalStackResult(
        operation="two-date-red-scl-stack",
        input_asset_ids=INPUT_IDS,
        input_sha256=[hashlib.sha256(value.encode()).hexdigest() for value in INPUT_IDS],
        before_item_id=BEFORE_ITEM,
        after_item_id=AFTER_ITEM,
        before_acquired=BEFORE_TIME,
        after_acquired=AFTER_TIME,
        crs="EPSG:32650",
        transform=[10.0, 0.0, 480000.0, 0.0, -10.0, 3540000.0],
        bbox_wgs84=[BBOX.west, BBOX.south, BBOX.east, BBOX.north],
        width=4,
        height=4,
        count=4,
        dtype="uint16",
        nodata=65535,
        band_order=STACK_BANDS,
        scales=[0.0001, 1.0, 0.0001, 1.0],
        offsets=[0.0, 0.0, 0.0, 0.0],
        before_valid_pixels=16,
        after_valid_pixels=16,
        aligned_valid_pixels=16,
        total_pixels=16,
        before_coverage_fraction=1.0,
        after_coverage_fraction=1.0,
        aligned_coverage_fraction=1.0,
        before_cloud_fraction=0.05,
        after_cloud_fraction=0.08,
        cloud_policy=CLOUD_POLICY,
        alignment_method="exact-red-grid-and-nearest-scl",
        invalid_policy="source-mask-or-nodata-or-scl-zero-or-outside-source-or-cross-date-gap",
    )
    return TemporalToolResult(
        selection=TemporalSelectionResult(
            status="selected",
            reason="selected",
            before=before,
            after=after,
            **common,
        ),
        stack=stack,
    )


def _tool_result(reason: str) -> dict:
    result = _selection(reason)
    return {
        "tool_id": TOOL_ID,
        "tool_version": VERSION,
        "status": "completed",
        **result.model_dump(mode="json"),
    }


class _Registry:
    def __init__(self, manifest: TaskManifest):
        self.manifest = manifest

    def get(self, task_id: str, task_version: str) -> TaskManifest:
        if (task_id, task_version) != (
            self.manifest.task.task_id,
            self.manifest.task.task_version,
        ):
            raise KeyError(task_id)
        return self.manifest


class _RejectedTemporalExecutor:
    tool_id = TOOL_ID
    tool_version = VERSION

    def __init__(self, reason: str):
        self.result = _selection(reason)

    def plan(self, action, manifest, accessible_asset_refs):
        return PreparedTool(
            tool_version=VERSION,
            input_bytes=0,
            max_output_bytes=0,
            invoke=lambda: ToolOutput(
                artifact=None,
                metadata=self.result.model_dump(mode="json"),
                input_bytes=0,
            ),
            metadata_only=True,
        )


class TemporalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.datasets = self.root / "datasets"
        self.datasets.mkdir()
        self.artifacts = ArtifactStore(str(self.root / "artifacts"))
        self.evaluator = EvaluatorRegistry(str(self.datasets), self.artifacts)

    def tearDown(self):
        self.tempdir.cleanup()

    def _state(self, manifest: TaskManifest):
        state, _ = create_initial_state(
            "ep2-00000000000000000000000000000001",
            manifest,
            42,
            timestamp="2026-09-20T00:00:00Z",
        )
        state.status = "terminated"
        return state

    def _artifact(self, result: TemporalToolResult):
        selection = result.selection
        stack = result.stack
        assert selection.before is not None
        assert selection.after is not None
        assert stack is not None
        descriptor = TemporalStackDescriptor(
            before=TemporalStackMember(
                item_id=selection.before.item_id,
                acquired=selection.before.acquired,
                platform=selection.before.platform,
                instrument=selection.before.instrument,
                red_asset_id=selection.before.red_asset_id,
                scl_asset_id=selection.before.scl_asset_id,
                bands=["red", "scl"],
                coverage_fraction=selection.before.coverage_fraction,
                cloud_fraction=stack.before_cloud_fraction,
            ),
            after=TemporalStackMember(
                item_id=selection.after.item_id,
                acquired=selection.after.acquired,
                platform=selection.after.platform,
                instrument=selection.after.instrument,
                red_asset_id=selection.after.red_asset_id,
                scl_asset_id=selection.after.scl_asset_id,
                bands=["red", "scl"],
                coverage_fraction=selection.after.coverage_fraction,
                cloud_fraction=stack.after_cloud_fraction,
            ),
            grid_crs=stack.crs,
            grid_transform=stack.transform,
            width=stack.width,
            height=stack.height,
            band_order=stack.band_order,
            cloud_policy=stack.cloud_policy,
            alignment_method=stack.alignment_method,
        )
        return self.artifacts.put_bytes(
            b"deterministic-temporal-stack",
            kind="raster",
            media_type="image/tiff",
            spatial=SpatialExtent(
                crs="EPSG:4326",
                bbox=BBOX,
                gsd_meters=10.0,
                shape=[4, 4, 4],
            ),
            temporal=TemporalExtent(start=BEFORE_TIME, end=AFTER_TIME),
            temporal_stack=descriptor,
            lineage=ArtifactLineage(
                tool_id=TOOL_ID,
                tool_version=VERSION,
                input_refs=INPUT_IDS,
                parameters_hash=hashlib.sha256(b"temporal-eval").hexdigest(),
            ),
        )

    def _evaluate(self, manifest, state, artifacts, reason):
        return self.evaluator.evaluate_safely(
            manifest,
            state,
            artifacts,
            renderer_calls=0,
            failed_actions=0,
            wall_time_ms=1000,
            tool_results=[_tool_result(reason)],
        )

    def test_selected_stack_and_full_extent_evidence_score_one(self):
        manifest = _manifest("selected")
        state = self._state(manifest)
        result = _selection("selected")
        artifact = self._artifact(result)
        evidence = EvidenceRef(
            evidence_id="ev-temporal-stack",
            claim_id="temporal-selection",
            source_ref=artifact.artifact_id,
            selector=EvidenceSelector(bbox=BBOX, time_range=artifact.temporal),
            description="Full aligned two-date stack.",
            frozen_sha256=artifact.sha256,
        )
        state.evidence_refs = [evidence]
        state.final_answer = AnswerRecord(
            outcome="submitted",
            answer={"label": "valid_pair", "claims": []},
            confidence=1.0,
            evidence_ids=[evidence.evidence_id],
        )
        state.step_count = 3
        evaluation = self._evaluate(
            manifest,
            state,
            {artifact.artifact_id: artifact},
            "selected",
        )
        self.assertEqual(evaluation.status, "completed", evaluation.model_dump())
        self.assertEqual(evaluation.aggregate_reward, 1.0)
        self.assertEqual(
            {metric.name: metric.value for metric in evaluation.metrics},
            {name: 1.0 for name in manifest.evaluator.metric_names},
        )
        self.assertFalse(evaluation.diagnostics["false_confidence"])

    def test_rejections_score_abstention_and_false_confidence_separately(self):
        for reason in ("cloudy", "insufficient_coverage", "wrong_date"):
            with self.subTest(reason=reason):
                manifest = _manifest(reason)
                state = self._state(manifest)
                state.final_answer = AnswerRecord(
                    outcome="abstained",
                    evidence_ids=[],
                    rationale="Temporal selection rejected: %s." % reason,
                )
                state.step_count = 2
                evaluation = self._evaluate(manifest, state, {}, reason)
                self.assertEqual(evaluation.aggregate_reward, 1.0)
                self.assertFalse(evaluation.diagnostics["false_confidence"])

        manifest = _manifest("cloudy")
        state = self._state(manifest)
        state.final_answer = AnswerRecord(
            outcome="submitted",
            answer={"label": "valid_pair", "claims": []},
            confidence=1.0,
            evidence_ids=[],
        )
        state.step_count = 2
        evaluation = self._evaluate(manifest, state, {}, "cloudy")
        values = {metric.name: metric.value for metric in evaluation.metrics}
        self.assertEqual(values["temporal.validity"], 1.0)
        self.assertEqual(values["spatial.coverage"], 1.0)
        self.assertEqual(values["answer.abstention_correctness"], 0.0)
        self.assertEqual(values["evidence.faithfulness"], 0.0)
        self.assertTrue(evaluation.diagnostics["false_confidence"])

    def test_temporal_tool_contract_tamper_fails_closed(self):
        manifest = _manifest("cloudy")
        state = self._state(manifest)
        state.final_answer = AnswerRecord(
            outcome="abstained",
            evidence_ids=[],
            rationale="Cloud rejection.",
        )
        value = _tool_result("cloudy")
        value["tool_version"] = "9.9.9"
        evaluation = self.evaluator.evaluate_safely(
            manifest,
            state,
            {},
            0,
            0,
            1,
            tool_results=[value],
        )
        self.assertEqual(evaluation.status, "failed")
        self.assertEqual(
            evaluation.diagnostics["code"], "temporal_tool_version_mismatch"
        )

    def test_rejection_refuses_untyped_temporal_output(self):
        manifest = _manifest("cloudy")
        state = self._state(manifest)
        state.final_answer = AnswerRecord(
            outcome="abstained",
            evidence_ids=[],
            rationale="Cloud rejection.",
        )
        state.step_count = 2
        unexpected = self.artifacts.put_bytes(
            b"untyped-temporal-output",
            kind="raster",
            media_type="image/tiff",
            lineage=ArtifactLineage(
                tool_id=TOOL_ID,
                tool_version=VERSION,
                input_refs=INPUT_IDS,
                parameters_hash=hashlib.sha256(b"unexpected").hexdigest(),
            ),
        )
        evaluation = self._evaluate(
            manifest,
            state,
            {unexpected.artifact_id: unexpected},
            "cloudy",
        )
        values = {metric.name: metric.value for metric in evaluation.metrics}
        self.assertEqual(values["temporal.validity"], 0.0)
        self.assertEqual(values["evidence.faithfulness"], 0.0)

    def test_metadata_rejection_is_evaluated_on_answer_abstain(self):
        manifest = _manifest("cloudy")
        store = V2EpisodeStore(
            str(self.root / "episodes.sqlite3"),
            _Registry(manifest),
            artifact_store=self.artifacts,
            evaluator_registry=self.evaluator,
            tool_executor=ToolRouter(
                temporal=_RejectedTemporalExecutor("cloudy")
            ),
        )
        episode = store.create_episode(manifest.task.task_id, "1.0.0", 42)
        selected = store.step(
            episode.episode_id,
            0,
            "temporal-reject",
            ToolInvokeAction(type="tool.invoke", tool_id=TOOL_ID, arguments={}),
        )
        self.assertIsNone(selected.state.evaluation)
        final = store.step(
            episode.episode_id,
            1,
            "abstain",
            AnswerAbstainAction(
                type="answer.abstain",
                rationale="Cloud policy rejected the pair.",
                evidence_ids=[],
            ),
        )
        self.assertIsNotNone(final.state.evaluation)
        self.assertEqual(final.state.evaluation.status, "completed")
        self.assertEqual(final.state.evaluation.aggregate_reward, 1.0)
        self.assertFalse(final.state.evaluation.diagnostics["false_confidence"])


if __name__ == "__main__":
    unittest.main()

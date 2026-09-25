import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy
import rasterio
from rasterio.io import MemoryFile

from app.core.artifacts import ArtifactStore
from app.core.domain import create_initial_state
from app.core.evaluation import EvaluatorRegistry
from app.core.execution_replay import read_snapshot, replay_episode
ROOT = Path(__file__).resolve().parents[2]

from app.core.schemas import (
    AnswerRecord,
    AnswerSubmitAction,
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
    TemporalExtent,
)
from app.core.store import V2EpisodeStore


WIDTH = 10
HEIGHT = 10
BEFORE_ID = "asset-whu-before"
AFTER_ID = "asset-whu-after"
BEFORE_LABEL_ID = "asset-whu-before-label"
AFTER_LABEL_ID = "asset-whu-after-label"


def _write_label(path: Path, values: numpy.ndarray) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=WIDTH,
        height=HEIGHT,
        count=1,
        dtype="uint8",
    ) as dataset:
        dataset.write(values.astype(numpy.uint8), 1)
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest(), len(content)


def _png(value: int) -> bytes:
    values = numpy.full((3, HEIGHT, WIDTH), value, dtype=numpy.uint8)
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


def _asset(
    asset_id: str,
    uri: str,
    digest: str,
    size: int,
    roles: list[str],
    timestamp: str,
    channels: int,
) -> PixelAssetRef:
    return PixelAssetRef(
        asset_id=asset_id,
        uri=uri,
        media_type="image/tiff",
        roles=roles,
        sha256=digest,
        size_bytes=size,
        spatial=None,
        pixel=PixelExtent(
            coordinate_system="pixel",
            width=WIDTH,
            height=HEIGHT,
            channels=channels,
        ),
        temporal=TemporalExtent(start=timestamp, end=timestamp),
        platform="aerial",
        instrument="optical",
        bands=["building_mask"] if channels == 1 else ["red", "green", "blue"],
        license="research-use-no-redistribution",
        source="WHU Building Change Detection official research copy",
    )


def _truth(before: numpy.ndarray, after: numpy.ndarray) -> dict:
    before_mask = before == 255
    after_mask = after == 255
    new_pixels = int(numpy.count_nonzero(~before_mask & after_mask))
    demolished_pixels = int(numpy.count_nonzero(before_mask & ~after_mask))
    changed_pixels = new_pixels + demolished_pixels
    fraction = changed_pixels / before.size
    if fraction == 0.0:
        change_class, direction = "no_change", "no_change"
    else:
        change_class = "minor_change" if fraction <= 0.1 else "major_change"
        if new_pixels > demolished_pixels * 1.5:
            direction = "expansion"
        elif demolished_pixels > new_pixels * 1.5:
            direction = "reduction"
        else:
            direction = "mixed"
    return {
        "changed_pixels": changed_pixels,
        "new_pixels": new_pixels,
        "demolished_pixels": demolished_pixels,
        "changed_fraction": fraction,
        "change_class": change_class,
        "change_direction": direction,
    }


def _manifest(
    dataset_root: Path,
    before: numpy.ndarray,
    after: numpy.ndarray,
    *,
    task_id: str = "whu-change-test",
    evaluation_profile: str | None = "whu-building-change-v1",
    evidence_required: bool = True,
    expected_outcome: str | None = None,
    minimum_coverage: float = 0.0,
) -> TaskManifest:
    before_hash, before_size = _write_label(
        dataset_root / "labels" / "before.tif", before
    )
    after_hash, after_size = _write_label(
        dataset_root / "labels" / "after.tif", after
    )
    assets = [
        _asset(
            BEFORE_ID,
            "local://approved-input/before.tif",
            hashlib.sha256(b"before-image").hexdigest(),
            1000,
            ["input_image", "temporal_before"],
            "2012-01-01T00:00:00Z",
            3,
        ),
        _asset(
            AFTER_ID,
            "local://approved-input/after.tif",
            hashlib.sha256(b"after-image").hexdigest(),
            1000,
            ["input_image", "temporal_after"],
            "2016-01-01T00:00:00Z",
            3,
        ),
        _asset(
            BEFORE_LABEL_ID,
            "local://dataset/labels/before.tif",
            before_hash,
            before_size,
            ["evaluator", "building_label", "temporal_before"],
            "2012-01-01T00:00:00Z",
            1,
        ),
        _asset(
            AFTER_LABEL_ID,
            "local://dataset/labels/after.tif",
            after_hash,
            after_size,
            ["evaluator", "building_label", "temporal_after"],
            "2016-01-01T00:00:00Z",
            1,
        ),
    ]
    metadata = {
        "observation_profile": "headless-tools-v1",
        "artifact_identity": "derivation-sha256-v1",
    }
    if evaluation_profile is not None:
        metadata["evaluation_profile"] = evaluation_profile
    if expected_outcome is not None:
        metadata["expected_outcome"] = expected_outcome
    truth = _truth(before, after)
    weights = {
        "task.change_class_accuracy": 0.3,
        "task.direction_accuracy": 0.2,
        "task.changed_fraction_score": 0.2,
        "evidence.faithfulness": 0.2,
        "process.efficiency": 0.1,
    }
    task = TaskSpec(
        task_id=task_id,
        task_version="1.0.0",
        family="temporal_change",
        prompt="Classify building change between the before and after images.",
        inputs=[BEFORE_ID, AFTER_ID],
        scenario_profile="whu-change-test-v1",
        answer_schema={
            "type": "object",
            "properties": {
                "change_class": {
                    "type": "string",
                    "enum": ["no_change", "minor_change", "major_change"],
                },
                "change_direction": {
                    "type": "string",
                    "enum": ["no_change", "expansion", "reduction", "mixed"],
                },
                "changed_fraction": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "claims": {"type": "array"},
            },
            "required": [
                "change_class",
                "change_direction",
                "changed_fraction",
                "claims",
            ],
        },
        evaluator="whu-building-change-v1",
        budget=BudgetSpec(
            max_steps=20,
            max_tool_calls=10,
            max_wall_time_ms=120000,
            max_input_bytes=10000000,
            max_artifact_bytes=10000000,
        ),
        seed=42,
        metric_aggregation=weights,
        metadata=metadata,
    )
    scenario = ScenarioProfile(
        profile_id="whu-change-test-v1",
        domain="building_change_detection",
        data_cutoff="2026-09-20T00:00:00Z",
        allowed_actions=["tool.invoke", "memory.save_evidence", "answer.*"],
        allowed_tools=["eo_gym.crop"],
        network_policy="none",
        evidence_required=evidence_required,
        abstention_allowed=True,
        human_review_policy="allowed",
    )
    evaluator = EvaluatorSpec(
        evaluator_id="whu-building-change-v1",
        evaluator_version="1.0.0",
        metric_names=list(weights),
        aggregate_weights=weights,
        config={
            "before_input_asset_id": BEFORE_ID,
            "after_input_asset_id": AFTER_ID,
            "before_label_asset_id": BEFORE_LABEL_ID,
            "after_label_asset_id": AFTER_LABEL_ID,
            "expected_width": WIDTH,
            "expected_height": HEIGHT,
            "expected_pixel_count": WIDTH * HEIGHT,
            "expected_changed_pixels": truth["changed_pixels"],
            "expected_new_pixels": truth["new_pixels"],
            "expected_demolished_pixels": truth["demolished_pixels"],
            "minor_change_max_fraction": 0.1,
            "direction_dominance_ratio": 1.5,
            "changed_fraction_tolerance": 0.05,
            "minimum_input_coverage_fraction": minimum_coverage,
            "required_evidence_tool_id": "eo_gym.crop",
            "efficiency": {
                "ideal_steps": 5,
                "ideal_renderer_calls": 2,
                "wall_time_soft_limit_ms": 10000,
            },
        },
    )
    return TaskManifest(
        task=task,
        scenario=scenario,
        assets=assets,
        evaluator=evaluator,
        task_manifest_hash="0" * 64,
    )


class _Registry:
    def __init__(self, *manifests: TaskManifest):
        self.manifests = {
            (manifest.task.task_id, manifest.task.task_version): manifest
            for manifest in manifests
        }

    def get(self, task_id: str, task_version: str) -> TaskManifest:
        return self.manifests[(task_id, task_version)]


class WHUChangeEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.datasets = self.root / "datasets"
        self.artifacts = ArtifactStore(str(self.root / "artifacts"))

    def tearDown(self):
        self.tempdir.cleanup()

    def _evaluate(
        self,
        before: numpy.ndarray,
        after: numpy.ndarray,
        include_after: bool = True,
    ):
        manifest = _manifest(self.datasets, before, after)
        state, _ = create_initial_state(
            "ep2-00000000000000000000000000000001",
            manifest,
            42,
            timestamp="2026-09-20T00:00:00Z",
        )
        artifact_refs = {}
        evidence_refs = []
        for index, asset_id in enumerate((BEFORE_ID, AFTER_ID)):
            artifact = self.artifacts.put_bytes(
                _png(index * 100),
                kind="image",
                media_type="image/png",
                lineage=ArtifactLineage(
                    tool_id="eo_gym.crop",
                    tool_version="1.1.0",
                    input_refs=[asset_id],
                    parameters_hash=hashlib.sha256(asset_id.encode()).hexdigest(),
                ),
                pixel=PixelExtent(
                    coordinate_system="pixel",
                    width=WIDTH,
                    height=HEIGHT,
                    channels=3,
                ),
            )
            artifact_refs[artifact.artifact_id] = artifact
            if asset_id == AFTER_ID and not include_after:
                continue
            evidence_refs.append(
                EvidenceRef(
                    evidence_id="ev-" + asset_id,
                    claim_id="claim-change",
                    source_ref=artifact.artifact_id,
                    selector=EvidenceSelector(
                        pixel_window=[0, 0, WIDTH, HEIGHT]
                    ),
                    description="Full temporal crop.",
                    frozen_sha256=artifact.sha256,
                )
            )
        truth = _truth(before, after)
        state.evidence_refs = evidence_refs
        state.final_answer = AnswerRecord(
            outcome="submitted",
            answer={
                "change_class": truth["change_class"],
                "change_direction": truth["change_direction"],
                "changed_fraction": truth["changed_fraction"],
                "claims": [{"claim_id": "claim-change"}],
            },
            confidence=1.0,
            evidence_ids=[item.evidence_id for item in evidence_refs],
        )
        state.step_count = 5
        state.status = "terminated"
        result = EvaluatorRegistry(
            str(self.datasets), self.artifacts
        ).evaluate_safely(
            manifest,
            state,
            artifact_refs,
            renderer_calls=2,
            failed_actions=0,
            wall_time_ms=1000,
        )
        return result

    def test_outcome_semantics_and_abstention_metric(self):
        before = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.uint8)
        after = before.copy()
        after.flat[:4] = 255
        cases = [
            ("submitted", "submitted", 1.0, False, False),
            ("submitted", "abstained", 0.0, False, True),
            ("abstained", "abstained", 1.0, False, False),
            ("abstained", "submitted", 0.0, True, False),
        ]
        for expected_outcome, actual_outcome, score, false_confident, unnecessary in cases:
            with self.subTest(expected=expected_outcome, actual=actual_outcome):
                manifest = _manifest(self.datasets, before, after)
                manifest.evaluator.config["expected_outcome"] = expected_outcome
                if expected_outcome == "abstained":
                    manifest.evaluator.config["minimum_input_coverage_fraction"] = 0.8
                    next(
                        asset for asset in manifest.assets if asset.asset_id == AFTER_ID
                    ).quality.coverage_fraction = 0.25
                if "answer.abstention_correctness" not in manifest.evaluator.metric_names:
                    manifest.evaluator.metric_names.insert(
                        3, "answer.abstention_correctness"
                    )
                manifest.evaluator.aggregate_weights[
                    "answer.abstention_correctness"
                ] = 0.1
                state, _ = create_initial_state(
                    "ep2-" + "1" * 32, manifest, 42, timestamp="2026-09-20T00:00:00Z"
                )
                if actual_outcome == "submitted":
                    truth = _truth(before, after)
                    state.final_answer = AnswerRecord(
                        outcome="submitted",
                        answer={
                            "change_class": truth["change_class"],
                            "change_direction": truth["change_direction"],
                            "changed_fraction": truth["changed_fraction"],
                            "claims": [],
                        },
                        confidence=1.0,
                        evidence_ids=[],
                    )
                else:
                    state.final_answer = AnswerRecord(
                        outcome="abstained",
                        answer={},
                        confidence=0.0,
                        evidence_ids=[],
                    )
                result = EvaluatorRegistry(
                    str(self.datasets), self.artifacts
                ).evaluate_safely(manifest, state, {}, 0, 0, 1)
                metric = next(
                    item for item in result.metrics
                    if item.name == "answer.abstention_correctness"
                )
                self.assertEqual(metric.value, score)
                self.assertEqual(
                    metric.diagnostics["expected_outcome"], expected_outcome
                )
                self.assertEqual(
                    metric.diagnostics["actual_outcome"], actual_outcome
                )
                self.assertEqual(
                    result.diagnostics["false_confidence"], false_confident
                )
                self.assertEqual(
                    result.diagnostics["unnecessary_abstention"], unnecessary
                )

    def test_public_quality_metadata_controls_answerability(self):
        before = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.uint8)
        after = before.copy()
        manifest = _manifest(self.datasets, before, after)
        manifest.evaluator.config["expected_outcome"] = "abstained"
        manifest.evaluator.config["minimum_input_coverage_fraction"] = 0.8
        next(
            asset for asset in manifest.assets if asset.asset_id == AFTER_ID
        ).quality.coverage_fraction = 0.25
        manifest.evaluator.metric_names.insert(
            3, "answer.abstention_correctness"
        )
        manifest.evaluator.aggregate_weights[
            "answer.abstention_correctness"
        ] = 0.1
        state, _ = create_initial_state(
            "ep2-" + "2" * 32, manifest, 42, timestamp="2026-09-20T00:00:00Z"
        )
        state.final_answer = AnswerRecord(
            outcome="abstained", answer={}, confidence=0.0, evidence_ids=[]
        )
        result = EvaluatorRegistry(
            str(self.datasets), self.artifacts
        ).evaluate_safely(manifest, state, {}, 0, 0, 1)
        metric = next(
            item for item in result.metrics
            if item.name == "answer.abstention_correctness"
        )
        self.assertEqual(metric.value, 1.0)
        self.assertEqual(
            metric.diagnostics["insufficient_input_asset_ids"], [AFTER_ID]
        )
        self.assertEqual(
            result.diagnostics["insufficient_input_asset_ids"], [AFTER_ID]
        )

    def test_answerability_truth_must_match_public_metadata(self):
        before = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.uint8)
        after = before.copy()
        cases = [
            ("abstained", None, "expected_abstention_unsupported"),
            ("submitted", 0.5, "expected_submission_unsupported"),
        ]
        for outcome, coverage, code in cases:
            with self.subTest(code=code):
                manifest = _manifest(self.datasets, before, after)
                manifest.evaluator.config["expected_outcome"] = outcome
                manifest.evaluator.config["minimum_input_coverage_fraction"] = 0.8
                if coverage is not None:
                    next(
                        asset for asset in manifest.assets if asset.asset_id == AFTER_ID
                    ).quality.coverage_fraction = coverage
                state, _ = create_initial_state(
                    "ep2-" + "3" * 32, manifest, 42, timestamp="2026-09-20T00:00:00Z"
                )
                result = EvaluatorRegistry(
                    str(self.datasets), self.artifacts
                ).evaluate_safely(manifest, state, {}, 0, 0, 1)
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.diagnostics["code"], code)

    @staticmethod
    def _run_prepare(config: dict, source: Path):
        with tempfile.TemporaryDirectory() as temp:
            script = ROOT / "scripts" / "prepare_whu_change_smoke.py"
            env = os.environ.copy()
            env["EO_WHU_TEST_CONFIG"] = str(Path(temp) / "config.json")
            env["EO_WHU_TEST_ROOT"] = str(Path(temp) / "runtime")
            (Path(temp) / "config.json").write_text(json.dumps(config))
            return subprocess.run(
                [sys.executable, str(script), "--source", str(source), "--output-name", "invalid-answerability"],
                env=env, capture_output=True, text=True, check=False,
            )

    def test_task_config_quality_must_justify_answerability(self):
        config = json.loads((ROOT / "config" / "whu-change-samples.json").read_text())
        sample = config["samples"][0]
        sample["expected_outcome"] = "abstained"
        source = Path(tempfile.mkdtemp(dir=self.root)) / "reviewed-source"
        source.mkdir()
        result = self._run_prepare(config, source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected abstention lacks insufficient public input coverage", result.stderr)

    def test_truth_classes_and_directions(self):
        no_change = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.uint8)
        minor_after = no_change.copy()
        minor_after.flat[:4] = 255
        major_after = numpy.full((HEIGHT, WIDTH), 255, dtype=numpy.uint8)
        cases = [
            (no_change, no_change.copy(), "no_change", "no_change"),
            (no_change, minor_after, "minor_change", "expansion"),
            (no_change, major_after, "major_change", "expansion"),
        ]
        for before, after, expected_class, expected_direction in cases:
            with self.subTest(expected_class=expected_class):
                result = self._evaluate(before, after)
                self.assertEqual(
                    result.status, "completed", result.model_dump()
                )
                values = {
                    metric.name: metric.value for metric in result.metrics
                }
                self.assertEqual(
                    values["task.change_class_accuracy"], 1.0
                )
                self.assertEqual(values["task.direction_accuracy"], 1.0)
                self.assertEqual(
                    values["task.changed_fraction_score"], 1.0
                )
                self.assertEqual(values["evidence.faithfulness"], 1.0)
                self.assertEqual(
                    result.diagnostics["truth"]["change_class"],
                    expected_class,
                )
                self.assertEqual(
                    result.diagnostics["truth"]["change_direction"],
                    expected_direction,
                )

    def test_labels_fail_closed(self):
        before = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.uint8)
        after = before.copy()
        roots = {
            "gold_asset_size_mismatch": self.datasets,
            "gold_asset_checksum_mismatch": self.root / "hash-datasets",
            "gold_label_value_domain_mismatch": self.root / "domain-datasets",
            "temporal_order_mismatch": self.root / "order-datasets",
        }
        manifests = {}
        manifests["gold_asset_size_mismatch"] = _manifest(
            roots["gold_asset_size_mismatch"], before, after
        )
        next(
            asset
            for asset in manifests["gold_asset_size_mismatch"].assets
            if asset.asset_id == BEFORE_LABEL_ID
        ).size_bytes += 1
        manifests["gold_asset_checksum_mismatch"] = _manifest(
            roots["gold_asset_checksum_mismatch"], before, after
        )
        next(
            asset
            for asset in manifests["gold_asset_checksum_mismatch"].assets
            if asset.asset_id == BEFORE_LABEL_ID
        ).sha256 = "f" * 64
        invalid = before.copy()
        invalid[0, 0] = 127
        manifests["gold_label_value_domain_mismatch"] = _manifest(
            roots["gold_label_value_domain_mismatch"], invalid, after
        )
        manifests["temporal_order_mismatch"] = _manifest(
            roots["temporal_order_mismatch"], before, after
        )
        order_manifest = manifests["temporal_order_mismatch"]
        before_input = next(
            asset for asset in order_manifest.assets if asset.asset_id == BEFORE_ID
        )
        after_input = next(
            asset for asset in order_manifest.assets if asset.asset_id == AFTER_ID
        )
        before_input.temporal, after_input.temporal = (
            after_input.temporal,
            before_input.temporal,
        )
        for code, manifest in manifests.items():
            with self.subTest(code=code):
                state, _ = create_initial_state(
                    "ep2-00000000000000000000000000000002",
                    manifest,
                    42,
                    timestamp="2026-09-20T00:00:00Z",
                )
                state.final_answer = AnswerRecord(
                    outcome="submitted",
                    answer={
                        "change_class": "no_change",
                        "change_direction": "no_change",
                        "changed_fraction": 0.0,
                        "claims": [],
                    },
                    evidence_ids=[],
                )
                result = EvaluatorRegistry(
                    str(roots[code]), self.artifacts
                ).evaluate_safely(manifest, state, {}, 0, 0, 1)
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.diagnostics["code"], code)

    def test_faithfulness_requires_both_timepoints(self):
        before = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.uint8)
        after = before.copy()
        after.flat[:4] = 255
        result = self._evaluate(before, after, include_after=False)
        metric = next(
            item
            for item in result.metrics
            if item.name == "evidence.faithfulness"
        )
        self.assertEqual(metric.value, 0.0)
        self.assertEqual(
            metric.diagnostics["covered_input_asset_ids"], [BEFORE_ID]
        )
        self.assertEqual(
            metric.diagnostics["missing_input_asset_ids"], [AFTER_ID]
        )


class WHUChangeStoreAndReplayTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.datasets = self.root / "datasets"
        mask = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.uint8)
        self.whu = _manifest(
            self.datasets, mask, mask, evidence_required=False
        )
        self.placeholder = _manifest(
            self.root / "placeholder-datasets",
            mask,
            mask,
            task_id="headless-placeholder",
            evaluation_profile=None,
            evidence_required=False,
        )
        self.registry = _Registry(self.whu, self.placeholder)
        self.artifacts = ArtifactStore(str(self.root / "artifacts"))
        self.evaluator = EvaluatorRegistry(
            str(self.datasets), self.artifacts
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _submit(self, store: V2EpisodeStore, task_id: str):
        episode = store.create_episode(task_id, "1.0.0", 42)
        return store.step(
            episode.episode_id,
            0,
            "submit-change",
            AnswerSubmitAction.model_validate({
                "type": "answer.submit",
                "answer": {
                    "change_class": "no_change",
                    "change_direction": "no_change",
                    "changed_fraction": 0.0,
                    "claims": [],
                },
                "confidence": 1.0,
                "evidence_ids": [],
            }),
        )

    def test_explicit_profile_triggers_evaluator_only_for_whu(self):
        store = V2EpisodeStore(
            str(self.root / "episodes.sqlite3"),
            self.registry,
            artifact_store=self.artifacts,
            evaluator_registry=self.evaluator,
        )
        whu_result = self._submit(store, self.whu.task.task_id)
        self.assertIsNotNone(whu_result.state.evaluation)
        self.assertEqual(whu_result.state.evaluation.status, "completed")
        placeholder_result = self._submit(
            store, self.placeholder.task.task_id
        )
        self.assertIsNone(placeholder_result.state.evaluation)

    def test_execution_replay_recomputes_whu_evaluation(self):
        database = self.root / "replay-source.sqlite3"
        store = V2EpisodeStore(
            str(database),
            self.registry,
            artifact_store=self.artifacts,
            evaluator_registry=self.evaluator,
        )
        result = self._submit(store, self.whu.task.task_id)
        snapshot = read_snapshot(database, result.episode_id)
        report = replay_episode(
            snapshot,
            self.registry,
            self.root / "replay-work",
            evaluator_factory=lambda artifacts: EvaluatorRegistry(
                str(self.datasets), artifacts
            ),
        )
        self.assertEqual(report["status"], "passed", report)
        self.assertTrue(report["evaluation_matched"])
        self.assertTrue(report["semantic_evaluator_replayed"])


if __name__ == "__main__":
    unittest.main()

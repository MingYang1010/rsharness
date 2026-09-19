import copy
import gc
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v2.helpers import TASKS_ROOT
from v2.test_m2_runtime import AOI, FakeEvaluatorRegistry, FakeRenderer
from v2.test_tool_execution import FakeExecutor, make_tool_tasks
from app.v2.artifacts import ArtifactStore
from app.v2.capabilities import TaskRegistry
from app.v2.domain import V2DomainError
from app.v2.execution_replay import ReplayError, read_snapshot, replay_episode, semanticize
from app.v2.schemas import StepRequest
from app.v2.store import V2EpisodeStore
from app.v2.tools.runtime import ToolRouter, ToolOutput


class ExecutionReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tasks = make_tool_tasks(self.root)
        path = self.tasks / "crop-smoke" / "scenario.json"
        scenario = json.loads(path.read_text())
        scenario["allowed_tools"] += ["catalog.search", "catalog.inspect_asset"]
        path.write_text(json.dumps(scenario))
        self.registry = TaskRegistry(self.tasks)
        self.artifacts = ArtifactStore(str(self.root / "original-artifacts"))
        self.original_executor = FakeExecutor(self.artifacts)
        self.store = V2EpisodeStore(str(self.root / "original.db"), self.registry,
            artifact_store=self.artifacts, tool_executor=ToolRouter(self.original_executor))
        self.initial = self.store.create_episode("crop-smoke", "1.0.0", 42)
        self.version = 0

    def step(self, action, identity):
        body = StepRequest.model_validate({"client_action_id": identity, "expected_state_version": self.version, "action": action})
        result = self.store.step(self.initial.episode_id, self.version, identity, body.action)
        self.version = result.state.state_version
        return result

    def record(self, invalid_evidence=False):
        self.step({"type": "tool.invoke", "tool_id": "catalog.search", "arguments": {}}, "search")
        self.step({"type": "tool.invoke", "tool_id": "eo_gym.crop", "arguments": {
            "asset_id": "asset-worldcover-n30e120", "aoi": [0, 0, .5, .5]}}, "crop")
        if invalid_evidence:
            with self.assertRaises(V2DomainError):
                self.step({"type": "memory.save_evidence", "evidence": {"evidence_id": "ev-missing",
                    "claim_id": "claim-test", "source_ref": "asset-missing", "selector": {"bands": ["visual"]},
                    "description": "invalid source test", "frozen_sha256": "f" * 64}}, "invalid-evidence")
        self.step({"type": "answer.abstain", "rationale": "test interaction only", "evidence_ids": []}, "abstain")
        return read_snapshot(self.root / "original.db", self.initial.episode_id)

    def replay(self, snapshot, factory=None):
        self.replay_executor = None
        def default(artifacts):
            self.replay_executor = FakeExecutor(artifacts)
            return ToolRouter(self.replay_executor)
        return replay_episode(snapshot, self.registry, self.root / "replay", factory or default)

    def record_rendered(self):
        registry = TaskRegistry(TASKS_ROOT)
        artifacts = ArtifactStore(str(self.root / "rendered-original-artifacts"))
        store = V2EpisodeStore(
            str(self.root / "rendered-original.db"),
            registry,
            artifact_store=artifacts,
            renderer=FakeRenderer(artifacts),
            evaluator_registry=FakeEvaluatorRegistry(),
            renderer_config={"capture_action_prefixes": ["map."]},
        )
        initial = store.create_episode("worldcover-grounded-vqa", "1.1.0", 42)
        version = 0

        def step(action, identity):
            nonlocal version
            body = StepRequest.model_validate(
                {
                    "client_action_id": identity,
                    "expected_state_version": version,
                    "action": action,
                }
            )
            result = store.step(
                initial.episode_id,
                version,
                identity,
                body.action,
            )
            version = result.state.state_version
            return result

        rendered = step(
            {"type": "map.set_view", "bbox": AOI.model_dump(mode="json")},
            "render",
        )
        artifact_id = next(
            item.artifact_ref
            for item in rendered.observation.items
            if item.artifact_ref is not None
        )
        artifact = store.get_artifact(artifact_id, initial.episode_id).artifact
        manifest = registry.get("worldcover-grounded-vqa", "1.1.0")
        source = next(
            asset
            for asset in manifest.assets
            if asset.asset_id == "asset-worldcover-n30e120"
        )
        step(
            {
                "type": "memory.save_evidence",
                "evidence": {
                    "evidence_id": "ev-replay-source",
                    "claim_id": "claim-dominant",
                    "source_ref": source.asset_id,
                    "selector": {"bbox": AOI.model_dump(mode="json")},
                    "description": "Source evidence for rendered replay.",
                    "frozen_sha256": source.sha256,
                },
            },
            "source-evidence",
        )
        step(
            {
                "type": "memory.save_evidence",
                "evidence": {
                    "evidence_id": "ev-replay-rendered",
                    "claim_id": "claim-dominant",
                    "source_ref": artifact_id,
                    "selector": {"bbox": AOI.model_dump(mode="json")},
                    "description": "Rendered evidence for execution replay.",
                    "frozen_sha256": artifact.sha256,
                },
            },
            "rendered-evidence",
        )
        step(
            {
                "type": "answer.submit",
                "answer": {
                    "label": "built-up",
                    "confidence": 0.95,
                    "claims": [
                        {
                            "claim_id": "claim-dominant",
                            "text": "Built-up dominates.",
                        }
                    ],
                },
                "confidence": 0.95,
                "evidence_ids": ["ev-replay-source", "ev-replay-rendered"],
            },
            "answer",
        )
        return (
            read_snapshot(self.root / "rendered-original.db", initial.episode_id),
            registry,
        )

    def test_all_recorded_actions_reexecute_in_new_store_without_original_changes(self):
        snapshot = self.record()
        # Fixture writers use transaction context managers, not explicit close().
        # Settle their delayed WAL checkpoint before measuring physical bytes;
        # replay must still preserve both logical contents and this file hash.
        gc.collect()
        before = hashlib.sha256((self.root / "original.db").read_bytes()).hexdigest()
        # No old output content is available at its original location during replay.
        self.artifacts.root.rename(self.root / "old-artifacts-preserved")
        result = self.replay(snapshot)
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["executed_actions"], 3)
        self.assertEqual(self.replay_executor.calls, 1)
        self.assertEqual(self.original_executor.calls, 1)
        self.assertTrue(result["artifact_checks"][0]["content_verified"])
        self.assertEqual(snapshot.fingerprint, read_snapshot(self.root / "original.db", self.initial.episode_id).fingerprint)
        self.assertEqual(before, hashlib.sha256((self.root / "original.db").read_bytes()).hexdigest())

    def test_deterministic_recorded_non_tool_rejection_is_replayed(self):
        result = self.replay(self.record(invalid_evidence=True))
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["executed_actions"], 4)
        self.assertEqual(result["actions"][2]["actual_outcome"], "error")

    def test_unsupported_tool_version_does_not_execute(self):
        snapshot = self.record()
        run = json.loads(snapshot.tool_runs[0]["run_json"])
        run["tool_version"] = "99.0.0"
        snapshot.tool_runs[0]["run_json"] = json.dumps(run)
        result = self.replay(snapshot)
        self.assertEqual(result["reason"], "tool_version_not_supported")
        self.assertEqual(result["executed_actions"], 0)
        self.assertIsNone(self.replay_executor)

    def test_changed_manifest_is_not_silently_replayed(self):
        snapshot = self.record()
        self.registry.get("crop-smoke", "1.0.0").task_manifest_hash = "f" * 64
        self.assertEqual(self.replay(snapshot)["reason"], "task_manifest_mismatch")

    def test_changed_request_disagrees_with_accepted_event(self):
        snapshot = self.record()
        for row in snapshot.results:
            if row["client_action_id"] == "crop":
                request = json.loads(row["request_json"])
                request["action"]["arguments"]["aoi"] = [0, 0, 1, 1]
                row["request_json"] = json.dumps(request)
        self.assertEqual(self.replay(snapshot)["reason"], "request_trace_mismatch")

    def test_missing_action_result_is_not_partial_success(self):
        snapshot = self.record()
        snapshot.results.pop()
        result = self.replay(snapshot)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["reason"], "action_coverage_mismatch")

    def test_running_episode_and_pending_tool_refused(self):
        active = read_snapshot(self.root / "original.db", self.initial.episode_id)
        self.assertEqual(self.replay(active)["reason"], "episode_not_terminal")
        snapshot = self.record()
        snapshot.tool_runs[0]["status"] = "running"
        self.assertEqual(self.replay(snapshot)["reason"], "episode_not_terminal")

    def test_output_metadata_difference_fails_at_exact_action(self):
        snapshot = self.record()
        class Changed(FakeExecutor):
            def invoke(self, action, manifest):
                output = super().invoke(action, manifest)
                return ToolOutput(output.artifact, {"width": 2, "height": 1}, output.input_bytes)
        result = self.replay(snapshot, lambda artifacts: ToolRouter(Changed(artifacts)))
        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(result["executed_actions"], 2)
        self.assertEqual(result["actions"][-1]["client_action_id"], "crop")

    def test_recorded_artifact_provenance_must_match(self):
        snapshot = self.record()
        artifact = json.loads(snapshot.artifacts[0]["artifact_json"])
        artifact["lineage"]["parameters_hash"] = "f" * 64
        snapshot.artifacts[0]["artifact_json"] = json.dumps(artifact)
        result = self.replay(snapshot)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "final_state_or_artifact_mismatch")

    def test_snapshot_bounds_and_unknown_episode(self):
        self.record()
        with patch("app.v2.execution_replay.MAX_ACTIONS", 1), self.assertRaises(ReplayError) as error:
            read_snapshot(self.root / "original.db", self.initial.episode_id)
        self.assertEqual(error.exception.code, "snapshot_limit_exceeded")
        with self.assertRaises(ReplayError) as error:
            read_snapshot(self.root / "original.db", "ep2-" + "f" * 32)
        self.assertEqual(error.exception.code, "unknown_episode")

    def test_replay_workspace_is_never_overwritten(self):
        snapshot = self.record()
        work = self.root / "replay"
        work.mkdir()
        (work / "keep").write_text("existing")
        with self.assertRaises(ReplayError):
            self.replay(snapshot)
        self.assertEqual((work / "keep").read_text(), "existing")

    def test_semantic_comparison_preserves_evidence_identity(self):
        first = {"episode_id": "one", "evidence_ids": ["ev-one"], "created_at": "old"}
        second = {"episode_id": "two", "evidence_ids": ["ev-two"], "created_at": "new"}
        self.assertNotEqual(semanticize(first), semanticize(second))

    def test_semantic_comparison_preserves_opaque_answer_and_tool_fields(self):
        for field in ("answer", "arguments", "evidence", "inline"):
            with self.subTest(field=field):
                first = {field: {"created_at": "old", "episode_id": "one", "wall_time_ms": {"limit": 9, "used": 1}}}
                second = {field: {"created_at": "new", "episode_id": "two", "wall_time_ms": {"limit": 9, "used": 2}}}
                self.assertEqual(semanticize(first), first)
                self.assertNotEqual(semanticize(first), semanticize(second))

    def test_recorded_failed_tool_is_incomplete_not_false_success(self):
        self.original_executor.fail = True
        with self.assertRaises(V2DomainError):
            self.step({"type": "tool.invoke", "tool_id": "eo_gym.crop", "arguments": {
                "asset_id": "asset-worldcover-n30e120", "aoi": [0, 0, .5, .5]}}, "failed-crop")
        self.version = self.store.get_state(self.initial.episode_id).state.state_version
        self.step({"type": "answer.abstain", "rationale": "failed tool", "evidence_ids": []}, "abstain")
        result = self.replay(read_snapshot(self.root / "original.db", self.initial.episode_id))
        self.assertEqual(result["reason"], "historical_tool_failure_not_reproduced")
        self.assertEqual(result["executed_actions"], 0)

    def test_current_provider_failure_cannot_pass_output_comparison(self):
        snapshot = self.record()
        def unavailable(artifacts):
            executor = FakeExecutor(artifacts)
            executor.fail = True
            return ToolRouter(executor)
        result = self.replay(snapshot, unavailable)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["executed_actions"], 2)
        self.assertEqual(result["actions"][-1]["error_code"], "tool_timeout")

    def test_tampered_completion_event_is_not_hidden_by_action_cache(self):
        snapshot = self.record()
        for row in snapshot.events:
            event = json.loads(row["event_json"])
            if event["event_type"] == "action.completed" and "artifact_sha256" in event["payload"]:
                event["payload"]["artifact_sha256"] = "f" * 64
                row["event_json"] = json.dumps(event)
        result = self.replay(snapshot)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "trace_execution_mismatch")

    def test_cross_episode_initial_record_rejected_before_execution(self):
        snapshot = self.record()
        initial = json.loads(snapshot.episode["initial_state_json"])
        initial["episode_id"] = "ep2-" + "f" * 32
        snapshot.episode["initial_state_json"] = json.dumps(initial)
        self.assertEqual(self.replay(snapshot)["reason"], "episode_record_mismatch")

    def test_renderer_and_semantic_evaluator_reexecute(self):
        snapshot, registry = self.record_rendered()
        result = replay_episode(
            snapshot,
            registry,
            self.root / "rendered-replay",
            renderer_factory=lambda artifacts: FakeRenderer(artifacts),
            evaluator_factory=lambda artifacts: FakeEvaluatorRegistry(),
            renderer_config={"capture_action_prefixes": ["map."]},
        )
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["executed_actions"], 4)
        self.assertTrue(result["renderer_execution_replayed"])
        self.assertEqual(result["renderer_artifact_count"], 1)
        self.assertTrue(result["semantic_evaluator_replayed"])
        self.assertTrue(result["evaluation_matched"])

    def test_rendered_replay_requires_explicit_adapters(self):
        snapshot, registry = self.record_rendered()
        missing_renderer = replay_episode(
            snapshot,
            registry,
            self.root / "missing-renderer",
        )
        self.assertEqual(
            missing_renderer["reason"],
            "renderer_execution_not_supported",
        )
        missing_evaluator = replay_episode(
            snapshot,
            registry,
            self.root / "missing-evaluator",
            renderer_factory=lambda artifacts: FakeRenderer(artifacts),
            renderer_config={"capture_action_prefixes": ["map."]},
        )
        self.assertEqual(
            missing_evaluator["reason"],
            "semantic_evaluator_execution_not_supported",
        )

    def test_evaluator_identity_mismatch_is_rejected_before_execution(self):
        snapshot, registry = self.record_rendered()
        state = json.loads(snapshot.episode["state_json"])
        state["evaluation"]["evaluator_version"] = "9.9.9"
        snapshot.episode["state_json"] = json.dumps(state)
        result = replay_episode(
            snapshot,
            registry,
            self.root / "bad-evaluator-identity",
            renderer_factory=lambda artifacts: FakeRenderer(artifacts),
            evaluator_factory=lambda artifacts: FakeEvaluatorRegistry(),
            renderer_config={"capture_action_prefixes": ["map."]},
        )
        self.assertEqual(result["reason"], "evaluation_record_mismatch")
        self.assertEqual(result["executed_actions"], 0)

    def test_recomputed_evaluator_difference_fails_at_answer(self):
        snapshot, registry = self.record_rendered()

        class ChangedEvaluator(FakeEvaluatorRegistry):
            def evaluate_safely(self, *args, **kwargs):
                result = super().evaluate_safely(*args, **kwargs)
                result.metrics[0].value = 0.0
                result.aggregate_reward = 0.4
                return result

        result = replay_episode(
            snapshot,
            registry,
            self.root / "changed-evaluator",
            renderer_factory=lambda artifacts: FakeRenderer(artifacts),
            evaluator_factory=lambda artifacts: ChangedEvaluator(),
            renderer_config={"capture_action_prefixes": ["map."]},
        )
        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(result["reason"], "action_execution_mismatch")
        self.assertEqual(result["actions"][-1]["client_action_id"], "answer")

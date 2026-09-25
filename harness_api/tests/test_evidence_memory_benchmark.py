import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import test_evidence_memory as memory_fixtures
from app.core.artifacts import ArtifactStore
from app.core.capabilities import TaskRegistry
from app.core.domain import create_initial_state
from app.core.evaluation import EvaluatorRegistry
from app.core.evidence_memory import (
    EvidenceMemoryPolicy,
    EvidenceMemoryStore,
    MemorySearchArguments,
    build_evidence_memory_record,
)
from app.core.schemas import AnswerRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "config" / "evidence-memory-benchmark-v1.json"
MATRIX = PROJECT_ROOT / "scripts" / "prepare_evidence_memory_matrix.py"
ACTOR_CERTIFICATE = "1" * 64
NOW = "2026-08-02T00:00:00Z"


def load_preparer():
    path = PROJECT_ROOT / "scripts" / "prepare_evidence_memory_benchmark.py"
    specification = importlib.util.spec_from_file_location(
        "prepare_evidence_memory_benchmark_test", path
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class EvidenceMemoryBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        value = memory_fixtures.policy_value()
        value["reader_grants"] = [
            {
                "task_id": "evidence-memory-benchmark",
                "task_versions": ["1.0.0"],
            }
        ]
        content = json.dumps(value, sort_keys=True).encode()
        self.policy_path = self.root / "policy.json"
        self.policy_path.write_bytes(content)
        os.chmod(self.policy_path, 0o644)
        self.policy_sha256 = hashlib.sha256(content).hexdigest()
        self.policy = EvidenceMemoryPolicy.model_validate_json(content)
        self.store_path = self.root / "memory" / "events.sqlite3"
        self.store = EvidenceMemoryStore(self.store_path)
        manifest, state, evaluation, evidence, source = (
            memory_fixtures.EvidenceMemoryTests.source_episode()
        )
        self.record = build_evidence_memory_record(
            policy=self.policy,
            policy_sha256=self.policy_sha256,
            manifest=manifest,
            state=state,
            evaluation=evaluation,
            evidence=evidence,
            source=source,
            object_type="land-cover-assessment",
            public_summary="Built-up was dominant in the reviewed area.",
            ttl_seconds=14 * 24 * 60 * 60,
        )
        self.store.publish(
            self.policy,
            self.policy_sha256,
            self.record,
            actor_id="trusted-memory-curator",
            actor_certificate_sha256=ACTOR_CERTIFICATE,
            now=NOW,
        )
        self.module = load_preparer()

    def arguments(self, output, config=CONFIG):
        return argparse.Namespace(
            config=Path(config),
            policy=self.policy_path,
            policy_sha256=self.policy_sha256,
            store=self.store_path,
            tasks=PROJECT_ROOT / "tasks",
            output=Path(output),
        )

    @staticmethod
    def tree(root):
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_prepares_deterministic_matched_task_pair(self):
        first = self.root / "first"
        second = self.root / "second"
        report = self.module.prepare(self.arguments(first))
        self.module.prepare(self.arguments(second))
        self.assertEqual(self.tree(first), self.tree(second))
        self.assertEqual(report["expected_memory_id"], self.record.memory_id)
        self.assertEqual(report["snapshot_sequence"], 1)
        registry = TaskRegistry(str(first))
        self.assertEqual(registry.count(), 2)
        with_memory = registry.get("evidence-memory-benchmark", "1.0.0")
        without_memory = registry.get("evidence-memory-benchmark", "1.0.1")
        self.assertEqual(with_memory.task.prompt, without_memory.task.prompt)
        self.assertEqual(with_memory.task.inputs, without_memory.task.inputs)
        self.assertEqual(with_memory.task.budget, without_memory.task.budget)
        self.assertEqual(with_memory.task.answer_schema, without_memory.task.answer_schema)
        self.assertEqual(with_memory.scenario.allowed_tools, ["memory.search"])
        self.assertEqual(without_memory.scenario.allowed_tools, [])
        self.assertIn("evidence_memory", with_memory.task.metadata)
        self.assertNotIn("evidence_memory", without_memory.task.metadata)
        self.assertEqual(
            with_memory.evaluator.config["expected_memory_id"], self.record.memory_id
        )

    def test_evaluator_separates_accuracy_memory_faithfulness_and_control(self):
        output = self.root / "tasks"
        self.module.prepare(self.arguments(output))
        registry = TaskRegistry(str(output))
        evaluator = EvaluatorRegistry(str(self.root), ArtifactStore(str(self.root / "artifacts")))
        with_memory = registry.get("evidence-memory-benchmark", "1.0.0")
        state, _ = create_initial_state(
            "ep2-" + "a" * 32, with_memory, 42, "2026-08-02T00:00:00Z"
        )
        state = state.model_copy(
            update={
                "status": "terminated",
                "step_count": 2,
                "final_answer": AnswerRecord(
                    outcome="submitted",
                    answer={
                        "label": "built-up",
                        "memory_ids": [self.record.memory_id],
                        "confidence": 1.0,
                    },
                    confidence=1.0,
                    evidence_ids=[],
                ),
            }
        )
        query = MemorySearchArguments.model_validate(
            json.loads(CONFIG.read_text())["query"]
        )
        result = self.store.search(
            self.policy, self.policy_sha256, with_memory, query
        ).model_dump(mode="json")
        tool_result = {
            "tool_id": "memory.search",
            "tool_version": "1.0.0",
            "status": "completed",
            **result,
        }
        scored = evaluator.evaluate_safely(
            with_memory, state, {}, 0, 0, 100, [tool_result]
        )
        self.assertEqual(scored.status, "completed")
        self.assertEqual(scored.aggregate_reward, 1.0)
        self.assertEqual({item.name: item.value for item in scored.metrics}, {
            "task.accuracy": 1.0,
            "evidence.memory_faithfulness": 1.0,
            "process.efficiency": 1.0,
        })

        control = registry.get("evidence-memory-benchmark", "1.0.1")
        control_state, _ = create_initial_state(
            "ep2-" + "b" * 32, control, 42, "2026-08-02T00:00:00Z"
        )
        control_state = control_state.model_copy(
            update={
                "status": "terminated",
                "step_count": 1,
                "final_answer": AnswerRecord(
                    outcome="abstained",
                    answer=None,
                    confidence=None,
                    evidence_ids=[],
                    rationale="No governed evidence tool is available.",
                ),
            }
        )
        control_score = evaluator.evaluate_safely(
            control, control_state, {}, 0, 0, 100, []
        )
        self.assertAlmostEqual(control_score.aggregate_reward, 0.1)
        self.assertEqual(control_score.diagnostics["treatment"], "without_memory")

        guessed = control_state.model_copy(
            update={
                "final_answer": AnswerRecord(
                    outcome="submitted",
                    answer={
                        "label": "built-up",
                        "memory_ids": [self.record.memory_id],
                        "confidence": 1.0,
                    },
                    confidence=1.0,
                    evidence_ids=[],
                )
            }
        )
        guessed_score = evaluator.evaluate_safely(control, guessed, {}, 0, 0, 100, [])
        self.assertAlmostEqual(guessed_score.aggregate_reward, 0.7)
        self.assertEqual(
            {item.name: item.value for item in guessed_score.metrics}[
                "evidence.memory_faithfulness"
            ],
            0.0,
        )

    def test_wrong_policy_config_duplicate_match_and_existing_output_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.module.prepare(
                argparse.Namespace(
                    **{
                        **vars(self.arguments(self.root / "bad-sha")),
                        "policy_sha256": "f" * 64,
                    }
                )
            )
        writable = self.root / "writable-config.json"
        shutil.copyfile(CONFIG, writable)
        writable.chmod(0o666)
        with self.assertRaisesRegex(ValueError, "writable"):
            self.module.prepare(self.arguments(self.root / "writable", writable))

        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.module.prepare(self.arguments(existing))

        manifest, state, evaluation, evidence, source = (
            memory_fixtures.EvidenceMemoryTests.source_episode()
        )
        second = build_evidence_memory_record(
            policy=self.policy,
            policy_sha256=self.policy_sha256,
            manifest=manifest,
            state=state,
            evaluation=evaluation,
            evidence=evidence,
            source=source,
            object_type="land-cover-assessment",
            public_summary="A second reviewed summary for the same query.",
            ttl_seconds=14 * 24 * 60 * 60,
        )
        self.store.publish(
            self.policy,
            self.policy_sha256,
            second,
            actor_id="trusted-memory-curator",
            actor_certificate_sha256=ACTOR_CERTIFICATE,
            now=NOW,
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.module.prepare(self.arguments(self.root / "duplicate"))

    def test_matrix_derives_distinct_valid_records(self):
        spec = importlib.util.spec_from_file_location("memory_matrix", MATRIX)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest, state, evaluation, evidence, source = (
            memory_fixtures.EvidenceMemoryTests.source_episode()
        )
        base = build_evidence_memory_record(
            policy=self.policy,
            policy_sha256=self.policy_sha256,
            manifest=manifest,
            state=state,
            evaluation=evaluation,
            evidence=evidence,
            source=source,
            object_type="land-cover-assessment",
            public_summary="Reviewed built-up evidence.",
            ttl_seconds=14 * 24 * 60 * 60,
        )
        records = [
            module._derive(base),
            module._derive(base, public_summary="Conflicting summary."),
            module._derive(base, bbox={"west": 1, "south": 2, "east": 3, "north": 4}),
        ]
        self.assertEqual(len({record.memory_id for record in records}), 3)
        for record in records:
            self.assertTrue(record.memory_id.startswith("mem-"))


if __name__ == "__main__":
    unittest.main()

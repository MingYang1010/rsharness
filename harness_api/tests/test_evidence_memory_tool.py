import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from v2.test_tool_execution import make_tool_tasks
from app.main import create_app
from app.core.capabilities import TaskRegistry
from app.core.domain import V2DomainError, create_initial_state
from app.core.evidence_memory import (
    EvidenceMemoryBinding,
    EvidenceMemoryPolicy,
    EvidenceMemoryStore,
    MemorySearchArguments,
    build_evidence_memory_record,
)
from app.core.events import sha256_json
from app.core.schemas import (
    AnswerRecord,
    EvidenceRef,
    MetricResult,
    TaskManifest,
    ToolInvokeAction,
)
from app.core.tools.memory import MemorySearchExecutor
from app.core.tools.runtime import ToolRouter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_TASKS = PROJECT_ROOT / "tasks"
ACTOR_CERTIFICATE = "1" * 64
NOW = "2026-08-02T00:00:00Z"


def policy_value():
    return {
        "schema_version": "1.0.0",
        "policy_id": "research-memory-v1",
        "scope_id": "nanjing-temporal-memory",
        "valid_from": "2026-01-01T00:00:00Z",
        "expires_at": "2027-01-01T00:00:00Z",
        "writers": [
            {
                "actor_id": "trusted-memory-curator",
                "certificate_sha256": ACTOR_CERTIFICATE,
            }
        ],
        "source_grants": [
            {
                "task_id": "worldcover-grounded-vqa",
                "task_versions": ["1.1.0"],
                "evaluator_id": "worldcover-grounded-v1",
                "min_aggregate_reward": 0.8,
            }
        ],
        "reader_grants": [
            {"task_id": "crop-smoke", "task_versions": ["1.0.0"]}
        ],
        "max_record_ttl_seconds": 30 * 24 * 60 * 60,
        "max_active_records": 8,
        "max_query_results": 5,
    }


class EvidenceMemoryToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        content = json.dumps(policy_value(), sort_keys=True).encode()
        self.policy_path = self.root / "policy.json"
        self.policy_path.write_bytes(content)
        os.chmod(self.policy_path, 0o644)
        self.policy_sha256 = hashlib.sha256(content).hexdigest()
        self.policy = EvidenceMemoryPolicy.model_validate_json(content)
        self.memory_path = self.root / "memory" / "events.sqlite3"
        self.store = EvidenceMemoryStore(self.memory_path)
        self.record = self._source_record()
        self.snapshot = self.store.publish(
            self.policy,
            self.policy_sha256,
            self.record,
            actor_id="trusted-memory-curator",
            actor_certificate_sha256=ACTOR_CERTIFICATE,
            now=NOW,
        )
        self.tasks = make_tool_tasks(self.root)
        self._bind_reader_task()

    def _source_record(self):
        original = TaskRegistry(str(SOURCE_TASKS)).get(
            "worldcover-grounded-vqa", "1.1.0"
        )
        source = original.assets[0].model_copy(update={"instrument": "WorldCover-map"})
        body = {
            "task": original.task,
            "scenario": original.scenario,
            "assets": [source, *original.assets[1:]],
            "evaluator": original.evaluator,
        }
        manifest = TaskManifest(
            **body,
            task_manifest_hash=sha256_json(
                {
                    "task": body["task"].model_dump(mode="json"),
                    "scenario": body["scenario"].model_dump(mode="json"),
                    "assets": [item.model_dump(mode="json") for item in body["assets"]],
                    "evaluator": body["evaluator"].model_dump(mode="json"),
                }
            ),
        )
        state, _ = create_initial_state(
            "ep2-" + "a" * 32,
            manifest,
            42,
            "2026-08-01T00:00:00Z",
        )
        evidence = EvidenceRef(
            evidence_id="ev-memory-source",
            claim_id="dominant-class",
            source_ref=source.asset_id,
            selector={
                "bbox": {
                    "west": 120.5,
                    "south": 30.5,
                    "east": 121.0,
                    "north": 31.0,
                },
                "time_range": source.temporal,
                "bands": ["visual"],
            },
            description="Reviewed source evidence for cross-task memory.",
            frozen_sha256=source.sha256,
        )
        state = state.model_copy(
            update={
                "status": "terminated",
                "state_version": 2,
                "step_count": 2,
                "evidence_refs": [evidence],
                "final_answer": AnswerRecord(
                    outcome="submitted",
                    answer={"label": "built-up"},
                    confidence=1.0,
                    evidence_ids=[evidence.evidence_id],
                ),
                "updated_at": "2026-08-01T00:00:00Z",
            }
        )
        evaluation = MetricResult(
            evaluation_id="eval-memory-source",
            status="completed",
            metrics=[],
            aggregate_reward=1.0,
            evaluator_id="worldcover-grounded-v1",
            evaluator_version="1.1.0",
        )
        return build_evidence_memory_record(
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

    def _bind_reader_task(self):
        directory = self.tasks / "crop-smoke"
        task = json.loads((directory / "task.json").read_text())
        task["metadata"]["evidence_memory"] = EvidenceMemoryBinding(
            schema_version="1.0.0",
            policy_id=self.policy.policy_id,
            policy_sha256=self.policy_sha256,
            scope_id=self.policy.scope_id,
            snapshot_sequence=self.snapshot.sequence,
            snapshot_sha256=self.snapshot.snapshot_sha256,
            as_of=NOW,
        ).model_dump(mode="json")
        scenario = json.loads((directory / "scenario.json").read_text())
        scenario["allowed_tools"].append("memory.search")
        scenario["data_cutoff"] = "2026-08-03T00:00:00Z"
        (directory / "task.json").write_text(json.dumps(task))
        (directory / "scenario.json").write_text(json.dumps(scenario))

    def app(self):
        return create_app(
            database_path=str(self.root / "episodes.sqlite3"),
            v2_tasks_path=str(self.tasks),
            v2_artifacts_path=str(self.root / "artifacts"),
            v2_memory_store_path=str(self.memory_path),
            v2_memory_policy_path=str(self.policy_path),
            v2_memory_policy_sha256=self.policy_sha256,
        )

    @staticmethod
    def arguments():
        return {
            "bbox": {"west": 120.6, "south": 30.6, "east": 120.9, "north": 30.9},
            "time_range": {
                "start": "2021-06-01T00:00:00Z",
                "end": "2021-06-30T00:00:00Z",
            },
            "object_type": "land-cover-assessment",
            "limit": 5,
        }

    def test_search_is_pinned_budgeted_idempotent_and_restart_safe(self):
        with TestClient(self.app()) as client:
            capabilities = client.get("/v2/capabilities").json()["data"]
            self.assertEqual(capabilities["tools"], ["memory.search"])
            reset = client.post(
                "/v2/reset",
                json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}},
            ).json()["data"]
            episode_id = reset["episode_id"]
            request = {
                "client_action_id": "memory-search-1",
                "expected_state_version": 0,
                "action": {
                    "type": "tool.invoke",
                    "tool_id": "memory.search",
                    "arguments": self.arguments(),
                },
            }
            first = client.post(f"/v2/episodes/{episode_id}/step", json=request)
            self.assertEqual(first.status_code, 200, first.text)
            data = first.json()["data"]
            inline = data["observation"]["items"][0]["inline"]
            self.assertEqual(inline["matched_count"], 1)
            self.assertEqual(inline["records"][0]["memory_id"], self.record.memory_id)
            public_result = json.dumps(inline)
            self.assertNotIn("episode_id", public_result)
            self.assertNotIn("evidence_id", public_result)
            self.assertEqual(
                data["state"]["budget"]["input_bytes"]["used"],
                inline["cost"]["input_bytes"],
            )
            self.assertEqual(data["state"]["budget"]["tool_calls"]["used"], 1)
            self.assertEqual(client.post(f"/v2/episodes/{episode_id}/step", json=request).json()["data"], data)
        with TestClient(self.app()) as restarted:
            repeat = restarted.post(f"/v2/episodes/{episode_id}/step", json=request)
            self.assertEqual(repeat.status_code, 200, repeat.text)
            self.assertEqual(repeat.json()["data"], data)
        with sqlite3.connect(self.root / "episodes.sqlite3") as connection:
            run = json.loads(connection.execute("SELECT run_json FROM v2_tool_runs").fetchone()[0])
            self.assertTrue(run["metadata_only"])
            self.assertEqual(run["logical_input_bytes"], inline["cost"]["input_bytes"])
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM v2_artifacts").fetchone()[0], 0)

    def test_invalid_query_pin_and_budget_fail_closed(self):
        manifest = TaskRegistry(str(self.tasks)).get("crop-smoke", "1.0.0")
        executor = MemorySearchExecutor(self.store, self.policy, self.policy_sha256)
        with self.assertRaisesRegex(V2DomainError, "invalid evidence memory query") as invalid:
            executor.plan(
                ToolInvokeAction(
                    type="tool.invoke",
                    tool_id="memory.search",
                    arguments={"limit": 0},
                ),
                manifest,
                [],
            )
        self.assertEqual(invalid.exception.code, "invalid_tool_arguments")
        task = manifest.task.model_copy(
            update={
                "metadata": {
                    **manifest.task.metadata,
                    "evidence_memory": {
                        **manifest.task.metadata["evidence_memory"],
                        "snapshot_sha256": "f" * 64,
                    },
                }
            }
        )
        wrong = manifest.model_copy(update={"task": task})
        query = MemorySearchArguments.model_validate(self.arguments())
        with self.assertRaisesRegex(V2DomainError, "unavailable") as unavailable:
            executor.plan(
                ToolInvokeAction(
                    type="tool.invoke",
                    tool_id="memory.search",
                    arguments=query.model_dump(mode="json"),
                ),
                wrong,
                [],
            )
        self.assertEqual(unavailable.exception.code, "policy_rejected")

        with TestClient(self.app()) as client:
            reset = client.post(
                "/v2/reset",
                json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}},
            ).json()["data"]
            episode_id = reset["episode_id"]
            with sqlite3.connect(self.root / "episodes.sqlite3") as connection:
                state = json.loads(connection.execute(
                    "SELECT state_json FROM v2_episodes WHERE episode_id=?", (episode_id,)
                ).fetchone()[0])
                state["budget"]["input_bytes"]["remaining"] = 1
                connection.execute(
                    "UPDATE v2_episodes SET state_json=? WHERE episode_id=?",
                    (json.dumps(state), episode_id),
                )
            response = client.post(
                f"/v2/episodes/{episode_id}/step",
                json={
                    "client_action_id": "memory-budget",
                    "expected_state_version": 0,
                    "action": {
                        "type": "tool.invoke",
                        "tool_id": "memory.search",
                        "arguments": self.arguments(),
                    },
                },
            )
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(response.json()["error"]["code"], "tool_budget_exceeded")
            with sqlite3.connect(self.root / "episodes.sqlite3") as connection:
                self.assertEqual(connection.execute(
                    "SELECT COUNT(*) FROM v2_tool_runs WHERE episode_id=?", (episode_id,)
                ).fetchone()[0], 0)

    def test_partial_main_configuration_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "configured together"):
            create_app(
                database_path=str(self.root / "partial.sqlite3"),
                v2_tasks_path=str(self.tasks),
                v2_memory_store_path=str(self.memory_path),
            )


if __name__ == "__main__":
    unittest.main()

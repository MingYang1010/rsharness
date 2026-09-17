import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from .helpers import TASKS_ROOT
from .test_m2_runtime import PNG_BYTES
from app.main import create_app
from app.v2.artifacts import ArtifactStore
from app.v2.domain import V2DomainError
from app.v2.schemas import ArtifactLineage
from app.v2.tools.eo_gym import EOGymExecutor, ToolOutput


def make_tool_tasks(root: Path):
    source = root / "tasks"
    shutil.copytree(TASKS_ROOT, source)
    task_dir = source / "crop-smoke"
    shutil.copytree(source / "worldcover-grounded-vqa", task_dir)
    task = json.loads((task_dir / "task.json").read_text())
    task.update(task_id="crop-smoke", metadata={"observation_profile": "headless-tools-v1"})
    task["budget"]["max_artifact_bytes"] = 128 * 1024 * 1024
    task["budget"]["max_input_bytes"] = 512 * 1024 * 1024
    (task_dir / "task.json").write_text(json.dumps(task))
    scenario = json.loads((task_dir / "scenario.json").read_text())
    scenario["allowed_actions"] = ["tool.invoke", "memory.save_evidence", "answer.*"]
    scenario["allowed_tools"] = ["eo_gym.crop"]
    (task_dir / "scenario.json").write_text(json.dumps(scenario))
    return source


class FakeExecutor(EOGymExecutor):
    max_output_bytes = 1024

    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = 0
        self.fail = False
        self.entered = threading.Event()
        self.release = None

    def invoke(self, action, manifest):
        self.calls += 1
        self.entered.set()
        if self.release:
            self.release.wait(5)
        if self.fail:
            raise V2DomainError("tool_timeout", "test timeout", 504, True, "tool")
        _, asset = self.prepare(action, manifest)
        artifact = self.artifacts.put_bytes(PNG_BYTES, "image", "image/png", ArtifactLineage(
            tool_id=self.tool_id, tool_version=self.tool_version, input_refs=[asset.asset_id], parameters_hash=hashlib.sha256(b"args").hexdigest()))
        return ToolOutput(artifact, {"width": 1, "height": 1}, asset.size_bytes)


class ToolExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tasks = make_tool_tasks(self.root)
        self.artifacts = ArtifactStore(str(self.root / "artifacts"))
        self.executor = FakeExecutor(self.artifacts)
        self.app = create_app(database_path=str(self.root / "state.db"), v2_tasks_path=str(self.tasks),
            v2_tool_executor=self.executor, v2_artifacts_path=str(self.artifacts.root))
        self.client = TestClient(self.app)
        reset = self.client.post("/v2/reset", json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}, "seed": 42})
        self.assertEqual(reset.status_code, 201, reset.text)
        self.episode_id = reset.json()["data"]["episode_id"]
        self.assertEqual(reset.json()["data"]["observation"]["primary_type"], "asset_metadata")
        self.url = f"/v2/episodes/{self.episode_id}/step"
        self.request = {"client_action_id": "crop-1", "expected_state_version": 0,
            "action": {"type": "tool.invoke", "tool_id": "eo_gym.crop", "arguments": {"asset_id": "asset-worldcover-n30e120", "aoi": [0, 0, 0.5, 0.5]}}}

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def test_real_step_budget_artifact_idempotency_and_restart(self):
        first = self.client.post(self.url, json=self.request)
        self.assertEqual(first.status_code, 200, first.text)
        data = first.json()["data"]
        self.assertEqual(data["state"]["budget"]["tool_calls"]["used"], 1)
        self.assertEqual(data["observation"]["primary_type"], "tool_result")
        self.assertEqual(self.client.post(self.url, json=self.request).json()["data"], data)
        self.assertEqual(self.executor.calls, 1)
        restarted = create_app(database_path=str(self.root / "state.db"), v2_tasks_path=str(self.tasks), v2_tool_executor=self.executor,
            v2_artifacts_path=str(self.artifacts.root))
        with TestClient(restarted) as client:
            self.assertEqual(client.post(self.url, json=self.request).json()["data"], data)
        conflict = {**self.request, "expected_state_version": 1}
        self.assertEqual(self.client.post(self.url, json=conflict).status_code, 409)
        self.assertEqual(self.executor.calls, 1)

    def test_failed_execution_is_cached_and_counted_once(self):
        self.executor.fail = True
        self.assertEqual(self.client.post(self.url, json=self.request).status_code, 504)
        self.assertEqual(self.client.post(self.url, json=self.request).status_code, 504)
        state = self.client.get(f"/v2/episodes/{self.episode_id}/state").json()["data"]["state"]
        self.assertEqual(state["budget"]["tool_calls"]["used"], 1)
        self.assertEqual(state["state_version"], 1)
        self.assertEqual(self.executor.calls, 1)

    def test_other_episode_is_not_locked_during_tool_io(self):
        self.executor.release = threading.Event()
        outcomes = []
        thread = threading.Thread(target=lambda: outcomes.append(self.client.post(self.url, json=self.request)))
        thread.start()
        try:
            self.assertTrue(self.executor.entered.wait(3))
            # Independent writer can commit while tool execution waits.
            with sqlite3.connect(self.root / "state.db", timeout=0.5) as connection:
                connection.execute("BEGIN IMMEDIATE")
            blocked = self.client.post(self.url, json={"client_action_id": "abstain", "expected_state_version": 0,
                "action": {"type": "answer.abstain", "rationale": "test", "evidence_ids": []}})
            self.assertEqual(blocked.status_code, 409)
            self.assertEqual(blocked.json()["error"]["code"], "tool_in_progress")
            self.assertEqual(self.client.post(self.url, json=self.request).status_code, 409)
        finally:
            self.executor.release.set()
            thread.join(5)
        self.assertEqual(outcomes[0].status_code, 200, outcomes[0].text)
        self.assertEqual(self.executor.calls, 1)

    def test_unknown_asset_and_exhausted_budget_are_rejected(self):
        invalid = json.loads(json.dumps(self.request))
        invalid["action"]["arguments"]["asset_id"] = "asset-hidden"
        self.assertEqual(self.client.post(self.url, json=invalid).status_code, 403)
        with sqlite3.connect(self.root / "state.db") as connection:
            state = json.loads(connection.execute("SELECT state_json FROM v2_episodes WHERE episode_id=?", (self.episode_id,)).fetchone()[0])
            state["budget"]["tool_calls"].update(used=10, remaining=0)
            connection.execute("UPDATE v2_episodes SET state_json=? WHERE episode_id=?", (json.dumps(state), self.episode_id))
        self.assertEqual(self.client.post(self.url, json=self.request).status_code, 422)
        self.assertEqual(self.executor.calls, 0)

    def test_expired_lease_is_finalized_without_reexecuting(self):
        request_json = json.dumps({"expected_state_version": 0, "action": self.request["action"]}, sort_keys=True, separators=(",", ":"))
        run = {"client_action_id": "crop-1", "request_json": request_json, "lease_until": time.time()-1,
               "input_bytes_reserved": 0, "output_bytes_reserved": 1024, "tool_version": "1.0.0", "expected_state_version": 0}
        with sqlite3.connect(self.root / "state.db") as connection:
            connection.execute("INSERT INTO v2_tool_runs VALUES (?,?,?,?,?,?)", ("tool-crashed", self.episode_id, "eo_gym.crop", "running", json.dumps(run), "2026-09-17T00:00:00Z"))
        response = self.client.post(self.url, json=self.request)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"]["code"], "tool_interrupted")
        self.assertEqual(self.executor.calls, 0)
        self.assertEqual(self.client.post(self.url, json=self.request).json()["error"], response.json()["error"])

    def test_artifact_provenance_conflict_finishes_without_pending_lease(self):
        first = self.client.post(self.url, json=self.request)
        self.assertEqual(first.status_code, 200)
        with sqlite3.connect(self.root / "state.db") as connection:
            artifact_id, value = connection.execute("SELECT artifact_id,artifact_json FROM v2_artifacts").fetchone()
            artifact = json.loads(value)
            artifact["lineage"]["parameters_hash"] = "f" * 64
            connection.execute("UPDATE v2_artifacts SET artifact_json=? WHERE artifact_id=?", (json.dumps(artifact, sort_keys=True, separators=(",", ":")), artifact_id))
        repeat = {**self.request, "expected_state_version": 1, "client_action_id": "crop-2"}
        response = self.client.post(self.url, json=repeat)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "artifact_metadata_conflict")
        with sqlite3.connect(self.root / "state.db") as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM v2_tool_runs WHERE status='running'").fetchone()[0], 0)

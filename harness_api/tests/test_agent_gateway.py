import copy
import asyncio
import contextlib
import hashlib
import importlib.util
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from v2.test_tool_execution import FakeExecutor, make_tool_tasks
from app.main import create_app as create_backend
from app.agent_gateway import AgentBinding, build_binding, create_app, public_state, public_observation
from app.agent_gateway import AgentGuard
from app.v2.artifacts import ArtifactStore
from app.v2.capabilities import TaskRegistry
from app.v2.schemas import V2EpisodeState
from app.v2.tools.runtime import ToolRouter, ToolOutput

TOKEN = "1" * 64  # Isolated test credential, never deployed.


class PublicFakeExecutor(FakeExecutor):
    def invoke(self, action, manifest):
        output = super().invoke(action, manifest)
        _, asset = self.prepare(action, manifest)
        return ToolOutput(output.artifact, {"width": 1, "height": 1, "bbox_px": [0, 0, 1, 1],
            "aoi_norm": list(action.arguments["aoi"]), "input_asset_id": asset.asset_id,
            "input_sha256": asset.sha256, "upstream_revision": "f" * 40,
            "source": "/private/SECRET-source.tif", "diagnostics": {"gold": "SECRET-label"}}, output.input_bytes)


class AgentGatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tasks = make_tool_tasks(self.root)
        directory = self.tasks / "crop-smoke"
        scenario = json.loads((directory / "scenario.json").read_text())
        scenario["allowed_tools"] += ["catalog.search", "catalog.inspect_asset"]
        (directory / "scenario.json").write_text(json.dumps(scenario))
        assets = json.loads((directory / "assets.json").read_text())
        assets[0].update(uri="file:///private/SECRET-image.tif", source="SECRET-source")
        assets.append({**copy.deepcopy(assets[0]), "asset_id": "SECRET-label", "roles": ["label"]})
        (directory / "assets.json").write_text(json.dumps(assets))
        self.executor = PublicFakeExecutor(ArtifactStore(str(self.root / "artifacts")))
        self.backend = create_backend(database_path=str(self.root / "db.sqlite3"), v2_tasks_path=str(self.tasks),
            v2_artifacts_path=str(self.root / "artifacts"), v2_tool_executor=ToolRouter(self.executor))
        self.operator = TestClient(self.backend)
        self.addCleanup(self.operator.close)
        reset = self.operator.post("/v2/reset", json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
        self.episode = reset["episode_id"]
        self.manifest = TaskRegistry(self.tasks).get("crop-smoke", "1.0.0")
        self.binding = build_binding(self.manifest, V2EpisodeState.model_validate(reset["state"]),
            self.operator.get("/v2/capabilities").json()["data"], hashlib.sha256(TOKEN.encode()).hexdigest())
        self.client = self.make_client()
        self.addCleanup(self.client.close)

    def make_client(self, transport=None, binding=None):
        return TestClient(create_app(binding or self.binding, "http://operator", transport or httpx.ASGITransport(app=self.backend)),
                          headers={"Authorization": "Bearer " + TOKEN})

    def step(self, tool="catalog.search", arguments=None, version=0, action_id="action-1"):
        return {"client_action_id": action_id, "expected_state_version": version,
                "action": {"type": "tool.invoke", "tool_id": tool, "arguments": arguments or {}}}

    def test_authentication_precedes_routing_and_body_processing(self):
        for path in ("/agent/session", "/v2/tasks/secret", "/docs", "/openapi.json"):
            response = self.client.get(path, headers={"Authorization": "Bearer wrong"})
            self.assertEqual(response.status_code, 401, response.text)
        self.assertEqual(self.client.get("/healthz", headers={"Authorization": ""}).status_code, 200)
        with self.make_client() as client:
            client.headers.pop("authorization")
            self.assertEqual(client.get("/agent/session").status_code, 401)
            self.assertEqual(client.post("/agent/step", content="x" * 140000).status_code, 401)

    def test_no_operator_routes_query_override_or_arbitrary_reset(self):
        for path in ("/v2/capabilities", "/v2/tasks/crop-smoke/versions/1.0.0", "/agent/reset", "/openapi.json", "/docs",
                     f"/v2/episodes/{self.episode}/trace", f"/v2/episodes/{self.episode}/evaluation"):
            self.assertEqual(self.client.get(path).status_code, 404, path)
        self.assertEqual(self.client.get("/agent/state?episode_id=other").status_code, 422)
        self.assertEqual(self.client.post("/agent/reset", json={}).status_code, 404)

    def test_public_session_and_catalog_exclude_private_fields(self):
        session = self.client.get("/agent/session")
        self.assertEqual(session.status_code, 200, session.text)
        data = session.json()
        self.assertEqual(data["state"]["episode_id"], self.episode)
        self.assertIn("catalog.search", data["tool_schemas"])
        self.assertNotIn("evaluation", data["state"])
        self.assertNotIn("provenance", data["observation"])
        self.assertNotIn("token_sha256", session.text)
        self.assertNotIn("SECRET", session.text)
        self.assertNotIn("source", data["task"])
        response = self.client.post("/agent/step", json=self.step())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("SECRET", response.text)
        record = response.json()["observation"]["items"][0]["inline"]["assets"][0]
        self.assertNotIn("uri", record)
        self.assertNotIn("source", record)

    def test_evaluation_diagnostics_and_warning_paths_are_never_projected(self):
        state = self.operator.get(f"/v2/episodes/{self.episode}/state").json()["data"]["state"]
        state["evaluation"] = {"evaluation_id": "eval-secret", "status": "completed", "metrics": [],
            "aggregate_reward": 1.0, "evaluator_id": "secret", "evaluator_version": "1.0.0", "diagnostics": {"label": "SECRET"}}
        self.assertNotIn("SECRET", json.dumps(public_state(state, self.binding)))
        observation = self.client.get("/agent/session").json()["observation"]
        original = self.operator.get(f"/v2/episodes/{self.episode}/observations/{observation['observation_id']}").json()["data"]["observation"]
        original.update(provenance={"path": "SECRET"}, warnings=["SECRET"])
        self.assertNotIn("SECRET", json.dumps(public_observation(original, self.binding)))

    def test_step_idempotency_and_gateway_restart_do_not_reexecute(self):
        body = self.step("eo_gym.crop", {"asset_id": self.binding.task.input_asset_refs[0], "aoi": [0, 0, .5, .5]})
        first = self.client.post("/agent/step", json=body)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertNotIn("SECRET", first.text)
        self.assertEqual(self.client.post("/agent/step", json=body).json(), first.json())
        with self.make_client() as restarted:
            self.assertEqual(restarted.post("/agent/step", json=body).json(), first.json())
        self.assertEqual(self.executor.calls, 1)
        conflict = self.client.post("/agent/step", json={**body, "expected_state_version": 1})
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.json()["error"]["code"], "idempotency_conflict")

    def test_scoped_artifacts_validate_hash_and_hide_storage_uri(self):
        body = self.step("eo_gym.crop", {"asset_id": self.binding.task.input_asset_refs[0], "aoi": [0, 0, .5, .5]})
        result = self.client.post("/agent/step", json=body).json()
        ref = result["observation"]["items"][1]["artifact_ref"]
        metadata = self.client.get("/agent/artifacts/" + ref)
        self.assertEqual(metadata.status_code, 200, metadata.text)
        self.assertNotIn("uri", metadata.json()["artifact"])
        content = self.client.get("/agent/artifacts/" + ref + "/content")
        self.assertEqual(content.status_code, 200)
        self.assertEqual(hashlib.sha256(content.content).hexdigest(), ref[4:])
        self.assertEqual(content.headers["cache-control"], "no-store")
        other = self.operator.post("/v2/reset", json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]["state"]
        binding = self.binding.model_copy(update={"episode_id": other["episode_id"]})
        with self.make_client(binding=binding) as foreign:
            self.assertEqual(foreign.get("/agent/artifacts/" + ref + "/content").status_code, 403)
        self.assertEqual(self.client.get("/agent/artifacts/not-an-id/content").status_code, 422)

    def test_hidden_assets_and_unreviewed_tools_denied_before_backend_step(self):
        for body in (self.step("catalog.inspect_asset", {"asset_id": "SECRET-label"}), self.step("shell.exec", {"cmd": "pwd"}),
                     self.step("catalog.search", {"uri": "http://private"})):
            response = self.client.post("/agent/step", json=body)
            self.assertIn(response.status_code, (403, 422), response.text)
        state = self.client.get("/agent/state").json()["state"]
        self.assertEqual(state["state_version"], 0)
        self.assertEqual(self.executor.calls, 0)

    def test_oversized_malformed_and_scope_override_requests_fail(self):
        self.assertEqual(self.client.post("/agent/step", content=b"x" * 131073).status_code, 413)
        self.assertEqual(self.client.post("/agent/step", content=b"{").status_code, 422)
        self.assertEqual(self.client.post("/agent/step", json={**self.step(), "episode_id": "other"}).status_code, 422)
        self.assertEqual(self.client.get("/agent/observations/not-an-id").status_code, 422)

    def test_other_episode_observation_cannot_be_read(self):
        other = self.operator.post("/v2/reset", json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
        response = self.client.get("/agent/observations/" + other["observation"]["observation_id"])
        self.assertIn(response.status_code, (403, 404), response.text)

    def test_pinned_task_mismatch_fails_closed(self):
        binding = self.binding.model_copy(update={"task_manifest_hash": "f" * 64})
        with self.make_client(binding=binding) as client:
            response = client.get("/agent/session")
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(response.json()["error"]["code"], "session_pin_mismatch")

    def test_upstream_errors_redirects_oversize_and_malformed_are_sanitized(self):
        responses = [httpx.Response(500, json={"error": {"code": "SECRET", "message": "/private/SECRET", "details": ["SECRET"]}}),
                     httpx.Response(302, headers={"location": "http://private/SECRET"}),
                     httpx.Response(200, content=b"SECRET"), httpx.Response(200, json={"data": {}})]
        for response in responses:
            with self.make_client(httpx.MockTransport(lambda request: response)) as client:
                result = client.get("/agent/session")
                self.assertEqual(result.status_code, 502, result.text)
                self.assertNotIn("SECRET", result.text)
        with patch("app.agent_gateway.MAX_JSON", 4), self.make_client(httpx.MockTransport(lambda request: httpx.Response(200, content=b"SECRET"))) as client:
            self.assertEqual(client.get("/agent/session").json()["error"]["code"], "upstream_response_too_large")

    def test_private_inputs_and_unsupported_binding_tools_rejected(self):
        state = self.operator.get(f"/v2/episodes/{self.episode}/state").json()["data"]["state"]
        self.manifest.assets[0].roles = ["labels"]
        with self.assertRaises(ValueError):
            build_binding(self.manifest, V2EpisodeState.model_validate(state), {"actions": [], "tools": []}, "f" * 64)
        value = self.binding.model_dump(mode="json")
        value["task"]["allowed_tools"].append("unsafe.tool")
        with self.assertRaises(ValueError):
            AgentBinding.model_validate(value)

    def test_issuance_is_private_and_rerun_does_not_create_another_episode(self):
        path = Path(__file__).resolve().parents[2] / "scripts" / "issue_agent_session.py"
        spec = importlib.util.spec_from_file_location("issue_agent_session_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        (self.root / "runtime").mkdir()
        job = self.root / "job.json"
        job.write_text(json.dumps({"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}}))
        output = self.root / "runtime" / "session"
        arguments = [str(path), "--job", str(job), "--output", str(output), "--reviewed-public-task"]
        captured = io.StringIO()
        with patch.object(module, "ROOT", self.root), patch("sys.argv", arguments), contextlib.redirect_stdout(captured):
            with patch.object(module.httpx, "Client", return_value=self.operator):
                module.main()
            token = (output / "agent-token").read_text().strip()
            self.assertNotIn(token, captured.getvalue())
            self.assertEqual((output / "agent-token").stat().st_mode & 0o777, 0o600)
            with sqlite3.connect(self.root / "db.sqlite3") as connection:
                before = connection.execute("SELECT COUNT(*) FROM v2_episodes").fetchone()[0]
            module.main()
            with sqlite3.connect(self.root / "db.sqlite3") as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM v2_episodes").fetchone()[0], before)
            job.write_text(json.dumps({"task_ref": {"task_id": "other", "task_version": "1.0.0"}}))
            with self.assertRaises(SystemExit):
                module.main()
        pending = self.root / "runtime" / "pending"
        pending.mkdir()
        (pending / "reset-pending.json").write_text("{}")
        arguments[arguments.index(str(output))] = str(pending)
        with patch.object(module, "ROOT", self.root), patch("sys.argv", arguments), self.assertRaises(SystemExit):
            module.main()

    def test_content_hash_mismatch_is_not_returned_to_agent(self):
        body = self.step("eo_gym.crop", {"asset_id": self.binding.task.input_asset_refs[0], "aoi": [0, 0, .5, .5]})
        result = self.client.post("/agent/step", json=body).json()
        ref = result["observation"]["items"][1]["artifact_ref"]
        backend = self.backend

        class Tamper(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                if request.url.path.endswith("/content"):
                    return httpx.Response(200, content=b"SECRET-corrupt-image")
                return await httpx.ASGITransport(app=backend).handle_async_request(request)

        with self.make_client(Tamper()) as client:
            response = client.get("/agent/artifacts/" + ref + "/content")
            self.assertEqual(response.status_code, 502, response.text)
            self.assertEqual(response.json()["error"]["code"], "artifact_checksum_mismatch")
            self.assertNotIn("SECRET", response.text)


class AgentGuardConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_slots_are_held_until_response_finishes(self):
        active = 0
        entered = asyncio.Event()
        release = asyncio.Event()

        async def app(scope, receive, send):
            nonlocal active
            active += 1
            if active == 4:
                entered.set()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await release.wait()
            await send({"type": "http.response.body", "body": b"ok"})

        guard = AgentGuard(app, hashlib.sha256(TOKEN.encode()).hexdigest())
        scope = {"type": "http", "path": "/agent/state", "method": "GET", "query_string": b"",
                 "headers": [(b"authorization", ("Bearer " + TOKEN).encode())]}

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            pass

        requests = [asyncio.create_task(guard(scope, receive, send)) for _ in range(5)]
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.sleep(0)
            self.assertEqual(active, 4)
        finally:
            release.set()
            await asyncio.gather(*requests)
        self.assertEqual(active, 5)

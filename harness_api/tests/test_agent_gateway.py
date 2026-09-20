import copy
import asyncio
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import ssl
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from v2.test_tool_execution import FakeExecutor, make_tool_tasks
from app.main import create_app as create_backend
from app.agent_credentials import (AgentCredentialRegistry, AgentSessionCredential,
                                   load_agent_registry)
from app.agent_gateway import (AgentBinding, AgentGuard, build_backend_ssl_context,
                               TOOL_ARGUMENTS, build_binding, create_app,
                               public_observation, public_state, validate_agent_binding)
from app.control_plane import AgentIssuancePolicy, verify_control_audit
from app.v2.artifacts import ArtifactStore
from app.v2.capabilities import TaskRegistry
from app.v2.schemas import V2EpisodeState
from app.v2.tools.runtime import ToolRouter, ToolOutput

TOKEN = "1" * 64  # Isolated test credential, never deployed.
TOKEN2 = "2" * 64
CERTIFICATE_SHA = "3" * 64
OPERATOR_CERTIFICATE_DER = b"operator-cert"
OPERATOR_CERTIFICATE_SHA = hashlib.sha256(OPERATOR_CERTIFICATE_DER).hexdigest()
OPERATOR_CERTIFICATE_PEM = (
    b"-----BEGIN CERTIFICATE-----\n"
    b"b3BlcmF0b3ItY2VydA==\n"
    b"-----END CERTIFICATE-----\n"
)


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

    def credential(self, binding=None, status="active", generation=1):
        issued = datetime.now(timezone.utc) - timedelta(minutes=1)
        return AgentSessionCredential(binding=binding or self.binding, issued_at=issued,
            expires_at=issued + timedelta(hours=1), status=status,
            revoked_at=issued if status == "revoked" else None, generation=generation)

    def write_registry(self, path, sessions):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_text(AgentCredentialRegistry(sessions=sessions).model_dump_json(indent=2))
        path.chmod(0o600)

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

    def test_memory_schema_and_results_are_strictly_public(self):
        self.assertIn("memory.search", TOOL_ARGUMENTS)
        self.assertNotIn("episode_id", TOOL_ARGUMENTS["memory.search"].model_json_schema()["properties"])
        task = self.binding.task.model_copy(
            update={"allowed_tools": [*self.binding.task.allowed_tools, "memory.search"]}
        )
        binding = self.binding.model_copy(update={"task": task})
        observation_id = self.binding.episode_id.replace("ep2-", "obs-")
        value = {
            "observation_id": observation_id,
            "sequence": 1,
            "primary_type": "tool_result",
            "items": [
                {
                    "type": "tool_result",
                    "inline": {
                        "tool_id": "memory.search",
                        "tool_version": "1.0.0",
                        "status": "completed",
                        "records": [
                            {
                                "memory_id": "mem-" + "a" * 64,
                                "object_type": "land-cover-assessment",
                                "public_summary": "Reviewed built-up evidence.",
                                "bbox": {"west": 120.5, "south": 30.5, "east": 121.0, "north": 31.0},
                                "time_range": {"start": "2021-01-01T00:00:00Z", "end": "2021-12-31T23:59:59Z"},
                                "platform": "ESA WorldCover",
                                "instrument": "WorldCover-map",
                                "source_task": {"task_id": "worldcover-grounded-vqa", "task_version": "1.1.0"},
                                "source_sha256": "b" * 64,
                                "provenance_sha256": "c" * 64,
                                "available_at": "2026-08-01T00:00:00Z",
                                "expires_at": "2026-08-15T00:00:00Z",
                            }
                        ],
                        "matched_count": 1,
                        "next_offset": None,
                        "snapshot_sequence": 1,
                        "snapshot_sha256": "d" * 64,
                        "cost": {"model": "logical-evidence-memory-v1", "records_scanned": 1, "input_bytes": 1024},
                        "episode_id": "SECRET-episode",
                        "source_ref": "/private/SECRET-source.tif",
                    },
                }
            ],
            "state_hash": "e" * 64,
            "semantic_state_hash": "f" * 64,
            "provenance": {"path": "/private/SECRET"},
            "warnings": ["SECRET"],
        }
        public = public_observation(value, binding)
        encoded = json.dumps(public)
        self.assertIn("memory.search", encoded)
        self.assertIn("Reviewed built-up evidence", encoded)
        self.assertNotIn("SECRET", encoded)
        self.assertNotIn("source_ref", encoded)

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
            validate_agent_binding(AgentBinding.model_validate(value))

    def test_two_registry_tokens_resolve_to_separate_episode_scopes(self):
        reset = self.operator.post("/v2/reset", json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}}).json()["data"]
        other = build_binding(self.manifest, V2EpisodeState.model_validate(reset["state"]),
            self.operator.get("/v2/capabilities").json()["data"], hashlib.sha256(TOKEN2.encode()).hexdigest())
        registry = AgentCredentialRegistry(sessions=[self.credential(), self.credential(other)])
        app = create_app(None, "http://operator", httpx.ASGITransport(app=self.backend), registry=registry)
        with TestClient(app, headers={"Authorization": "Bearer " + TOKEN}) as first, \
                TestClient(app, headers={"Authorization": "Bearer " + TOKEN2}) as second:
            self.assertEqual(first.get("/agent/state").json()["state"]["episode_id"], self.episode)
            self.assertEqual(second.get("/agent/state").json()["state"]["episode_id"], other.episode_id)
            other_observation = second.get("/agent/session").json()["observation"]["observation_id"]
            self.assertIn(first.get("/agent/observations/" + other_observation).status_code, (403, 404))
            body = self.step(action_id="session-a")
            self.assertEqual(first.post("/agent/step", json=body).status_code, 200)
            self.assertEqual(second.get("/agent/state").json()["state"]["state_version"], 0)

    def test_certificate_bound_session_requires_trusted_matching_mtls_identity(self):
        credential = self.credential().model_copy(update={
            "issuer_id": "trusted-operator",
            "subject_id": "approved-runner",
            "issuance_policy_id": "certificate-policy-v1",
            "issuance_policy_sha256": "4" * 64,
            "subject_certificate_sha256": CERTIFICATE_SHA,
        })
        registry = AgentCredentialRegistry(
            schema_version="1.2.0", sessions=[credential]
        )
        unavailable = create_app(
            None,
            "http://operator",
            httpx.ASGITransport(app=self.backend),
            registry=registry,
            trusted_mtls_header=False,
        )
        with TestClient(
            unavailable, headers={"Authorization": "Bearer " + TOKEN}
        ) as client:
            response = client.get("/agent/state")
            self.assertEqual(
                (response.status_code, response.json()["error"]["code"]),
                (503, "client_identity_unavailable"),
            )

        def certificate_digest(value):
            if value == "authorized-certificate":
                return CERTIFICATE_SHA
            if value == "other-certificate":
                return "5" * 64
            raise ValueError("missing certificate")

        app = create_app(
            None,
            "http://operator",
            httpx.ASGITransport(app=self.backend),
            registry=registry,
            trusted_mtls_header=True,
        )
        with patch(
            "app.agent_gateway.client_certificate_sha256",
            side_effect=certificate_digest,
        ), TestClient(
            app, headers={"Authorization": "Bearer " + TOKEN}
        ) as client:
            response = client.get("/agent/state")
            self.assertEqual(
                (response.status_code, response.json()["error"]["code"]),
                (401, "client_identity_required"),
            )
            response = client.get(
                "/agent/state",
                headers={"x-eo-client-cert": "other-certificate"},
            )
            self.assertEqual(
                (response.status_code, response.json()["error"]["code"]),
                (401, "client_identity_mismatch"),
            )
            response = client.get(
                "/agent/state",
                headers={"x-eo-client-cert": "authorized-certificate"},
            )
            self.assertEqual(response.status_code, 200, response.text)

    def test_backend_mtls_configuration_is_explicit_and_fail_closed(self):
        context = ssl.create_default_context()
        with self.assertRaisesRegex(ValueError, "HTTPS origin"):
            create_app(
                self.binding,
                "http://operator",
                httpx.ASGITransport(app=self.backend),
                require_backend_mtls=True,
                backend_ssl_context=context,
            )
        with patch.dict(os.environ, {
            "EO_AGENT_BACKEND_CA_FILE": "",
            "EO_AGENT_BACKEND_CERT_FILE": "",
            "EO_AGENT_BACKEND_KEY_FILE": "",
        }):
            with self.assertRaisesRegex(ValueError, "backend CA is required"):
                create_app(
                    self.binding,
                    "https://harness:8443",
                    httpx.ASGITransport(app=self.backend),
                    require_backend_mtls=True,
                )
        with patch.dict(os.environ, {"EO_AGENT_REQUIRE_BACKEND_MTLS": "invalid"}):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                create_app(
                    self.binding,
                    "https://harness:8443",
                    httpx.ASGITransport(app=self.backend),
                )
        for name in ("ca.crt", "gateway.crt", "gateway.key"):
            (self.root / name).write_text("not-a-certificate")
        (self.root / "gateway.key").chmod(0o644)
        with self.assertRaisesRegex(ValueError, "owner-private"):
            build_backend_ssl_context(
                str(self.root / "ca.crt"),
                str(self.root / "gateway.crt"),
                str(self.root / "gateway.key"),
            )
        app = create_app(
            self.binding,
            "https://harness:8443",
            httpx.ASGITransport(app=self.backend),
            require_backend_mtls=True,
            backend_ssl_context=context,
        )
        with TestClient(
            app, headers={"Authorization": "Bearer " + TOKEN}
        ) as client:
            response = client.get("/agent/state")
            self.assertEqual(response.status_code, 200, response.text)

    def test_agent_backend_interface_denies_operator_routes(self):
        app = create_backend(
            database_path=str(self.root / "db.sqlite3"),
            v2_tasks_path=str(self.tasks),
            v2_artifacts_path=str(self.root / "artifacts"),
            v2_tool_executor=ToolRouter(self.executor),
            interface_role="agent-backend",
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            self.assertEqual(client.get("/healthz").status_code, 200)
            self.assertEqual(
                client.get(f"/v2/episodes/{self.episode}/state").status_code,
                200,
            )
            observation = self.operator.get(
                f"/v2/episodes/{self.episode}/state"
            ).json()["data"]["state"]["observation_refs"][-1]
            self.assertEqual(
                client.get(
                    f"/v2/episodes/{self.episode}/observations/{observation}"
                ).status_code,
                200,
            )
            self.assertEqual(
                client.post(
                    f"/v2/episodes/{self.episode}/step",
                    json={},
                ).status_code,
                422,
            )
            for path in (
                "/v2/capabilities",
                "/v2/openapi.json",
                "/v2/tasks/crop-smoke/versions/1.0.0",
                f"/v2/episodes/{self.episode}/trace",
                f"/v2/episodes/{self.episode}/evaluation",
                f"/v2/episodes/{self.episode}/replay",
                "/docs",
            ):
                self.assertEqual(client.get(path).status_code, 404, path)
            self.assertEqual(
                client.post(
                    "/v2/reset",
                    json={
                        "task_ref": {
                            "task_id": "crop-smoke",
                            "task_version": "1.0.0",
                        }
                    },
                ).status_code,
                404,
            )
        with self.assertRaisesRegex(ValueError, "interface role"):
            create_backend(
                database_path=str(self.root / "invalid-role.sqlite3"),
                interface_role="invalid",
            )

    def test_operator_backend_mtls_configuration_is_complete(self):
        path = Path(__file__).resolve().parents[2] / "scripts" / "issue_agent_session.py"
        spec = importlib.util.spec_from_file_location("operator_mtls_issue_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.operator_backend_verify(
            "http://harness:8000", None, None, None
        ), (True, None))
        with self.assertRaisesRegex(SystemExit, "CA, certificate and key"):
            module.operator_backend_verify(
                "https://operator-harness:8444",
                self.root / "ca.crt",
                None,
                None,
            )
        with self.assertRaisesRegex(SystemExit, "HTTPS origin"):
            module.operator_backend_verify(
                "http://operator-harness:8444",
                self.root / "ca.crt",
                self.root / "operator.crt",
                self.root / "operator.key",
            )
        context = ssl.create_default_context()
        certificate = self.root / "operator.crt"
        certificate.write_bytes(OPERATOR_CERTIFICATE_PEM)
        with patch.object(
            module,
            "build_backend_ssl_context",
            return_value=context,
        ) as builder:
            verify, identity = module.operator_backend_verify(
                    "https://operator-harness:8444",
                    self.root / "ca.crt",
                    certificate,
                    self.root / "operator.key",
                )
            self.assertIs(verify, context)
            self.assertEqual(identity, OPERATOR_CERTIFICATE_SHA)
            builder.assert_called_once()

    def test_registry_expiry_revocation_rotation_and_restart_fail_closed(self):
        path = self.root / "credentials" / "registry.json"
        self.write_registry(path, [self.credential()])
        app = create_app(None, "http://operator", httpx.ASGITransport(app=self.backend), registry_path=path)
        with TestClient(app, headers={"Authorization": "Bearer " + TOKEN}) as client:
            self.assertEqual(client.get("/agent/state").status_code, 200)
            expired = self.credential().model_copy(update={
                "issued_at": datetime.now(timezone.utc) - timedelta(hours=2),
                "expires_at": datetime.now(timezone.utc) - timedelta(hours=1)})
            self.write_registry(path, [expired])
            response = client.post("/agent/step", content="x" * 140000)
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (401, "session_expired"))
            self.write_registry(path, [self.credential(status="revoked")])
            response = client.get("/agent/state")
            self.assertEqual((response.status_code, response.json()["error"]["code"]), (401, "session_revoked"))
            rotated = self.binding.model_copy(update={"token_sha256": hashlib.sha256(TOKEN2.encode()).hexdigest()})
            self.write_registry(path, [self.credential(rotated, generation=2)])
            self.assertEqual(client.get("/agent/state").status_code, 401)
            response = client.get("/agent/state", headers={"Authorization": "Bearer " + TOKEN2})
            self.assertEqual(response.status_code, 200, response.text)
        with TestClient(create_app(None, "http://operator", httpx.ASGITransport(app=self.backend), registry_path=path),
                        headers={"Authorization": "Bearer " + TOKEN2}) as restarted:
            self.assertEqual(restarted.get("/agent/state").status_code, 200)
            path.write_text("{")
            response = restarted.get("/agent/state")
            self.assertEqual((response.status_code, response.json()["error"]["code"]),
                             (503, "credential_registry_unavailable"))
            self.assertEqual(restarted.get("/healthz", headers={"Authorization": ""}).status_code, 503)

    def test_registry_rejects_duplicate_scope_insecure_mode_and_symlink(self):
        with self.assertRaises(ValueError):
            AgentCredentialRegistry(sessions=[self.credential(), self.credential()])
        path = self.root / "credentials" / "registry.json"
        self.write_registry(path, [self.credential()])
        path.chmod(0o644)
        with self.assertRaises(ValueError):
            create_app(None, "http://operator", httpx.ASGITransport(app=self.backend), registry_path=path)
        path.chmod(0o600)
        link = self.root / "credentials-link.json"
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            create_app(None, "http://operator", httpx.ASGITransport(app=self.backend), registry_path=link)

    def test_operator_registry_management_revokes_and_rotates_without_printing_token(self):
        registry = self.root / "runtime" / "credentials" / "registry.json"
        self.write_registry(registry, [self.credential()])
        script = Path(__file__).resolve().parents[2] / "scripts" / "manage_agent_registry.py"
        spec = importlib.util.spec_from_file_location("manage_agent_registry_test", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        app = create_app(None, "http://operator", httpx.ASGITransport(app=self.backend), registry_path=registry)
        with TestClient(app, headers={"Authorization": "Bearer " + TOKEN}) as client:
            captured = io.StringIO()
            arguments = [str(script), "--registry", str(registry),
                         "--episode-id", self.episode, "--revoke"]
            with patch.object(module, "ROOT", self.root), patch("sys.argv", arguments), \
                    contextlib.redirect_stdout(captured):
                module.main()
            self.assertEqual(client.get("/agent/state").json()["error"]["code"], "session_revoked")
            token_output = self.root / "runtime" / "rotation-2" / "agent-token"
            arguments = [str(script), "--registry", str(registry),
                         "--episode-id", self.episode, "--rotate",
                         "--token-output", str(token_output), "--ttl-seconds", "3600"]
            with patch.object(module, "ROOT", self.root), patch("sys.argv", arguments), \
                    contextlib.redirect_stdout(captured):
                module.main()
            new_token = token_output.read_text().strip()
            self.assertNotIn(new_token, captured.getvalue())
            self.assertEqual(token_output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(client.get("/agent/state").status_code, 401)
            self.assertEqual(client.get("/agent/state", headers={
                "Authorization": "Bearer " + new_token}).status_code, 200)
            current = load_agent_registry(registry).sessions[0]
            self.assertEqual((current.status, current.generation), ("active", 2))

    def test_issuance_is_private_and_rerun_does_not_create_another_episode(self):
        path = Path(__file__).resolve().parents[2] / "scripts" / "issue_agent_session.py"
        spec = importlib.util.spec_from_file_location("issue_agent_session_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        (self.root / "runtime").mkdir()
        job = self.root / "job.json"
        job.write_text(json.dumps({"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}}))
        output = self.root / "runtime" / "session"
        registry = self.root / "runtime" / "credentials" / "registry.json"
        arguments = [str(path), "--job", str(job), "--output", str(output),
                     "--registry", str(registry), "--ttl-seconds", "3600",
                     "--reviewed-public-task"]
        captured = io.StringIO()
        with patch.object(module, "ROOT", self.root), patch("sys.argv", arguments), contextlib.redirect_stdout(captured):
            with patch.object(module.httpx, "Client", return_value=self.operator):
                module.main()
            token = (output / "agent-token").read_text().strip()
            issued_binding = AgentBinding.model_validate_json((output / "binding.json").read_bytes())
            self.assertNotIn(token, captured.getvalue())
            self.assertNotIn(token, registry.read_text())
            self.assertEqual(load_agent_registry(registry).sessions[0].binding.episode_id,
                             issued_binding.episode_id)
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

    def test_governed_issuance_rotation_revocation_are_policy_pinned_and_audited(self):
        issue_path = Path(__file__).resolve().parents[2] / "scripts" / "issue_agent_session.py"
        issue_spec = importlib.util.spec_from_file_location("governed_issue_test", issue_path)
        issue = importlib.util.module_from_spec(issue_spec)
        issue_spec.loader.exec_module(issue)
        manage_path = Path(__file__).resolve().parents[2] / "scripts" / "manage_agent_registry.py"
        manage_spec = importlib.util.spec_from_file_location("governed_manage_test", manage_path)
        manage = importlib.util.module_from_spec(manage_spec)
        manage_spec.loader.exec_module(manage)
        (self.root / "runtime").mkdir()
        job = self.root / "job.json"
        job.write_text(json.dumps({"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}}))
        current = datetime.now(timezone.utc)
        operator_certificate = self.root / "operator.crt"
        operator_certificate.write_bytes(OPERATOR_CERTIFICATE_PEM)
        backend_ca = self.root / "operator-ca.crt"
        backend_key = self.root / "operator.key"
        backend_ca.write_text("test-only")
        backend_key.write_text("test-only")
        policy = AgentIssuancePolicy(
            schema_version="1.2.0",
            policy_id="test-research-policy",
            valid_from=current - timedelta(days=1),
            expires_at=current + timedelta(days=1),
            issuers=["trusted-operator"],
            subjects=["approved-runner"],
            issuer_certificates=[{
                "issuer_id": "trusted-operator",
                "certificate_sha256": OPERATOR_CERTIFICATE_SHA,
            }],
            subject_certificates=[{
                "subject_id": "approved-runner",
                "certificate_sha256": CERTIFICATE_SHA,
            }],
            grants=[{"task_id": "crop-smoke", "task_version": "1.0.0",
                     "max_ttl_seconds": 3600}],
            max_active_sessions_per_subject=1,
        )
        policy_path = self.root / "policy.json"
        policy_bytes = policy.model_dump_json(indent=2).encode()
        policy_path.write_bytes(policy_bytes)
        policy_sha = hashlib.sha256(policy_bytes).hexdigest()
        output = self.root / "runtime" / "governed-session"
        registry = self.root / "runtime" / "governed-registry" / "registry.json"
        audit = self.root / "runtime" / "governed-audit" / "events.jsonl"
        governance = ["--issuance-policy", str(policy_path),
                      "--issuance-policy-sha256", policy_sha,
                      "--actor-id", "trusted-operator",
                      "--subject-id", "approved-runner",
                      "--subject-certificate-sha256", CERTIFICATE_SHA,
                      "--audit-log", str(audit)]
        issue_common = [
            "--backend", "https://operator-harness:8444",
            "--backend-ca-file", str(backend_ca),
            "--backend-certificate-file", str(operator_certificate),
            "--backend-key-file", str(backend_key),
            *governance,
        ]
        manage_common = [
            *governance,
            "--actor-certificate-file", str(operator_certificate),
        ]
        arguments = [str(issue_path), "--job", str(job), "--output", str(output),
                     "--registry", str(registry), "--ttl-seconds", "3600",
                     "--reviewed-public-task", *issue_common]
        with patch.object(issue, "ROOT", self.root), patch("sys.argv", arguments), \
                patch.object(issue, "build_backend_ssl_context", return_value=True), \
                patch.object(issue.httpx, "Client", return_value=self.operator):
            issue.main()
            token = (output / "agent-token").read_text().strip()
            issue.main()
        governed = load_agent_registry(registry)
        self.assertEqual(governed.schema_version, "1.3.0")
        self.assertEqual(
            (governed.sessions[0].issuer_id, governed.sessions[0].subject_id),
            ("trusted-operator", "approved-runner"),
        )
        self.assertEqual(
            governed.sessions[0].subject_certificate_sha256,
            CERTIFICATE_SHA,
        )
        self.assertEqual(
            governed.sessions[0].issuer_certificate_sha256,
            OPERATOR_CERTIFICATE_SHA,
        )
        self.assertEqual(
            [item.event_type for item in verify_control_audit(audit)],
            ["issuance_started", "issuance_completed", "binding_reused"],
        )
        events = verify_control_audit(audit)
        self.assertEqual(events[0].operation_id, events[1].operation_id)
        self.assertNotEqual(events[1].operation_id, events[2].operation_id)
        second_binding = governed.sessions[0].binding.model_copy(
            update={
                "episode_id": "ep2-" + "f" * 32,
                "token_sha256": "e" * 64,
            }
        )
        with self.assertRaisesRegex(SystemExit, "active-session limit"):
            issue.publish_binding(
                self.root / "runtime",
                registry,
                second_binding,
                3600,
                {
                    "policy": policy,
                    "policy_sha256": policy_sha,
                    "actor_id": "trusted-operator",
                    "actor_certificate_sha256": OPERATOR_CERTIFICATE_SHA,
                    "subject_id": "approved-runner",
                    "subject_certificate_sha256": CERTIFICATE_SHA,
                },
            )
        self.assertEqual(len(load_agent_registry(registry).sessions), 1)
        rotated_token = self.root / "runtime" / "governed-rotation" / "agent-token"
        arguments = [str(manage_path), "--registry", str(registry),
                     "--episode-id", governed.sessions[0].binding.episode_id,
                     "--rotate", "--token-output", str(rotated_token),
                     *manage_common]
        with patch.object(manage, "ROOT", self.root), patch("sys.argv", arguments):
            manage.main()
        arguments = [str(manage_path), "--registry", str(registry),
                     "--episode-id", governed.sessions[0].binding.episode_id,
                     "--revoke", *manage_common]
        with patch.object(manage, "ROOT", self.root), patch("sys.argv", arguments):
            manage.main()
        events = verify_control_audit(audit)
        self.assertEqual(
            [item.event_type for item in events],
            ["issuance_started", "issuance_completed", "binding_reused",
             "rotation_started", "rotation_completed",
             "revocation_started", "revocation_completed"],
        )
        self.assertEqual(events[3].operation_id, events[4].operation_id)
        self.assertEqual(events[5].operation_id, events[6].operation_id)
        audit_content = audit.read_text()
        self.assertNotIn(token, audit_content)
        self.assertNotIn(rotated_token.read_text().strip(), audit_content)
        self.assertTrue(all(
            item.actor_certificate_sha256 == OPERATOR_CERTIFICATE_SHA
            for item in events
        ))
        final = load_agent_registry(registry).sessions[0]
        self.assertEqual((final.status, final.generation), ("revoked", 2))

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

        class Resolver:
            def resolve_with_certificate(self, token_sha256):
                return None, None

        guard = AgentGuard(app, Resolver())
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

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.v2.api import build_openapi_schema

from .helpers import (
    ANSWER_REQUEST,
    EVIDENCE_REQUEST,
    PROJECT_ROOT,
    RESET_REQUEST,
    ZOOM_REQUEST,
    make_app,
    normalize_dynamic,
    read_fixture,
)


class V2ContractTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        database = Path(self.tempdir.name) / "episodes.sqlite3"
        self.application = make_app(database)
        self.client = TestClient(self.application, raise_server_exceptions=False)

    def tearDown(self):
        self.client.close()
        self.tempdir.cleanup()

    def test_openapi_and_golden_fixtures(self):
        capabilities = self.client.get(
            "/v2/capabilities",
            headers={"X-Request-ID": "req-v2-capabilities"},
        )
        task = self.client.get(
            "/v2/tasks/worldcover-grounded-vqa/versions/1.0.0",
            headers={"X-Request-ID": "req-v2-task"},
        )
        reset = self.client.post(
            "/v2/reset",
            headers={"X-Request-ID": "req-v2-reset"},
            json=RESET_REQUEST,
        )
        self.assertEqual(reset.status_code, 201, reset.text)
        episode_id = reset.json()["data"]["episode_id"]

        zoom = self.client.post(
            "/v2/episodes/%s/step" % episode_id,
            headers={"X-Request-ID": "req-v2-zoom"},
            json=ZOOM_REQUEST,
        )
        evidence = self.client.post(
            "/v2/episodes/%s/step" % episode_id,
            headers={"X-Request-ID": "req-v2-evidence"},
            json=EVIDENCE_REQUEST,
        )
        answer = self.client.post(
            "/v2/episodes/%s/step" % episode_id,
            headers={"X-Request-ID": "req-v2-answer"},
            json=ANSWER_REQUEST,
        )
        state = self.client.get(
            "/v2/episodes/%s/state" % episode_id,
            headers={"X-Request-ID": "req-v2-state"},
        )
        trace = self.client.get(
            "/v2/episodes/%s/trace" % episode_id,
            headers={"X-Request-ID": "req-v2-trace"},
        )
        replay = self.client.post(
            "/v2/episodes/%s/replay" % episode_id,
            headers={"X-Request-ID": "req-v2-replay"},
        )
        invalid = self.client.post(
            "/v2/episodes/%s/step" % episode_id,
            headers={"X-Request-ID": "req-v2-error"},
            json={
                "client_action_id": "invalid-1",
                "expected_state_version": 3,
                "action": {
                    "type": "map.zoom",
                    "direction": "in",
                    "factor": "2.0",
                },
            },
        )

        responses = {
            "capabilities-response.json": capabilities,
            "task-response.json": task,
            "reset-response.json": reset,
            "zoom-step-response.json": zoom,
            "evidence-step-response.json": evidence,
            "answer-step-response.json": answer,
            "state-response.json": state,
            "trace-response.json": trace,
            "replay-response.json": replay,
            "validation-error-response.json": invalid,
        }
        for name, response in responses.items():
            expected_status = 422 if name == "validation-error-response.json" else 200
            if name == "reset-response.json":
                expected_status = 201
            self.assertEqual(response.status_code, expected_status, response.text)
            actual = normalize_dynamic(response.json())
            expected = read_fixture(name)
            # Request IDs are route-specific operator inputs; the canonical fixture
            # comparison covers body semantics rather than transient routing IDs.
            actual["meta"]["request_id"] = expected["meta"]["request_id"]
            self.assertEqual(actual, expected, name)

        frozen_path = PROJECT_ROOT / "contracts" / "v2" / "openapi-v2.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        self.assertEqual(build_openapi_schema(), frozen)

    def test_v1_openapi_remains_frozen(self):
        frozen_path = PROJECT_ROOT / "contracts" / "openapi-v1.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        self.assertEqual(self.application.openapi(), frozen)

    def test_v2_error_shape_and_headers(self):
        response = self.client.post(
            "/v2/reset",
            headers={"X-Request-ID": "req-v2-invalid"},
            json={"task_ref": {"task_id": "missing"}},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(set(response.json()), {"meta", "error"})
        self.assertEqual(response.json()["meta"]["api_version"], "v2")
        self.assertEqual(response.json()["meta"]["schema_version"], "2.0.0")
        self.assertEqual(response.json()["error"]["phase"], "request")
        self.assertFalse(response.json()["error"]["retryable"])
        self.assertEqual(response.headers["X-EO-Harness-API-Version"], "v2")
        self.assertEqual(response.headers["X-EO-Harness-Schema-Version"], "2.0.0")

    def test_operator_can_disable_v2_without_migrating_or_breaking_v1(self):
        database = Path(self.tempdir.name) / "v1-only.sqlite3"
        application = make_app(database)
        application.state.v2_store = None
        with TestClient(application, raise_server_exceptions=False) as client:
            v1 = client.get("/v1/action-space")
            disabled = client.get("/v2/capabilities")
        self.assertEqual(v1.status_code, 200, v1.text)
        self.assertEqual(disabled.status_code, 503, disabled.text)
        self.assertEqual(disabled.json()["error"]["code"], "v2_disabled")

        no_migration_database = Path(self.tempdir.name) / "no-v2-migration.sqlite3"
        from app.main import create_app

        disabled_application = create_app(
            database_path=str(no_migration_database),
            v2_enabled=False,
        )
        with TestClient(
            disabled_application,
            raise_server_exceptions=False,
        ) as client:
            disabled_response = client.get("/v2/capabilities")
            v1_response = client.get("/v1/action-space")
        self.assertEqual(v1_response.status_code, 200, v1_response.text)
        self.assertEqual(disabled_response.status_code, 503, disabled_response.text)
        self.assertEqual(disabled_response.json()["error"]["code"], "v2_disabled")
        with sqlite3.connect(no_migration_database) as connection:
            names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        self.assertNotIn("v2_episodes", names)


if __name__ == "__main__":
    unittest.main()

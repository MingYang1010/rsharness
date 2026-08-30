import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HARNESS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(HARNESS_ROOT))
os.environ.setdefault(
    "EO_HARNESS_DB",
    str(Path(tempfile.gettempdir()) / "eo-harness-test-import.sqlite3"),
)

from fastapi.testclient import TestClient

from app.domain import canonical_json
from app.main import create_app
from scripts.export_contracts import (
    RESET_REQUEST,
    STEP_REQUEST,
    normalize_dynamic,
)


class HttpContractTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Path(self.tempdir.name) / "episodes.sqlite3"
        self.application = create_app(str(self.database))
        self.client = TestClient(self.application, raise_server_exceptions=False)

    def tearDown(self):
        self.client.close()
        self.tempdir.cleanup()

    def reset(self, request_id="req-test-reset", body=None):
        response = self.client.post(
            "/v1/reset",
            headers={"X-Request-ID": request_id},
            json=body or RESET_REQUEST,
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response

    def fixture(self, name):
        path = PROJECT_ROOT / "contracts" / "fixtures" / name
        return json.loads(path.read_text(encoding="utf-8"))

    def assert_error(self, response, status_code, code):
        self.assertEqual(response.status_code, status_code, response.text)
        payload = response.json()
        self.assertEqual(set(payload), {"meta", "error"})
        self.assertEqual(payload["meta"]["api_version"], "v1")
        self.assertEqual(payload["meta"]["schema_version"], "1.0.0")
        self.assertEqual(payload["error"]["code"], code)
        self.assertIn("details", payload["error"])

    def test_golden_http_contracts_and_openapi_snapshot(self):
        reset = self.reset("req-contract-reset")
        episode_id = reset.json()["data"]["episode_id"]
        step = self.client.post(
            "/v1/episodes/%s/step" % episode_id,
            headers={"X-Request-ID": "req-contract-step"},
            json=STEP_REQUEST,
        )
        state = self.client.get(
            "/v1/episodes/%s/state" % episode_id,
            headers={"X-Request-ID": "req-contract-state"},
        )
        trace = self.client.get(
            "/v1/episodes/%s/trace" % episode_id,
            headers={"X-Request-ID": "req-contract-trace"},
        )
        action_space = self.client.get(
            "/v1/action-space",
            headers={"X-Request-ID": "req-contract-actions"},
        )
        validation_error = self.client.post(
            "/v1/episodes/%s/step" % episode_id,
            headers={"X-Request-ID": "req-contract-error"},
            json={
                "action": {
                    "type": "zoom",
                    "direction": "in",
                    "factor": "2.0",
                }
            },
        )

        for response in (step, state, trace, action_space):
            self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(validation_error.status_code, 422, validation_error.text)

        actual = {
            "reset-response.json": reset.json(),
            "step-response.json": step.json(),
            "state-response.json": state.json(),
            "trace-response.json": trace.json(),
            "action-space-response.json": action_space.json(),
            "validation-error-response.json": validation_error.json(),
        }
        for name, payload in actual.items():
            self.assertEqual(normalize_dynamic(payload), self.fixture(name), name)

        openapi_path = PROJECT_ROOT / "contracts" / "openapi-v1.json"
        expected_openapi = json.loads(openapi_path.read_text(encoding="utf-8"))
        self.assertEqual(self.application.openapi(), expected_openapi)

    def test_errors_share_one_envelope(self):
        invalid = self.client.post(
            "/v1/reset",
            headers={"X-Request-ID": "req-invalid"},
            json={"max_steps": 0, "unexpected": True},
        )
        self.assert_error(invalid, 422, "validation_error")
        self.assertEqual(invalid.json()["meta"]["request_id"], "req-invalid")
        self.assertGreaterEqual(len(invalid.json()["error"]["details"]), 2)

        oversized = self.client.post(
            "/v1/reset",
            headers={"X-Request-ID": "req-oversized"},
            json={"metadata": {"text": "界" * 22000}},
        )
        self.assert_error(oversized, 422, "validation_error")

        missing = self.client.get(
            "/v1/episodes/ep-00000000000000000000000000000000/state",
            headers={"X-Request-ID": "req-missing"},
        )
        self.assert_error(missing, 404, "episode_not_found")

        route_missing = self.client.get(
            "/v1/not-a-route",
            headers={"X-Request-ID": "req-route-missing"},
        )
        self.assert_error(route_missing, 404, "not_found")

        with patch.object(
            self.application.state.episode_store,
            "health",
            side_effect=RuntimeError("contract test failure"),
        ):
            with self.assertLogs("app.main", level="ERROR"):
                internal = self.client.get(
                    "/healthz",
                    headers={"X-Request-ID": "req-internal"},
                )
        self.assert_error(internal, 500, "internal_error")
        self.assertNotIn("contract test failure", internal.text)

        reset = self.reset()
        episode_id = reset.json()["data"]["episode_id"]
        first = self.client.post(
            "/v1/episodes/%s/step" % episode_id,
            json=STEP_REQUEST,
        )
        self.assertEqual(first.status_code, 200, first.text)
        conflict = self.client.post(
            "/v1/episodes/%s/step" % episode_id,
            json={
                "client_action_id": STEP_REQUEST["client_action_id"],
                "action": {"type": "zoom", "direction": "out", "factor": 2.0},
            },
        )
        self.assert_error(conflict, 409, "idempotency_conflict")

    def test_request_id_and_idempotent_data_are_stable(self):
        reset = self.reset("caller-reset-1")
        self.assertEqual(reset.headers["X-Request-ID"], "caller-reset-1")
        self.assertEqual(reset.json()["meta"]["request_id"], "caller-reset-1")
        episode_id = reset.json()["data"]["episode_id"]

        first = self.client.post(
            "/v1/episodes/%s/step" % episode_id,
            headers={"X-Request-ID": "caller-step-1"},
            json=STEP_REQUEST,
        )
        second = self.client.post(
            "/v1/episodes/%s/step" % episode_id,
            headers={"X-Request-ID": "caller-step-2"},
            json=STEP_REQUEST,
        )
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(first.json()["data"], second.json()["data"])
        self.assertNotEqual(first.json()["meta"], second.json()["meta"])

        trace = self.client.get("/v1/episodes/%s/trace" % episode_id)
        self.assertEqual(trace.json()["data"]["transition_count"], 1)

    def test_semantic_hash_matches_equivalent_episodes(self):
        first_id = self.reset("req-semantic-1").json()["data"]["episode_id"]
        second_id = self.reset("req-semantic-2").json()["data"]["episode_id"]
        for episode_id, action_id in (
            (first_id, "semantic-a"),
            (second_id, "semantic-b"),
        ):
            response = self.client.post(
                "/v1/episodes/%s/step" % episode_id,
                json={
                    "client_action_id": action_id,
                    "action": STEP_REQUEST["action"],
                },
            )
            self.assertEqual(response.status_code, 200, response.text)

        first = self.client.get("/v1/episodes/%s/trace" % first_id).json()["data"]
        second = self.client.get("/v1/episodes/%s/trace" % second_id).json()["data"]
        self.assertNotEqual(first["trace_hash"], second["trace_hash"])
        self.assertEqual(first["semantic_trace_hash"], second["semantic_trace_hash"])
        self.assertEqual(
            first["transitions"][0]["semantic_state_hash"],
            second["transitions"][0]["semantic_state_hash"],
        )

    def test_old_persisted_step_response_is_projected_to_v1(self):
        reset = self.reset()
        episode_id = reset.json()["data"]["episode_id"]
        first = self.client.post(
            "/v1/episodes/%s/step" % episode_id,
            json=STEP_REQUEST,
        )
        self.assertEqual(first.status_code, 200, first.text)

        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                "SELECT response_json FROM transitions WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            persisted = json.loads(row[0])
            persisted.pop("reward", None)
            persisted["observation"].pop("semantic_state_hash", None)
            persisted["info"].pop("semantic_state_hash", None)
            connection.execute(
                "UPDATE transitions SET response_json = ? WHERE episode_id = ?",
                (canonical_json(persisted), episode_id),
            )

        replay = self.client.post(
            "/v1/episodes/%s/step" % episode_id,
            json=STEP_REQUEST,
        )
        self.assertEqual(replay.status_code, 200, replay.text)
        data = replay.json()["data"]
        self.assertIsNone(data["reward"])
        self.assertEqual(len(data["info"]["semantic_state_hash"]), 64)
        self.assertEqual(len(data["observation"]["semantic_state_hash"]), 64)


if __name__ == "__main__":
    unittest.main()

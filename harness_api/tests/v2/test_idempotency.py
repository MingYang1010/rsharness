import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from .helpers import RESET_REQUEST, ZOOM_REQUEST, make_app


class V2IdempotencyTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.client = TestClient(
            make_app(Path(self.tempdir.name) / "episodes.sqlite3"),
            raise_server_exceptions=False,
        )
        reset = self.client.post("/v2/reset", json=RESET_REQUEST)
        self.episode_id = reset.json()["data"]["episode_id"]
        self.path = "/v2/episodes/%s/step" % self.episode_id

    def tearDown(self):
        self.client.close()
        self.tempdir.cleanup()

    def test_same_request_returns_identical_data_and_no_new_events(self):
        first = self.client.post(
            self.path,
            headers={"X-Request-ID": "request-a"},
            json=ZOOM_REQUEST,
        )
        second = self.client.post(
            self.path,
            headers={"X-Request-ID": "request-b"},
            json=ZOOM_REQUEST,
        )
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(first.json()["data"], second.json()["data"])
        self.assertNotEqual(first.json()["meta"], second.json()["meta"])
        trace = self.client.get(
            "/v2/episodes/%s/trace" % self.episode_id
        ).json()["data"]
        self.assertEqual(trace["total_events"], 5)

    def test_conflicting_reuse_returns_409(self):
        first = self.client.post(self.path, json=ZOOM_REQUEST)
        self.assertEqual(first.status_code, 200, first.text)
        changed = dict(ZOOM_REQUEST)
        changed["action"] = dict(ZOOM_REQUEST["action"])
        changed["action"]["direction"] = "out"
        conflict = self.client.post(self.path, json=changed)
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.json()["error"]["code"], "idempotency_conflict")

    def test_stale_expected_state_returns_retryable_409(self):
        first = self.client.post(self.path, json=ZOOM_REQUEST)
        self.assertEqual(first.status_code, 200, first.text)
        stale = {
            "client_action_id": "stale-2",
            "expected_state_version": 0,
            "action": {
                "type": "map.zoom",
                "direction": "out",
                "factor": 2.0,
            },
        }
        response = self.client.post(self.path, json=stale)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "state_version_conflict")
        self.assertTrue(response.json()["error"]["retryable"])
        self.assertEqual(response.json()["error"]["phase"], "state")


if __name__ == "__main__":
    unittest.main()

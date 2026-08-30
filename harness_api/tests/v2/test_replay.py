import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from .helpers import RESET_REQUEST, ZOOM_REQUEST, make_app


class V2ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.client = TestClient(
            make_app(Path(self.tempdir.name) / "episodes.sqlite3"),
            raise_server_exceptions=False,
        )

    def tearDown(self):
        self.client.close()
        self.tempdir.cleanup()

    def create_zoom_episode(self, action_id):
        reset = self.client.post("/v2/reset", json=RESET_REQUEST)
        episode_id = reset.json()["data"]["episode_id"]
        body = dict(ZOOM_REQUEST)
        body["client_action_id"] = action_id
        step = self.client.post(
            "/v2/episodes/%s/step" % episode_id,
            json=body,
        )
        self.assertEqual(step.status_code, 200, step.text)
        return episode_id

    def test_equivalent_episodes_have_equal_semantic_trace_hash(self):
        first_id = self.create_zoom_episode("zoom-instance-a")
        second_id = self.create_zoom_episode("zoom-instance-b")
        first = self.client.get(
            "/v2/episodes/%s/trace" % first_id
        ).json()["data"]
        second = self.client.get(
            "/v2/episodes/%s/trace" % second_id
        ).json()["data"]
        self.assertNotEqual(first["trace_hash"], second["trace_hash"])
        self.assertEqual(first["semantic_trace_hash"], second["semantic_trace_hash"])

    def test_cursor_pagination_is_stable_and_complete(self):
        episode_id = self.create_zoom_episode("zoom-pagination")
        cursor = None
        sequences = []
        hashes = set()
        while True:
            query = "?limit=2"
            if cursor is not None:
                query += "&cursor=%s" % cursor
            response = self.client.get(
                "/v2/episodes/%s/trace%s" % (episode_id, query)
            )
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()["data"]
            sequences.extend(event["sequence"] for event in data["events"])
            hashes.add((data["trace_hash"], data["semantic_trace_hash"]))
            if not data["has_more"]:
                break
            cursor = data["next_cursor"]
        self.assertEqual(sequences, list(range(5)))
        self.assertEqual(len(hashes), 1)

    def test_structural_replay_passes(self):
        episode_id = self.create_zoom_episode("zoom-replay")
        response = self.client.post(
            "/v2/episodes/%s/replay" % episode_id
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertEqual(data["status"], "passed")
        self.assertEqual(data["checked_event_count"], 5)
        self.assertTrue(all(item["passed"] for item in data["checks"]))


if __name__ == "__main__":
    unittest.main()

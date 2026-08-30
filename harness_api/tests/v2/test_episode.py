import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from .helpers import (
    ANSWER_REQUEST,
    EVIDENCE_REQUEST,
    RESET_REQUEST,
    ZOOM_REQUEST,
    make_app,
)


class V2EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.client = TestClient(
            make_app(Path(self.tempdir.name) / "episodes.sqlite3"),
            raise_server_exceptions=False,
        )
        reset = self.client.post("/v2/reset", json=RESET_REQUEST)
        self.assertEqual(reset.status_code, 201, reset.text)
        self.episode_id = reset.json()["data"]["episode_id"]

    def tearDown(self):
        self.client.close()
        self.tempdir.cleanup()

    def step(self, body):
        return self.client.post(
            "/v2/episodes/%s/step" % self.episode_id,
            json=body,
        )

    def test_worldcover_structural_episode(self):
        zoom = self.step(ZOOM_REQUEST)
        self.assertEqual(zoom.status_code, 200, zoom.text)
        self.assertEqual(zoom.json()["data"]["state"]["state_version"], 1)
        self.assertEqual(
            zoom.json()["data"]["observation"]["primary_type"],
            "map_state",
        )

        evidence = self.step(EVIDENCE_REQUEST)
        self.assertEqual(evidence.status_code, 200, evidence.text)
        self.assertEqual(len(evidence.json()["data"]["state"]["evidence_refs"]), 1)

        answer = self.step(ANSWER_REQUEST)
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertTrue(answer.json()["data"]["terminated"])
        self.assertFalse(answer.json()["data"]["truncated"])
        self.assertEqual(
            answer.json()["data"]["state"]["final_answer"]["outcome"],
            "submitted",
        )
        self.assertIsNone(answer.json()["data"]["state"]["evaluation"])

        closed = self.step(
            {
                "client_action_id": "after-close",
                "expected_state_version": 3,
                "action": {
                    "type": "map.pan",
                    "delta_longitude": 0.1,
                    "delta_latitude": 0.0,
                },
            }
        )
        self.assertEqual(closed.status_code, 409)
        self.assertEqual(closed.json()["error"]["code"], "episode_closed")

    def test_evidence_must_be_frozen_and_within_asset(self):
        invalid = dict(EVIDENCE_REQUEST)
        invalid["action"] = dict(EVIDENCE_REQUEST["action"])
        invalid["action"]["evidence"] = dict(
            EVIDENCE_REQUEST["action"]["evidence"]
        )
        invalid["action"]["evidence"]["frozen_sha256"] = "0" * 64
        invalid["expected_state_version"] = 0
        response = self.step(invalid)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "invalid_evidence")

    def test_declared_but_unimplemented_tool_is_policy_rejected(self):
        body = {
            "client_action_id": "tool-1",
            "expected_state_version": 0,
            "action": {
                "type": "tool.invoke",
                "tool_id": "raster.crop",
                "arguments": {},
            },
        }
        response = self.step(body)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "policy_rejected")
        self.assertEqual(response.json()["error"]["phase"], "policy")
        retry = self.step(body)
        self.assertEqual(retry.status_code, 403)
        self.assertEqual(retry.json()["error"], response.json()["error"])
        trace = self.client.get(
            "/v2/episodes/%s/trace" % self.episode_id
        ).json()["data"]
        self.assertEqual(trace["total_events"], 4)
        self.assertEqual(
            [event["event_type"] for event in trace["events"][-2:]],
            ["action.accepted", "action.failed"],
        )


if __name__ == "__main__":
    unittest.main()

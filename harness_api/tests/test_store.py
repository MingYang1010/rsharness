import tempfile
import unittest
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domain import DomainError
from app.store import EpisodeStore


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Path(self.tempdir.name) / "episodes.sqlite3"
        self.store = EpisodeStore(str(self.database))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_trace_persists_across_store_restart(self):
        reset = self.store.create_episode(
            {"task_id": "persist", "prompt": "Test persistence.", "max_steps": 5}
        )
        episode_id = reset["episode_id"]
        self.store.step(
            episode_id,
            {"type": "zoom", "direction": "in", "factor": 2.0},
            "action-1",
        )

        reopened = EpisodeStore(str(self.database))
        state = reopened.get_state(episode_id)
        trace = reopened.get_trace(episode_id)
        self.assertEqual(state["state"]["step_count"], 1)
        self.assertEqual(trace["transition_count"], 1)
        self.assertEqual(trace["transitions"][0]["client_action_id"], "action-1")

    def test_client_action_id_is_idempotent(self):
        reset = self.store.create_episode({"max_steps": 5})
        episode_id = reset["episode_id"]
        action = {"type": "zoom", "direction": "in", "factor": 2.0}
        first = self.store.step(episode_id, action, "retry-safe")
        second = self.store.step(episode_id, action, "retry-safe")
        self.assertEqual(first, second)
        self.assertEqual(list(first), list(second))
        self.assertEqual(self.store.get_trace(episode_id)["transition_count"], 1)

    def test_client_action_id_rejects_different_action(self):
        reset = self.store.create_episode({"max_steps": 5})
        episode_id = reset["episode_id"]
        self.store.step(
            episode_id,
            {"type": "zoom", "direction": "in", "factor": 2.0},
            "conflict",
        )
        with self.assertRaises(DomainError) as context:
            self.store.step(
                episode_id,
                {"type": "zoom", "direction": "out", "factor": 2.0},
                "conflict",
            )
        self.assertEqual(context.exception.status_code, 409)
        self.assertEqual(self.store.get_trace(episode_id)["transition_count"], 1)

    def test_unknown_episode_returns_not_found(self):
        with self.assertRaises(DomainError) as context:
            self.store.get_state("ep-missing")
        self.assertEqual(context.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()

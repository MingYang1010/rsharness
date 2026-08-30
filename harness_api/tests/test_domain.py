import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domain import DomainError, apply_action, create_initial_state, state_hash


class DomainTests(unittest.TestCase):
    def state(self, max_steps=20):
        return create_initial_state(
            "ep-test",
            {
                "task_id": "test-task",
                "prompt": "Inspect WorldCover.",
                "seed": 7,
                "max_steps": max_steps,
            },
        )

    def test_initial_state_is_active_and_hashable(self):
        state = self.state()
        self.assertEqual(state["status"], "active")
        self.assertEqual(state["step_count"], 0)
        self.assertEqual(len(state_hash(state)), 64)
        self.assertIn("esa-worldcover-2021", state["layers"])

    def test_zoom_in_and_out_are_inverse(self):
        state = self.state()
        initial = state["view"]["bbox"]
        zoomed, _ = apply_action(
            state, {"type": "zoom", "direction": "in", "factor": 2.0}
        )
        restored, _ = apply_action(
            zoomed, {"type": "zoom", "direction": "out", "factor": 2.0}
        )
        self.assertAlmostEqual(
            zoomed["view"]["bbox"]["east"] - zoomed["view"]["bbox"]["west"],
            (initial["east"] - initial["west"]) / 2.0,
        )
        for key, value in initial.items():
            self.assertAlmostEqual(restored["view"]["bbox"][key], value)

    def test_pan_preserves_span_and_clamps_to_world(self):
        state = self.state()
        initial = state["view"]["bbox"]
        panned, _ = apply_action(
            state,
            {"type": "pan", "delta_longitude": 300.0, "delta_latitude": 100.0},
        )
        bbox = panned["view"]["bbox"]
        self.assertEqual(bbox["east"], 180.0)
        self.assertEqual(bbox["north"], 90.0)
        self.assertAlmostEqual(
            bbox["east"] - bbox["west"], initial["east"] - initial["west"]
        )
        self.assertAlmostEqual(
            bbox["north"] - bbox["south"], initial["north"] - initial["south"]
        )

    def test_layer_visibility_and_opacity(self):
        state = self.state()
        hidden, observation = apply_action(
            state,
            {
                "type": "set_layer_visibility",
                "layer_id": "esa-worldcover-2021",
                "visible": False,
            },
        )
        self.assertEqual(observation["visible_layers"], [])
        changed, _ = apply_action(
            hidden,
            {
                "type": "set_layer_opacity",
                "layer_id": "esa-worldcover-2021",
                "opacity": 0.4,
            },
        )
        self.assertEqual(changed["layers"]["esa-worldcover-2021"]["opacity"], 0.4)

    def test_budget_truncates_and_rejects_later_steps(self):
        state = self.state(max_steps=2)
        state, _ = apply_action(
            state, {"type": "zoom", "direction": "in", "factor": 2.0}
        )
        state, _ = apply_action(
            state, {"type": "zoom", "direction": "out", "factor": 2.0}
        )
        self.assertEqual(state["status"], "truncated")
        with self.assertRaises(DomainError) as context:
            apply_action(
                state, {"type": "zoom", "direction": "in", "factor": 2.0}
            )
        self.assertEqual(context.exception.status_code, 409)

    def test_submit_answer_terminates_episode(self):
        state, observation = apply_action(
            self.state(),
            {
                "type": "submit_answer",
                "answer": "The active tile contains urban land cover.",
                "evidence_refs": ["layer://esa-worldcover-2021"],
            },
        )
        self.assertEqual(state["status"], "terminated")
        self.assertTrue(observation["message"].startswith("Answer submitted"))


if __name__ == "__main__":
    unittest.main()

import json
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx

RUNNER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_qwen_agent.py"
SPEC = importlib.util.spec_from_file_location("qwen_agent_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
artifact_refs = RUNNER.artifact_refs
content_message = RUNNER.content_message
decode_tool_arguments = RUNNER.decode_tool_arguments
run = RUNNER.run
MAX_IMAGE_BYTES = RUNNER.MAX_IMAGE_BYTES


ROOT = Path(__file__).resolve().parents[2]


class FakeModel:
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, messages, tools, **kwargs):
        self.calls += 1
        self.seen_messages = messages
        self.seen_tools = tools
        if self.calls == 1:
            function = {"name": "eo_gym.crop", "arguments": '{"asset_id":"asset-one","aoi":[0.25,0.25,0.75,0.75]}'}
            message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="call-1", function=function)])
        else:
            function = {"name": "answer.submit", "arguments": '{"answer":{"label":"crop","confidence":0.9,"claims":[]},"confidence":0.9,"evidence_ids":["ev-one"]}'}
            message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="call-2", function=function)])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class QwenAgentRunnerTests(unittest.TestCase):
    def test_utility_projection(self):
        refs = artifact_refs({"items": [{"artifact_ref": "art-one"}, {"artifact_ref": "art-one"}]})
        self.assertEqual(refs, ["art-one", "art-one"])
        self.assertEqual(decode_tool_arguments({"function": {"name": "x", "arguments": "{\"x\":1}"}}), {"x": 1})
        message = content_message("text", {"size_bytes": 2, "sha256": "f" * 64, "media_type": "image/png"}, b"ab", "image/png")
        self.assertIn("data:image/png;base64,YWI=", message["content"][1]["image_url"]["url"])
        with self.assertRaises(ValueError):
            content_message("x", {"size_bytes": MAX_IMAGE_BYTES + 1, "sha256": "f" * 64}, b"x" * (MAX_IMAGE_BYTES + 1))

    def test_model_tool_calls_execute_and_verified_image_is_returned(self):
        model = FakeModel()
        image = b"fake-png"

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["authorization"], "Bearer token")
            if request.url.path == "/agent/session":
                return httpx.Response(200, json={"task": {"prompt": "crop", "input_asset_refs": ["asset-one"]}, "state": {"episode_id": "ep2-" + "1" * 32, "state_version": 0}, "observation": {}, "tool_schemas": {"eo_gym.crop": {"type": "object", "properties": {}, "additionalProperties": False}}})
            if request.url.path == "/agent/step" and json.loads(request.content)["action"]["tool_id"] == "eo_gym.crop":
                return httpx.Response(200, json={"terminated": False, "observation": {"items": [{"artifact_ref": "art-one"}]}, "state": {"episode_id": "ep2-" + "1" * 32, "state_version": 1}})
            if request.url.path == "/agent/step":
                return httpx.Response(200, json={"terminated": True, "observation": {}, "state": {"episode_id": "ep2-" + "1" * 32, "state_version": 2, "status": "terminal"}})
            if request.url.path == "/agent/artifacts/art-one":
                return httpx.Response(200, json={"artifact": {"artifact_id": "art-one", "size_bytes": len(image), "sha256": "a" * 64, "media_type": "image/png"}})
            if request.url.path == "/agent/artifacts/art-one/content":
                return httpx.Response(200, content=image, headers={"content-type": "image/png"})
            raise AssertionError(request.url.path)

        transport = httpx.MockTransport(handler)
        original = httpx.Client
        try:
            httpx.Client = lambda **kwargs: original(transport=transport, **kwargs)
            report = run("http://gateway", "token", model)
        finally:
            httpx.Client = original
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["model_tool_calls"], 2)
        self.assertEqual(report["image_hashes"], ["a" * 64])
        self.assertTrue(any(item.get("name") == "eo_gym.crop" for item in report["transcript"]))

    def test_compose_profile_is_agent_front_only_and_minimally_mounted(self):
        path = Path(__file__).resolve().parents[2] / "compose.qwen-runner.yaml"
        content = path.read_text()
        networks = content.split("networks:", 1)[1]
        self.assertIn("- agent-front", networks)
        self.assertNotIn("- isolated", networks)
        self.assertIn("read_only: true", content)
        volumes = content.split("volumes:", 1)[1].split("tmpfs:", 1)[0]
        self.assertIn(":/run/agent-token:ro", volumes)
        self.assertIn("/reports:rw", volumes)
        self.assertNotIn("tasks:", volumes)
        self.assertNotIn("state:", volumes)
        self.assertNotIn("datasets:", volumes)


if __name__ == "__main__":
    unittest.main()

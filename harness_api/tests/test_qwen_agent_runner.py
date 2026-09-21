import json
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

RUNNER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_qwen_agent.py"
SPEC = importlib.util.spec_from_file_location("qwen_agent_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
artifact_refs = RUNNER.artifact_refs
action_from_tool_call = RUNNER.action_from_tool_call
compact_messages = RUNNER.compact_messages
content_message = RUNNER.content_message
decode_tool_arguments = RUNNER.decode_tool_arguments
openai_tools = RUNNER.openai_tools
run = RUNNER.run
model_client = RUNNER.model_client
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
        elif self.calls == 2:
            function = {"name": "memory.save_evidence", "arguments": '{"evidence_id":"ev-one","claim_id":"claim-one","source_ref":"art-one","selector":{"pixel_window":[0,0,1,1]},"description":"crop","frozen_sha256":"' + "a" * 64 + '"}'}
            message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="call-2", function=function)])
        else:
            function = {"name": "answer.submit", "arguments": '{"answer":{"label":"crop","confidence":0.9,"claims":[]},"confidence":0.9,"evidence_ids":["ev-one"]}'}
            message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="call-3", function=function)])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    @property
    def last_messages(self):
        return self.seen_messages


class QwenAgentRunnerTests(unittest.TestCase):
    def test_model_client_ignores_inherited_proxy_environment(self):
        with patch.dict(os.environ, {
            "HTTP_PROXY": "http://proxy.invalid", "http_proxy": "http://proxy.invalid",
        }):
            client = model_client("http://172.17.0.1:18000/v1")
        try:
            self.assertFalse(client._client.trust_env)
        finally:
            client.close()

    def test_utility_projection(self):
        refs = artifact_refs({"items": [{"artifact_ref": "art-one"}, {"artifact_ref": "art-one"}]})
        self.assertEqual(refs, ["art-one", "art-one"])
        self.assertEqual(decode_tool_arguments({"function": {"name": "x", "arguments": "{\"x\":1}"}}), {"x": 1})
        message = content_message("text", {"size_bytes": 2, "sha256": "f" * 64, "media_type": "image/png"}, b"ab", "image/png")
        self.assertIn("data:image/png;base64,YWI=", message["content"][1]["image_url"]["url"])
        self.assertEqual(compact_messages([{"role": "user", "content": "keep"}, message]),
                         [{"role": "user", "content": "keep"}])
        with self.assertRaises(ValueError):
            content_message("x", {"size_bytes": MAX_IMAGE_BYTES + 1, "sha256": "f" * 64}, b"x" * (MAX_IMAGE_BYTES + 1))

    def test_native_actions_are_exposed_and_not_wrapped_as_tool_invoke(self):
        session = {"task": {"allowed_actions": ["tool.invoke", "memory.save_evidence", "answer.submit"],
                            "answer_schema": {"type": "object", "properties": {"label": {"type": "string"}}}},
                   "tool_schemas": {"eo_gym.crop": {"type": "object", "properties": {
                       "aoi": {"type": "array", "items": {"type": "number"}}}}}}
        names = [item["function"]["name"] for item in openai_tools(session)]
        self.assertEqual(names, ["eo_gym.crop", "memory.save_evidence", "answer.submit"])
        crop_aoi = openai_tools(session)[0]["function"]["parameters"]["properties"]["aoi"]
        self.assertEqual(crop_aoi["items"]["maximum"], 1.0)
        self.assertIn("not pixel coordinates", crop_aoi["description"])
        evidence_schema = openai_tools(session)[1]["function"]["parameters"]
        self.assertIn("pattern", evidence_schema["properties"]["evidence_id"])
        self.assertEqual(evidence_schema["properties"]["selector"]["properties"]["bbox"]["type"], "object")
        pixel_window = evidence_schema["properties"]["selector"]["properties"]["pixel_window"]
        self.assertIn("[x,y,width,height]", pixel_window["description"])
        self.assertEqual(action_from_tool_call(session, "eo_gym.crop", {"x": 1}),
                         {"type": "tool.invoke", "tool_id": "eo_gym.crop", "arguments": {"x": 1}})
        evidence = {"evidence_id": "ev-one", "claim_id": "claim-one"}
        self.assertEqual(action_from_tool_call(session, "memory.save_evidence", evidence),
                         {"type": "memory.save_evidence", "evidence": evidence})
        self.assertEqual(action_from_tool_call(session, "answer.submit", {"answer": {}, "evidence_ids": []}),
                         {"type": "answer.submit", "answer": {}, "evidence_ids": []})
        with self.assertRaises(ValueError):
            action_from_tool_call(session, "unknown", {})

    def test_model_tool_calls_execute_and_verified_image_is_returned(self):
        model = FakeModel()
        image = b"fake-png"

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["authorization"], "Bearer token")
            if request.url.path == "/agent/session":
                return httpx.Response(200, json={"task": {"prompt": "crop", "input_asset_refs": ["asset-one"],
                    "allowed_actions": ["tool.invoke", "memory.save_evidence", "answer.submit"],
                    "answer_schema": {"type": "object", "properties": {"label": {"type": "string"}}}},
                    "state": {"episode_id": "ep2-" + "1" * 32, "state_version": 0}, "observation": {},
                    "tool_schemas": {"eo_gym.crop": {"type": "object", "properties": {
                        "aoi": {"type": "array", "items": {"type": "number"}}}, "additionalProperties": False}}})
            action = json.loads(request.content)["action"] if request.url.path == "/agent/step" else None
            if action and action["type"] == "tool.invoke":
                return httpx.Response(200, json={"terminated": False, "observation": {"items": [{"artifact_ref": "art-one"}]}, "state": {"episode_id": "ep2-" + "1" * 32, "state_version": 1}})
            if action and action["type"] == "memory.save_evidence":
                return httpx.Response(200, json={"terminated": False, "observation": {}, "state": {"episode_id": "ep2-" + "1" * 32, "state_version": 2, "status": "active"}})
            if action and action["type"] == "answer.submit":
                return httpx.Response(200, json={"terminated": True, "observation": {}, "state": {"episode_id": "ep2-" + "1" * 32, "state_version": 3, "status": "terminated"}})
            if request.url.path == "/agent/artifacts/art-one":
                return httpx.Response(200, json={"artifact": {"artifact_id": "art-one", "size_bytes": len(image), "sha256": "a" * 64, "media_type": "image/png", "pixel": {"width": 1, "height": 1}}})
            if request.url.path == "/agent/artifacts/art-one/content":
                return httpx.Response(200, content=image, headers={"content-type": "image/png"})
            raise AssertionError(request.url.path)

        transport = httpx.MockTransport(handler)
        original = httpx.Client
        try:
            httpx.Client = lambda **kwargs: original(transport=transport, **kwargs)
            with __import__("tempfile").TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "checkpoint.json"
                report = run("http://gateway", "token", model, checkpoint_path=checkpoint)
                checkpoint_text = checkpoint.read_text()
                self.assertNotIn("image_url", checkpoint_text)
                self.assertIn('"artifact_ids": ["art-one"]', checkpoint_text)
        finally:
            httpx.Client = original
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["model_tool_calls"], 3)
        self.assertEqual(report["image_hashes"], ["a" * 64])
        self.assertTrue(any(item.get("name") == "eo_gym.crop" for item in report["transcript"]))
        self.assertTrue(any(item.get("name") == "memory.save_evidence" for item in report["transcript"]))

    def test_checkpoint_resumes_without_repeating_completed_action(self):
        model = FakeModel()
        original_run = RUNNER.run
        gateway_calls = []
        image = b"fake-png"

        def handler(request: httpx.Request) -> httpx.Response:
            gateway_calls.append((request.method, request.url.path))
            if request.url.path == "/agent/session":
                return httpx.Response(200, json={"task": {"allowed_actions": ["tool.invoke"]},
                    "state": {"episode_id": "ep2-" + "2" * 32, "state_version": 0}, "observation": {},
                    "tool_schemas": {"eo_gym.crop": {"type": "object", "properties": {
                        "aoi": {"type": "array", "items": {"type": "number"}}}}}})
            if request.url.path == "/agent/step":
                return httpx.Response(200, json={"terminated": True, "observation": {}, "state": {"episode_id": "ep2-" + "2" * 32, "state_version": 1, "status": "terminated"}})
            raise AssertionError(request.url.path)

        transport = httpx.MockTransport(handler)
        original_client = httpx.Client
        with __import__("tempfile").TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            try:
                httpx.Client = lambda **kwargs: original_client(transport=transport, **kwargs)
                report = RUNNER.run("http://gateway", "token", model, checkpoint_path=checkpoint)
            finally:
                httpx.Client = original_client
            self.assertTrue(checkpoint.is_file())
            self.assertIn(("GET", "/agent/session"), gateway_calls)
            gateway_calls.clear()
            second = FakeModel()
            try:
                httpx.Client = lambda **kwargs: original_client(transport=transport, **kwargs)
                resumed = RUNNER.run("http://gateway", "token", second, checkpoint_path=checkpoint)
            finally:
                httpx.Client = original_client
            self.assertEqual(resumed["status"], "passed")
            self.assertNotIn(("GET", "/agent/session"), gateway_calls)
            self.assertNotIn(("POST", "/agent/step"), gateway_calls)

    def test_error_report_is_fail_closed_and_typed(self):
        report = RUNNER.error_report(httpx.ConnectError("down"), episode_id="ep", turns=2, tool_calls=1,
                                      image_hashes=["a" * 64], elapsed_ms=12, transcript=[{"x": 1}])
        self.assertEqual(report["reason"], "gateway_network_error")
        self.assertEqual(report["error"]["type"], "ConnectError")

    def test_step_rejection_preserves_model_call_context(self):
        model = FakeModel()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/agent/session":
                return httpx.Response(200, json={
                    "task": {"allowed_actions": ["tool.invoke"]},
                    "state": {"episode_id": "ep2-" + "3" * 32, "state_version": 0},
                    "observation": {}, "tool_schemas": {"eo_gym.crop": {
                        "type": "object", "properties": {
                            "aoi": {"type": "array", "items": {"type": "number"}},
                        },
                    }},
                })
            if request.url.path == "/agent/step":
                return httpx.Response(403, json={"error": {"code": "policy_rejected"}})
            raise AssertionError(request.url.path)

        transport = httpx.MockTransport(handler)
        original = httpx.Client
        try:
            httpx.Client = lambda **kwargs: original(transport=transport, **kwargs)
            report = run("http://gateway", "token", model)
        finally:
            httpx.Client = original
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["episode_id"], "ep2-" + "3" * 32)
        self.assertEqual(report["model_tool_calls"], 1)
        self.assertEqual(report["transcript"][0]["assistant"]["tool_calls"][0]["function"]["name"], "eo_gym.crop")

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

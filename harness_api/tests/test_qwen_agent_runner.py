import json
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

RUNNER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_qwen_agent.py"
BATCH_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_qwen_whu_batch.py"
SPEC = importlib.util.spec_from_file_location("qwen_agent_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
BATCH_SPEC = importlib.util.spec_from_file_location("qwen_whu_batch", BATCH_PATH)
BATCH = importlib.util.module_from_spec(BATCH_SPEC)
BATCH_SPEC.loader.exec_module(BATCH)
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
FIXTURE_IMAGE = b"fake-png"
FIXTURE_IMAGE_SHA256 = __import__("hashlib").sha256(FIXTURE_IMAGE).hexdigest()


class FakeModel:
    def __init__(self):
        self.calls = 0
        self.tool_history = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, messages, tools, **kwargs):
        self.calls += 1
        self.seen_messages = messages
        self.seen_tools = tools
        self.tool_history.append(tools)
        if self.calls == 1:
            function = {"name": "eo_gym.crop", "arguments": '{"asset_id":"asset-one","aoi":[0.25,0.25,0.75,0.75]}'}
            message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="call-1", function=function)])
        elif self.calls == 2:
            function = {"name": "memory.save_evidence", "arguments": '{"evidence_id":"ev-one","claim_id":"claim-one","artifact_index":0,"pixel_window":[0,0,1,1],"description":"crop"}'}
            message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="call-2", function=function)])
        else:
            function = {"name": "answer.submit", "arguments": '{"answer":{"label":"crop","confidence":0.9,"claims":[]},"confidence":0.9,"evidence_ids":["ev-one"]}'}
            message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="call-3", function=function)])
        usage = SimpleNamespace(model_dump=lambda mode="json": {
            "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
        })
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    @property
    def last_messages(self):
        return self.seen_messages


class QwenAgentRunnerTests(unittest.TestCase):
    def test_whu_batch_preserves_reports_and_writes_incremental_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime" / "whu"
            (runtime / "jobs").mkdir(parents=True)
            job = {
                "sample_id": "0_224",
                "truth": {"change_class": "no_change", "change_direction": "no_change"},
            }
            (runtime / "jobs" / "0_224.json").write_text(json.dumps(job))
            (runtime / "job.json").write_text(json.dumps({"jobs": [job]}))
            report = {
                "status": "passed",
                "cost": {"total_tokens": 12},
                "elapsed_ms": 100,
                "terminal_state": {"evaluation": {"diagnostics": {
                    "unnecessary_abstention": False, "false_confidence": True}}},
                "transcript": [{"name": "answer.abstain"}],
            }
            (runtime / "reports").mkdir()
            (runtime / "reports" / "qwen-0_224.json").write_text(json.dumps(report))
            args = SimpleNamespace(
                runtime=runtime, samples=None, manifest=root / "manifest.json",
                gateway="http://gateway", backend="http://backend",
                openai_base_url="http://model", registry=root / "registry.json",
                max_turns=2, ttl_seconds=3600,
                model_client_factory=lambda _: None, run_episode=lambda *args: None,
                runtime_boundary=root / "runtime",
            )
            result = BATCH.batch_run(args)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(result["reports"]["0_224"], report)
            self.assertEqual(manifest["completed"], ["0_224"])
            self.assertEqual(manifest["summary"]["total"], 1)
            self.assertEqual(manifest["summary"]["abstained"], 1)
            self.assertEqual(manifest["summary"]["groups"][0]["false_confidence"], 1)
            self.assertTrue((runtime / "reports" / "qwen-0_224.json").is_file())

    def test_whu_batch_enriches_report_from_readonly_terminal_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime" / "whu"
            state_root = runtime / "state"
            state_root.mkdir(parents=True)
            report = {
                "status": "passed",
                "episode_id": "ep2-" + "7" * 32,
                "terminal_state": {
                    "episode_id": "ep2-" + "7" * 32,
                    "status": "terminated",
                    "final_answer": {"outcome": "submitted"},
                },
            }
            state = {
                "status": "terminated",
                "final_answer": {"outcome": "submitted"},
                "evaluation": {"status": "completed", "diagnostics": {
                    "false_confidence": True, "unnecessary_abstention": False,
                }},
            }
            import sqlite3
            with sqlite3.connect(state_root / "episodes.sqlite3") as connection:
                connection.execute(
                    "CREATE TABLE v2_episodes (episode_id TEXT PRIMARY KEY, state_json TEXT)"
                )
                connection.execute(
                    "INSERT INTO v2_episodes VALUES (?, ?)",
                    ("ep2-" + "7" * 32, json.dumps(state)),
                )
            enriched = BATCH._enrich_terminal_state(runtime, report)
            self.assertEqual(
                enriched["terminal_state"]["evaluation"]["diagnostics"]["false_confidence"],
                True,
            )

    def test_model_client_ignores_inherited_proxy_environment(self):
        self.assertIn("multiple input", RUNNER.SYSTEM_PROMPT)

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self._client = kwargs["http_client"]

            def close(self):
                self._client.close()

        with patch.dict(os.environ, {
            "HTTP_PROXY": "http://proxy.invalid", "http_proxy": "http://proxy.invalid",
        }):
            client = model_client(
                "http://172.17.0.1:18000/v1", openai_client_class=FakeOpenAI,
            )
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
        self.assertEqual([item["function"]["name"] for item in openai_tools(session)],
                         ["eo_gym.crop", "answer.submit"])
        artifacts = [{"artifact_id": "art-one", "sha256": "a" * 64,
                      "pixel": {"width": 1, "height": 1}}]
        names = [item["function"]["name"] for item in openai_tools(session, artifacts)]
        self.assertEqual(names, ["eo_gym.crop", "memory.save_evidence", "answer.submit"])
        crop_aoi = openai_tools(session)[0]["function"]["parameters"]["properties"]["aoi"]
        self.assertEqual(crop_aoi["items"]["maximum"], 1.0)
        self.assertIn("not pixel coordinates", crop_aoi["description"])
        evidence_schema = openai_tools(session, artifacts)[1]["function"]["parameters"]
        self.assertIn("pattern", evidence_schema["properties"]["evidence_id"])
        self.assertEqual(evidence_schema["properties"]["artifact_index"]["enum"], [0])
        self.assertNotIn("source_ref", evidence_schema["properties"])
        self.assertNotIn("frozen_sha256", evidence_schema["properties"])
        self.assertNotIn("selector", evidence_schema["properties"])
        pixel_window = evidence_schema["properties"]["pixel_window"]
        self.assertIn("[x,y,width,height]", pixel_window["description"])
        raster_session = {**session, "tool_schemas": {
            "raster.band_math": {"type": "object", "properties": {}}}}
        raster_evidence = openai_tools(raster_session, artifacts)[1]["function"]["parameters"]
        self.assertIn("bbox", raster_evidence["properties"]["selector"]["properties"])
        self.assertEqual(action_from_tool_call(session, "eo_gym.crop", {"x": 1}),
                         {"type": "tool.invoke", "tool_id": "eo_gym.crop", "arguments": {"x": 1}})
        evidence = {"evidence_id": "ev-one", "claim_id": "claim-one", "artifact_index": 0,
                    "pixel_window": [0, 0, 1, 1]}
        bound = {"evidence_id": "ev-one", "claim_id": "claim-one",
                 "selector": {"pixel_window": [0, 0, 1, 1]},
                 "source_ref": "art-one", "frozen_sha256": "a" * 64}
        self.assertEqual(action_from_tool_call(session, "memory.save_evidence", evidence, artifacts),
                         {"type": "memory.save_evidence", "evidence": bound})
        with self.assertRaises(ValueError):
            action_from_tool_call(session, "memory.save_evidence", {**evidence, "artifact_index": 1}, artifacts)
        with self.assertRaises(ValueError):
            action_from_tool_call(session, "memory.save_evidence",
                                  {**evidence, "pixel_window": [0, 0, 2, 1]}, artifacts)
        self.assertEqual(action_from_tool_call(session, "answer.submit", {"answer": {}, "evidence_ids": []}),
                         {"type": "answer.submit", "answer": {}, "evidence_ids": []})
        map_session = {"task": {"allowed_actions": ["map.set_view"]}, "tool_schemas": {}}
        map_tools = openai_tools(map_session)
        self.assertEqual([item["function"]["name"] for item in map_tools], ["map.set_view"])
        bbox = {"west": 121.45, "south": 31.2, "east": 121.55, "north": 31.3}
        self.assertEqual(map_tools[0]["function"]["parameters"]["properties"]["bbox"]["required"],
                         ["west", "south", "east", "north"])
        self.assertEqual(action_from_tool_call(map_session, "map.set_view", {"bbox": bbox}),
                         {"type": "map.set_view", "bbox": bbox})
        with self.assertRaises(ValueError):
            action_from_tool_call(session, "unknown", {})

    def test_model_tool_calls_execute_and_verified_image_is_returned(self):
        model = FakeModel()
        image = FIXTURE_IMAGE

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
                self.assertEqual(action["evidence"]["source_ref"], "art-one")
                self.assertEqual(action["evidence"]["frozen_sha256"], FIXTURE_IMAGE_SHA256)
                self.assertNotIn("artifact_index", action["evidence"])
                return httpx.Response(200, json={"terminated": False, "observation": {}, "state": {"episode_id": "ep2-" + "1" * 32, "state_version": 2, "status": "active"}})
            if action and action["type"] == "answer.submit":
                evaluation = {"status": "completed", "diagnostics": {"false_confidence": True,
                                                                    "unnecessary_abstention": False}}
                return httpx.Response(200, json={"terminated": True, "observation": {}, "state": {"episode_id": "ep2-" + "1" * 32, "state_version": 3, "status": "terminated", "final_answer": {"outcome": "submitted"}, "evaluation": evaluation}})
            if request.url.path == "/agent/artifacts/art-one":
                return httpx.Response(200, json={"artifact": {"artifact_id": "art-one", "size_bytes": len(image), "sha256": FIXTURE_IMAGE_SHA256, "media_type": "image/png", "pixel": {"width": 1, "height": 1}}})
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
        self.assertEqual(report["image_hashes"], [FIXTURE_IMAGE_SHA256])
        self.assertEqual(report["cost"]["model_calls"], 3)
        self.assertEqual(report["cost"]["prompt_tokens"], 30)
        self.assertEqual(report["cost"]["completion_tokens"], 6)
        self.assertEqual(report["cost"]["total_tokens"], 36)
        self.assertEqual(report["attempt"], {"phase": "completed", "resumed": False,
                                             "new_model_calls": 3})
        self.assertEqual(report["terminal_state"], {
            "episode_id": "ep2-" + "1" * 32,
            "status": "terminated",
            "final_answer": {"outcome": "submitted"},
            "evaluation": {"status": "completed", "diagnostics": {
                "false_confidence": True, "unnecessary_abstention": False,
            }},
        })
        first_names = [item["function"]["name"] for item in model.tool_history[0]]
        second_names = [item["function"]["name"] for item in model.tool_history[1]]
        self.assertNotIn("memory.save_evidence", first_names)
        self.assertIn("memory.save_evidence", second_names)
        self.assertTrue(any(item.get("name") == "eo_gym.crop" for item in report["transcript"]))
        self.assertTrue(any(item.get("name") == "memory.save_evidence" for item in report["transcript"]))

    def test_multiple_tool_calls_execute_sequentially_with_updated_state(self):
        class TwoCallModel:
            def __init__(self):
                self.calls = 0
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    calls = [
                        SimpleNamespace(id="multi-inspect-one", function={"name": "catalog.inspect_asset", "arguments": '{"asset_id":"asset-one"}'}),
                        SimpleNamespace(id="multi-inspect-two", function={"name": "catalog.inspect_asset", "arguments": '{"asset_id":"asset-two"}'}),
                    ]
                else:
                    calls = [SimpleNamespace(id="multi-submit", function={"name": "answer.submit", "arguments": '{"answer":{"label":"ok"},"confidence":0.9,"evidence_ids":[]}'})]
                message = SimpleNamespace(content=None, tool_calls=calls)
                usage = SimpleNamespace(model_dump=lambda mode="json": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
                return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

        model = TwoCallModel()
        state_versions = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/agent/session":
                return httpx.Response(200, json={"task": {"allowed_actions": ["tool.invoke", "answer.submit"]}, "state": {"episode_id": "ep2-" + "5" * 32, "state_version": 0}, "observation": {}, "tool_schemas": {"catalog.inspect_asset": {"type": "object", "properties": {"asset_id": {"type": "string"}}}}})
            if request.url.path == "/agent/step":
                body = json.loads(request.content)
                state_versions.append((body["expected_state_version"], body["action"]["type"], body["client_action_id"]))
                version = len(state_versions)
                return httpx.Response(200, json={"terminated": version == 3, "observation": {}, "state": {"episode_id": "ep2-" + "5" * 32, "state_version": version, "status": "terminated" if version == 3 else "active"}})
            raise AssertionError(request.url.path)

        transport = httpx.MockTransport(handler)
        original = httpx.Client
        try:
            httpx.Client = lambda **kwargs: original(transport=transport, **kwargs)
            report = run("http://gateway", "token", model)
        finally:
            httpx.Client = original
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(model.calls, 2)
        self.assertEqual(state_versions, [
            (0, "tool.invoke", "qwen-" + "5" * 8 + "-0"),
            (1, "tool.invoke", "qwen-" + "5" * 8 + "-1"),
            (2, "answer.submit", "qwen-" + "5" * 8 + "-2"),
        ])

    def test_received_artifact_checksum_mismatch_fails_closed(self):
        model = FakeModel()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/agent/session":
                return httpx.Response(200, json={"task": {"allowed_actions": ["tool.invoke"]},
                    "state": {"episode_id": "ep2-" + "4" * 32, "state_version": 0},
                    "observation": {}, "tool_schemas": {"eo_gym.crop": {
                        "type": "object", "properties": {"aoi": {
                            "type": "array", "items": {"type": "number"}}}}}})
            if request.url.path == "/agent/step":
                return httpx.Response(200, json={"terminated": False,
                    "observation": {"items": [{"artifact_ref": "art-wrong"}]},
                    "state": {"episode_id": "ep2-" + "4" * 32, "state_version": 1}})
            if request.url.path == "/agent/artifacts/art-wrong":
                return httpx.Response(200, json={"artifact": {"size_bytes": 9,
                    "sha256": "b" * 64, "media_type": "image/png"}})
            if request.url.path == "/agent/artifacts/art-wrong/content":
                return httpx.Response(200, content=b"same-size")
            raise AssertionError(request.url.path)

        transport = httpx.MockTransport(handler)
        original = httpx.Client
        try:
            httpx.Client = lambda **kwargs: original(transport=transport, **kwargs)
            with __import__("tempfile").TemporaryDirectory() as directory:
                report = run("http://gateway", "token", model,
                             checkpoint_path=Path(directory) / "checkpoint.json")
        finally:
            httpx.Client = original
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason"], "runner_contract_error")
        self.assertIn("checksum mismatch", report["error"]["message"])

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
            self.assertEqual(resumed["attempt"], {"phase": "resume_terminal",
                                                  "resumed": True, "new_model_calls": 0})
            self.assertEqual(resumed["cost"]["model_calls"], report["cost"]["model_calls"])
            self.assertEqual(resumed["cost"]["total_tokens"], report["cost"]["total_tokens"])

    def test_error_report_is_fail_closed_and_typed(self):
        report = RUNNER.error_report(httpx.ConnectError("down"), episode_id="ep", turns=2, tool_calls=1,
                                      image_hashes=["a" * 64], elapsed_ms=12, transcript=[{"x": 1}])
        self.assertEqual(report["reason"], "gateway_network_error")
        self.assertEqual(report["error"]["type"], "ConnectError")
        self.assertEqual(report["cost"]["model_calls"], 0)

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

        worldcover_path = Path(__file__).resolve().parents[2] / "compose.worldcover-qwen.yaml"
        worldcover = worldcover_path.read_text()
        self.assertIn("EO_HARNESS_V2_RENDERER_URL: http://renderer:8090", worldcover)
        self.assertIn("./datasets:/datasets:ro", worldcover)
        self.assertIn("${EO_SMOKE_ROOT}/artifacts:/artifacts:rw", worldcover)
        self.assertIn("harness-internal", worldcover)


if __name__ == "__main__":
    unittest.main()

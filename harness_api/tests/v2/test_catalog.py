import copy
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from .test_tool_execution import make_tool_tasks, FakeExecutor
from app.main import create_app
from app.v2.artifacts import ArtifactStore
from app.v2.capabilities import TaskRegistry
from app.v2.domain import V2DomainError
from app.v2.events import canonical_json
from app.v2.schemas import ToolInvokeAction
from app.v2.tools.catalog import CatalogExecutor
from app.v2.tools.runtime import ToolRouter


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tasks = make_tool_tasks(self.root)
        task_dir = self.tasks / "crop-smoke"
        base = json.loads((task_dir / "assets.json").read_text())[0]
        base.update(asset_id="asset-a", uri="file:///private/image.tif", source="SECRET_SOURCE", platform="Sentinel-2")
        base["quality"]["cloud_cover_percent"] = 10
        pixel = {**copy.deepcopy(base), "asset_id": "asset-b", "spatial": None, "temporal": None,
                 "pixel": {"coordinate_system": "pixel", "width": 32, "height": 24, "channels": 3}}
        pixel["quality"]["cloud_cover_percent"] = None
        hidden = {**copy.deepcopy(base), "asset_id": "asset-hidden"}
        label = {**copy.deepcopy(base), "asset_id": "asset-label", "roles": ["data", "LABELS"]}
        task = json.loads((task_dir / "task.json").read_text())
        task["inputs"] = ["asset-b", "asset-a", "asset-label"]
        task["budget"].update(max_artifact_bytes=0, max_wall_time_ms=3600000)
        scenario = json.loads((task_dir / "scenario.json").read_text())
        scenario["allowed_tools"] = ["catalog.search", "catalog.inspect_asset", "eo_gym.crop"]
        (task_dir / "task.json").write_text(json.dumps(task))
        (task_dir / "scenario.json").write_text(json.dumps(scenario))
        (task_dir / "assets.json").write_text(json.dumps([pixel, hidden, label, base]))
        self.manifest = TaskRegistry(self.tasks).get("crop-smoke", "1.0.0")
        self.catalog = CatalogExecutor()
        self.client = TestClient(self.app())
        self.addCleanup(self.client.close)
        reset = self.client.post("/v2/reset", json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}})
        self.assertEqual(reset.status_code, 201, reset.text)
        self.episode_id = reset.json()["data"]["episode_id"]
        self.url = f"/v2/episodes/{self.episode_id}/step"

    def app(self, executor=None):
        return create_app(database_path=str(self.root / "state.db"), v2_tasks_path=str(self.tasks),
                          v2_artifacts_path=str(self.root / "artifacts"),
                          v2_tool_executor=executor or ToolRouter())

    def plan(self, tool="catalog.search", arguments=None, accessible=None):
        return self.catalog.plan(ToolInvokeAction(type="tool.invoke", tool_id=tool, arguments=arguments or {}),
                                 self.manifest, self.manifest.task.inputs if accessible is None else accessible)

    def request(self, tool="catalog.search", arguments=None, version=0, action_id="search-1"):
        return {"client_action_id": action_id, "expected_state_version": version,
                "action": {"type": "tool.invoke", "tool_id": tool, "arguments": arguments or {}}}

    def test_search_is_sorted_scoped_and_paginated_without_paths_or_labels(self):
        first = self.plan(arguments={"limit": 1}).invoke()
        self.assertEqual(first.metadata["assets"][0]["asset_id"], "asset-a")
        self.assertEqual(first.metadata["next_offset"], 1)
        second = self.plan(arguments={"limit": 1, "offset": 1}).invoke()
        self.assertEqual(second.metadata["assets"][0]["asset_id"], "asset-b")
        self.assertIsNone(second.metadata["next_offset"])
        self.assertEqual(first.input_bytes, second.input_bytes)
        encoded = canonical_json(first.metadata)
        for private in ("SECRET_SOURCE", "private/image", "asset-label", "asset-hidden", '"uri"', '"roles"', '"source"'):
            self.assertNotIn(private, encoded)
        self.assertIsNone(first.artifact)
        self.assertEqual(self.plan(arguments={"offset": 1000}).invoke().metadata["assets"], [])

    def test_inspect_rejects_hidden_unknown_label_and_inaccessible_identically(self):
        errors = []
        for asset_id in ("asset-hidden", "asset-missing", "asset-label", "asset-b"):
            with self.assertRaises(V2DomainError) as error:
                self.plan("catalog.inspect_asset", {"asset_id": asset_id}, accessible=["asset-a", "asset-label"])
            errors.append((error.exception.code, error.exception.message, error.exception.status_code))
        self.assertEqual(len(set(errors)), 1)
        self.assertEqual(self.plan(accessible=[]).invoke().metadata["matched_count"], 0)

    def test_pixel_inspection_does_not_invent_geography(self):
        out = self.plan("catalog.inspect_asset", {"asset_id": "asset-b"}).invoke()
        self.assertIsNone(out.metadata["asset"]["spatial"])
        self.assertIsNone(out.metadata["asset"]["temporal"])
        self.assertEqual(out.metadata["asset"]["pixel"]["width"], 32)
        self.assertEqual(out.input_bytes, len(canonical_json(out.metadata["asset"]).encode()))

    def test_spatial_temporal_quality_and_band_filters_exclude_unknowns(self):
        queries = [
            {"bbox": {"west": 121, "east": 122, "south": 31, "north": 32}},
            {"time_range": {"start": "2021-06-01T00:00:00Z", "end": "2022-01-01T00:00:00Z"}},
            {"max_cloud_cover_percent": 10},
        ]
        for query in queries:
            self.assertEqual([a["asset_id"] for a in self.plan(arguments=query).invoke().metadata["assets"]], ["asset-a"])
        for query in ({"max_cloud_cover_percent": 9}, {"bands": ["NIR"]}, {"platform": "Other"},
                      {"bbox": {"west": 0, "east": 1, "south": 0, "north": 1}}):
            self.assertEqual(self.plan(arguments=query).invoke().metadata["matched_count"], 0)
        self.assertEqual(self.plan(arguments={"platform": "Sentinel-2", "bands": ["visual"]}).invoke().metadata["matched_count"], 2)

    def test_unknown_fields_invalid_bounds_and_invalid_dates_rejected(self):
        invalid = [{"uri": "/etc/passwd"}, {"limit": 51}, {"limit": True}, {"offset": -1}, {"limit": "1"},
                   {"max_cloud_cover_percent": 101}, {"bands": ["x"] * 33},
                   {"bbox": {"west": 170, "east": -170, "south": 0, "north": 1}},
                   {"bbox": {"west": "0", "east": 1, "south": 0, "north": 1}},
                   {"time_range": {"start": "2021-02-30T00:00:00Z", "end": "2021-03-01T00:00:00Z"}},
                   {"time_range": {"start": "2021-01-02T00:00:00Z", "end": "2021-01-01T00:00:00Z"}}]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(V2DomainError) as error:
                self.plan(arguments=arguments)
            self.assertEqual(error.exception.code, "invalid_tool_arguments")

    def test_non_wgs84_is_not_reinterpreted(self):
        self.manifest.assets[-1].spatial.crs = "EPSG:3857"
        query = {"bbox": {"west": 121, "east": 122, "south": 31, "north": 32}}
        self.assertEqual(self.plan(arguments=query).invoke().metadata["matched_count"], 0)

    def test_fractional_timestamps_are_compared_chronologically(self):
        temporal = self.manifest.assets[-1].temporal
        temporal.start = "2021-01-01T00:00:00Z"
        temporal.end = "2021-01-01T00:00:00.100000Z"
        query = {"time_range": {"start": "2021-01-01T00:00:00.050000Z", "end": "2021-01-01T00:00:00.060000Z"}}
        self.assertEqual(self.plan(arguments=query).invoke().metadata["matched_count"], 1)

    def test_http_scope_comes_from_current_episode_not_another_episode(self):
        reset = self.client.post("/v2/reset", json={"task_ref": {"task_id": "crop-smoke", "task_version": "1.0.0"}})
        other = reset.json()["data"]["episode_id"]
        with sqlite3.connect(self.root / "state.db") as connection:
            for episode, asset in ((self.episode_id, "asset-a"), (other, "asset-b")):
                state = json.loads(connection.execute("SELECT state_json FROM v2_episodes WHERE episode_id=?", (episode,)).fetchone()[0])
                state["accessible_asset_refs"] = [asset]
                connection.execute("UPDATE v2_episodes SET state_json=? WHERE episode_id=?", (json.dumps(state), episode))
        denied = self.client.post(self.url, json=self.request("catalog.inspect_asset", {"asset_id": "asset-b"}))
        self.assertEqual(denied.status_code, 403, denied.text)
        result = self.client.post(f"/v2/episodes/{other}/step", json=self.request("catalog.inspect_asset", {"asset_id": "asset-b"}))
        self.assertEqual(result.status_code, 200, result.text)
        result = self.client.post(self.url, json=self.request()).json()["data"]
        self.assertEqual([a["asset_id"] for a in result["observation"]["items"][0]["inline"]["assets"]], ["asset-a"])

    def test_metadata_and_result_bounds(self):
        self.manifest.assets[-1].platform = "x" * 8192
        with self.assertRaises(V2DomainError) as error:
            self.plan()
        self.assertEqual(error.exception.code, "catalog_metadata_too_large")
        self.manifest.assets[-1].platform = "x"
        with patch("app.v2.tools.catalog.MAX_RESULT_BYTES", 1), self.assertRaises(V2DomainError) as error:
            self.plan()
        self.assertEqual(error.exception.code, "catalog_result_too_large")

    def test_http_metadata_budget_retry_restart_and_no_artifact_registration(self):
        request = self.request()
        response = self.client.post(self.url, json=request)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertEqual(len(data["observation"]["items"]), 1)
        budget = data["state"]["budget"]
        self.assertEqual(budget["input_bytes"]["used"], self.plan().input_bytes)
        self.assertEqual(budget["tool_calls"]["used"], 1)
        self.assertEqual(budget["artifact_bytes"]["used"], 0)
        self.assertEqual(self.client.post(self.url, json=request).json()["data"], data)
        with TestClient(self.app()) as restarted:
            self.assertEqual(restarted.post(self.url, json=request).json()["data"], data)
        conflict = self.request(arguments={"limit": 1})
        self.assertEqual(self.client.post(self.url, json=conflict).status_code, 409)
        with sqlite3.connect(self.root / "state.db") as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM v2_artifacts").fetchone()[0], 0)
            run = json.loads(connection.execute("SELECT run_json FROM v2_tool_runs").fetchone()[0])
            self.assertTrue(run["metadata_only"])
            self.assertEqual(run["logical_input_bytes"], self.plan().input_bytes)
        trace = self.client.get(f"/v2/episodes/{self.episode_id}/trace").json()["data"]["events"]
        self.assertFalse(any(e["event_type"] == "artifact.created" for e in trace))

    def test_insufficient_metadata_budget_rejected_before_reservation(self):
        with sqlite3.connect(self.root / "state.db") as connection:
            state = json.loads(connection.execute("SELECT state_json FROM v2_episodes WHERE episode_id=?", (self.episode_id,)).fetchone()[0])
            state["budget"]["input_bytes"].update(remaining=1)
            connection.execute("UPDATE v2_episodes SET state_json=? WHERE episode_id=?", (json.dumps(state), self.episode_id))
        response = self.client.post(self.url, json=self.request())
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(response.json()["error"]["code"], "tool_budget_exceeded")
        with sqlite3.connect(self.root / "state.db") as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM v2_tool_runs").fetchone()[0], 0)

    def test_task_allowlist_and_disabled_capabilities(self):
        self.manifest.scenario.allowed_tools = []
        with self.assertRaises(V2DomainError):
            self.plan()
        self.assertEqual(self.client.get("/v2/capabilities").json()["data"]["tools"], CatalogExecutor.tool_ids)
        with patch.dict("os.environ", {"EO_HARNESS_CATALOG_ENABLED": "0", "EO_HARNESS_EO_GYM_URL": ""}):
            app = create_app(database_path=str(self.root / "disabled.db"), v2_tasks_path=str(self.tasks), v2_artifacts_path=str(self.root / "disabled-artifacts"))
            with TestClient(app) as client:
                self.assertNotIn("catalog.search", client.get("/v2/capabilities").json()["data"]["tools"])

    def test_opt_in_without_provider_and_router_crop_compatibility(self):
        with patch.dict("os.environ", {"EO_HARNESS_CATALOG_ENABLED": "1", "EO_HARNESS_EO_GYM_URL": ""}):
            app = create_app(database_path=str(self.root / "enabled.db"), v2_tasks_path=str(self.tasks), v2_artifacts_path=str(self.root / "enabled-artifacts"))
            with TestClient(app) as client:
                self.assertEqual(client.get("/v2/capabilities").json()["data"]["tools"], CatalogExecutor.tool_ids)
        provider = FakeExecutor(ArtifactStore(str(self.root / "provider-artifacts")))
        plan = ToolRouter(provider).plan(ToolInvokeAction(type="tool.invoke", tool_id="eo_gym.crop",
            arguments={"asset_id": "asset-a", "aoi": [0, 0, 0.5, 0.5]}), self.manifest, ["asset-a"])
        self.assertFalse(plan.metadata_only)
        self.assertIsNotNone(plan.invoke().artifact)
        self.assertEqual(provider.calls, 1)

    def test_expired_metadata_lease_does_not_repeat_or_double_charge(self):
        request = self.request()
        request_json = canonical_json({"expected_state_version": 0, "action": request["action"]})
        run = {"client_action_id": "search-1", "request_json": request_json, "lease_until": time.time()-1,
               "input_bytes_reserved": self.plan().input_bytes, "output_bytes_reserved": 0,
               "metadata_only": True, "tool_version": "1.0.0", "expected_state_version": 0}
        with sqlite3.connect(self.root / "state.db") as connection:
            connection.execute("INSERT INTO v2_tool_runs VALUES (?,?,?,?,?,?)", ("tool-crashed", self.episode_id, "catalog.search", "running", json.dumps(run), "2026-09-17T00:00:00Z"))
        first = self.client.post(self.url, json=request)
        self.assertEqual(first.status_code, 409, first.text)
        self.assertEqual(first.json()["error"]["code"], "tool_interrupted")
        self.assertEqual(self.client.post(self.url, json=request).json()["error"], first.json()["error"])
        state = self.client.get(f"/v2/episodes/{self.episode_id}/state").json()["data"]["state"]
        self.assertEqual(state["budget"]["tool_calls"]["used"], 1)

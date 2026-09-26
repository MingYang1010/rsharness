import json
import importlib.util
import hashlib
import sqlite3
import tempfile
import unittest
from app.core.events import sha256_json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("qwen_result_manifest", Path(__file__).resolve().parents[2] / "scripts" / "summarize_qwen_results.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class QwenResultManifestTests(unittest.TestCase):
    def _write_fixture(self, root: Path, *, task_id_suffix: str = "correct") -> None:
        reports = root / "reports"
        reports.mkdir()
        for index, ((dataset, sample_id), sample) in enumerate(sorted(module.matrix_samples().items())):
            episode_id = "ep2-" + hashlib.sha256(f"{dataset}/{sample_id}".encode()).hexdigest()
            asset_ids = [sample["asset_id"]] if sample.get("asset_id") else list(sample["asset_ids"])
            hashes = (list(sample["content_sha256s"]) if sample.get("content_sha256s")
                      else [sample["content_sha256"]])
            task_id = f"task-{index}-{task_id_suffix}"
            task = {
                "task_id": task_id, "task_version": "1.0.0", "family": "fixture",
                "prompt": "fixture task", "inputs": asset_ids,
                "scenario_profile": "fixture", "answer_schema": {"type": "object"},
                "evaluator": "fixture", "seed": 0, "metadata": {},
                "metric_aggregation": {},
                "budget": {"max_steps": 20, "max_tool_calls": 5,
                           "max_wall_time_ms": 300000, "max_input_bytes": 1048576,
                           "max_artifact_bytes": 1048576},
            }
            scenario = {
                "profile_id": "fixture", "domain": "remote-sensing",
                "data_cutoff": "2020-01-01T00:00:00Z",
                "allowed_actions": ["tool.invoke", "map.set_view"],
                "allowed_tools": [],
                "network_policy": "none", "evidence_required": True,
                "abstention_allowed": True, "human_review_policy": "never",
            }
            assets = [{"asset_id": asset_id, "uri": "asset://" + asset_id,
                       "media_type": "application/octet-stream", "roles": ["input_image"],
                       "sha256": digest, "size_bytes": 1, "spatial": None,
                       "pixel": {"coordinate_system": "pixel", "width": 1,
                                 "height": 1, "channels": 1},
                       "license": "fixture", "source": "fixture"}
                      for asset_id, digest in zip(asset_ids, hashes)]
            evaluator = {"evaluator_id": "fixture", "evaluator_version": "1.0.0",
                         "metric_names": []}
            manifest_dir = root / sample["task_root"]
            manifest_dir.mkdir(parents=True)
            for name, value in (("task.json", task), ("scenario.json", scenario),
                                ("assets.json", assets), ("evaluator.json", evaluator)):
                (manifest_dir / name).write_text(json.dumps(value))
            task, assets, manifest_hash = module.load_manifest(manifest_dir)
            report = {
                "status": "passed", "episode_id": episode_id, "model_tool_calls": 1,
                "image_hashes": ["b" * 64], "resume_checked": True,
                "transcript": [{"assistant": {}, "model_request": {
                    "schema_version": "qwen-model-request-receipt-v1",
                    "model": module.MODEL_NAME,
                    "image_hashes": [{"payload_sha256": "b" * 64}],
                }, "provider_response": {
                    "id": f"response-{index}", "model": module.MODEL_NAME,
                }, "adapter_projection": {
                    "schema_version": "qwen-response-adapter-projection-v1",
                    "provider_response_sha256": "placeholder",
                    "assistant": {},
                }, "model_response": {
                    "model": module.MODEL_NAME, "id": f"response-{index}",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 0,
                              "total_tokens": 1},
                }}],
            }
            (reports / f"{dataset}__{sample_id}.json").write_text(json.dumps(report))
            database = reports / f"{dataset}__{sample_id}.sqlite3"
            connection = sqlite3.connect(database)
            connection.executescript("""
            CREATE TABLE v2_episodes(episode_id TEXT,task_id TEXT,task_version TEXT,task_manifest_hash TEXT,state_json TEXT);
            CREATE TABLE v2_events(episode_id TEXT,sequence INTEGER,event_json TEXT);
            CREATE TABLE v2_action_results(episode_id TEXT,client_action_id TEXT,outcome TEXT,request_json TEXT,response_json TEXT);
            CREATE TABLE v2_tool_runs(episode_id TEXT,tool_run_id TEXT,client_action_id TEXT,status TEXT,run_json TEXT);
            CREATE TABLE v2_artifacts(artifact_id TEXT,episode_id TEXT,artifact_json TEXT);
            CREATE TABLE v2_episode_artifacts(episode_id TEXT,artifact_id TEXT);
            CREATE TABLE v2_evidence(episode_id TEXT,evidence_id TEXT,evidence_json TEXT);
            """)
            state = {"status": "terminated", "state_version": 2,
                     "accessible_asset_refs": asset_ids,
                     "evaluation": None,
                     "final_answer": {"outcome": "submitted",
                                      "evidence_ids": ["ev"]}}
            if dataset == "ESA-WorldCover-2021":
                state["evaluation"] = {
                    "status": "completed", "aggregate_reward": 0.1,
                    "evaluator_id": "fixture-evaluator", "evaluator_version": "1.0.0",
                    "metrics": [{"name": "task.accuracy", "value": 0.0, "weight": 1.0}],
                }
            connection.execute("INSERT INTO v2_episodes VALUES(?,?,?,?,?)",
                               (episode_id, task_id, "1.0.0", manifest_hash, json.dumps(state)))
            connection.execute("INSERT INTO v2_events VALUES(?,?,?)", (episode_id, 0, "{}"))
            rendered = dataset == "ESA-WorldCover-2021"
            request = ({"action": {"type": "map.set_view"}} if rendered
                       else {"action": {"arguments": {"asset_id": asset_ids[0]}, "type": "tool.invoke"}})
            connection.execute("INSERT INTO v2_action_results VALUES(?,?,?,?,?)",
                               (episode_id, "a", "success", json.dumps(request), "{}"))
            evidence_request = {"action": {"type": "memory.save_evidence",
                                           "evidence": {"evidence_id": "ev"}}}
            connection.execute(
                "INSERT INTO v2_action_results VALUES(?,?,?,?,?)",
                (episode_id, "evidence-action", "success",
                 json.dumps(evidence_request), "{}"),
            )
            if not rendered:
                run = {"request_json": json.dumps(request),
                       "expected_state_version": 0}
                connection.execute("INSERT INTO v2_tool_runs VALUES(?,?,?,?,?)",
                                   (episode_id, "run", "a", "completed", json.dumps(run)))
            artifact = {"sha256": "b" * 64, "lineage": {"input_refs": asset_ids}}
            artifact["artifact_id"] = f"art-{index}"
            if not rendered:
                artifact["lineage"]["tool_id"] = "fixture.crop"
            if rendered:
                artifact["lineage"]["tool_id"] = "renderer.terriamap.capture"
            connection.execute("INSERT INTO v2_artifacts VALUES(?,?,?)",
                               (f"art-{index}", episode_id, json.dumps(artifact)))
            connection.execute("INSERT INTO v2_episode_artifacts VALUES(?,?)",
                               (episode_id, f"art-{index}"))
            connection.execute("INSERT INTO v2_evidence VALUES(?,?,?)",
                               (episode_id, "ev", json.dumps({
                                   "evidence_id": "ev",
                                   "source_ref": f"art-{index}",
                               })))
            connection.commit(); connection.close()

        for path in reports.glob("*.json"):
            value = json.loads(path.read_text())
            provider = value["transcript"][0]["provider_response"]
            value["transcript"][0]["adapter_projection"][
                "provider_response_sha256"
            ] = module.sha256_json(provider)
            path.write_text(json.dumps(value))

    def test_complete_fixture_passes_and_missing_report_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            reports = root / "reports"
            complete = module.summarize(reports)
            self.assertTrue(complete["real_model_acceptance"])
            self.assertEqual(complete["passed_samples"], len(module.matrix_samples()))
            self.assertTrue(complete["checks"]["episode_ids_unique"])
            self.assertEqual(complete["semantic"]["scored"], 2)
            self.assertEqual(complete["semantic"]["unscored"], len(module.matrix_samples()) - 2)
            self.assertEqual(complete["semantic"]["task_correct"], 0)
            self.assertEqual(complete["cost"]["model_calls"], len(module.matrix_samples()))
            self.assertEqual(complete["cost"]["total_tokens"], len(module.matrix_samples()))
            first = next(iter(sorted(reports.glob("*.json"))))
            first.unlink()
            missing = module.summarize(reports)
            self.assertFalse(missing["real_model_acceptance"])
            self.assertEqual(len(missing["missing_reports"]), 1)

    def test_wrong_task_identity_or_content_hash_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root, task_id_suffix="wrong")
            reports = root / "reports"
            first_report_path = next(iter(sorted(reports.glob("*.json"))))
            first = json.loads(first_report_path.read_text())
            key = first_report_path.stem.split("__", 1)
            sample = module.matrix_samples()[tuple(key)]
            task_path = root / sample["task_root"] / "task.json"
            task = json.loads(task_path.read_text())
            task["task_id"] += "-unexpected"
            task_path.write_text(json.dumps(task))
            result = module.summarize(reports)
            self.assertFalse(result["status"] == "passed")
            self.assertIn("episode_task_match", result["failed_checks"])
            self.assertIn("episode_manifest_binding", result["failed_checks"])

    def test_semantic_error_is_not_counted_as_correct(self):
        evaluation = {
            "status": "completed", "aggregate_reward": 0.2,
            "evaluator_id": "fixture", "evaluator_version": "1.0.0",
            "metrics": [
                {"name": "task.change_class_accuracy", "value": 1.0, "weight": 1.0},
                {"name": "task.direction_accuracy", "value": 0.0, "weight": 1.0},
            ],
        }
        outcome = module.semantic_outcome(evaluation)
        self.assertEqual(outcome["status"], "completed")
        self.assertFalse(outcome["task_correct"])
        self.assertEqual(module.semantic_outcome(None)["status"], "unscored")

    def test_failed_runner_or_missing_model_metadata_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); reports = root / "reports"; reports.mkdir()
            sample = next(iter(module.matrix_samples().values()))
            state = root / sample["task_root"] / "state"; state.mkdir(parents=True)
            report = {"status":"passed","episode_id":"ep2-"+"a"*32,"model_tool_calls":1,
                      "image_hashes":[],"transcript":[],"checkpoint":{}}
            (reports / "x.json").write_text(json.dumps(report))
            state.mkdir(parents=True, exist_ok=True)
            state.joinpath("episodes.sqlite3").write_bytes(b"not sqlite")
            result = module.validate_report("dataset", "sample", sample, report, state / "episodes.sqlite3")
        self.assertFalse(result["checks"]["real_image_input_to_model"])
        self.assertFalse(result["checks"]["model_response_metadata_present"])

    def test_inconsistent_action_or_tool_join_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            reports = root / "reports"
            path = next(iter(sorted(reports.glob("*.json"))))
            report = json.loads(path.read_text())
            key = path.stem.split("__", 1)
            sample = module.matrix_samples()[tuple(key)]
            del report
            database_path = reports / f"{key[0]}__{key[1]}.sqlite3"
            with sqlite3.connect(database_path) as connection:
                connection.execute(
                    "UPDATE v2_action_results SET outcome='error' WHERE episode_id=("
                    "SELECT episode_id FROM v2_episodes LIMIT 1)"
                )
            result = module.summarize(reports)
            self.assertIn("episode_actions_consistent", result["failed_checks"])
            self.assertFalse(result["real_model_acceptance"])

    def test_unbound_final_answer_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            reports = root / "reports"
            path = next(iter(sorted(reports.glob("*.json"))))
            key = path.stem.split("__", 1)
            with sqlite3.connect(
                reports / f"{key[0]}__{key[1]}.sqlite3"
            ) as connection:
                connection.execute(
                    "UPDATE v2_evidence SET evidence_json=? "
                    "WHERE episode_id=(SELECT episode_id FROM v2_episodes LIMIT 1)",
                    (json.dumps({"evidence_id": "unknown",
                                 "source_ref": "missing-artifact"}),),
                )
            result = module.summarize(reports)
            self.assertIn("episode_evidence_consistent", result["failed_checks"])

    def test_duplicate_episode_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            reports = root / "reports"
            paths = sorted(reports.glob("*.json"))
            if len(paths) < 2:
                self.skipTest("matrix fixture requires two samples")
            source = json.loads(paths[0].read_text())
            second_report = json.loads(paths[1].read_text())
            second_report["episode_id"] = source["episode_id"]
            paths[1].write_text(json.dumps(second_report))
            source_db = reports / (paths[0].stem + ".sqlite3")
            target_db = reports / (paths[1].stem + ".sqlite3")
            if source_db != target_db:
                source_db.replace(target_db)
            result = module.summarize(reports)
            self.assertIn("episode_ids_unique", result["failed_checks"])

    def test_tampered_provider_adapter_binding_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            reports = root / "reports"
            path = next(iter(sorted(reports.glob("*.json"))))
            value = json.loads(path.read_text())
            value["transcript"][0]["adapter_projection"][
                "provider_response_sha256"
            ] = "0" * 64
            path.write_text(json.dumps(value))
            result = module.summarize(reports)
            self.assertIn("provider_adapter_binding", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()

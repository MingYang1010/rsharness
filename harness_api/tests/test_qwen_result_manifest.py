import json
import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("qwen_result_manifest", Path(__file__).resolve().parents[2] / "scripts" / "summarize_qwen_results.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class QwenResultManifestTests(unittest.TestCase):
    def test_complete_fixture_passes_and_missing_report_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            samples = module.matrix_samples()
            for (dataset, sample_id), sample in samples.items():
                report_root = root / sample["task_root"]
                state = report_root / "state"
                state.mkdir(parents=True, exist_ok=True)
                report = {"status": "passed", "episode_id": "ep2-" + "a" * 32,
                          "model_tool_calls": 1, "image_hashes": ["b" * 64],
                          "transcript": [{"model_response": {"id": "x"}}], "resume_checked": True}
                (reports / f"{dataset}__{sample_id}.json").write_text(json.dumps(report))
                database = reports / f"{dataset}__{sample_id}.sqlite3"
                connection = sqlite3.connect(database)
                connection.executescript("""
                CREATE TABLE v2_episodes(episode_id TEXT,task_id TEXT,task_version TEXT,task_manifest_hash TEXT,state_json TEXT);
                CREATE TABLE v2_events(episode_id TEXT,sequence INTEGER,event_json TEXT);
                CREATE TABLE v2_action_results(episode_id TEXT,client_action_id TEXT,outcome TEXT,request_json TEXT,response_json TEXT);
                CREATE TABLE v2_tool_runs(episode_id TEXT,tool_run_id TEXT,run_json TEXT);
                CREATE TABLE v2_artifacts(artifact_id TEXT,episode_id TEXT,artifact_json TEXT);
                CREATE TABLE v2_episode_artifacts(episode_id TEXT,artifact_id TEXT);
                CREATE TABLE v2_evidence(episode_id TEXT,evidence_id TEXT,evidence_json TEXT);
                """)
                connection.execute("INSERT INTO v2_episodes VALUES(?,?,?,?,?)", ("ep2-" + "a"*32, "task", "1.0.0", "f"*64, json.dumps({"status":"terminated","state_version":2})))
                connection.execute("INSERT INTO v2_events VALUES(?,?,?)", ("ep2-"+"a"*32,0,"{}"))
                rendered = dataset == "ESA-WorldCover-2021"
                request = {"action": {"type": "map.set_view"}} if rendered else {}
                connection.execute("INSERT INTO v2_action_results VALUES(?,?,?,?,?)",
                                   ("ep2-"+"a"*32,"a","success",json.dumps(request),"{}"))
                if not rendered:
                    connection.execute("INSERT INTO v2_tool_runs VALUES(?,?,?)", ("ep2-"+"a"*32,"run","{}"))
                artifact = json.dumps({"sha256":"b"*64, **({
                    "lineage": {"tool_id": "renderer.terriamap.capture"},
                } if rendered else {})})
                connection.execute("INSERT INTO v2_artifacts VALUES(?,?,?)", ("art","ep2-"+"a"*32,artifact))
                connection.execute("INSERT INTO v2_episode_artifacts VALUES(?,?)", ("ep2-"+"a"*32,"art"))
                connection.execute("INSERT INTO v2_evidence VALUES(?,?,?)", ("ep2-"+"a"*32,"ev","{}"))
                connection.commit(); connection.close()
            complete = module.summarize(reports)
            self.assertTrue(complete["real_model_acceptance"])
            self.assertEqual(complete["passed_samples"], len(samples))
            first = next(iter(sorted(reports.glob("*.json"))))
            first.unlink()
            missing = module.summarize(reports)
            self.assertFalse(missing["real_model_acceptance"])
            self.assertEqual(len(missing["missing_reports"]), 1)

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


if __name__ == "__main__":
    unittest.main()

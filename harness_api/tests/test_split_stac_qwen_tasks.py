import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("split_stac_qwen_tasks", PROJECT / "scripts" / "split_stac_qwen_tasks.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SplitSTACQwenTasksTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_source(self):
        source = self.root / "runtime" / "source"
        (source / "tasks" / "stac-window-smoke").mkdir(parents=True)
        (source / "receipts").mkdir(parents=True)
        (source / "inputs").mkdir(parents=True)
        assets, manifest, records = [], {}, []
        for index in range(2):
            content = (str(index) * 128).encode()
            digest = __import__("hashlib").sha256(content).hexdigest()
            filename = f"item-{index}-visual.tif"
            (source / "inputs" / filename).write_bytes(content)
            record = {"item_id": f"item-{index}", "asset_key": "visual", "filename": filename,
                      "sha256": digest, "size_bytes": len(content), "source_snapshot_hash": "a" * 64,
                      "agent_admitted": True, "bbox_wgs84": [118, 31, 119, 32], "width": 10, "height": 10,
                      "acquired": "2024-04-05T00:00:00Z", "scene_cloud_cover_percent": 1,
                      "nodata_fraction": 0}
            records.append(record)
            identity_payload = [record["item_id"], "visual", digest]
            identity_bytes = (json.dumps(identity_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
            identity = "asset-" + __import__("hashlib").sha256(identity_bytes).hexdigest()
            asset = {"asset_id": identity, "sha256": digest, "roles": ["input_image"]}
            assets.append(asset)
            manifest[identity] = {"filename": filename, "sha256": digest}
        task = {"task_id": "original", "task_version": "1.0.0", "inputs": [a["asset_id"] for a in assets],
                "metadata": {"observation_profile": "headless-tools-v1", "artifact_identity": "derivation-sha256-v1"}}
        (source / "tasks" / "stac-window-smoke" / "task.json").write_text(json.dumps(task))
        (source / "tasks" / "stac-window-smoke" / "assets.json").write_text(json.dumps(assets))
        (source / "tasks" / "stac-window-smoke" / "scenario.json").write_text("{}")
        (source / "tasks" / "stac-window-smoke" / "evaluator.json").write_text("{}")
        (source / "inputs.json").write_text(json.dumps(manifest))
        (source / "receipts" / "admission.json").write_text(json.dumps({"status": "admitted", "windows": records, "agent_visual_inputs": 2}))
        return source

    def test_split_creates_two_distinct_one_window_tasks(self):
        source = self.make_source()
        output = self.root / "runtime" / "split"
        result = MODULE.split(source, output)
        self.assertEqual(len(result), 2)
        self.assertNotEqual(result[0]["content_sha256"], result[1]["content_sha256"])
        for sample in result:
            task_root = self.root / sample["task_root"]
            task = json.loads((task_root / "task.json").read_text())
            self.assertEqual(task["inputs"], [sample["asset_id"]])
            self.assertIn("display image", task["prompt"])
            self.assertEqual(json.loads((task_root.parent.parent / "job.json").read_text())["asset_id"], sample["asset_id"])
        receipt = json.loads((output / "split-receipt.json").read_text())
        self.assertEqual(receipt["selected"], result)

    def test_tampered_input_fails_before_output(self):
        source = self.make_source()
        path = next((source / "inputs").iterdir())
        path.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            MODULE.split(source, self.root / "runtime" / "split")


if __name__ == "__main__":
    unittest.main()

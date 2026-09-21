import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("split_native_ndvi_qwen", PROJECT / "scripts" / "split_native_ndvi_qwen_tasks.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SplitNativeNDVIQwenTests(unittest.TestCase):
    def make_source(self, count=2):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        source = Path(self.temp.name) / "source"
        task_dir = source / "tasks" / "native-ndvi"
        (source / "native-inputs").mkdir(parents=True)
        task_dir.mkdir(parents=True)
        assets, profiles, inputs = [], {}, []
        for index in range(count):
            item = {}
            for band in ("nir", "red"):
                content = (str(index) + band).encode()
                name = f"item-{index}-{band}.tif"
                (source / "native-inputs" / name).write_bytes(content)
                import hashlib
                digest = hashlib.sha256(content).hexdigest()
                asset_id = f"asset-{index}-{band}"
                asset = {"asset_id": asset_id, "bands": [band], "sha256": digest,
                         "uri": "local://approved-native/" + name}
                assets.append(asset)
                inputs.append(asset_id)
                item[band] = {"asset_id": asset_id, "sha256": digest, "item_id": f"item-{index}"}
                profiles[asset_id] = item[band]
        task = {"task_id": "source", "task_version": "1.0.0", "inputs": inputs,
                "metadata": {"artifact_identity": "derivation-sha256-v1", "raster_inputs": profiles},
                "budget": {}}
        (task_dir / "task.json").write_text(json.dumps(task))
        (task_dir / "assets.json").write_text(json.dumps(assets))
        (task_dir / "scenario.json").write_text("{}")
        (task_dir / "evaluator.json").write_text("{}")
        return source

    def test_split_keeps_two_independent_dates_and_tool_profile(self):
        source = self.make_source()
        output = source.parent / "out"
        result = MODULE.split(source, output)
        self.assertEqual(len(result), 2)
        self.assertNotEqual(result[0]["content_sha256"], result[1]["content_sha256"])
        for sample in result:
            task = json.loads((Path(sample["task_root"]) / "task.json").read_text())
            self.assertEqual(task["inputs"], sample["asset_ids"])
            self.assertEqual(task["metadata"]["raster_inputs"].keys(), set(sample["asset_ids"]))
            scenario = json.loads((Path(sample["task_root"]) / "scenario.json").read_text())
            self.assertIn("raster.band_math", scenario["allowed_tools"])

    def test_requires_two_dates(self):
        source = self.make_source(count=1)
        with self.assertRaises(ValueError):
            MODULE.split(source, source.parent / "out")


if __name__ == "__main__":
    unittest.main()

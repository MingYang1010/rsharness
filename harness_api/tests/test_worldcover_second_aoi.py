import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("worldcover_second_aoi", PROJECT / "scripts" / "prepare_worldcover_second_aoi.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WorldCoverSecondAOITests(unittest.TestCase):
    def prepare_root(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source = root / "tasks" / "worldcover-grounded-vqa-1.1.0"
        source.mkdir(parents=True)
        for name in ("task.json", "assets.json", "scenario.json", "evaluator.json"):
            shutil_copy = Path(PROJECT / "tasks" / "worldcover-grounded-vqa-1.1.0" / name)
            (source / name).write_bytes(shutil_copy.read_bytes())
        raster = root / "datasets" / "worldcover-2021" / "ESA_WorldCover_10m_2021_v200_N30E120_Map.tif"
        raster.parent.mkdir(parents=True)
        raster.write_bytes(b"canonical")
        assets = json.loads((source / "assets.json").read_text())
        canonical = next(item for item in assets if item["asset_id"].endswith("canonical"))
        canonical.update(sha256=hashlib.sha256(b"canonical").hexdigest(), size_bytes=len(b"canonical"))
        (source / "assets.json").write_text(json.dumps(assets))
        return root

    def test_prepare_fixes_second_aoi_and_distribution(self):
        root = self.prepare_root()
        observed = {"10": 31172, "30": 3999, "40": 92, "50": 2178, "60": 2250, "80": 1400309}
        pixels = [80] * MODULE.EXPECTED_DISTRIBUTION["80"]
        pixels.extend(int(value) for value, count in MODULE.EXPECTED_DISTRIBUTION.items() for _ in range(count) if value != "80")
        values = type("Values", (), {"compressed": lambda self: type("Array", (), {"tolist": lambda self: pixels, "size": len(pixels)})()})()
        class FakeDataset:
            crs = type("CRS", (), {"to_string": staticmethod(lambda: "EPSG:4326")})()
            def window(self, *args, **kwargs):
                return self
            def round_offsets(self):
                return self
            def round_lengths(self):
                return self
            def read(self, *args, masked=False, **kwargs):
                return values
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        class FakeRasterio:
            @staticmethod
            def open(path):
                return FakeDataset()
        import sys
        with patch.dict(sys.modules, {"rasterio": FakeRasterio}):
            result = MODULE.prepare(root / "tasks", root / "runtime" / "out")
        self.assertEqual(result["evaluation_aoi"], MODULE.SECOND_AOI)
        task = json.loads((root / "runtime" / "out" / "tasks" / "worldcover-water-qwen" / "task.json").read_text())
        evaluator = json.loads((root / "runtime" / "out" / "tasks" / "worldcover-water-qwen" / "evaluator.json").read_text())
        self.assertEqual(task["metadata"]["evaluation_aoi"], MODULE.SECOND_AOI)
        self.assertEqual(evaluator["config"]["expected_distribution"], observed)
        self.assertIn("qwen", task["metadata"]["acceptance"])

    def test_canonical_hash_mismatch_fails(self):
        root = self.prepare_root()
        (root / "datasets" / "worldcover-2021" / "ESA_WorldCover_10m_2021_v200_N30E120_Map.tif").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            MODULE.prepare(root / "tasks", root / "runtime" / "out")


if __name__ == "__main__":
    unittest.main()

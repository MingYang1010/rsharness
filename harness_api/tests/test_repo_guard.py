import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard = load_script("check_git_payload")
fetcher = load_script("fetch_eo_gym")


class RepositoryGuardTests(unittest.TestCase):
    def test_source_and_small_config_are_allowed(self):
        for path in ["app/tool.py", "config/eo-gym-source.json", "tasks/demo/assets.json", ".env.example"]:
            self.assertEqual(guard.check_entry(path, 100), [])

    def test_bulk_metadata_models_and_data_are_rejected(self):
        for path in ["datasets/a.jpg", "runtime/catalog.json", "metadata/all.json", "index_files/a.csv", "weights/a.safetensors", "data/train.jsonl", "a.tif", "x.zip", ".env.local"]:
            self.assertTrue(guard.check_entry(path, 100), path)

    def test_big_files_and_symlinks_are_rejected(self):
        self.assertTrue(guard.check_entry("small.txt", 1, "120000"))
        self.assertTrue(guard.check_entry("huge.json", 2 * 1024 * 1024 + 1))

    def test_source_selection_does_not_fetch_data(self):
        self.assertTrue(fetcher.source_file("src/eo_gym/server/app.py"))
        for path in ["EO_GYM_DATA.zip", "image_cache.zip", "index_files/labels.json", "datasets/train.jsonl"]:
            self.assertFalse(fetcher.source_file(path))

    def test_upstream_path_traversal_rejected(self):
        for path in ["../a.py", "/tmp/a.py", "src/../../a", "src\\a.py"]:
            with self.assertRaises(ValueError):
                fetcher.safe_relative(path)

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_repo_guard import load_script

inventory = load_script("inventory_datasets")


class InventoryTests(unittest.TestCase):
    def test_missing_root_does_not_claim_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {"dataset_base": str(root), "roots": ["missing"], "sample_count": 3, "sample_seed": 42, "sample_max_file_bytes": 1000}
            result = inventory.run(config, root / "out")
            self.assertEqual(result["datasets"][0]["status"], "missing")

    def test_labels_symlinks_and_bad_images_not_exposed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "dataset"
            images.mkdir()
            (images / "image.jpg").write_bytes(b"not-a-real-jpeg")
            (images / "image_mask.png").write_bytes(b"label")
            (images / "escape.jpg").symlink_to(root / "secret.jpg")
            config = {"dataset_base": str(root), "roots": ["dataset"], "sample_count": 3, "sample_seed": 42, "sample_max_file_bytes": 1000}
            with patch.object(inventory, "probe_image", side_effect=ValueError("invalid image")):
                result = inventory.run(config, root / "out")
            report = result["datasets"][0]
            self.assertEqual(report["candidate_images"], 1)
            self.assertEqual(report["label_like_images"], 1)
            self.assertEqual(report["symlinks_skipped"], 1)
            self.assertEqual(report["status"], "unavailable")
            self.assertEqual(report["agent_access"], "not_granted")

    def test_sampling_is_stable_and_content_is_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "dataset"
            images.mkdir()
            for i in range(8):
                (images / f"{i}.png").write_bytes(str(i).encode())
            config = {"dataset_base": str(root), "roots": ["dataset"], "sample_count": 3, "sample_seed": 42, "sample_max_file_bytes": 1000}
            with patch.object(inventory, "probe_image", return_value={"spatial": None, "reference_kind": "pixel"}):
                first = inventory.run(config, root / "a")
                second = inventory.run(config, root / "b")
            self.assertEqual(first, second)
            self.assertEqual(len(first["datasets"][0]["samples"]), 3)
            self.assertTrue(all(s["sha256"] for s in first["datasets"][0]["samples"]))
            self.assertEqual(first["datasets"][0]["status"], "sampled_readable")

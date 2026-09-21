import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("cloudsen12_public_windows", PROJECT / "scripts" / "admit_cloudsen12_public_windows.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CloudSEN12PublicWindowTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((PROJECT / "config" / "cloudsen12-public-samples-v1.json").read_text())

    def test_configuration_is_bounded_and_label_safe(self):
        MODULE.review_config(self.config)
        self.assertEqual(len(self.config["samples"]), 2)
        self.assertNotIn("manual_hq", json.dumps(self.config["samples"]))
        self.assertNotIn("sen2cor", json.dumps(self.config["samples"]))

    def test_review_rejects_more_than_two_or_bad_profiles(self):
        bad = json.loads(json.dumps(self.config))
        bad["samples"].append(bad["samples"][0])
        with self.assertRaises(ValueError):
            MODULE.review_config(bad)
        bad = json.loads(json.dumps(self.config))
        bad["samples"][0]["label_profile"]["width"] = 510
        with self.assertRaises(ValueError):
            MODULE.review_config(bad)

    def test_item_time_is_derived_from_sample_id(self):
        config = json.loads(json.dumps(self.config))
        value = MODULE.item_config(config, config["samples"][0])
        self.assertTrue(value["start"].startswith("2019-05-19T00"))
        self.assertTrue(value["end"].startswith("2019-05-19T23"))
        self.assertEqual(value["items"], [config["samples"][0]["sentinel_item_id"]])


if __name__ == "__main__":
    unittest.main()

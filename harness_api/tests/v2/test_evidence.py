import unittest

from pydantic import ValidationError

from app.v2.capabilities import TaskRegistry
from app.v2.evidence import validate_evidence
from app.v2.schemas import EvidenceRef

from .helpers import EVIDENCE_REQUEST, TASKS_ROOT


class V2EvidenceTests(unittest.TestCase):
    def setUp(self):
        registry = TaskRegistry(str(TASKS_ROOT))
        manifest = registry.get("worldcover-grounded-vqa", "1.0.0")
        self.assets = {asset.asset_id: asset for asset in manifest.assets}

    def test_valid_selector_replays_to_pinned_asset(self):
        evidence = EvidenceRef.model_validate(
            EVIDENCE_REQUEST["action"]["evidence"]
        )
        validate_evidence(evidence, self.assets)

    def test_empty_selector_is_rejected(self):
        value = dict(EVIDENCE_REQUEST["action"]["evidence"])
        value["selector"] = {
            "geometry": None,
            "bbox": None,
            "time_range": None,
            "bands": [],
            "pixel_window": None,
        }
        with self.assertRaises(ValidationError):
            EvidenceRef.model_validate(value)

    def test_selector_outside_asset_is_rejected(self):
        value = dict(EVIDENCE_REQUEST["action"]["evidence"])
        value["selector"] = dict(value["selector"])
        value["selector"]["bbox"] = {
            "west": 130.0,
            "south": 30.0,
            "east": 131.0,
            "north": 31.0,
        }
        evidence = EvidenceRef.model_validate(value)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_evidence(evidence, self.assets)


if __name__ == "__main__":
    unittest.main()

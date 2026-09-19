import copy
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from app.v2.cloud_policy_validation import evaluate_cloud_policy


PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prepare_cloud_policy_benchmark",
    PROJECT / "scripts" / "prepare_cloud_policy_benchmark.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _label(values: numpy.ndarray, *, transform=None) -> bytes:
    transform = transform or from_origin(257470.0, 4095840.0, 10.0, 10.0)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=509,
            height=509,
            count=1,
            dtype="float32",
            crs="EPSG:32618",
            transform=transform,
            nodata=-3.3999999521443642e38,
        ) as image:
            image.write(values.astype(numpy.float32), 1)
        return memory.read()


class CloudPolicyMetricTests(unittest.TestCase):
    def test_metrics_and_false_accept_are_exact(self):
        result = evaluate_cloud_policy(
            numpy.array([[0, 1], [2, 3]], dtype=numpy.float32),
            numpy.array([[4, 8], [4, 3]], dtype=numpy.float32),
            thresholds=[0.6],
        )
        self.assertEqual(
            result["confusion"],
            {
                "true_negative": 1,
                "false_positive": 0,
                "false_negative": 1,
                "true_positive": 2,
            },
        )
        self.assertEqual(result["fractions"], {"manual_invalid": 0.75, "policy_invalid": 0.5})
        self.assertEqual(result["metrics"]["precision"], 1.0)
        self.assertAlmostEqual(result["metrics"]["recall"], 2 / 3)
        self.assertAlmostEqual(result["metrics"]["intersection_over_union"], 2 / 3)
        self.assertAlmostEqual(result["metrics"]["balanced_accuracy"], 5 / 6)
        self.assertEqual(result["threshold_decisions"][0]["outcome"], "false_accept")

    def test_empty_metric_denominators_are_null(self):
        result = evaluate_cloud_policy(
            numpy.zeros((2, 2), dtype=numpy.float32),
            numpy.full((2, 2), 4, dtype=numpy.float32),
            thresholds=[0.2],
        )
        self.assertIsNone(result["metrics"]["precision"])
        self.assertIsNone(result["metrics"]["recall"])
        self.assertIsNone(result["metrics"]["intersection_over_union"])
        self.assertEqual(result["metrics"]["specificity"], 1.0)
        self.assertIsNone(result["metrics"]["balanced_accuracy"])


class CloudPolicyPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.output = self.runtime / "cloud-pack"
        self.config = json.loads(
            (PROJECT / "config" / "cloud-policy-benchmark-v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.config["samples"] = [copy.deepcopy(self.config["samples"][0])]
        manual_values = numpy.zeros((509, 509), dtype=numpy.float32)
        manual_values.flat[:4] = [0, 1, 2, 3]
        scl_values = numpy.full((509, 509), 4, dtype=numpy.float32)
        scl_values.flat[:4] = [4, 8, 4, 3]
        self.manual_values = manual_values
        self.scl_values = scl_values
        self._write_archive(_label(manual_values), _label(scl_values))

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_archive(self, manual: bytes, scl: bytes, *, duplicate=False) -> None:
        sample = self.config["samples"][0]
        sample["manual"].update(size_bytes=len(manual), sha256=_sha256(manual))
        sample["scl"].update(size_bytes=len(scl), sha256=_sha256(scl))
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for role, content in (("manual", manual), ("scl", scl)):
                info = tarfile.TarInfo(sample[role]["path"])
                info.size = len(content)
                info.mode = 0o600
                archive.addfile(info, io.BytesIO(content))
                if duplicate and role == "manual":
                    info = tarfile.TarInfo(sample["manual"]["path"])
                    info.size = len(manual)
                    archive.addfile(info, io.BytesIO(manual))
        content = stream.getvalue()
        self.archive = self.root / "prefix.tar"
        self.archive.write_bytes(content)
        self.config["archive"]["prefix"] = {
            "size_bytes": len(content),
            "sha256": _sha256(content),
        }

    def _prepare(self):
        return MODULE.prepare(
            self.archive,
            self.output,
            self.config,
            runtime_root=self.runtime,
        )

    def test_prepares_deterministic_private_label_pack(self):
        report = self._prepare()
        self.assertEqual(report["selected_sample_count"], 1)
        self.assertEqual(report["selected_member_count"], 2)
        self.assertFalse(report["scope_limits"]["global_accuracy_claim"])
        self.assertFalse(report["data_policy"]["labels_committed_to_git"])
        self.assertEqual(report["aggregate"]["confusion"]["true_positive"], 2)
        report_path = self.output / "cloud-policy-validation.json"
        self.assertEqual(json.loads(report_path.read_text()), report)
        for role in ("manual", "scl"):
            member = self.config["samples"][0][role]
            admitted = self.output / "labels" / member["path"]
            self.assertEqual(hashlib.sha256(admitted.read_bytes()).hexdigest(), member["sha256"])

    def test_archive_prefix_drift_is_rejected_before_output(self):
        self.config["archive"]["prefix"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "archive prefix size or checksum changed"):
            self._prepare()
        self.assertFalse(self.output.exists())

    def test_duplicate_member_is_rejected_before_output(self):
        self._write_archive(_label(self.manual_values), _label(self.scl_values), duplicate=True)
        with self.assertRaisesRegex(ValueError, "duplicate reviewed member"):
            self._prepare()
        self.assertFalse(self.output.exists())

    def test_member_hash_drift_is_rejected_before_output(self):
        self.config["samples"][0]["manual"]["sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "member checksum changed"):
            self._prepare()
        self.assertFalse(self.output.exists())

    def test_class_drift_is_rejected_before_output(self):
        invalid = self.manual_values.copy()
        invalid[0, 0] = 4
        self._write_archive(_label(invalid), _label(self.scl_values))
        with self.assertRaisesRegex(ValueError, "outside the reviewed class domain"):
            self._prepare()
        self.assertFalse(self.output.exists())

    def test_grid_drift_is_rejected_before_output(self):
        shifted = _label(
            self.scl_values,
            transform=from_origin(257480.0, 4095840.0, 10.0, 10.0),
        )
        self._write_archive(_label(self.manual_values), shifted)
        with self.assertRaisesRegex(ValueError, "TIFF profile changed"):
            self._prepare()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

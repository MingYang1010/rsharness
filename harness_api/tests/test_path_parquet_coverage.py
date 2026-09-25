import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.data.packed import AdmissionError
from app.core.data.path_coverage import audit_path_coverage, validate_coverage_output


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "requires isolated packed-requirements")
class PathParquetCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.images = self.dataset / "extracted/train/images"
        self.images.mkdir(parents=True)
        self.parquet = self.dataset / "samples.parquet"
        self.spec = {
            "dataset_id": "test-path-coverage",
            "root": str(self.dataset),
            "parquet": "samples.parquet",
            "image_root": "extracted",
            "image_column": "image_path",
            "allowed_path_prefixes": ["train/images"],
            "reviewed_role": "input_image",
            "review_id": "test-reviewed-coverage",
            "license": "test-fixture",
        }

    def tearDown(self):
        self.temp.cleanup()

    def write_png(self, name: str, width: int = 8, height: int = 6) -> None:
        import numpy as np
        import rasterio
        path = self.images / name
        with rasterio.open(path, "w", driver="PNG", width=width, height=height, count=3, dtype="uint8") as dst:
            dst.write(np.zeros((3, height, width), dtype="uint8"))

    def write_tiff(self, name: str, width: int, height: int, count: int = 1) -> None:
        import rasterio
        path = self.images / name
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=count,
            dtype="uint8",
            tiled=True,
            compress="deflate",
            SPARSE_OK="TRUE",
        ):
            pass

    def write_parquet(self, values: list[str]) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.table({
            "image_path": values,
            "answer": ["PRIVATE_GOLD"] * len(values),
            "annotation_path": ["PRIVATE_LABEL"] * len(values),
        })
        pq.write_table(table, self.parquet, row_group_size=2)

    def test_complete_aggregate_separates_whole_window_duplicate_and_failures(self):
        self.write_png("small.png")
        self.write_tiff("large.tif", width=5000, height=5000)
        self.write_tiff("five-band.tif", width=8, height=6, count=5)
        (self.images / "unsupported.bin").write_bytes(b"not an image")
        values = [
            "train/images/small.png",
            "train/images/small.png",
            "train/images/large.tif",
            "train/images/five-band.tif",
            "train/images/unsupported.bin",
            "train/images/missing.png",
            "private/hidden.png",
        ]
        self.write_parquet(values)
        report, receipt = audit_path_coverage(self.spec)
        self.assertTrue(report["scan_complete"])
        self.assertEqual(report["rows_scanned"], len(values))
        self.assertEqual(report["duplicate_path_rows"], 1)
        self.assertEqual(report["unique_files_by_outcome"], {
            "reviewed_window_header_candidate": 1,
            "whole_image_header_eligible": 1,
        })
        self.assertEqual(report["failures_by_code"], {
            "image_path_missing": 1,
            "image_path_prefix_forbidden": 1,
            "provider_image_dimensions_limit": 1,
            "unsupported_image_encoding": 1,
        })
        public = json.dumps(report)
        self.assertNotIn("small.png", public)
        self.assertNotIn("PRIVATE_", public)
        self.assertFalse(report["label_columns_read"])
        self.assertFalse(report["pixel_content_read"])
        self.assertFalse(report["image_content_hashes_verified"])
        self.assertFalse(receipt["row_values_retained"])

    def test_max_rows_is_explicitly_partial(self):
        self.write_png("small.png")
        self.write_parquet(["train/images/small.png"] * 3)
        progress = []
        report, _ = audit_path_coverage(self.spec, max_rows=2, progress=lambda rows, limit: progress.append((rows, limit)))
        self.assertFalse(report["scan_complete"])
        self.assertEqual(report["rows_total"], 3)
        self.assertEqual(report["rows_scanned"], 2)
        self.assertEqual(progress[-1], (2, 2))

    def test_changed_image_aborts_coverage_claim(self):
        self.write_png("small.png")
        self.write_parquet(["train/images/small.png"])
        with patch(
            "app.core.data.path_coverage._unchanged",
            side_effect=AdmissionError("image_changed_during_header_audit"),
        ):
            with self.assertRaisesRegex(AdmissionError, "image_changed_during_header_audit"):
                audit_path_coverage(self.spec)

    def test_invalid_limit_and_unreviewed_column_fail_before_scan(self):
        self.write_png("small.png")
        self.write_parquet(["train/images/small.png"])
        with self.assertRaisesRegex(AdmissionError, "coverage_row_limit_invalid"):
            audit_path_coverage(self.spec, max_rows=0)
        with self.assertRaises(AdmissionError):
            audit_path_coverage({**self.spec, "image_column": "annotation_path"})

    def test_output_must_be_fresh_and_outside_source(self):
        outside = self.root / "runtime/coverage"
        validate_coverage_output(outside, self.dataset)
        with self.assertRaisesRegex(AdmissionError, "output_must_not_modify_source_root"):
            validate_coverage_output(self.dataset / "runtime/coverage", self.dataset)
        outside.mkdir(parents=True)
        with self.assertRaisesRegex(AdmissionError, "output_exists_preserve_previous_run"):
            validate_coverage_output(outside, self.dataset)


if __name__ == "__main__":
    unittest.main()

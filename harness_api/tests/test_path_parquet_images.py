import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from app.core.data.packed import AdmissionError
from app.core.data.path_parquet import (
    checked_local_image,
    extract_path_samples,
    reviewed_relative_path,
)


def png_bytes(width=8, height=6):
    def chunk(kind, content):
        return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content))

    pixels = bytes(index % 251 for index in range(width * height * 3))
    stride = width * 3
    rows = b"".join(b"\x00" + pixels[offset:offset + stride] for offset in range(0, len(pixels), stride))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "requires isolated packed-requirements")
class PathParquetImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.images = self.dataset / "extracted"
        (self.images / "train/images").mkdir(parents=True)
        self.parquet = self.dataset / "samples.parquet"
        self.output = self.root / "runtime/path-samples"
        self.spec = {
            "dataset_id": "test-path-parquet",
            "root": str(self.dataset),
            "parquet": "samples.parquet",
            "image_root": "extracted",
            "image_column": "image_path",
            "allowed_path_prefixes": ["train/images"],
            "reviewed_role": "input_image",
            "review_id": "test-reviewed-relative-path",
            "license": "test-fixture",
        }
        self.payloads = [png_bytes(width=value, height=6) for value in (8, 9, 10)]

    def tearDown(self):
        self.temp.cleanup()

    def write_fixture(self, values=None):
        import pyarrow as pa
        import pyarrow.parquet as pq
        values = values or [f"train/images/{index}.png" for index in range(3)]
        for index, content in enumerate(self.payloads):
            (self.images / f"train/images/{index}.png").write_bytes(content)
        table = pa.table({
            "image_path": values,
            "answer": ["PRIVATE_GOLD_ANSWER"] * len(values),
            "annotation_path": ["PRIVATE_GOLD_LABEL"] * len(values),
        })
        pq.write_table(table, self.parquet, row_group_size=1)

    def test_extracts_three_images_and_keeps_paths_private(self):
        self.write_fixture()
        report = extract_path_samples(self.spec, self.output)
        self.assertEqual(report["unique_images"], 3)
        self.assertFalse(report["label_columns_exported"])
        public = (self.output / "coverage.json").read_text() + (self.output / "inputs.json").read_text()
        self.assertNotIn("train/images", public)
        self.assertNotIn("PRIVATE_GOLD", public)
        receipt = json.loads((self.output / "private/receipt.json").read_text())
        self.assertEqual(receipt["parquet"]["sha256"], hashlib.sha256(self.parquet.read_bytes()).hexdigest())
        self.assertEqual(len(receipt["origins"]), 3)
        for sample in report["datasets"][0]["samples"]:
            staged = self.output / "inputs" / sample["relative_path"]
            self.assertEqual(hashlib.sha256(staged.read_bytes()).hexdigest(), sample["sha256"])

    def test_rejects_url_absolute_traversal_windows_and_non_normal_paths(self):
        denied = [
            "https://example.com/a.png", "/etc/passwd", "../a.png",
            "train/images/../a.png", "train//images/a.png", "./train/images/a.png",
            "C:\\private\\a.png", "//server/share/a.png", "train\\images\\a.png",
        ]
        for value in denied:
            with self.subTest(value=value), self.assertRaises(AdmissionError):
                reviewed_relative_path(value)

    def test_requires_allowed_component_prefix(self):
        self.write_fixture(["private/0.png"])
        (self.images / "private").mkdir()
        (self.images / "private/0.png").write_bytes(self.payloads[0])
        report = extract_path_samples(self.spec, self.output, sample_count=1)
        self.assertEqual(report["unique_images"], 0)
        self.assertEqual(report["failures"], [{"row": 0, "code": "image_path_prefix_forbidden"}])

    def test_symlink_file_and_parent_are_rejected(self):
        self.write_fixture()
        outside = self.root / "outside.png"
        outside.write_bytes(self.payloads[0])
        link = self.images / "train/images/link.png"
        link.symlink_to(outside)
        with self.assertRaisesRegex(AdmissionError, "symlink_image_path"):
            checked_local_image(self.images, PurePosixPath("train/images/link.png"))
        parent = self.images / "linked"
        parent.symlink_to(self.images / "train", target_is_directory=True)
        with self.assertRaisesRegex(AdmissionError, "symlink_image_path"):
            checked_local_image(self.images, PurePosixPath("linked/images/0.png"))

    def test_missing_image_is_a_sanitized_partial_failure(self):
        self.write_fixture(["train/images/missing.png"])
        report = extract_path_samples(self.spec, self.output, sample_count=1)
        self.assertEqual(report["unique_images"], 0)
        self.assertEqual(report["failures"], [{"row": 0, "code": "image_path_missing"}])
        self.assertNotIn("missing.png", json.dumps(report))

    def test_label_column_and_unreviewed_spec_are_denied_before_output(self):
        self.write_fixture()
        variants = [
            {**self.spec, "image_column": "annotation_path"},
            {**self.spec, "reviewed_role": "label"},
            {**self.spec, "review_id": ""},
            {**self.spec, "allowed_path_prefixes": []},
        ]
        for spec in variants:
            with self.subTest(spec=spec), self.assertRaises(AdmissionError):
                extract_path_samples(spec, self.output)
            self.assertFalse(self.output.exists())

    def test_output_source_and_byte_limits(self):
        self.write_fixture()
        with self.assertRaisesRegex(AdmissionError, "output_must_not_modify_source_root"):
            extract_path_samples(self.spec, self.dataset / "runtime")
        report = extract_path_samples(self.spec, self.output, max_output_bytes=1)
        self.assertEqual(report["unique_images"], 0)
        self.assertTrue(all(value["code"] == "output_byte_limit" for value in report["failures"]))
        with self.assertRaisesRegex(AdmissionError, "output_exists_preserve_previous_run"):
            extract_path_samples(self.spec, self.output)

    def test_changed_image_does_not_publish_manifest(self):
        self.write_fixture()
        original = Path.read_bytes

        def changed(path):
            content = original(path)
            if path.suffix == ".png":
                path.write_bytes(content + b"x")
            return content

        with patch("pathlib.Path.read_bytes", changed):
            with self.assertRaisesRegex(AdmissionError, "image_changed_during_read"):
                extract_path_samples(self.spec, self.output)
        self.assertFalse((self.output / "coverage.json").exists())
        self.assertFalse((self.output / "inputs.json").exists())


if __name__ == "__main__":
    unittest.main()

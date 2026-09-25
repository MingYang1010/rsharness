import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from app.core.data.packed import (
    AdmissionError, checked_shard, embedded_images, extract_samples,
    probe_bytes, projected_rows, validate_column,
)


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "requires isolated packed-requirements")
class PackedImageTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        self.pa = pa
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "runtime" / "samples"
        self.spec = {"dataset_id": "test-packed", "root": str(self.source), "image_column": "image",
                     "reviewed_role": "input_image", "review_id": "test-reviewed-image",
                     "license": "test-fixture"}
        from v2.test_pixel_artifacts import png_bytes
        self.images = [png_bytes(width=w, height=6) for w in (8, 9, 10)]

    def tearDown(self):
        self.temp.cleanup()

    def write(self, kind="stream", images=None, sequence=False):
        import pyarrow.parquet as pq
        pa = self.pa
        values = [{"bytes": b, "path": "PRIVATE_GOLD_PATH_must_not_export"} for b in (images or self.images)]
        image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
        if sequence:
            values = [[v] for v in values]
            image_type = pa.list_(image_type)
        table = pa.table({"image": pa.array(values, type=image_type),
                          "answer": ["PRIVATE_GOLD_ANSWER_must_not_export"] * len(values),
                          "bbox": [[1, 2, 3, 4]] * len(values)})
        path = self.source / ("data.parquet" if kind == "parquet" else "data.arrow")
        if kind == "parquet":
            pq.write_table(table, path, row_group_size=1)
        else:
            writer = pa.ipc.new_stream if kind == "stream" else pa.ipc.new_file
            with pa.OSFile(str(path), "wb") as sink, writer(sink, table.schema) as stream:
                stream.write_table(table)
        return path

    def assert_ready(self, report, path):
        self.assertEqual(report["unique_images"], 3)
        self.assertFalse(report["label_columns_exported"])
        receipt = json.loads((self.output / "private" / "receipt.json").read_text())
        self.assertEqual(receipt["sources"][0]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        combined = (self.output / "coverage.json").read_text() + (self.output / "private" / "receipt.json").read_text() + (self.output / "inputs.json").read_text()
        self.assertNotIn("PRIVATE_GOLD", combined)
        self.assertEqual(len(list((self.output / "inputs").iterdir())), 3)
        for sample in report["datasets"][0]["samples"]:
            self.assertIsNone(sample["spatial"])
            staged = self.output / "inputs" / sample["relative_path"]
            self.assertEqual(hashlib.sha256(staged.read_bytes()).hexdigest(), sample["sha256"])
            self.assertEqual(sample["asset_id"], "asset-" + sample["sha256"])

    def test_arrow_stream_projected_images_and_private_receipts(self):
        path = self.write()
        self.assert_ready(extract_samples(self.spec, self.output), path)

    def test_arrow_file_sequence_images(self):
        path = self.write("file", sequence=True)
        self.assert_ready(extract_samples(self.spec, self.output), path)

    def test_parquet_projects_only_image_column(self):
        path = self.write("parquet")
        self.assertEqual(len(list(projected_rows(path, "image", 2))), 2)
        self.assert_ready(extract_samples(self.spec, self.output), path)

    def test_duplicates_share_one_input_but_keep_origins(self):
        self.write(images=[self.images[0], self.images[0], self.images[1]])
        report = extract_samples(self.spec, self.output)
        self.assertEqual(report["unique_images"], 2)
        self.assertEqual(report["duplicate_images"], 1)
        self.assertEqual(report["datasets"][0]["status"], "partial")
        receipt = json.loads((self.output / "private" / "receipt.json").read_text())
        self.assertEqual(len(receipt["origins"]), 3)
        self.assertEqual(len(list((self.output / "inputs").iterdir())), 2)

    def test_no_review_or_label_column_cannot_be_admitted(self):
        self.write()
        for spec in ({**self.spec, "review_id": ""}, {**self.spec, "reviewed_role": "label"},
                     {**self.spec, "image_column": "answer"}):
            with self.subTest(spec=spec), self.assertRaises(AdmissionError):
                extract_samples(spec, self.output)
        self.assertFalse(self.output.exists())
        for column in ("label", "answer_image", "masks", "image.bytes", "../image"):
            with self.subTest(column=column), self.assertRaises(AdmissionError):
                validate_column(column)

    def test_external_paths_and_non_image_payload_never_opened(self):
        for value in ({"bytes": None, "path": "/etc/passwd"}, {"path": "https://example.com/secret"}, None):
            with self.subTest(value=value), self.assertRaises(AdmissionError):
                embedded_images(value)
        with self.assertRaises(AdmissionError):
            probe_bytes(b'<VRTDataset><SourceFilename>/etc/passwd</SourceFilename></VRTDataset>')
        with self.assertRaises(AdmissionError):
            probe_bytes(self.images[0][:33])
        for value in ([self.images[0]] * 17, [[self.images[0]] * 16] * 2, [[[[self.images[0]]]]]):
            with self.subTest(sequence="bounded"), self.assertRaises(AdmissionError):
                embedded_images(value)

    def test_size_row_and_output_limits(self):
        self.write()
        report = extract_samples(self.spec, self.output, max_rows=1)
        self.assertEqual(report["unique_images"], 1)
        small_output = self.root / "runtime" / "limited"
        report = extract_samples(self.spec, small_output, max_output_bytes=1)
        self.assertEqual(report["output_bytes"], 0)
        self.assertEqual(report["unique_images"], 0)
        self.assertEqual(json.loads((small_output / "inputs.json").read_text()), {})
        self.assertTrue(all(f["code"] == "output_byte_limit" for f in report["failures"]))

    def test_source_path_and_existing_output_protection(self):
        path = self.write()
        link = self.source / "link.arrow"
        link.symlink_to(path)
        with self.assertRaises(AdmissionError):
            checked_shard(self.source, link)
        outside = self.root / "outside.arrow"
        outside.write_bytes(path.read_bytes())
        with self.assertRaises(AdmissionError):
            checked_shard(self.source, outside)
        with self.assertRaises(AdmissionError):
            extract_samples(self.spec, self.source / "extracted")
        report = extract_samples(self.spec, self.output)
        with self.assertRaises(AdmissionError):
            extract_samples(self.spec, self.output)
        self.assertEqual(json.loads((self.output / "coverage.json").read_text()), report)

    def test_corrupt_container_records_error_without_row_content(self):
        (self.source / "bad.arrow").write_bytes(b"PRIVATE_GOLD_broken_container")
        report = extract_samples(self.spec, self.output)
        self.assertEqual(report["unique_images"], 0)
        self.assertTrue(report["failures"])
        self.assertNotIn("PRIVATE_GOLD", json.dumps(report))
        self.assertEqual(json.loads((self.output / "inputs.json").read_text()), {})

    def test_source_change_does_not_publish_admission_manifest(self):
        import os
        from unittest.mock import patch
        path = self.write()
        original = projected_rows
        def changed(source_path, column, max_rows):
            current = source_path.stat()
            os.utime(source_path, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000))
            yield from original(source_path, column, max_rows)
        with patch("app.core.data.packed.projected_rows", changed):
            with self.assertRaisesRegex(AdmissionError, "source_changed_during_read"):
                extract_samples(self.spec, self.output)
        self.assertFalse((self.output / "inputs.json").exists())
        self.assertFalse((self.output / "coverage.json").exists())

    def test_source_change_then_parser_error_does_not_publish(self):
        import os
        from unittest.mock import patch
        self.write()
        def changed(source_path, column, max_rows):
            yield 0, self.images[0]
            current = source_path.stat()
            os.utime(source_path, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000))
            raise self.pa.ArrowInvalid("PRIVATE_GOLD_parser_error")
        with patch("app.core.data.packed.projected_rows", changed):
            with self.assertRaisesRegex(AdmissionError, "source_changed_during_read"):
                extract_samples(self.spec, self.output)
        self.assertFalse((self.output / "inputs.json").exists())
        self.assertFalse((self.output / "coverage.json").exists())

    def test_receipt_budget_precedes_any_metadata_publication(self):
        from unittest.mock import patch
        self.write()
        with patch("app.core.data.packed.MAX_RECEIPT_BYTES", 1):
            with self.assertRaisesRegex(AdmissionError, "receipt_size_limit"):
                extract_samples(self.spec, self.output)
        self.assertFalse((self.output / "inputs.json").exists())
        self.assertFalse((self.output / "coverage.json").exists())
        self.assertFalse((self.output / "private/receipt.json").exists())


if __name__ == "__main__":
    unittest.main()

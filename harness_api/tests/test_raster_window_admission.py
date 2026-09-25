import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.data.packed import AdmissionError, digest_file as real_digest_file
from app.core.data.raster_windows import (
    POLICY_ID,
    extract_raster_windows,
    iter_grid_windows,
    verify_raster_window_output,
)


@unittest.skipUnless(
    importlib.util.find_spec("pyarrow") and importlib.util.find_spec("rasterio"),
    "requires isolated packed-requirements and rasterio",
)
class RasterWindowAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.images = self.dataset / "extracted/train/images"
        self.images.mkdir(parents=True)
        self.parquet = self.dataset / "samples.parquet"
        self.output = self.root / "runtime/windows"
        self.spec = {
            "dataset_id": "test-reviewed-window",
            "root": str(self.dataset),
            "parquet": "samples.parquet",
            "image_root": "extracted",
            "image_column": "image_path",
            "allowed_path_prefixes": ["train/images"],
            "reviewed_role": "input_image",
            "review_id": "test-reviewed-window-policy",
            "window_policy": POLICY_ID,
            "license": "test-fixture",
        }

    def tearDown(self):
        self.temp.cleanup()

    def write_raster(self, name: str, value: int, width: int = 5, height: int = 5) -> Path:
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        path = self.images / name
        pixels = np.full((3, height, width), value, dtype="uint8")
        pixels[1] += 1
        pixels[2] += 2
        mask = np.full((height, width), 255, dtype="uint8")
        mask[-1, -1] = 0
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK="YES"):
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=3,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_origin(10 + value / 100, 20, 0.01, 0.01),
                nodata=0,
            ) as image:
                image.write(pixels)
                image.write_mask(mask)
        return path

    def write_fixture(self) -> list[Path]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        small = self.write_raster("small.tif", 1, width=4, height=4)
        sources = [self.write_raster(f"source-{index}.tif", 10 + index) for index in range(3)]
        values = [
            "train/images/small.tif",
            "train/images/source-0.tif",
            "train/images/source-1.tif",
            "train/images/source-1.tif",
            "train/images/source-2.tif",
        ]
        table = pa.table({
            "image_path": values,
            "answer": ["PRIVATE_GOLD_ANSWER"] * len(values),
            "annotation_path": ["PRIVATE_GOLD_LABEL"] * len(values),
        })
        pq.write_table(table, self.parquet, row_group_size=1)
        return [small, *sources]

    def test_grid_is_row_major_and_clips_edges(self):
        windows = list(iter_grid_windows(5000, 5000))
        self.assertEqual(len(windows), 9)
        self.assertEqual(windows[:3], [
            (0, 0, 2048, 2048),
            (2048, 0, 2048, 2048),
            (4096, 0, 904, 2048),
        ])
        self.assertEqual(windows[-1], (4096, 4096, 904, 904))

    def test_round_robin_sources_preserve_pixels_grid_mask_and_private_paths(self):
        import numpy as np
        import rasterio

        paths = self.write_fixture()[1:]
        with patch("app.core.data.raster_windows.MAX_IMAGE_PIXELS", 16):
            report = extract_raster_windows(self.spec, self.output, sample_count=3)
        self.assertEqual(report["unique_windows"], 3)
        self.assertEqual(report["unique_sources"], 3)
        self.assertEqual(report["duplicate_source_rows"], 1)
        self.assertEqual(report["ineligible_whole_images"], 1)
        self.assertFalse(report["label_columns_exported"])
        public = (self.output / "coverage.json").read_text() + (self.output / "inputs.json").read_text()
        self.assertNotIn("train/images", public)
        self.assertNotIn("PRIVATE_GOLD", public)
        receipt = json.loads((self.output / "private/receipt.json").read_text())
        self.assertEqual([value["parquet_rows"] for value in receipt["sources"]], [[1], [2, 3], [4]])
        for value in receipt["windows"]:
            canonical = json.dumps(value["derivation"], sort_keys=True, separators=(",", ":")).encode()
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), value["source_snapshot_hash"])
        for path, sample in zip(paths, report["datasets"][0]["samples"]):
            staged = self.output / "inputs" / sample["relative_path"]
            self.assertTrue(staged.exists())
            self.assertFalse(Path(str(staged) + ".msk").exists())
            with rasterio.open(path) as source, rasterio.open(staged) as derived:
                self.assertTrue(np.array_equal(source.read(), derived.read()))
                self.assertTrue(np.array_equal(source.read_masks(1), derived.read_masks(1)))
                self.assertEqual(source.crs, derived.crs)
                self.assertEqual(source.transform, derived.transform)
                self.assertEqual(source.nodata, derived.nodata)
                self.assertEqual(source.colorinterp, derived.colorinterp)
            self.assertEqual(sample["pixel_window"], {"col_off": 0, "row_off": 0, "width": 5, "height": 5})
            self.assertEqual(sample["admission_profile"], POLICY_ID)
            self.assertEqual(real_digest_file(staged), sample["sha256"])
        self.assertEqual(verify_raster_window_output(self.output), {
            "status": "passed",
            "policy_id": POLICY_ID,
            "verified_windows": 3,
            "verified_sources": 3,
        })

    def test_derivation_is_repeatable_and_binds_parquet_source_and_window(self):
        self.write_fixture()
        second = self.root / "runtime/windows-second"
        with patch("app.core.data.raster_windows.MAX_IMAGE_PIXELS", 16):
            first_report = extract_raster_windows(self.spec, self.output, sample_count=3)
            second_report = extract_raster_windows(self.spec, second, sample_count=3)
        first = first_report["datasets"][0]["samples"]
        repeated = second_report["datasets"][0]["samples"]
        self.assertEqual(
            [(value["source_snapshot_hash"], value["sha256"]) for value in first],
            [(value["source_snapshot_hash"], value["sha256"]) for value in repeated],
        )

        changed_source = self.write_raster("source-0.tif", 90)
        third = self.root / "runtime/windows-third"
        with patch("app.core.data.raster_windows.MAX_IMAGE_PIXELS", 16):
            changed_report = extract_raster_windows(self.spec, third, sample_count=3)
        changed = changed_report["datasets"][0]["samples"]
        self.assertNotEqual(first[0]["source_snapshot_hash"], changed[0]["source_snapshot_hash"])
        self.assertEqual(changed_source, self.images / "source-0.tif")

    def test_requires_reviewed_fixed_policy_before_output(self):
        self.write_fixture()
        spec = {**self.spec, "window_policy": "arbitrary-window-v0"}
        with self.assertRaisesRegex(AdmissionError, "window_policy_review_required"):
            extract_raster_windows(spec, self.output)
        self.assertFalse(self.output.exists())

    def test_output_byte_limit_is_sanitized_and_does_not_publish_assets(self):
        self.write_fixture()
        with patch("app.core.data.raster_windows.MAX_IMAGE_PIXELS", 16):
            report = extract_raster_windows(self.spec, self.output, sample_count=1, max_output_bytes=1)
        self.assertEqual(report["unique_windows"], 0)
        self.assertTrue(any(value["code"] == "output_byte_limit" for value in report["failures"]))
        self.assertEqual(json.loads((self.output / "inputs.json").read_text()), {})

    def test_source_mutation_aborts_publication(self):
        paths = self.write_fixture()
        target = paths[1]
        mutated = False

        def digest_then_mutate(path):
            nonlocal mutated
            value = real_digest_file(path)
            if path == target and not mutated:
                mutated = True
                path.write_bytes(path.read_bytes() + b"x")
            return value

        with patch("app.core.data.raster_windows.MAX_IMAGE_PIXELS", 16), patch(
            "app.core.data.raster_windows.digest_file", side_effect=digest_then_mutate
        ):
            with self.assertRaisesRegex(AdmissionError, "source_changed_during_hash"):
                extract_raster_windows(self.spec, self.output, sample_count=1)
        self.assertFalse((self.output / "coverage.json").exists())
        self.assertFalse((self.output / "inputs.json").exists())

    def test_verifier_rejects_tampered_staged_window(self):
        self.write_fixture()
        with patch("app.core.data.raster_windows.MAX_IMAGE_PIXELS", 16):
            report = extract_raster_windows(self.spec, self.output, sample_count=1)
        sample = report["datasets"][0]["samples"][0]
        staged = self.output / "inputs" / sample["relative_path"]
        staged.write_bytes(staged.read_bytes() + b"tamper")
        with self.assertRaisesRegex(AdmissionError, "staged_window_checksum_mismatch"):
            verify_raster_window_output(self.output)


if __name__ == "__main__":
    unittest.main()

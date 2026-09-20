#!/usr/bin/env python3
"""Independent pixel-centre reference for real continuous grid alignment."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.v2.storage.quota import StorageQuota

NODATA = -9999.


def bilinear(source, target):
    """Manual bilinear interpolation; deliberately no production warp call."""
    raw = source.read(1)
    physical = raw.astype("float64") * source.scales[0] + source.offsets[0]
    source_valid = source.read_masks(1) > 0
    if source.nodata is not None:
        source_valid &= raw != source.nodata
    expected = np.full((target.height, target.width), NODATA, dtype="float32")
    expected_valid = np.zeros((target.height, target.width), dtype=bool)
    for row in range(target.height):
        for column in range(target.width):
            x, y = target.transform * (column + .5, row + .5)
            if source.crs != target.crs:
                from rasterio.warp import transform
                transformed = transform(target.crs, source.crs, [x], [y])
                x, y = transformed[0][0], transformed[1][0]
            source_column, source_row = ~source.transform * (x, y)
            if not (0 <= source_column <= source.width
                    and 0 <= source_row <= source.height):
                continue
            centered_column = source_column - .5
            centered_row = source_row - .5
            left, top = int(np.floor(centered_column)), int(np.floor(centered_row))
            right, bottom = left + 1, top + 1
            dx, dy = centered_column - left, centered_row - top
            samples = ((top, left, (1 - dx) * (1 - dy)),
                       (top, right, dx * (1 - dy)),
                       (bottom, left, (1 - dx) * dy),
                       (bottom, right, dx * dy))
            value = 0.
            valid_weight = 0.
            for source_row_index, source_column_index, weight in samples:
                source_row_index = min(max(source_row_index, 0), source.height - 1)
                source_column_index = min(max(source_column_index, 0), source.width - 1)
                if source_valid[source_row_index, source_column_index]:
                    value += physical[source_row_index, source_column_index] * weight
                    valid_weight += weight
            if valid_weight >= 1 - 1e-6:
                expected[row, column] = np.float32(value / valid_weight)
                expected_valid[row, column] = True
    return expected, expected_valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    arguments = parser.parse_args()
    run = arguments.run.resolve()
    if not run.is_relative_to((ROOT / "runtime").resolve()):
        raise SystemExit("runtime run required")
    manifest = json.loads((run / "native-inputs.json").read_text())
    saved = json.loads((run / "reports/continuous-grid-checkpoint.json").read_text())
    if len(saved.get("artifacts", [])) != 1 or len(saved.get("results", [])) != 1:
        raise SystemExit("one completed continuous artifact required")
    artifact, result = saved["artifacts"][0], saved["results"][0]
    source_entry, reference_entry = [manifest[key]
                                     for key in result["input_asset_ids"]]
    source_path = run / "native-inputs" / source_entry["filename"]
    reference_path = run / "native-inputs" / reference_entry["filename"]
    for path, entry in ((source_path, source_entry),
                        (reference_path, reference_entry)):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["native"]["sha256"]
    with rasterio.open(source_path, driver="GTiff") as source, rasterio.open(
            reference_path, driver="GTiff") as reference:
        expected, expected_valid = bilinear(source, reference)
        target_transform, target_crs = reference.transform, reference.crs
    digest = artifact["sha256"]
    content_path = ROOT / "runtime/managed-artifacts" / digest[:2] / digest / "content"
    assert hashlib.sha256(content_path.read_bytes()).hexdigest() == digest
    with rasterio.open(content_path, driver="GTiff") as output:
        actual = output.read(1)
        actual_valid = output.read_masks(1) > 0
        assert output.transform == target_transform and output.crs == target_crs
        assert np.array_equal(actual_valid, expected_valid)
        difference = np.abs(actual[actual_valid].astype("float64")
                            - expected[actual_valid].astype("float64"))
        maximum_error = float(difference.max()) if difference.size else 0.
        assert maximum_error <= 1e-7
        assert np.all(actual[~actual_valid] == NODATA)
    report = {"status": "passed", "artifact_id": artifact["artifact_id"],
              "sha256": digest, "valid_pixels": int(expected_valid.sum()),
              "total_pixels": int(expected_valid.size),
              "maximum_absolute_error": maximum_error,
              "mask_exact": True, "target_grid_exact": True,
              "scope": ("numeric B11-to-B08 alignment reference only; not "
                        "an NDMI, cloud or land-cover validation")}
    report_path = run / "reference/continuous-grid-numeric.json"
    if report_path.exists():
        raise SystemExit("preserve prior report")
    with StorageQuota(ROOT / "runtime").hold(
            report_path.parent, 1024 * 1024, "continuous-grid-reference-report"):
        report_path.parent.mkdir()
        with report_path.open("x") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()

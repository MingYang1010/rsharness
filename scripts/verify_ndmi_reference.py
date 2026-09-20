#!/usr/bin/env python3
"""Independent scalar B11 alignment, fixed NDMI and zonal reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.v2.storage.quota import StorageQuota

NODATA = -9999.


def bilinear(source, target):
    raw = source.read(1)
    physical = raw.astype("float64") * source.scales[0] + source.offsets[0]
    source_valid = source.read_masks(1) > 0
    if source.nodata is not None:
        source_valid &= raw != source.nodata
    expected = np.full((target.height, target.width), NODATA, dtype="float32")
    expected_valid = np.zeros(expected.shape, dtype=bool)
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
            dx, dy = centered_column - left, centered_row - top
            samples = ((top, left, (1 - dx) * (1 - dy)),
                       (top, left + 1, dx * (1 - dy)),
                       (top + 1, left, (1 - dx) * dy),
                       (top + 1, left + 1, dx * dy))
            value = 0.
            valid_weight = 0.
            for sample_row, sample_column, weight in samples:
                sample_row = min(max(sample_row, 0), source.height - 1)
                sample_column = min(max(sample_column, 0), source.width - 1)
                if source_valid[sample_row, sample_column]:
                    value += physical[sample_row, sample_column] * weight
                    valid_weight += weight
            if valid_weight >= 1 - 1e-6:
                expected[row, column] = np.float32(value / valid_weight)
                expected_valid[row, column] = True
    return expected, expected_valid


def on_segment(x, y, first, second):
    x1, y1 = first
    x2, y2 = second
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    tolerance = max(1., abs(x), abs(y), abs(x1), abs(y1),
                    abs(x2), abs(y2)) * 1e-12
    return (abs(cross) <= tolerance
            and min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
            and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance)


def contains(x, y, coordinates):
    inside = False
    for first, second in zip(coordinates[:-1], coordinates[1:]):
        if on_segment(x, y, first, second):
            return True
        x1, y1 = first
        x2, y2 = second
        if (y1 > y) != (y2 > y):
            intersection = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersection:
                inside = not inside
    return inside


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    arguments = parser.parse_args()
    run = arguments.run.resolve()
    if not run.is_relative_to((ROOT / "runtime").resolve()):
        raise SystemExit("runtime run required")
    provider = json.loads((run / "native-inputs.json").read_text())
    checkpoint = json.loads((run / "reports/ndmi-checkpoint.json").read_text())
    task = json.loads((run / "tasks/ndmi/task.json").read_text())
    if len(checkpoint.get("artifacts", [])) != 2:
        raise SystemExit("aligned SWIR and NDMI artifacts required")
    profiles = {entry["native"]["band"]: entry for entry in provider.values()}
    if set(profiles) != {"swir16", "nir"}:
        raise SystemExit("reviewed B11 and B08 inputs required")
    source_path = run / "native-inputs" / profiles["swir16"]["filename"]
    nir_path = run / "native-inputs" / profiles["nir"]["filename"]
    for path, entry in ((source_path, profiles["swir16"]),
                        (nir_path, profiles["nir"])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["native"]["sha256"]:
            raise SystemExit("native input checksum differs")
    aligned_artifact, ndmi_artifact = checkpoint["artifacts"]
    aligned_path = (ROOT / "runtime/managed-artifacts"
                    / aligned_artifact["sha256"][:2]
                    / aligned_artifact["sha256"] / "content")
    ndmi_path = (ROOT / "runtime/managed-artifacts"
                 / ndmi_artifact["sha256"][:2]
                 / ndmi_artifact["sha256"] / "content")
    for path, artifact in ((aligned_path, aligned_artifact),
                           (ndmi_path, ndmi_artifact)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise SystemExit("managed artifact checksum differs")

    with rasterio.open(source_path, driver="GTiff") as source, \
            rasterio.open(nir_path, driver="GTiff") as nir:
        aligned_expected, aligned_valid = bilinear(source, nir)
        nir_raw = nir.read(1)
        nir_values = nir_raw.astype("float64") * nir.scales[0] + nir.offsets[0]
        nir_valid = nir.read_masks(1) > 0
        if nir.nodata is not None:
            nir_valid &= nir_raw != nir.nodata
        target_transform, target_crs = nir.transform, nir.crs
    denominator = nir_values + aligned_expected.astype("float64")
    expected_valid = (nir_valid & aligned_valid & np.isfinite(nir_values)
                      & np.isfinite(aligned_expected) & (nir_values >= 0)
                      & (aligned_expected >= 0) & (denominator > 1e-6))
    expected = np.full(nir_values.shape, NODATA, dtype="float32")
    expected[expected_valid] = ((nir_values[expected_valid]
                                 - aligned_expected[expected_valid])
                                / denominator[expected_valid]).astype("float32")

    with rasterio.open(aligned_path, driver="GTiff") as aligned:
        actual_aligned = aligned.read(1)
        actual_aligned_valid = aligned.read_masks(1) > 0
        aligned_difference = np.abs(
            actual_aligned[aligned_valid].astype("float64")
            - aligned_expected[aligned_valid].astype("float64"))
        aligned_error = float(aligned_difference.max()) \
            if aligned_difference.size else 0.
        if (aligned.transform != target_transform or aligned.crs != target_crs
                or not np.array_equal(actual_aligned_valid, aligned_valid)
                or aligned_error > 1e-7):
            raise SystemExit("independent B11 alignment differs")
    with rasterio.open(ndmi_path, driver="GTiff") as ndmi:
        actual = ndmi.read(1)
        actual_valid = ndmi.read_masks(1) > 0
        difference = np.abs(actual[expected_valid].astype("float64")
                            - expected[expected_valid].astype("float64"))
        maximum_error = float(difference.max()) if difference.size else 0.
        if (ndmi.transform != target_transform or ndmi.crs != target_crs
                or not np.array_equal(actual_valid, expected_valid)
                or maximum_error > 1e-7
                or not np.all(actual[~actual_valid] == NODATA)):
            raise SystemExit("independent fixed NDMI differs")
        zone = task["metadata"]["zonal_inputs"][
            checkpoint["zonal_result"]["zone_id"]]
        selected = []
        zone_pixels = 0
        for row in range(ndmi.height):
            for column in range(ndmi.width):
                x, y = ndmi.transform * (column + .5, row + .5)
                if not contains(x, y, zone["coordinates"]):
                    continue
                zone_pixels += 1
                if actual_valid[row, column]:
                    selected.append(float(actual[row, column]))
    selected = np.asarray(selected, dtype="float64")
    valid_pixels = int(selected.size)
    expected_zonal = {
        "zone_pixels": zone_pixels,
        "valid_pixels": valid_pixels,
        "invalid_pixels": zone_pixels - valid_pixels,
        "valid_fraction": valid_pixels / zone_pixels,
        "minimum": float(selected.min()) if valid_pixels else None,
        "maximum": float(selected.max()) if valid_pixels else None,
        "mean": float(selected.mean()) if valid_pixels else None,
    }
    actual_zonal = checkpoint["zonal_result"]
    for key in ("zone_pixels", "valid_pixels", "invalid_pixels"):
        if actual_zonal[key] != expected_zonal[key]:
            raise SystemExit(key + " differs from scalar zonal reference")
    for key in ("valid_fraction", "minimum", "maximum", "mean"):
        if not math.isclose(actual_zonal[key], expected_zonal[key],
                            rel_tol=0., abs_tol=1e-15):
            raise SystemExit(key + " differs from scalar zonal reference")
    report = {
        "status": "passed",
        "aligned_artifact_id": aligned_artifact["artifact_id"],
        "aligned_sha256": aligned_artifact["sha256"],
        "ndmi_artifact_id": ndmi_artifact["artifact_id"],
        "ndmi_sha256": ndmi_artifact["sha256"],
        "aligned_maximum_absolute_error": aligned_error,
        "ndmi_maximum_absolute_error": maximum_error,
        "aligned_mask_exact": True,
        "ndmi_mask_exact": True,
        "target_grid_exact": True,
        **expected_zonal,
        "zonal_statistics_float64_exact": all(
            actual_zonal[key] == value for key, value in expected_zonal.items()),
        "scope": ("independent numeric B11 alignment, B08 conversion, fixed NDMI "
                  "and pinned-zone reference; not a moisture or drought label"),
    }
    report_path = run / "reference/fixed-ndmi.json"
    if report_path.exists():
        raise SystemExit("preserve prior report")
    with StorageQuota(ROOT / "runtime").hold(
            report_path.parent, 1024 * 1024, "fixed-ndmi-reference-report"):
        report_path.parent.mkdir()
        with report_path.open("x") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()

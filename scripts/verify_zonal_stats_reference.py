#!/usr/bin/env python3
"""Independent scalar point-in-polygon and float64 zonal reference."""
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
from app.core.storage.quota import StorageQuota


def on_segment(x: float, y: float, first: list[float], second: list[float]) -> bool:
    x1, y1 = first
    x2, y2 = second
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    tolerance = max(1., abs(x), abs(y), abs(x1), abs(y1), abs(x2), abs(y2)) * 1e-12
    return (abs(cross) <= tolerance
            and min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
            and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance)


def contains(x: float, y: float, coordinates: list[list[float]]) -> bool:
    """Independent scalar boundary-inclusive even-odd test."""
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
    checkpoint = json.loads((run / "reports/zonal-stats-checkpoint.json").read_text())
    task = json.loads((run / "tasks/zonal-stats/task.json").read_text())
    admission = json.loads((run / "receipts/admission.json").read_text())
    if len(checkpoint.get("artifacts", [])) != 1:
        raise SystemExit("one completed continuous artifact required")
    artifact = checkpoint["artifacts"][0]
    result = checkpoint["zonal_result"]
    zone = task["metadata"]["zonal_inputs"][result["zone_id"]]
    digest = artifact["sha256"]
    content_path = ROOT / "runtime/managed-artifacts" / digest[:2] / digest / "content"
    content = content_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise SystemExit("managed artifact digest differs")
    selected = []
    zone_pixels = 0
    with rasterio.open(content_path, driver="GTiff") as image:
        values = image.read(1)
        valid = image.read_masks(1) > 0
        for row in range(image.height):
            for column in range(image.width):
                x, y = image.transform * (column + .5, row + .5)
                if not contains(x, y, zone["coordinates"]):
                    continue
                zone_pixels += 1
                value = values[row, column]
                if (valid[row, column] and np.isfinite(value)
                        and (image.nodata is None or value != image.nodata)):
                    selected.append(float(value))
    expected_boundary = admission["boundary_pixel_window"]["expected_zone_pixels"]
    if zone_pixels != expected_boundary:
        raise SystemExit("boundary-inclusive pixel count differs")
    values64 = np.asarray(selected, dtype="float64")
    valid_pixels = int(values64.size)
    invalid_pixels = zone_pixels - valid_pixels
    expected = {
        "zone_pixels": zone_pixels,
        "valid_pixels": valid_pixels,
        "invalid_pixels": invalid_pixels,
        "valid_fraction": valid_pixels / zone_pixels,
        "minimum": float(values64.min()) if valid_pixels else None,
        "maximum": float(values64.max()) if valid_pixels else None,
        "mean": float(values64.mean()) if valid_pixels else None,
    }
    for key in ("zone_pixels", "valid_pixels", "invalid_pixels"):
        if result[key] != expected[key]:
            raise SystemExit(key + " differs from independent reference")
    for key in ("valid_fraction", "minimum", "maximum", "mean"):
        if result[key] is None or expected[key] is None:
            if result[key] != expected[key]:
                raise SystemExit(key + " differs from independent reference")
        elif not math.isclose(result[key], expected[key], rel_tol=0., abs_tol=1e-15):
            raise SystemExit(key + " differs from independent reference")
    report = {
        "status": "passed",
        "artifact_id": artifact["artifact_id"],
        "sha256": digest,
        **expected,
        "boundary_inclusive_exact": True,
        "statistics_float64_exact": all(result[key] == expected[key]
                                         for key in expected),
        "scope": ("independent zonal membership and statistics reference only; "
                  "not a land-cover, cloud or change validation"),
    }
    report_path = run / "reference/zonal-statistics.json"
    if report_path.exists():
        raise SystemExit("preserve prior report")
    with StorageQuota(ROOT / "runtime").hold(
            report_path.parent, 1024 * 1024, "zonal-statistics-reference-report"):
        report_path.parent.mkdir()
        with report_path.open("x") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()

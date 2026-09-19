#!/usr/bin/env python3
"""Operator-only independent pixel-centre reference for real SCL alignments."""
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


def nearest(source, target):
    values, source_valid = source.read(1), source.read_masks(1) > 0
    rr, cc = np.indices((target.height, target.width))
    xx, yy = target.transform * (cc.flatten() + .5, rr.flatten() + .5)
    if source.crs != target.crs:
        from rasterio.warp import transform
        xx, yy = transform(target.crs, source.crs, xx.tolist(), yy.tolist())
    sx, sy = ~source.transform * (np.asarray(xx), np.asarray(yy))
    columns, rows = np.floor(sx).astype(int), np.floor(sy).astype(int)
    inside = ((columns >= 0) & (columns < source.width)
              & (rows >= 0) & (rows < source.height))
    expected = np.full(target.width * target.height, 255, dtype="uint8")
    positions = np.flatnonzero(inside)
    keep = source_valid[rows[positions], columns[positions]] & (values[rows[positions], columns[positions]] != 0)
    positions = positions[keep]
    expected[positions] = values[rows[positions], columns[positions]]
    return expected.reshape(target.height, target.width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to((ROOT / "runtime").resolve()):
        raise SystemExit("runtime run required")
    manifest = json.loads((run / "native-inputs.json").read_text())
    saved = json.loads((run / "reports/grid-checkpoint.json").read_text())
    report_path = run / "reference/grid-numeric.json"
    if report_path.exists():
        raise SystemExit("preserve prior report")
    results = []
    for artifact, result in zip(saved["artifacts"], saved["results"]):
        source_entry, reference_entry = [manifest[key] for key in result["input_asset_ids"]]
        source_path = run / "native-inputs" / source_entry["filename"]
        reference_path = run / "native-inputs" / reference_entry["filename"]
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_entry["native"]["sha256"]
        assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == reference_entry["native"]["sha256"]
        with rasterio.open(source_path, driver="GTiff") as source, rasterio.open(reference_path, driver="GTiff") as reference:
            expected = nearest(source, reference)
            grid, crs = reference.transform, reference.crs
        digest = artifact["sha256"]
        path = ROOT / "runtime/managed-artifacts" / digest[:2] / digest / "content"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        with rasterio.open(path, driver="GTiff") as output:
            actual = output.read(1)
            assert output.transform == grid and output.crs == crs
            assert np.array_equal(output.read_masks(1) > 0, expected != 255)
            assert np.array_equal(actual, expected)
        results.append({"artifact_id": artifact["artifact_id"], "sha256": digest,
                        "valid_pixels": int((expected != 255).sum()),
                        "total_pixels": int(expected.size), "exact_pixel_equality": True,
                        "class_counts": np.bincount(expected[expected != 255], minlength=12).tolist()})
    assert len(results) == 3
    with StorageQuota(ROOT / "runtime").hold(report_path.parent, 1024 * 1024, "grid-reference-report"):
        report_path.parent.mkdir()
        with report_path.open("x") as stream:
            json.dump({"status": "passed", "rasters": results,
                       "scope": "categorical alignment only; not cloud/change ground truth"}, stream, indent=2)
    print(json.dumps({"status": "passed", "rasters": len(results), "exact_pixel_equality": True}))


if __name__ == "__main__":
    main()

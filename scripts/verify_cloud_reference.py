#!/usr/bin/env python3
"""Independent real-data reference for SCL-grid to policy-masked NDVI chain."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.core.storage.quota import StorageQuota

POLICY = "sentinel-2-scl-cloud-shadow-cirrus-snow-saturation-v1"
EXCLUDED = (1, 3, 8, 9, 10, 11)


def nearest(source, target):
    values, source_valid = source.read(1), source.read_masks(1) > 0
    rows, columns = np.indices((target.height, target.width))
    x, y = target.transform * (columns.flatten() + .5, rows.flatten() + .5)
    if source.crs != target.crs:
        from rasterio.warp import transform
        x, y = transform(target.crs, source.crs, x.tolist(), y.tolist())
    source_columns, source_rows = ~source.transform * (np.asarray(x), np.asarray(y))
    source_columns, source_rows = (np.floor(source_columns).astype(int),
                                   np.floor(source_rows).astype(int))
    inside = ((source_columns >= 0) & (source_columns < source.width)
              & (source_rows >= 0) & (source_rows < source.height))
    expected = np.full(target.width * target.height, 255, dtype="uint8")
    positions = np.flatnonzero(inside)
    keep = (source_valid[source_rows[positions], source_columns[positions]]
            & (values[source_rows[positions], source_columns[positions]] != 0))
    positions = positions[keep]
    expected[positions] = values[source_rows[positions], source_columns[positions]]
    return expected.reshape(target.height, target.width)


def artifact_path(sha256: str) -> Path:
    return ROOT / "runtime/managed-artifacts" / sha256[:2] / sha256 / "content"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    arguments = parser.parse_args()
    run = arguments.run.resolve()
    if not run.is_relative_to((ROOT / "runtime").resolve()):
        raise SystemExit("runtime run required")
    manifest = json.loads((run / "native-inputs.json").read_text())
    saved = json.loads((run / "reports/cloud-checkpoint.json").read_text())
    report_path = run / "reference/cloud-numeric.json"
    if report_path.exists():
        raise SystemExit("preserve prior report")
    audited = []

    for record in saved["results"]:
        grid_result, ndvi_result = record["grid"], record["ndvi"]
        if (ndvi_result["cloud_policy_version"] != POLICY
                or ndvi_result["mask_artifact_id"] != record["mask_artifact"]["artifact_id"]):
            raise ValueError("checkpoint policy or lineage mismatch")
        source_id, reference_id = grid_result["input_asset_ids"]
        red_id, nir_id = ndvi_result["input_asset_ids"]
        if reference_id != red_id:
            raise ValueError("mask target is not the NDVI red grid")
        entries = {key: manifest[key] for key in (source_id, red_id, nir_id)}
        paths = {key: run / "native-inputs" / entry["filename"]
                 for key, entry in entries.items()}
        for key, path in paths.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != entries[key]["native"]["sha256"]:
                raise ValueError("native input changed")

        with (rasterio.open(paths[source_id], driver="GTiff") as source,
              rasterio.open(paths[red_id], driver="GTiff") as red,
              rasterio.open(paths[nir_id], driver="GTiff") as nir):
            expected_scl = nearest(source, red)
            red_values = red.read(1).astype(np.float64) * red.scales[0] + red.offsets[0]
            nir_values = nir.read(1).astype(np.float64) * nir.scales[0] + nir.offsets[0]
            clear = ((expected_scl != 255) & ~np.isin(expected_scl, EXCLUDED))
            excluded = ((expected_scl != 255) & np.isin(expected_scl, EXCLUDED))
            denominator = nir_values + red_values
            valid = (red.read_masks(1) > 0) & (nir.read_masks(1) > 0) & clear
            valid &= (np.isfinite(red_values) & np.isfinite(nir_values)
                      & (red_values >= 0) & (nir_values >= 0) & (denominator > 1e-6))
            expected_ndvi = np.full(red_values.shape, -9999., dtype="float32")
            ratio = np.divide(nir_values - red_values, denominator,
                              out=np.zeros(red_values.shape), where=valid)
            expected_ndvi[valid] = ratio[valid].astype("float32")
            grid, crs = red.transform, red.crs

        mask_sha = record["mask_artifact"]["sha256"]
        mask_path = artifact_path(mask_sha)
        if hashlib.sha256(mask_path.read_bytes()).hexdigest() != mask_sha:
            raise ValueError("stored mask checksum mismatch")
        with rasterio.open(mask_path, driver="GTiff") as output:
            if (output.transform != grid or output.crs != crs
                    or not np.array_equal(output.read(1), expected_scl)
                    or not np.array_equal(output.read_masks(1) > 0, expected_scl != 255)):
                raise ValueError("categorical grid differs from independent reference")

        ndvi_sha = record["ndvi_artifact"]["sha256"]
        ndvi_path = artifact_path(ndvi_sha)
        if hashlib.sha256(ndvi_path.read_bytes()).hexdigest() != ndvi_sha:
            raise ValueError("stored NDVI checksum mismatch")
        with rasterio.open(ndvi_path, driver="GTiff") as output:
            actual = output.read(1)
            if (output.transform != grid or output.crs != crs
                    or not np.array_equal(output.read_masks(1) > 0, valid)
                    or not np.array_equal(actual, expected_ndvi)):
                raise ValueError("masked NDVI differs from independent reference")
        expected_counts = {"mask_valid_pixels": int((expected_scl != 255).sum()),
            "clear_mask_pixels": int(clear.sum()),
            "cloud_excluded_pixels": int(excluded.sum()), "valid_pixels": int(valid.sum())}
        if any(ndvi_result[key] != value for key, value in expected_counts.items()):
            raise ValueError("reported mask/NDVI counts differ from reference")
        audited.append({"acquired": record["acquired"],
            "mask_artifact_id": record["mask_artifact"]["artifact_id"],
            "mask_sha256": mask_sha,
            "ndvi_artifact_id": record["ndvi_artifact"]["artifact_id"],
            "ndvi_sha256": ndvi_sha, **expected_counts,
            "mean": float(expected_ndvi[valid].astype(np.float64).mean()) if valid.any() else None,
            "exact_scl_pixel_equality": True, "exact_ndvi_pixel_equality": True,
            "maximum_absolute_error": float(np.max(np.abs(actual - expected_ndvi)))})

    if len(audited) != 3:
        raise ValueError("three audited dates required")
    with StorageQuota(ROOT / "runtime").hold(report_path.parent, 1024 * 1024,
                                               "cloud-reference-report"):
        report_path.parent.mkdir()
        with report_path.open("x") as stream:
            json.dump({"status": "passed", "cloud_mask_policy": POLICY,
                "rasters": audited,
                "scope": "implementation equality for fixed SCL policy; not cloud or change ground truth"},
                stream, indent=2)
    print(json.dumps({"status": "passed", "dates": len(audited),
                      "exact_scl_pixel_equality": True,
                      "exact_ndvi_pixel_equality": True}))


if __name__ == "__main__":
    main()

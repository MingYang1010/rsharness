#!/usr/bin/env python3
"""Private, streaming dataset inventory. Candidate discovery is NOT agent access."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import sqlite3
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.core.storage.quota import CONTROL_ALLOWANCE, StorageQuota

MAX_CATALOG_BYTES = 8 * 1024 * 1024 * 1024
MAX_COVERAGE_BYTES = 8 * 1024 * 1024

IMAGE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".jp2", ".bmp", ".webp"}
SKIP_DIRS = {".git", ".cache", "__pycache__", "logs"}
LABEL_WORDS = {"label", "labels", "mask", "masks", "annotation", "annotations", "gt", "groundtruth", "ground_truth"}


def probable_label(path: Path) -> bool:
    tokens = set(re.split(r"[^a-z0-9]+", str(path).lower()))
    return bool(tokens & LABEL_WORDS)


def roots_from_config(config: dict) -> dict[str, Path]:
    base = Path(config["dataset_base"]).resolve()
    roots = {name: base / name for name in config["roots"]}
    for group in config.get("expand_children", []):
        parent = base / group
        if parent.is_dir():
            for child in sorted(parent.iterdir()):
                if child.is_dir() and not child.is_symlink():
                    roots[f"{group}/{child.name}"] = child
    roots.update({name: Path(path) for name, path in config.get("extra_roots", {}).items()})
    return roots


def probe_image(path: Path) -> dict:
    import rasterio
    from rasterio.warp import transform_bounds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(path) as dataset:
            # Read only a small window, not a full enormous orthophoto.
            chip = dataset.read(1, window=((0, min(16, dataset.height)), (0, min(16, dataset.width))))
            if chip.size == 0:
                raise ValueError("empty raster")
            spatial = None
            if dataset.crs:
                bbox = list(transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds))
                spatial = {"native_crs": str(dataset.crs), "bbox_wgs84": bbox}
            return {"width": dataset.width, "height": dataset.height, "bands": dataset.count,
                    "dtypes": list(dataset.dtypes), "spatial": spatial,
                    "reference_kind": "geographic" if spatial else "pixel"}


def content_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_dataset(connection: sqlite3.Connection, dataset_id: str, root: Path, config: dict) -> dict:
    report = {"dataset_id": dataset_id, "root": str(root), "candidate_images": 0,
              "label_like_images": 0, "bytes": 0, "archives": 0, "symlinks_skipped": 0,
              "scan_errors": [], "failure_markers": [], "samples": [],
              "agent_access": "not_granted", "license": "needs_source_review"}
    if not root.is_dir():
        return {**report, "status": "missing"}
    root = root.resolve()
    samples = []
    limit = config["sample_count"]
    def onerror(error):
        report["scan_errors"].append(str(error))
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
        parent = Path(directory)
        allowed_dirs = []
        for name in dirs:
            if (parent / name).is_symlink():
                report["symlinks_skipped"] += 1
            elif name not in SKIP_DIRS:
                allowed_dirs.append(name)
        dirs[:] = sorted(allowed_dirs)
        for name in sorted(files):
            path = parent / name
            relative = path.relative_to(root)
            if path.is_symlink():
                report["symlinks_skipped"] += 1
                continue
            if "DOWNLOAD_FAILED" in name.upper():
                report["failure_markers"].append(str(relative))
            if path.suffix.lower() in {".zip", ".7z", ".tar", ".gz", ".parquet", ".arrow"}:
                report["archives"] += 1
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                size = path.stat().st_size
            except OSError as error:
                report["scan_errors"].append(str(error))
                continue
            is_label = probable_label(relative)
            report["label_like_images" if is_label else "candidate_images"] += 1
            report["bytes"] += size
            asset_id = "asset-" + hashlib.sha256((dataset_id + "\0" + str(relative)).encode()).hexdigest()
            connection.execute("INSERT INTO assets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                               (asset_id, dataset_id, str(relative), size,
                                "label_like" if is_label else "unreviewed_image", None, None, 0))
            if not is_label and 0 < size <= config["sample_max_file_bytes"]:
                rank = int(hashlib.sha256((str(config["sample_seed"]) + asset_id).encode()).hexdigest(), 16)
                item = (-rank, str(relative), asset_id)
                if len(samples) < limit:
                    heapq.heappush(samples, item)
                elif rank < -samples[0][0]:
                    heapq.heapreplace(samples, item)
    for _, relative, asset_id in sorted(samples, reverse=True):
        path = root / relative
        sample = {"asset_id": asset_id, "relative_path": relative}
        try:
            sample.update(probe_image(path))
            sample["sha256"] = content_hash(path)
            sample["status"] = "readable_sample"
            connection.execute("UPDATE assets SET sha256=?, probe_json=? WHERE asset_id=?",
                               (sample["sha256"], json.dumps(sample), asset_id))
        except Exception as error:
            sample.update(status="unreadable", error=f"{type(error).__name__}: {error}")
        report["samples"].append(sample)
    readable = sum(sample["status"] == "readable_sample" for sample in report["samples"])
    report["status"] = (
        "sampled_readable" if readable == limit and not report["scan_errors"] and not report["failure_markers"]
        else "partial" if readable else "unavailable"
    )
    connection.commit()
    return report


def run(config: dict, output: Path, max_catalog_bytes: int = MAX_CATALOG_BYTES) -> dict:
    if type(max_catalog_bytes) is not int or not 4096 <= max_catalog_bytes <= MAX_CATALOG_BYTES:
        raise ValueError("invalid catalog byte limit")
    output.mkdir(parents=True, exist_ok=False)
    connection = sqlite3.connect(output / "catalog.sqlite3")
    try:
        page_size = connection.execute("PRAGMA page_size").fetchone()[0]
        connection.execute("PRAGMA max_page_count=%d" % (max_catalog_bytes // page_size))
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("CREATE TABLE assets (asset_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, relative_path TEXT NOT NULL, size_bytes INTEGER NOT NULL, role TEXT NOT NULL, sha256 TEXT, probe_json TEXT, agent_visible INTEGER NOT NULL CHECK(agent_visible IN (0,1)))")
        connection.execute("CREATE INDEX asset_dataset ON assets(dataset_id)")
        reports = []
        for dataset_id, root in roots_from_config(config).items():
            report = scan_dataset(connection, dataset_id, root, config)
            reports.append(report)
            print(json.dumps({key: report[key] for key in ("dataset_id", "status", "candidate_images", "label_like_images")}), flush=True)
        result = {"schema_version": "inventory-1", "seed": config["sample_seed"],
                  "discovery_not_agent_access": True, "datasets": reports,
                  "validation": "three seeded image samples per dataset, not full-file validation"}
        payload = (json.dumps(result, indent=2) + "\n").encode()
        if len(payload) > MAX_COVERAGE_BYTES:
            raise ValueError("coverage report byte limit exceeded")
        (output / "coverage.json").write_bytes(payload)
        return result
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or "runtime" not in args.output.parts:
        parser.error("output must be an absolute ignored runtime directory")
    quota = StorageQuota(ROOT / "runtime")
    # Bound both the database and its rollback journal; reports are bounded too.
    with quota.hold(args.output, 2 * MAX_CATALOG_BYTES + CONTROL_ALLOWANCE, "dataset-inventory"):
        run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()

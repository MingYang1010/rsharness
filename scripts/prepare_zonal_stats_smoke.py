#!/usr/bin/env python3
"""Build one immutable zonal-statistics task from the accepted continuous run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.v2.capabilities import TaskRegistry
from app.v2.data.stac import json_bytes
from app.v2.raster_grid import ContinuousBand, checked_continuous_grids
from app.v2.raster_zonal import INCLUSION_POLICY, ZoneSpec, validate_zone_bounds
from app.v2.storage.quota import StorageQuota

ZONE_ID = "zone-central-pixel-centres"


def read_json(path: Path, maximum: int = 2 * 1024 * 1024) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("bounded regular JSON required")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def write_json(path: Path, value: object) -> None:
    content = json_bytes(value)
    if len(content) > 2 * 1024 * 1024:
        raise ValueError("metadata exceeds bound")
    with path.open("xb") as stream:
        stream.write(content)


def checked_source(source: Path) -> tuple[Path, dict, dict, dict, dict]:
    runtime = (ROOT / "runtime").resolve()
    resolved = source.resolve()
    if source.is_symlink() or not resolved.is_relative_to(runtime):
        raise ValueError("reviewed runtime source required")
    admission = read_json(resolved / "receipts/admission.json")
    checkpoint = read_json(resolved / "reports/continuous-grid-checkpoint.json")
    reference = read_json(resolved / "reference/continuous-grid-numeric.json")
    task = read_json(resolved / "tasks/continuous-grid/task.json")
    if (admission.get("status") != "admitted"
            or checkpoint.get("state", {}).get("status") != "terminated"
            or len(checkpoint.get("artifacts", [])) != 1
            or reference.get("status") != "passed"
            or reference.get("maximum_absolute_error") != 0.0
            or task.get("metadata", {}).get("acceptance")
            != "scripted-continuous-grid-numeric-check"):
        raise ValueError("accepted continuous-grid source required")
    manifest = read_json(resolved / "native-inputs.json")
    if len(manifest) != 2:
        raise ValueError("exactly two native inputs required")
    return resolved, admission, checkpoint, task, manifest


def centre(profile: ContinuousBand, column: int, row: int) -> list[float]:
    a, b, c, d, e, f = profile.transform
    return [c + a * (column + .5) + b * (row + .5),
            f + d * (column + .5) + e * (row + .5)]


def prepare(source: Path, out: Path) -> str:
    source, admission, checkpoint, task, provider = checked_source(source)
    if out.exists() or out.is_symlink():
        raise ValueError("fresh output required")
    profiles = {asset_id: ContinuousBand.model_validate(entry["native"])
                for asset_id, entry in provider.items()}
    bands = {profile.band: profile for profile in profiles.values()}
    if set(bands) != {"swir16", "nir"}:
        raise ValueError("pinned B11 and B08 profiles required")
    checked_continuous_grids(bands["swir16"], bands["nir"])
    reference = bands["nir"]
    left_column, right_column = reference.width // 4, reference.width * 3 // 4
    top_row, bottom_row = reference.height // 4, reference.height * 3 // 4
    if right_column >= reference.width or bottom_row >= reference.height:
        raise ValueError("invalid reviewed zone window")
    corners = [centre(reference, left_column, top_row),
               centre(reference, right_column, top_row),
               centre(reference, right_column, bottom_row),
               centre(reference, left_column, bottom_row)]
    coordinates = [*corners, corners[0]]
    xs, ys = zip(*coordinates)
    bbox = list(transform_bounds(
        reference.crs, "EPSG:4326", min(xs), min(ys), max(xs), max(ys),
        densify_pts=21))
    zone = ZoneSpec(
        zone_id=ZONE_ID,
        crs=reference.crs,
        coordinates=coordinates,
        bbox_wgs84=bbox,
        inclusion_policy=INCLUSION_POLICY,
    )
    validate_zone_bounds(zone)

    with StorageQuota(ROOT / "runtime").hold(
            out, 16 * 1024 * 1024, "zonal-statistics-task-admission"):
        for name in ("inputs", "native-inputs", "state", "reports", "receipts",
                     "tasks/zonal-stats"):
            (out / name).mkdir(parents=True)
        linked = []
        for asset_id, entry in provider.items():
            filename = entry.get("filename")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("invalid native input filename")
            source_path = source / "native-inputs" / filename
            profile = profiles[asset_id]
            content = source_path.read_bytes()
            if (source_path.is_symlink() or hashlib.sha256(content).hexdigest()
                    != profile.sha256):
                raise ValueError("native input differs from accepted source")
            destination = out / "native-inputs" / filename
            os.link(source_path, destination)
            if (destination.stat().st_ino != source_path.stat().st_ino
                    or destination.stat().st_dev != source_path.stat().st_dev):
                raise ValueError("read-only hard-link staging failed")
            linked.append({"asset_id": asset_id, "filename": filename,
                           "sha256": profile.sha256,
                           "size_bytes": len(content)})

        source_task_dir = source / "tasks/continuous-grid"
        assets_path = source_task_dir / "assets.json"
        if (assets_path.is_symlink() or not assets_path.is_file()
                or assets_path.stat().st_size > 2 * 1024 * 1024):
            raise ValueError("bounded regular asset list required")
        assets = json.loads(assets_path.read_text())
        if not isinstance(assets, list) or len(assets) != 2:
            raise ValueError("exactly two reviewed assets required")
        scenario = read_json(source_task_dir / "scenario.json")
        evaluator = read_json(source_task_dir / "evaluator.json")
        identity = hashlib.sha256(json_bytes({
            "source_task_manifest_hash": admission["task_manifest_hash"],
            "zone": zone.model_dump(mode="json"),
        })).hexdigest()[:16]
        task.update(
            task_id="zonal-stats-" + identity,
            prompt=("Inspect the pinned same-scene B11 20 m and B08 10 m assets, "
                    "align physical B11 reflectance to the B08 grid with bilinear "
                    f"resampling, then compute zonal statistics for {ZONE_ID}. "
                    "Cite the aligned raster and report the pinned zone's pixel "
                    "counts, valid fraction, minimum, maximum and mean."),
            metadata={
                "observation_profile": "headless-tools-v1",
                "artifact_identity": "derivation-sha256-v1",
                "acceptance": "scripted-zonal-statistics-reference-check",
                "grid_inputs": {key: value.model_dump(mode="json")
                                for key, value in profiles.items()},
                "zonal_inputs": {ZONE_ID: zone.model_dump(mode="json")},
            },
        )
        task["budget"].update(
            max_steps=20, max_tool_calls=10, max_wall_time_ms=300000,
            max_input_bytes=32 * 1024 * 1024,
            max_artifact_bytes=16 * 1024 * 1024,
        )
        scenario.update(
            allowed_actions=["tool.invoke", "memory.save_evidence", "answer.*"],
            allowed_tools=["catalog.search", "catalog.inspect_asset",
                           "raster.resample", "raster.zonal_stats"],
        )
        directory = out / "tasks/zonal-stats"
        for name, value in (("task.json", task), ("assets.json", assets),
                            ("scenario.json", scenario),
                            ("evaluator.json", evaluator)):
            write_json(directory / name, value)
        write_json(out / "inputs.json", {})
        write_json(out / "native-inputs.json", provider)
        write_json(out / "job.json", {"seed": 42, "task_ref": {
            "task_id": task["task_id"], "task_version": task["task_version"]}})
        manifest = TaskRegistry(out / "tasks").get(
            task["task_id"], task["task_version"])
        write_json(out / "receipts/admission.json", {
            "status": "admitted",
            "task_manifest_hash": manifest.task_manifest_hash,
            "source_task_manifest_hash": admission["task_manifest_hash"],
            "source_episode_id": checkpoint["state"]["episode_id"],
            "source_checkpoint_sha256": hashlib.sha256(
                (source / "reports/continuous-grid-checkpoint.json").read_bytes()
            ).hexdigest(),
            "source_reference_sha256": hashlib.sha256(
                (source / "reference/continuous-grid-numeric.json").read_bytes()
            ).hexdigest(),
            "source_artifact_reused": False,
            "native_inputs": linked,
            "staging": "same-filesystem hard links mounted read-only",
            "zone": zone.model_dump(mode="json"),
            "boundary_pixel_window": {
                "left_column": left_column, "right_column": right_column,
                "top_row": top_row, "bottom_row": bottom_row,
                "expected_zone_pixels": ((right_column - left_column + 1)
                                         * (bottom_row - top_row + 1)),
            },
            "network_requests": 0,
            "license_scope": "local-research; not redistribution authorization",
        })
        return manifest.task_manifest_hash


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-name", required=True)
    arguments = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", arguments.output_name):
        raise SystemExit("bounded fresh output name required")
    digest = prepare(arguments.source, ROOT / "runtime" / arguments.output_name)
    print(json.dumps({"status": "admitted", "task_manifest_hash": digest,
                      "zone_id": ZONE_ID, "network_requests": 0}))

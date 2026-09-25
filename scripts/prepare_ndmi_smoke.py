#!/usr/bin/env python3
"""Build one immutable fixed-NDMI task from accepted local Sentinel-2 inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.core.capabilities import TaskRegistry
from app.core.data.stac import json_bytes
from app.core.raster_grid import ContinuousBand, checked_continuous_grids
from app.core.raster_math import NDMI_FORMULA
from app.core.raster_zonal import ZoneSpec, validate_zone_bounds
from app.core.storage.quota import StorageQuota


def read_json(path: Path, maximum: int = 2 * 1024 * 1024):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("bounded regular JSON required")
    value = json.loads(path.read_text())
    if not isinstance(value, (dict, list)):
        raise ValueError("JSON object or array required")
    return value


def write_json(path: Path, value: object) -> None:
    content = json_bytes(value)
    if len(content) > 2 * 1024 * 1024:
        raise ValueError("metadata exceeds bound")
    with path.open("xb") as stream:
        stream.write(content)


def checked_source(source: Path):
    runtime = (ROOT / "runtime").resolve()
    resolved = source.resolve()
    if source.is_symlink() or not resolved.is_relative_to(runtime):
        raise ValueError("reviewed runtime source required")
    admission = read_json(resolved / "receipts/admission.json")
    checkpoint = read_json(resolved / "reports/zonal-stats-checkpoint.json")
    reference = read_json(resolved / "reference/zonal-statistics.json")
    task = read_json(resolved / "tasks/zonal-stats/task.json")
    provider = read_json(resolved / "native-inputs.json")
    if (admission.get("status") != "admitted"
            or checkpoint.get("state", {}).get("status") != "terminated"
            or len(checkpoint.get("artifacts", [])) != 1
            or reference.get("status") != "passed"
            or reference.get("boundary_inclusive_exact") is not True
            or reference.get("statistics_float64_exact") is not True
            or task.get("metadata", {}).get("acceptance")
            != "scripted-zonal-statistics-reference-check"
            or not isinstance(provider, dict) or len(provider) != 2):
        raise ValueError("accepted zonal-statistics source required")
    return resolved, admission, checkpoint, task, provider


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
    source_zone = task["metadata"]["zonal_inputs"]
    if not isinstance(source_zone, dict) or len(source_zone) != 1:
        raise ValueError("one pinned source zone required")
    zone = ZoneSpec.model_validate(next(iter(source_zone.values())))
    validate_zone_bounds(zone)

    with StorageQuota(ROOT / "runtime").hold(
            out, 16 * 1024 * 1024, "fixed-ndmi-task-admission"):
        for name in ("inputs", "native-inputs", "state", "reports", "receipts",
                     "tasks/ndmi"):
            (out / name).mkdir(parents=True)
        linked = []
        for asset_id, entry in provider.items():
            filename = entry.get("filename")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("invalid native input filename")
            source_path = source / "native-inputs" / filename
            profile = profiles[asset_id]
            content = source_path.read_bytes()
            if (source_path.is_symlink()
                    or hashlib.sha256(content).hexdigest() != profile.sha256):
                raise ValueError("native input differs from accepted source")
            destination = out / "native-inputs" / filename
            os.link(source_path, destination)
            if (destination.stat().st_ino != source_path.stat().st_ino
                    or destination.stat().st_dev != source_path.stat().st_dev):
                raise ValueError("read-only hard-link staging failed")
            linked.append({"asset_id": asset_id, "filename": filename,
                           "sha256": profile.sha256,
                           "size_bytes": len(content)})

        source_task = source / "tasks/zonal-stats"
        assets = read_json(source_task / "assets.json")
        scenario = read_json(source_task / "scenario.json")
        evaluator = read_json(source_task / "evaluator.json")
        if not isinstance(assets, list) or len(assets) != 2:
            raise ValueError("exactly two reviewed assets required")
        identity = hashlib.sha256(json_bytes({
            "source_task_manifest_hash": admission["task_manifest_hash"],
            "formula_id": NDMI_FORMULA,
            "zone": zone.model_dump(mode="json"),
        })).hexdigest()[:16]
        task.update(
            task_id="fixed-ndmi-" + identity,
            prompt=("Inspect the pinned same-scene Sentinel-2 B11 20 m and B08 "
                    "10 m assets. Align B11 physical reflectance to the B08 grid, "
                    "compute the fixed reviewed NDMI, then report NDMI statistics "
                    f"for {zone.zone_id}. Cite the NDMI artifact."),
            metadata={
                "observation_profile": "headless-tools-v1",
                "artifact_identity": "derivation-sha256-v1",
                "acceptance": "scripted-fixed-ndmi-reference-check",
                "band_math_formula": NDMI_FORMULA,
                "grid_inputs": {key: value.model_dump(mode="json")
                                for key, value in profiles.items()},
                "zonal_inputs": {zone.zone_id: zone.model_dump(mode="json")},
            },
        )
        task["budget"].update(
            max_steps=24, max_tool_calls=12, max_wall_time_ms=300000,
            max_input_bytes=48 * 1024 * 1024,
            max_artifact_bytes=24 * 1024 * 1024,
        )
        scenario.update(
            allowed_actions=["tool.invoke", "memory.save_evidence", "answer.*"],
            allowed_tools=["catalog.search", "catalog.inspect_asset",
                           "raster.resample", "raster.band_math",
                           "raster.zonal_stats"],
        )
        directory = out / "tasks/ndmi"
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
                (source / "reports/zonal-stats-checkpoint.json").read_bytes()
            ).hexdigest(),
            "source_reference_sha256": hashlib.sha256(
                (source / "reference/zonal-statistics.json").read_bytes()
            ).hexdigest(),
            "source_artifact_reused": False,
            "native_inputs": linked,
            "staging": "same-filesystem hard links mounted read-only",
            "formula_id": NDMI_FORMULA,
            "zone": zone.model_dump(mode="json"),
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
                      "formula_id": NDMI_FORMULA, "network_requests": 0}))

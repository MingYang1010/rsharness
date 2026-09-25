#!/usr/bin/env python3
"""Admit three reviewed SCL/red/NIR triples for artifact-chained NDVI."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.core.capabilities import TaskRegistry
from app.core.data.stac import json_bytes
from app.core.raster_grid import NativeSCL, checked_grids
from app.core.raster_math import CLOUD_POLICY, MAX_INPUT, NativeBand, checked_pair
from app.core.storage.quota import StorageQuota


def write(path: Path, value) -> None:
    content = json_bytes(value)
    if len(content) > 1024 * 1024:
        raise ValueError("metadata too large")
    with path.open("xb") as stream:
        stream.write(content)


def prepare(source: Path, out: Path) -> str:
    import rasterio

    runtime = (ROOT / "runtime").resolve()
    if (source.is_symlink() or not source.resolve().is_relative_to(runtime)
            or out.exists() or not out.resolve().is_relative_to(runtime)):
        raise ValueError("existing reviewed runtime source and fresh runtime output required")
    receipt_path = source / "receipts/admission.json"
    if receipt_path.is_symlink() or receipt_path.stat().st_size > 1024 * 1024:
        raise ValueError("receipt exceeds bound")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "admitted" or len(receipt.get("windows", [])) != 12:
        raise ValueError("three reviewed Sentinel dates required")
    records = [record for record in receipt["windows"]
               if record["asset_key"] in {"scl", "red", "nir"}]
    if len(records) != 9:
        raise ValueError("nine reviewed SCL/red/NIR inputs required")

    with StorageQuota(ROOT / "runtime").hold(out, 192 * 1024 * 1024,
                                               "cloud-chain-task-admission"):
        for name in ("inputs", "native-inputs", "state", "reports",
                     "tasks/cloud-masked-ndvi"):
            (out / name).mkdir(parents=True)
        template = ROOT / "tasks/worldcover-grounded-vqa"
        assets, profiles, raster_inputs, provider, triples = [], {}, {}, {}, {}
        for record in records:
            filename = record["filename"]
            if Path(filename).name != filename:
                raise ValueError("invalid approved filename")
            path = source / "native-bands" / filename
            if (path.is_symlink() or not path.resolve().is_relative_to(source.resolve())
                    or path.stat().st_size > MAX_INPUT):
                raise ValueError("bounded native file required")
            content = path.read_bytes()
            if (len(content) != record["size_bytes"]
                    or hashlib.sha256(content).hexdigest() != record["sha256"]):
                raise ValueError("approved content changed")
            with rasterio.open(path, driver="GTiff") as image:
                expected_dtype = "uint8" if record["asset_key"] == "scl" else "uint16"
                if (image.count != 1 or image.dtypes != (expected_dtype,)
                        or image.crs is None or image.crs.to_string() != record["crs"]
                        or list(image.transform)[:6] != record["transform"]
                        or [image.width, image.height] != [record["width"], record["height"]]
                        or list(image.scales) != record["scales"]
                        or list(image.offsets) != record["offsets"]):
                    raise ValueError("approved native metadata changed")
                asset_id = "asset-" + hashlib.sha256(json_bytes(
                    [record["item_id"], record["asset_key"], record["sha256"]])).hexdigest()
                common = dict(asset_id=asset_id, sha256=record["sha256"],
                    item_id=record["item_id"], acquired=record["acquired"],
                    crs=record["crs"], transform=record["transform"], width=image.width,
                    height=image.height, dtype=expected_dtype, scale=image.scales[0],
                    offset=image.offsets[0], nodata=image.nodata)
                profile = (NativeSCL(band="scl", **common) if record["asset_key"] == "scl"
                           else NativeBand(band=record["asset_key"], **common))
            with (out / "native-inputs" / filename).open("xb") as stream:
                stream.write(content)
            serialized = profile.model_dump(mode="json")
            profiles[asset_id] = serialized
            if profile.band in {"red", "nir"}:
                raster_inputs[asset_id] = serialized
            provider[asset_id] = {"filename": filename, "native": serialized}
            triples.setdefault(profile.item_id, {})[profile.band] = profile
            west, south, east, north = record["bbox_wgs84"]
            roles = (["input_image", "scene_classification"] if profile.band == "scl"
                     else ["input_image", "reflectance"])
            assets.append({"asset_id": asset_id,
                "uri": "local://approved-cloud-chain/" + filename,
                "media_type": "image/tiff", "roles": roles,
                "sha256": record["sha256"], "size_bytes": len(content),
                "spatial": {"crs": "EPSG:4326",
                    "bbox": {"west": west, "south": south, "east": east, "north": north},
                    "gsd_meters": profile.transform[0],
                    "shape": [profile.height, profile.width, 1]},
                "temporal": {"start": profile.acquired, "end": profile.acquired},
                "platform": "sentinel-2", "bands": [profile.band],
                "quality": {"cloud_cover_percent": record["scene_cloud_cover_percent"],
                            "nodata_fraction": record["nodata_fraction"]},
                "license": receipt["config"]["license_review"]["attribution"],
                "source": "Reviewed Sentinel-2 SCL or native reflectance window",
                "source_snapshot_hash": record["source_snapshot_hash"]})
        if len(triples) != 3:
            raise ValueError("three unique acquisition triples required")
        for triple in triples.values():
            if set(triple) != {"scl", "red", "nir"}:
                raise ValueError("complete SCL/red/NIR triple required")
            checked_grids(triple["scl"], triple["red"])
            checked_pair(triple["red"], triple["nir"])

        assets.sort(key=lambda asset: asset["asset_id"])
        task = json.loads((template / "task.json").read_text())
        identity = hashlib.sha256(json_bytes(profiles)).hexdigest()[:16]
        task.update(task_id="cloud-masked-ndvi-" + identity,
            inputs=[asset["asset_id"] for asset in assets],
            prompt="For each of three dates, explicitly align SCL to the red grid with nearest-neighbor, then compute NDVI using that episode-local artifact and the fixed reviewed SCL cloud policy. Cite all three masked NDVI rasters and report final valid means plus mask-valid, policy-excluded and clear counts. This policy is not cloud ground truth or change detection.",
            metadata={"observation_profile": "headless-tools-v1",
                      "artifact_identity": "derivation-sha256-v1",
                      "acceptance": "scripted-cloud-chain-numeric-check",
                      "cloud_mask_policy": CLOUD_POLICY,
                      "grid_inputs": profiles, "raster_inputs": raster_inputs})
        task["budget"].update(max_steps=50, max_tool_calls=35,
            max_wall_time_ms=900000, max_input_bytes=512 * 1024 * 1024,
            max_artifact_bytes=128 * 1024 * 1024)
        scenario = json.loads((template / "scenario.json").read_text())
        scenario.update(data_cutoff=receipt["config"]["cutoff"],
            freshness_max_age_seconds=None,
            allowed_actions=["tool.invoke", "memory.save_evidence", "answer.*"],
            allowed_tools=["catalog.search", "catalog.inspect_asset",
                           "raster.resample", "raster.band_math"])
        directory = out / "tasks/cloud-masked-ndvi"
        for name, value in (("task.json", task), ("assets.json", assets),
                            ("scenario.json", scenario),
                            ("evaluator.json", json.loads((template / "evaluator.json").read_text()))):
            write(directory / name, value)
        write(out / "inputs.json", {})
        write(out / "native-inputs.json", provider)
        write(out / "job.json", {"seed": 42, "task_ref": {
            "task_id": task["task_id"], "task_version": task["task_version"]}})
        manifest = TaskRegistry(out / "tasks").get(task["task_id"], task["task_version"])
        write(out / "admission.json", {
            "source_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "task_manifest_hash": manifest.task_manifest_hash, "inputs": 9,
            "date_triples": 3, "grid_method": "nearest", "cloud_mask_applied": True,
            "cloud_mask_policy": CLOUD_POLICY})
        return manifest.task_manifest_hash


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-name", required=True)
    arguments = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", arguments.output_name):
        raise SystemExit("simple fresh output name required")
    digest = prepare(arguments.source, ROOT / "runtime" / arguments.output_name)
    print(json.dumps({"task_manifest_hash": digest, "inputs": 9,
                      "date_triples": 3, "cloud_mask_policy": CLOUD_POLICY}))

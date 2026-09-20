#!/usr/bin/env python3
"""Acquire one pinned B11 window and build a reviewed B11-to-B08 task."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.v2.capabilities import TaskRegistry
from app.v2.data.http_range import BoundedHTTP
from app.v2.data.stac import (SWIR16_KEY, checked_config, json_bytes,
                              validate_item, validate_swir16_asset)
from app.v2.data.stac_windows import extract_window
from app.v2.raster_grid import ContinuousBand, checked_continuous_grids
from app.v2.raster_math import MAX_INPUT
from app.v2.storage.quota import StorageQuota


def write_json(path: Path, value: object) -> None:
    content = json_bytes(value)
    if len(content) > 2 * 1024 * 1024:
        raise ValueError("metadata exceeds bound")
    with path.open("xb") as stream:
        stream.write(content)


def read_json(path: Path, maximum: int = 2 * 1024 * 1024) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("bounded regular JSON required")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def checked_runtime_source(source: Path) -> tuple[Path, dict, dict]:
    runtime = (ROOT / "runtime").resolve()
    resolved = source.resolve()
    if source.is_symlink() or not resolved.is_relative_to(runtime):
        raise ValueError("reviewed runtime source required")
    admission_path = resolved / "receipts/admission.json"
    admission = read_json(admission_path)
    if admission.get("status") != "admitted" or len(admission.get("windows", [])) != 12:
        raise ValueError("complete three-date Sentinel admission required")
    config = checked_config(admission["config"])
    license_review = read_json(resolved / "receipts/license-review.json")
    if (license_review.get("legal_notice_sha256")
            != "fa2955ff48a1d82e77fc7296d63681670ecdb9d2811a0505ae60d0683b62fa64"
            or license_review.get("not_redistribution_authorization") is not True):
        raise ValueError("reviewed local-research license receipt required")
    return resolved, admission, config


def native_profile(record: dict, path: Path, band: str) -> ContinuousBand:
    import rasterio

    content = path.read_bytes()
    if (len(content) != record["size_bytes"]
            or hashlib.sha256(content).hexdigest() != record["sha256"]
            or len(content) > MAX_INPUT):
        raise ValueError("native input content differs from receipt")
    with rasterio.open(path, driver="GTiff") as image:
        if (image.count != 1 or image.dtypes != ("uint16",)
                or image.crs is None or image.crs.to_string() != record["crs"]
                or list(image.transform)[:6] != record["transform"]
                or [image.width, image.height] != [record["width"], record["height"]]
                or list(image.scales) != record["scales"]
                or list(image.offsets) != record["offsets"]):
            raise ValueError("native input metadata differs from receipt")
        asset_id = "asset-" + hashlib.sha256(json_bytes(
            [record["item_id"], record["asset_key"], record["sha256"]])).hexdigest()
        return ContinuousBand(
            asset_id=asset_id, sha256=record["sha256"],
            item_id=record["item_id"], band=band, acquired=record["acquired"],
            crs=record["crs"], transform=record["transform"],
            width=image.width, height=image.height, dtype="uint16",
            scale=image.scales[0], offset=image.offsets[0], nodata=image.nodata)


def prepare(source: Path, out: Path, item_id: str) -> str:
    source, admission, config = checked_runtime_source(source)
    if out.exists() or out.is_symlink():
        raise ValueError("fresh output required")
    if item_id not in config["items"]:
        raise ValueError("item must be pinned by the reviewed admission")
    reference_records = [record for record in admission["windows"]
                         if record["item_id"] == item_id
                         and record["asset_key"] == "nir"]
    if len(reference_records) != 1:
        raise ValueError("exactly one reviewed B08 reference required")
    reference_record = reference_records[0]
    item_path = source / "receipts" / (item_id + ".json")
    item = read_json(item_path)
    selected = validate_item(item, config, expected_id=item_id)
    if selected["snapshot_sha256"] != reference_record["source_snapshot_hash"]:
        raise ValueError("item receipt changed from the reference admission")
    selected["assets"][SWIR16_KEY] = validate_swir16_asset(item, selected)

    with StorageQuota(ROOT / "runtime").hold(
            out, 64 * 1024 * 1024, "continuous-grid-task-admission"):
        for name in ("inputs", "native-inputs", "state", "reports",
                     "receipts", "tasks/continuous-grid"):
            (out / name).mkdir(parents=True)
        http = BoundedHTTP(max_bytes=32 * 1024 * 1024,
                           max_requests=128, seconds=300)
        try:
            swir_path = out / "native-inputs" / (item_id + "-swir16.tif")
            swir_record = extract_window(
                http, selected, SWIR16_KEY, config["bbox"], swir_path)
        except Exception as error:
            write_json(out / "receipts/failure.json", {
                "status": "incomplete", "error_type": type(error).__name__,
                "requests": http.requests, "payload_bytes": http.bytes,
                "http_receipts": http.receipts})
            raise
        finally:
            http.close()

        reference_name = reference_record["filename"]
        if Path(reference_name).name != reference_name:
            raise ValueError("invalid reference filename")
        source_reference = source / "native-bands" / reference_name
        if (source_reference.is_symlink() or not source_reference.is_file()
                or not source_reference.resolve().is_relative_to(source)):
            raise ValueError("reviewed B08 reference path required")
        reference_content = source_reference.read_bytes()
        reference_path = out / "native-inputs" / reference_name
        with reference_path.open("xb") as stream:
            stream.write(reference_content)

        swir = native_profile(swir_record, swir_path, "swir16")
        reference = native_profile(reference_record, reference_path, "nir")
        checked_continuous_grids(swir, reference)
        records = [(swir_record, swir), (reference_record, reference)]
        profiles = {profile.asset_id: profile.model_dump(mode="json")
                    for _, profile in records}
        provider = {profile.asset_id: {
            "filename": path.name, "native": profiles[profile.asset_id]}
            for (_, profile), path in zip(records, (swir_path, reference_path))}
        assets = []
        for record, profile in records:
            west, south, east, north = record["bbox_wgs84"]
            assets.append({
                "asset_id": profile.asset_id,
                "uri": "local://approved-continuous-grid/" + record["filename"],
                "media_type": "image/tiff",
                "roles": ["input_image", "reflectance"],
                "sha256": profile.sha256, "size_bytes": record["size_bytes"],
                "spatial": {"crs": "EPSG:4326",
                    "bbox": {"west": west, "south": south,
                             "east": east, "north": north},
                    "gsd_meters": profile.transform[0],
                    "shape": [profile.height, profile.width, 1]},
                "temporal": {"start": profile.acquired,
                             "end": profile.acquired},
                "platform": "sentinel-2", "bands": [profile.band],
                "quality": {
                    "cloud_cover_percent": record["scene_cloud_cover_percent"],
                    "nodata_fraction": record["nodata_fraction"]},
                "license": config["license_review"]["attribution"],
                "source": "Reviewed Sentinel-2 L2A continuous reflectance window",
                "source_snapshot_hash": record["source_snapshot_hash"]})
        assets.sort(key=lambda asset: asset["asset_id"])
        template = ROOT / "tasks/worldcover-grounded-vqa"
        task = read_json(template / "task.json")
        identity = hashlib.sha256(json_bytes(profiles)).hexdigest()[:16]
        task.update(
            task_id="continuous-grid-" + identity,
            inputs=[asset["asset_id"] for asset in assets],
            prompt=("Inspect the pinned same-scene B11 20 m and B08 10 m assets, "
                    "align physical B11 reflectance to the B08 grid with explicit "
                    "bilinear resampling, cite the output and report coverage and "
                    "summary statistics. The B08 values/mask are not a science mask."),
            metadata={"observation_profile": "headless-tools-v1",
                      "artifact_identity": "derivation-sha256-v1",
                      "acceptance": "scripted-continuous-grid-numeric-check",
                      "grid_inputs": profiles})
        task["budget"].update(
            max_steps=16, max_tool_calls=8, max_wall_time_ms=300000,
            max_input_bytes=32 * 1024 * 1024,
            max_artifact_bytes=16 * 1024 * 1024)
        scenario = read_json(template / "scenario.json")
        scenario.update(
            data_cutoff=config["cutoff"], freshness_max_age_seconds=None,
            allowed_actions=["tool.invoke", "memory.save_evidence", "answer.*"],
            allowed_tools=["catalog.search", "catalog.inspect_asset",
                           "raster.resample"])
        directory = out / "tasks/continuous-grid"
        for name, value in (
                ("task.json", task), ("assets.json", assets),
                ("scenario.json", scenario),
                ("evaluator.json", read_json(template / "evaluator.json"))):
            write_json(directory / name, value)
        write_json(out / "inputs.json", {})
        write_json(out / "native-inputs.json", provider)
        write_json(out / "job.json", {"seed": 42, "task_ref": {
            "task_id": task["task_id"], "task_version": task["task_version"]}})
        manifest = TaskRegistry(out / "tasks").get(
            task["task_id"], task["task_version"])
        write_json(out / "receipts/admission.json", {
            "status": "admitted", "task_manifest_hash": manifest.task_manifest_hash,
            "source_admission_sha256": hashlib.sha256(
                (source / "receipts/admission.json").read_bytes()).hexdigest(),
            "item_receipt_sha256": hashlib.sha256(item_path.read_bytes()).hexdigest(),
            "item_id": item_id, "inputs": 2, "source_band": "swir16",
            "reference_band": "nir", "method": "bilinear",
            "network": "A800-system-proxy", "requests": http.requests,
            "payload_bytes": http.bytes, "http_receipts": http.receipts,
            "swir16_window": swir_record,
            "license_scope": "local-research; not redistribution authorization"})
        return manifest.task_manifest_hash


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--reviewed-license", action="store_true")
    arguments = parser.parse_args()
    if (not arguments.reviewed_license
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}",
                                arguments.output_name)
            or not re.fullmatch(r"S2[ABC]_[0-9]{2}[A-Z]{3}_[0-9]{8}_[0-9]+_L2A",
                                arguments.item_id)):
        raise SystemExit("fresh output, pinned item and reviewed license required")
    digest = prepare(arguments.source,
                     ROOT / "runtime" / arguments.output_name,
                     arguments.item_id)
    print(json.dumps({"status": "admitted", "task_manifest_hash": digest,
                      "inputs": 2, "method": "bilinear"}))

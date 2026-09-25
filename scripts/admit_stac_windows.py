#!/usr/bin/env python3
"""Operator-reviewed, bounded public Sentinel COG admission through system proxy."""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.core.data.http_range import BoundedHTTP
from app.core.data.stac import ASSETS, COLLECTION, ENDPOINT, checked_config, discover, fetch_json, json_bytes, validate_item
from app.core.data.stac_windows import extract_window
from app.core.storage.quota import StorageQuota
from app.core.capabilities import TaskRegistry


def write_json(path: Path, value: object) -> None:
    content = json_bytes(value)
    if len(content) > 2 * 1024 * 1024:
        raise ValueError("metadata receipt exceeds bound")
    with path.open("xb") as stream:
        stream.write(content)


def prepare_task(out: Path, config: dict, records: list[dict]) -> str:
    directory = out / "tasks" / "stac-window-smoke"
    directory.mkdir(parents=True)
    template = ROOT / "tasks/worldcover-grounded-vqa"
    task = json.loads((template / "task.json").read_text())
    scenario = json.loads((template / "scenario.json").read_text())
    evaluator = json.loads((template / "evaluator.json").read_text())
    assets, provider = [], {}
    for record in records:
        if not record["agent_admitted"]:
            continue  # DN bands/SCL retained privately pending dedicated tools.
        identity = hashlib.sha256(json_bytes([record["item_id"], record["asset_key"], record["sha256"]])).hexdigest()
        asset_id = "asset-" + identity
        west, south, east, north = record["bbox_wgs84"]
        assets.append({"asset_id": asset_id, "uri": "local://approved-input/" + record["filename"],
            "media_type": "image/tiff", "roles": ["input_image"], "sha256": record["sha256"],
            "size_bytes": record["size_bytes"], "spatial": {"crs": "EPSG:4326",
                "bbox": {"west": west, "south": south, "east": east, "north": north},
                "gsd_meters": 10, "shape": [record["height"], record["width"], 3]},
            "temporal": {"start": record["acquired"], "end": record["acquired"]},
            "platform": "sentinel-2", "bands": ["red", "green", "blue"],
            "quality": {"cloud_cover_percent": record["scene_cloud_cover_percent"], "nodata_fraction": record["nodata_fraction"]},
            "license": "Copernicus Sentinel legal notice; local research; " + config["license_review"]["attribution"],
            "source": "Earth Search Sentinel-2 L2A visual native-resolution AOI window; not reflectance bands",
            "source_snapshot_hash": record["source_snapshot_hash"]})
        provider[asset_id] = {"role": "input_image", "filename": record["filename"], "sha256": record["sha256"]}
    assets.sort(key=lambda a: a["asset_id"])
    identity = hashlib.sha256(json_bytes([config, assets])).hexdigest()[:16]
    task.update(task_id="stac-sentinel-window-" + identity, inputs=[a["asset_id"] for a in assets],
        prompt="Search the three dated Sentinel-2 visual windows, inspect and crop each central half, then cite the frozen crop dimensions. These RGB products are display images, not reflectance measurements.",
        metadata={"observation_profile": "headless-tools-v1", "artifact_identity": "derivation-sha256-v1",
                  "acceptance": "scripted-public-data-interaction-only", "cutoff_semantics": "acquisition-time; historical availability unverified"})
    task["budget"].update(max_steps=30, max_tool_calls=20, max_wall_time_ms=600000,
                          max_input_bytes=512 * 1024 * 1024, max_artifact_bytes=256 * 1024 * 1024)
    scenario.update(data_cutoff=config["cutoff"], freshness_max_age_seconds=None,
                    allowed_actions=["tool.invoke", "memory.save_evidence", "answer.*"],
                    allowed_tools=["catalog.search", "catalog.inspect_asset", "eo_gym.crop"])
    for filename, value in (("task.json", task), ("scenario.json", scenario), ("assets.json", assets), ("evaluator.json", evaluator)):
        write_json(directory / filename, value)
    write_json(out / "inputs.json", provider)
    write_json(out / "job.json", {"task_ref": {"task_id": task["task_id"], "task_version": task["task_version"]}, "seed": 42})
    return TaskRegistry(out / "tasks").get(task["task_id"], task["task_version"]).task_manifest_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/stac-sentinel-nanjing.json")
    parser.add_argument("--output-name")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--reviewed-license", action="store_true")
    args = parser.parse_args()
    if args.config.is_symlink() or args.config.stat().st_size > 65536:
        raise SystemExit("configuration must be a bounded regular file")
    config = checked_config(json.loads(args.config.read_text()))
    if args.discover_only:
        http = BoundedHTTP(max_bytes=4 * 1024 * 1024, max_requests=4, seconds=60)
        try:
            print(json.dumps(discover(http, config)))
        finally:
            http.close()
        return
    if not args.reviewed_license or not args.output_name or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", args.output_name):
        raise SystemExit("admission requires a fresh output name and operator license review")
    out = ROOT / "runtime" / args.output_name
    if out.exists():
        raise SystemExit("preserve existing run; never implicitly resume changed STAC metadata")
    with StorageQuota(ROOT / "runtime").hold(out, 256 * 1024 * 1024, "stac-window-admission"):
        for directory in ("inputs", "native-bands", "receipts", "state", "reports"):
            (out / directory).mkdir(parents=True, exist_ok=True)
        http = BoundedHTTP()
        records = []
        try:
            collection, raw = fetch_json(http, ENDPOINT + "/collections/" + COLLECTION)
            if collection.get("id") != COLLECTION or collection.get("license") != config["license_review"]["collection_license"]:
                raise ValueError("collection identity/license changed; manual review required")
            write_json(out / "receipts/collection.json", collection)
            terms, _ = http.fetch(config["license_review"]["terms_url"], 256 * 1024)
            if b"free, full and open" not in terms:
                raise ValueError("official terms no longer contain reviewed access statement")
            legal, _ = http.fetch(config["license_review"]["legal_notice_url"], 256 * 1024)
            legal_hash = hashlib.sha256(legal).hexdigest()
            if legal_hash != "fa2955ff48a1d82e77fc7296d63681670ecdb9d2811a0505ae60d0683b62fa64":
                raise ValueError("official legal notice changed; manual review required")
            write_json(out / "receipts/license-review.json", {**config["license_review"],
                "legal_notice_sha256": legal_hash,
                "terms_sha256": hashlib.sha256(terms).hexdigest(), "collection_response_sha256": hashlib.sha256(raw).hexdigest(),
                "reviewed_by": "operator", "not_redistribution_authorization": True})
            for item_id in config["items"]:
                item, _ = fetch_json(http, ENDPOINT + "/collections/" + COLLECTION + "/items/" + item_id)
                selected = validate_item(item, config, expected_id=item_id)
                write_json(out / "receipts" / (item_id + ".json"), item)
                for key in ASSETS:
                    destination = out / ("inputs" if key == "visual" else "native-bands") / (item_id + "-" + key + ".tif")
                    record = extract_window(http, selected, key, config["bbox"], destination)
                    records.append(record)
                    print(json.dumps({"item": item_id, "asset": key, "output_bytes": record["size_bytes"],
                                      "total_network_payload": http.bytes, "windows": len(records)}), flush=True)
            task_hash = prepare_task(out, config, records)
            write_json(out / "receipts/admission.json", {"status": "admitted", "config": config,
                "retrieved_at": datetime.now(timezone.utc).isoformat(), "task_manifest_hash": task_hash,
                "items": len(config["items"]), "windows": records, "agent_visual_inputs": len(config["items"]),
                "network": "A800-system-proxy", "requests": http.requests, "payload_bytes": http.bytes,
                "cutoff_semantics": "acquisition-time-only; not proof of historical availability", "http_receipts": http.receipts})
            print(json.dumps({"status": "admitted", "output": str(out), "visual_inputs": len(config["items"]), "windows": len(records)}))
        except Exception as error:
            write_json(out / "receipts/failure.json", {"status": "incomplete", "error_type": type(error).__name__,
                "completed_windows": records, "requests": http.requests, "payload_bytes": http.bytes, "http_receipts": http.receipts})
            raise
        finally:
            http.close()


if __name__ == "__main__":
    main()

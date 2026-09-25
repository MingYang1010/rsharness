#!/usr/bin/env python3
"""Admit two CloudSEN12-aligned public Sentinel visual windows for Qwen."""
from __future__ import annotations

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
from app.core.data.stac import ENDPOINT, COLLECTION, fetch_json, json_bytes, validate_item
from app.core.data.stac_windows import extract_window
from app.core.schemas import EvaluatorSpec, ScenarioProfile, TaskSpec
from app.core.storage.quota import StorageQuota


REQUIRED_CONFIG = {"schema_version", "source", "samples", "license_review"}
EVALUATOR_ID = "worldcover-grounded-v1"
METRIC_WEIGHTS = {
    "task.accuracy": 0.6,
    "evidence.faithfulness": 0.3,
    "process.efficiency": 0.1,
}


def review_config(config: dict) -> None:
    if set(config) != REQUIRED_CONFIG or config.get("schema_version") != "cloudsen12-public-sentinel-windows-v1":
        raise ValueError("unsupported CloudSEN12 public-window configuration")
    source = config["source"]
    if (source.get("dataset_name") != "CloudSEN12" or source.get("dataset_doi") != "10.57760/sciencedb.06669"
            or source.get("not_redistribution_authorization") is not True):
        raise ValueError("CloudSEN12 source boundary changed")
    samples = config.get("samples")
    if not isinstance(samples, list) or len(samples) != 2:
        raise ValueError("exactly two reviewed samples are required")
    ids = set()
    for sample in samples:
        if not sample.get("sample_id") or sample["sample_id"] in ids:
            raise ValueError("sample IDs must be unique")
        ids.add(sample["sample_id"])
        profile = sample.get("label_profile", {})
        if (profile.get("width") != 509 or profile.get("height") != 509
                or profile.get("crs") not in {"EPSG:32618", "EPSG:32651"}
                or not isinstance(profile.get("transform"), list) or len(profile["transform"]) != 6):
            raise ValueError("reviewed CloudSEN12 label profile changed")
        item_id = sample.get("sentinel_item_id")
        if not isinstance(item_id, str) or not re.fullmatch(r"S2[ABC]_[0-9]{2}[A-Z]{3}_[0-9]{8}_[0-9]+_L2A", item_id):
            raise ValueError("invalid Sentinel item ID")
        aoi = sample.get("aoi")
        if (not isinstance(aoi, list) or len(aoi) != 4
                or not -180 <= aoi[0] < aoi[2] <= 180 or not -90 <= aoi[1] < aoi[3] <= 90
                or aoi[2] - aoi[0] > .05 or aoi[3] - aoi[1] > .05):
            raise ValueError("invalid CloudSEN12-aligned AOI")


def item_config(config: dict, sample: dict) -> dict:
    aoi = sample["aoi"]
    # CloudSEN12 sample IDs contain the label scene's sensing window, while the
    # public Earth Search product can use a different processing datetime.
    # Pin the exact reviewed item ID and use a one-day interval around its year/month/day.
    date_text = sample["sample_id"].split("--", 1)[1].split("_", 1)[0][:8]
    date = datetime.strptime(date_text, "%Y%m%d").date()
    start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    end = datetime(date.year, date.month, date.day, tzinfo=timezone.utc).timestamp() + 86399
    return {
        "collection": COLLECTION,
        "bbox": aoi,
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": datetime.fromtimestamp(end, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "cutoff": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "max_cloud_cover": 100,
        "items": [sample["sentinel_item_id"]],
        "assets": ["visual", "red", "nir", "scl"],
        "license_review": config["license_review"],
    }


def qwen_documents(task_id: str, asset_id: str, data_cutoff: str) -> tuple[dict, dict, dict]:
    task = {
        "task_id": task_id, "task_version": "1.0.0", "family": "grounded_vqa",
        "scenario_profile": "general-inspection-v1",
        "prompt": "Inspect the supplied Sentinel-2 visual window aligned to a CloudSEN12 ROI, crop the central half, save verified evidence, and report frozen pixel dimensions. Do not claim cloud truth.",
        "inputs": [asset_id], "seed": 42, "evaluator": EVALUATOR_ID,
        "metric_aggregation": METRIC_WEIGHTS,
        "answer_schema": {"type": "object", "required": ["label", "confidence", "claims"], "properties": {
            "label": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "claims": {"type": "array", "items": {"type": "object"}}}},
        "budget": {"max_steps": 20, "max_tool_calls": 5, "max_wall_time_ms": 300000,
                   "max_input_bytes": 536870912, "max_artifact_bytes": 134217728},
        "metadata": {"observation_profile": "headless-tools-v1",
                     "artifact_identity": "derivation-sha256-v1",
                     "acceptance": "qwen-real-interaction-public-window-not-cloud-truth"},
    }
    scenario = {
        "profile_id": "general-inspection-v1", "domain": "general_inspection",
        "data_cutoff": data_cutoff, "freshness_max_age_seconds": None,
        "allowed_actions": ["tool.invoke", "memory.save_evidence", "answer.*"],
        "allowed_tools": ["catalog.search", "catalog.inspect_asset", "eo_gym.crop"],
        "network_policy": "none", "evidence_required": True, "abstention_allowed": True,
        "human_review_policy": "allowed",
    }
    evaluator = {
        "evaluator_id": EVALUATOR_ID, "evaluator_version": "1.0.0",
        "metric_names": list(METRIC_WEIGHTS), "aggregate_weights": METRIC_WEIGHTS,
        "config": {"implementation_status": "interaction-only",
                   "label_source": "public Sentinel visual crop; CloudSEN12 labels excluded"},
    }
    TaskSpec.model_validate(task)
    ScenarioProfile.model_validate(scenario)
    EvaluatorSpec.model_validate(evaluator)
    return task, scenario, evaluator


def write_json(path: Path, value: object) -> None:
    content = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def prepare(config: dict, output: Path) -> list[dict]:
    review_config(config)
    http = BoundedHTTP(max_bytes=64 * 1024 * 1024, max_requests=64, seconds=600)
    results = []
    try:
        for sample in config["samples"]:
            item_url = ENDPOINT + "/collections/" + COLLECTION + "/items/" + sample["sentinel_item_id"]
            item, raw = fetch_json(http, item_url)
            current = item_config(config, sample)
            selected = validate_item(item, current, expected_id=sample["sentinel_item_id"])
            write_json(output / "receipts" / (sample["sentinel_item_id"] + ".json"), item)
            sample_dir = output / ("sample-" + sample["sample_id"])
            (sample_dir / "inputs").mkdir(parents=True, exist_ok=True)
            (sample_dir / "tasks" / "cloudsen12-qwen").mkdir(parents=True, exist_ok=True)
            (sample_dir / "state").mkdir(exist_ok=True)
            (sample_dir / "reports").mkdir(exist_ok=True)
            filename = sample["sentinel_item_id"] + "-visual.tif"
            record = extract_window(http, selected, "visual", sample["aoi"], sample_dir / "inputs" / filename)
            identity = hashlib.sha256(json_bytes([sample["sample_id"], "public-visual", record["sha256"]])).hexdigest()
            asset_id = "asset-" + identity
            asset = {
                "asset_id": asset_id, "uri": "local://approved-input/" + filename,
                "media_type": "image/tiff", "roles": ["input_image"], "sha256": record["sha256"],
                "size_bytes": record["size_bytes"], "bands": ["red", "green", "blue"],
                "platform": "sentinel-2", "license": "Copernicus Sentinel legal notice; local research; "
                    + config["license_review"]["attribution"],
                "source": "CloudSEN12-aligned public Sentinel-2 visual AOI window; not CloudSEN12 labels",
                "source_snapshot_hash": record["source_snapshot_hash"],
                "spatial": {"crs": "EPSG:4326", "bbox": {
                    "west": sample["aoi"][0], "south": sample["aoi"][1], "east": sample["aoi"][2], "north": sample["aoi"][3]},
                    "gsd_meters": 10, "shape": [record["height"], record["width"], 3]},
                "temporal": {"start": record["acquired"], "end": record["acquired"]},
                "quality": {"cloud_cover_percent": selected["scene_cloud_cover_percent"], "nodata_fraction": record["nodata_fraction"]},
            }
            task_id = "cloudsen12-qwen-" + hashlib.sha256(json_bytes([asset_id, record["sha256"], sample["sample_id"]])).hexdigest()[:16]
            task, scenario, evaluator = qwen_documents(task_id, asset_id, record["acquired"])
            write_json(sample_dir / "tasks" / "cloudsen12-qwen" / "task.json", task)
            write_json(sample_dir / "tasks" / "cloudsen12-qwen" / "assets.json", [asset])
            write_json(sample_dir / "tasks" / "cloudsen12-qwen" / "scenario.json", scenario)
            write_json(sample_dir / "tasks" / "cloudsen12-qwen" / "evaluator.json", evaluator)
            write_json(sample_dir / "inputs.json", {asset_id: {"role": "input_image", "filename": filename, "sha256": record["sha256"]}})
            write_json(sample_dir / "job.json", {"task_ref": {"task_id": task_id, "task_version": "1.0.0"}, "seed": 42,
                        "cloudsen12_sample_id": sample["sample_id"], "asset_id": asset_id, "source_sha256": record["sha256"]})
            results.append({"sample_id": sample["sample_id"], "item_id": sample["sentinel_item_id"], "asset_id": asset_id,
                            "content_sha256": record["sha256"], "task_root": str((sample_dir / "tasks" / "cloudsen12-qwen").resolve())})
        write_json(output / "admission.json", {"schema_version": config["schema_version"], "source": config["source"],
                    "samples": results, "http": {"requests": http.requests, "bytes": http.bytes, "receipts": http.receipts},
                    "boundary": "public visual windows admitted; CloudSEN12 manual/SCL labels are evaluator-only and absent"})
        return results
    finally:
        http.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "cloudsen12-public-samples-v1.json")
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", args.output_name):
        raise SystemExit("output-name must be a simple runtime directory name")
    if args.config.is_symlink() or not args.config.is_file() or args.config.stat().st_size > 256 * 1024:
        raise SystemExit("configuration must be a bounded regular file")
    output = ROOT / "runtime" / args.output_name
    if output.exists():
        raise SystemExit("preserve existing output; choose a fresh name")
    config = json.loads(args.config.read_text())
    with StorageQuota(ROOT / "runtime").hold(output, 64 * 1024 * 1024, "cloudsen12-public-qwen-windows"):
        result = prepare(config, output)
    print(json.dumps({"output": str(output), "samples": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

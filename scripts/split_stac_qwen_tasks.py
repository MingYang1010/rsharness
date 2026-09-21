#!/usr/bin/env python3
"""Split an accepted three-window Sentinel STAC task into two Qwen tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.v2.storage.quota import StorageQuota


MAX_TASK_BYTES = 4 * 1024 * 1024


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def checked_source(source: Path) -> tuple[dict, list[dict]]:
    admission_path = source / "receipts" / "admission.json"
    task_path = source / "tasks" / "stac-window-smoke" / "task.json"
    assets_path = source / "tasks" / "stac-window-smoke" / "assets.json"
    for path in (admission_path, task_path, assets_path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_TASK_BYTES:
            raise ValueError("source admission/task is unavailable or oversized")
    admission = json.loads(admission_path.read_text())
    original_task = json.loads(task_path.read_text())
    assets = json.loads(assets_path.read_text())
    if admission.get("status") != "admitted" or admission.get("agent_visual_inputs", 0) < 2:
        raise ValueError("source STAC admission is not accepted or has fewer than two visual inputs")
    records = [record for record in admission.get("windows", []) if record.get("agent_admitted")]
    if len(records) < 2 or len(assets) < 2:
        raise ValueError("source admission does not contain two visual assets")
    if set(original_task.get("inputs", [])) != {item["asset_id"] for item in assets}:
        raise ValueError("source task and asset manifest disagree")
    return admission, records


def split(source: Path, output: Path) -> list[dict]:
    admission, records = checked_source(source)
    assets = json.loads((source / "tasks" / "stac-window-smoke" / "assets.json").read_text())
    manifest = json.loads((source / "inputs.json").read_text())
    selected = []
    seen_content = set()
    for record in sorted(records, key=lambda item: item["filename"]):
        filename = record["filename"]
        input_path = source / "inputs" / filename
        if input_path.is_symlink() or not input_path.is_file():
            raise ValueError("visual input is unavailable")
        digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if digest != record["sha256"] or digest in seen_content:
            raise ValueError("visual input changed or is not distinct")
        seen_content.add(digest)
        matches = [item for item in assets if item.get("sha256") == digest]
        if len(matches) != 1:
            raise ValueError("visual window is not uniquely represented by the source task")
        asset = matches[0]
        asset_id = asset["asset_id"]
        if asset_id not in manifest:
            raise ValueError("admission receipt, task asset, and provider manifest disagree")
        selected.append((record, input_path, asset))
    selected = selected[:2]
    if len(selected) != 2:
        raise ValueError("two distinct visual inputs are required")

    template = source / "tasks" / "stac-window-smoke"
    results = []
    for index, (record, input_path, asset) in enumerate(selected):
        sample = output / ("sample-" + str(index))
        for directory in ("inputs", "tasks/stac-qwen", "state", "reports"):
            (sample / directory).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, sample / "inputs" / record["filename"])
        if hashlib.sha256((sample / "inputs" / record["filename"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("copied STAC window changed")
        task = json.loads((template / "task.json").read_text())
        scenario = json.loads((template / "scenario.json").read_text())
        evaluator = json.loads((template / "evaluator.json").read_text())
        task_identity = ["sentinel-2-stac-qwen-v1", record["item_id"], record["sha256"], record["source_snapshot_hash"]]
        task_id = "stac-qwen-" + hashlib.sha256(canonical(task_identity)).hexdigest()[:16]
        task.update(task_id=task_id, inputs=[asset["asset_id"]],
                    prompt=("Inspect the supplied dated Sentinel-2 visual window, crop its central half, save the "
                            "verified crop evidence, then report the frozen pixel dimensions. This RGB product is a "
                            "display image, not calibrated reflectance."),
                    metadata={**task["metadata"], "acceptance": "qwen-real-interaction-only-not-semantic"})
        scenario.update(allowed_actions=["tool.invoke", "memory.save_evidence", "answer.*"],
                        allowed_tools=["catalog.search", "catalog.inspect_asset", "eo_gym.crop"])
        write_json(sample / "tasks" / "stac-qwen" / "task.json", task)
        write_json(sample / "tasks" / "stac-qwen" / "scenario.json", scenario)
        write_json(sample / "tasks" / "stac-qwen" / "assets.json", [asset])
        write_json(sample / "tasks" / "stac-qwen" / "evaluator.json", evaluator)
        write_json(sample / "inputs.json", {asset["asset_id"]: {"role": "input_image", "filename": record["filename"], "sha256": record["sha256"]}})
        write_json(sample / "job.json", {"task_ref": {"task_id": task_id, "task_version": "1.0.0"}, "seed": 42,
                                          "asset_id": asset["asset_id"], "item_id": record["item_id"],
                                          "source_sha256": record["sha256"], "sample_index": index})
        results.append({"sample_index": index, "task_root": str((sample / "tasks" / "stac-qwen").resolve()),
                        "asset_id": asset["asset_id"], "content_sha256": record["sha256"], "item_id": record["item_id"]})
    write_json(output / "split-receipt.json", {
        "schema_version": "stac-qwen-split-1",
        "source": str(source.resolve()),
        "source_task_manifest_hash": json.loads((source / "tasks" / "stac-window-smoke" / "task.json").read_text()).get("task_manifest_hash"),
        "source_windows": len(records),
        "selected": results,
        "model": "Qwen3.5-9B",
        "scope": "two distinct accepted visual windows; no network or source mutation",
    })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runtime" / "stac-sentinel-nanjing-20260917-01")
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", args.output_name):
        raise SystemExit("output-name must be a simple runtime directory name")
    source = args.source.resolve()
    output = ROOT / "runtime" / args.output_name
    if not source.is_relative_to(ROOT / "runtime") or source == ROOT / "runtime":
        raise SystemExit("source must be an existing project runtime admission")
    if output.exists():
        raise SystemExit("preserve existing split output; choose a fresh name")
    with StorageQuota(ROOT / "runtime").hold(output, 8 * 1024 * 1024, "stac-qwen-task-split"):
        result = split(source, output)
    print(json.dumps({"output": str(output), "samples": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

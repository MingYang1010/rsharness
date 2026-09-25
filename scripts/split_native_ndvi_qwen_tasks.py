#!/usr/bin/env python3
"""Split the accepted three-date native-NDVI task into two Qwen tasks."""
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
from app.core.storage.quota import StorageQuota


def write_json(path: Path, value: object) -> None:
    content = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if len(content) > 2 * 1024 * 1024:
        raise ValueError("task metadata exceeds bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def date_pairs(task: dict, assets: list[dict]) -> list[tuple[str, list[dict]]]:
    inputs = set(task.get("inputs", []))
    public = [item for item in assets if item["asset_id"] in inputs]
    profiles = task["metadata"]["raster_inputs"]
    groups: dict[str, list[dict]] = {}
    for asset in public:
        item_id = profiles[asset["asset_id"]]["item_id"]
        groups.setdefault(item_id, []).append(asset)
    result = []
    for item_id, items in sorted(groups.items()):
        if sorted(item["bands"] for item in items) != [["nir"], ["red"]]:
            raise ValueError(item_id + " does not contain one red/NIR pair")
        result.append((item_id, sorted(items, key=lambda item: item["bands"][0])))
    if len(result) < 2:
        raise ValueError("source has fewer than two dates")
    return result


def split(source: Path, output: Path) -> list[dict]:
    task_dir = source / "tasks" / "native-ndvi"
    for name in ("task.json", "assets.json", "scenario.json", "evaluator.json"):
        path = task_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("native-NDVI source task is unavailable or oversized")
    task = json.loads((task_dir / "task.json").read_text())
    assets = json.loads((task_dir / "assets.json").read_text())
    scenario = json.loads((task_dir / "scenario.json").read_text())
    evaluator = json.loads((task_dir / "evaluator.json").read_text())
    provider_path = source / "native-inputs.json"
    if provider_path.is_symlink() or not provider_path.is_file() or provider_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("native input manifest is unavailable or oversized")
    provider = json.loads(provider_path.read_text())
    if task.get("metadata", {}).get("artifact_identity") != "derivation-sha256-v1":
        raise ValueError("source task lacks derivation identity")
    pairs = date_pairs(task, assets)
    results = []
    seen = set()
    for index, (item_id, pair) in enumerate(pairs[:2]):
        identities = tuple(sorted(item["asset_id"] for item in pair))
        if identities in seen:
            raise ValueError("science sample assets are not distinct")
        seen.add(identities)
        sample = output / ("sample-" + str(index))
        (sample / "native-inputs").mkdir(parents=True, exist_ok=True)
        (sample / "tasks" / "native-ndvi-qwen").mkdir(parents=True, exist_ok=True)
        sample_assets = []
        sample_provider = {}
        for asset in pair:
            source_path = source / "native-inputs" / Path(asset["uri"]).name
            if source_path.is_symlink() or not source_path.is_file():
                raise ValueError("native input is unavailable")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if digest != asset["sha256"]:
                raise ValueError("native input changed")
            shutil.copyfile(source_path, sample / "native-inputs" / source_path.name)
            sample_assets.append(asset)
            entry = provider.get(asset["asset_id"])
            if not isinstance(entry, dict) or entry.get("filename") != source_path.name:
                raise ValueError("native input manifest does not bind the selected file")
            sample_provider[asset["asset_id"]] = entry
        selected_profiles = {item["asset_id"]: task["metadata"]["raster_inputs"][item["asset_id"]] for item in pair}
        if any(sample_provider[asset_id].get("native") != profile
               for asset_id, profile in selected_profiles.items()):
            raise ValueError("native input manifest metadata changed")
        task_identity = ["native-ndvi-qwen-v1", item_id, selected_profiles]
        task_id = "native-ndvi-qwen-" + hashlib.sha256(json.dumps(task_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        sample_task = {
            **task,
            "task_id": task_id,
            "inputs": [item["asset_id"] for item in pair],
            "prompt": ("For the supplied same-date Sentinel-2 red/NIR window, call raster.band_math with operation "
                       "ndvi, save the numeric artifact evidence, and report the valid-pixel mean. Apply DN scale and "
                       "offset first. Cloud masking is not applied; this is not vegetation-change truth."),
            "metadata": {**task["metadata"], "raster_inputs": selected_profiles,
                         "acceptance": "qwen-real-interaction-science-not-semantic"},
        }
        sample_task["budget"].update(max_steps=20, max_tool_calls=5, max_wall_time_ms=300000)
        sample_scenario = {**scenario, "allowed_actions": ["tool.invoke", "memory.save_evidence", "answer.*"],
                           "allowed_tools": ["catalog.search", "catalog.inspect_asset", "raster.band_math"]}
        write_json(sample / "tasks" / "native-ndvi-qwen" / "task.json", sample_task)
        write_json(sample / "tasks" / "native-ndvi-qwen" / "assets.json", sample_assets)
        write_json(sample / "tasks" / "native-ndvi-qwen" / "scenario.json", sample_scenario)
        write_json(sample / "tasks" / "native-ndvi-qwen" / "evaluator.json", evaluator)
        write_json(sample / "inputs.json", {})
        write_json(sample / "native-inputs.json", sample_provider)
        write_json(sample / "job.json", {"task_ref": {"task_id": task_id, "task_version": "1.0.0"}, "seed": 42,
                                          "item_id": item_id, "asset_ids": list(identities)})
        (sample / "state").mkdir(exist_ok=True)
        (sample / "reports").mkdir(exist_ok=True)
        results.append({"sample_index": index, "item_id": item_id,
                        "asset_ids": list(identities),
                        "content_sha256": selected_profiles[identities[0]]["sha256"],
                        "task_root": str((sample / "tasks" / "native-ndvi-qwen").resolve())})
    if len(results) != 2:
        raise ValueError("two science samples are required")
    write_json(output / "split-receipt.json", {"schema_version": "native-ndvi-qwen-split-1",
                "source": str(source.resolve()), "selected": results,
                "scope": "two independent dates; source task and state unchanged"})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runtime" / "native-ndvi-nanjing-20260917-01")
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", args.output_name):
        raise SystemExit("output-name must be a simple runtime directory name")
    output = ROOT / "runtime" / args.output_name
    if output.exists():
        raise SystemExit("preserve existing output; choose a fresh name")
    with StorageQuota(ROOT / "runtime").hold(output, 8 * 1024 * 1024, "native-ndvi-qwen-task-split"):
        result = split(args.source.resolve(), output)
    print(json.dumps({"output": str(output), "samples": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

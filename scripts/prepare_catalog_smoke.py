#!/usr/bin/env python3
"""Combine up to three already-reviewed smoke images into one catalog task."""
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


def read_json(path: Path):
    if path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError("smoke metadata must be a bounded regular file")
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-run", action="append", required=True, type=Path)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    if not 1 <= len(args.sample_run) <= 3 or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", args.output_name):
        raise SystemExit("use one to three reviewed runs and a simple output name")
    runtime = ROOT / "runtime"
    out = runtime / args.output_name
    if out.exists():
        raise SystemExit("preserve existing smoke directory")
    task_entries = list((ROOT / "tasks").rglob("*"))
    if (len(task_entries) > 1024 or any(p.is_symlink() for p in task_entries)
            or sum(p.stat().st_size for p in task_entries if p.is_file()) > 4 * 1024 * 1024):
        raise SystemExit("template task tree exceeds staging bounds")
    from rasterio.io import MemoryFile
    with StorageQuota(runtime).hold(out, 512 * 1024 * 1024, "catalog-smoke-staging"):
        assets, inputs, expected = [], {}, []
        for directory in ("inputs", "provider-out", "state", "artifacts", "reports"):
            (out / directory).mkdir(parents=True, exist_ok=True)
        for original in args.sample_run:
            run = original.resolve()
            if original.is_symlink() or not run.is_relative_to(runtime.resolve()):
                raise ValueError("reviewed run must be inside project runtime")
            job = read_json(run / "job.json")
            asset_id = job["asset_id"]
            if asset_id in inputs:
                raise ValueError("duplicate image identity")
            manifest = read_json(run / "inputs.json")[asset_id]
            source = (run / "inputs" / manifest["filename"]).resolve()
            if manifest["role"] != "input_image" or not source.is_relative_to(run / "inputs") or source.stat().st_size > 128 * 1024 * 1024:
                raise ValueError("invalid reviewed input")
            assets_path = run / "tasks" / "eo-gym-crop-smoke" / "assets.json"
            asset = next(a for a in read_json(assets_path) if a["asset_id"] == asset_id)
            if asset["roles"] != ["input_image"] or asset["spatial"] is not None:
                raise ValueError("catalog acceptance uses reviewed pixel images only")
            content = source.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != manifest["sha256"] or digest != asset["sha256"] or digest != job["source_sha256"] or len(content) != asset["size_bytes"]:
                raise ValueError("reviewed image hash changed")
            with MemoryFile(content) as memory, memory.open() as image:
                if (image.crs is not None or image.width * image.height > 20_000_000 or
                        (image.width, image.height, image.count) != (asset["pixel"]["width"], asset["pixel"]["height"], asset["pixel"]["channels"])):
                    raise ValueError("reviewed image dimensions changed")
            filename = digest + source.suffix.lower()
            (out / "inputs" / filename).write_bytes(content)
            asset["uri"] = "local://approved-input/" + filename
            assets.append(asset)
            inputs[asset_id] = {"role": "input_image", "filename": filename, "sha256": digest}
            expected.append({"asset_id": asset_id, "sha256": digest, "pixel": asset["pixel"]})
        expected.sort(key=lambda a: a["asset_id"])
        identity = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:16]
        task_dir = out / "tasks" / "catalog-smoke"
        shutil.copytree(ROOT / "tasks" / "worldcover-grounded-vqa", task_dir)
        task = read_json(task_dir / "task.json")
        task.update(task_id="catalog-smoke-" + identity, inputs=[a["asset_id"] for a in expected],
                    prompt="Search the available images, inspect and crop each central half, then cite the crop dimensions.",
                    metadata={"observation_profile": "headless-tools-v1", "acceptance": "scripted-catalog-interaction-only"})
        task["budget"].update(max_steps=30, max_tool_calls=20, max_wall_time_ms=600000,
                              max_input_bytes=512 * 1024 * 1024, max_artifact_bytes=256 * 1024 * 1024)
        scenario = read_json(task_dir / "scenario.json")
        scenario.update(allowed_actions=["tool.invoke", "memory.save_evidence", "answer.*"],
                        allowed_tools=["catalog.search", "catalog.inspect_asset", "eo_gym.crop"])
        for path, data in [(task_dir / "task.json", task), (task_dir / "scenario.json", scenario),
                           (task_dir / "assets.json", assets), (out / "inputs.json", inputs),
                           (out / "job.json", {"task_ref": {"task_id": task["task_id"], "task_version": "1.0.0"}, "seed": 42, "expected": expected})]:
            path.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"output": str(out), "images": len(assets)}))


if __name__ == "__main__":
    main()

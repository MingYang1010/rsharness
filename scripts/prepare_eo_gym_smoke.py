#!/usr/bin/env python3
"""Stage one audited georeferenced FAIR1M image for an isolated integration smoke."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "runtime" / "eo-gym-smoke"
    if out.exists():
        raise SystemExit("smoke directory exists; preserve prior run, do not overwrite")
    coverage = json.loads(args.inventory.read_text())
    dataset = next(d for d in coverage["datasets"] if d["dataset_id"] == "source_datasets/FAIR1M2_LittleCollections")
    sample = next(s for s in dataset["samples"] if s["status"] == "readable_sample" and s["spatial"] is not None
                  and s["width"] * s["height"] <= 20_000_000 and s["spatial"]["native_crs"] == "EPSG:4326")
    source = (Path(dataset["root"]) / sample["relative_path"]).resolve()
    if not source.is_relative_to(Path(dataset["root"]).resolve()) or source.stat().st_size > 128 * 1024 * 1024:
        raise SystemExit("invalid sample source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != sample["sha256"]:
        raise SystemExit("source changed after inventory")
    for directory in ("inputs", "provider-out", "state", "artifacts", "reports"):
        (out / directory).mkdir(parents=True)
    filename = sample["asset_id"] + source.suffix.lower()
    shutil.copyfile(source, out / "inputs" / filename)
    manifest = {sample["asset_id"]: {"role": "input_image", "filename": filename, "sha256": digest}}
    (out / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copytree(root / "tasks", out / "tasks")
    task_dir = out / "tasks" / "eo-gym-crop-smoke"
    shutil.copytree(root / "tasks" / "worldcover-grounded-vqa", task_dir)
    task = json.loads((task_dir / "task.json").read_text())
    task.update(task_id="eo-gym-crop-smoke", inputs=[sample["asset_id"]],
        prompt="Crop the central half of the supplied image, report its pixel dimensions and cite the frozen crop artifact.",
        metadata={"observation_profile": "headless-tools-v1", "acceptance": "interaction-only-not-semantic-benchmark"})
    task["budget"].update(max_input_bytes=512 * 1024 * 1024, max_artifact_bytes=128 * 1024 * 1024, max_wall_time_ms=300000)
    (task_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    scenario = json.loads((task_dir / "scenario.json").read_text())
    scenario.update(allowed_actions=["tool.invoke", "memory.save_evidence", "answer.*"], allowed_tools=["eo_gym.crop"])
    (task_dir / "scenario.json").write_text(json.dumps(scenario, indent=2) + "\n")
    west, south, east, north = sample["spatial"]["bbox_wgs84"]
    asset = {"asset_id": sample["asset_id"], "uri": "local://approved-input/" + filename,
        "media_type": "image/tiff", "roles": ["input_image"], "sha256": digest, "size_bytes": source.stat().st_size,
        "spatial": {"crs": "EPSG:4326", "bbox": {"west": west, "south": south, "east": east, "north": north},
                    "shape": [sample["height"], sample["width"], sample["bands"]]},
        "license": "existing-local-research-copy-redistribution-not-authorized",
        "source": "FAIR1M2 local image sample; original georeferencing retained"}
    (task_dir / "assets.json").write_text(json.dumps([asset], indent=2) + "\n")
    job = {"task_ref": {"task_id": "eo-gym-crop-smoke", "task_version": "1.0.0"}, "seed": 42,
           "asset_id": sample["asset_id"], "source_width": sample["width"], "source_height": sample["height"],
           "source_sha256": digest, "aoi": [0.25, 0.25, 0.75, 0.75]}
    (out / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    print(json.dumps({"output": str(out), "input_bytes": source.stat().st_size, "source_sha256": digest}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage one audited georeferenced or pixel-only image for isolated smoke tests."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--dataset-id", default="source_datasets/FAIR1M2_LittleCollections")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-name", default="eo-gym-smoke")
    args = parser.parse_args()
    root = args.root.resolve()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", args.output_name):
        raise SystemExit("output-name must be a simple runtime directory name")
    out = root / "runtime" / args.output_name
    if out.exists():
        raise SystemExit("smoke directory exists; preserve prior run, do not overwrite")
    coverage = json.loads(args.inventory.read_text())
    dataset = next((d for d in coverage["datasets"] if d["dataset_id"] == args.dataset_id), None)
    if dataset is None or not 0 <= args.sample_index < len(dataset["samples"]):
        raise SystemExit("dataset or sample index not present in inventory")
    sample = dataset["samples"][args.sample_index]
    if sample["status"] != "readable_sample" or sample["width"] * sample["height"] > 20_000_000:
        raise SystemExit("sample is unreadable or exceeds provider pixel limit")
    source = (Path(dataset["root"]) / sample["relative_path"]).resolve()
    if not source.is_relative_to(Path(dataset["root"]).resolve()) or source.stat().st_size > 128 * 1024 * 1024:
        raise SystemExit("invalid sample source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != sample["sha256"]:
        raise SystemExit("source changed after inventory")
    import rasterio
    with rasterio.open(source) as image:
        if (image.width, image.height, image.count) != (sample["width"], sample["height"], sample["bands"]):
            raise SystemExit("sample dimensions changed after inventory")
        if sample["spatial"] is None and image.crs is not None:
            raise SystemExit("cannot discard existing image georeferencing")
        if sample["spatial"] is not None and image.crs is None:
            raise SystemExit("inventory claims georeferencing absent from image")
    spatial = None
    if sample["spatial"] is not None:
        west, south, east, north = sample["spatial"]["bbox_wgs84"]
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise SystemExit("inventory WGS84 bounds are invalid; do not invent coordinates")
        spatial = {"crs": "EPSG:4326", "bbox": {"west": west, "south": south, "east": east, "north": north},
                   "shape": [sample["height"], sample["width"], sample["bands"]]}
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
    task_identity = [args.dataset_id, sample["asset_id"], digest, sample.get("source_snapshot_hash")]
    task_id = "eo-gym-crop-" + hashlib.sha256(json.dumps(task_identity, separators=(",", ":")).encode()).hexdigest()[:16]
    task.update(task_id=task_id, inputs=[sample["asset_id"]],
        prompt="Crop the central half of the supplied image, report its pixel dimensions and cite the frozen crop artifact.",
        metadata={"observation_profile": "headless-tools-v1", "acceptance": "interaction-only-not-semantic-benchmark"})
    task["budget"].update(max_input_bytes=512 * 1024 * 1024, max_artifact_bytes=128 * 1024 * 1024, max_wall_time_ms=300000)
    (task_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    scenario = json.loads((task_dir / "scenario.json").read_text())
    scenario.update(allowed_actions=["tool.invoke", "memory.save_evidence", "answer.*"], allowed_tools=["eo_gym.crop"])
    (task_dir / "scenario.json").write_text(json.dumps(scenario, indent=2) + "\n")
    asset = {"asset_id": sample["asset_id"], "uri": "local://approved-input/" + filename,
        "media_type": {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(source.suffix.lower(), "image/tiff"),
        "roles": ["input_image"], "sha256": digest, "size_bytes": source.stat().st_size,
        "spatial": spatial,
        "license": "existing-local-research-copy-redistribution-not-authorized",
        "source": args.dataset_id + " local image sample; no fabricated georeferencing"}
    if sample.get("source_snapshot_hash"):
        asset["source_snapshot_hash"] = sample["source_snapshot_hash"]
    if spatial is None:
        asset["pixel"] = {"coordinate_system": "pixel", "width": sample["width"],
                          "height": sample["height"], "channels": sample["bands"]}
    (task_dir / "assets.json").write_text(json.dumps([asset], indent=2) + "\n")
    job = {"task_ref": {"task_id": task_id, "task_version": "1.0.0"}, "seed": 42,
           "asset_id": sample["asset_id"], "source_width": sample["width"], "source_height": sample["height"],
           "source_sha256": digest, "coordinate_system": "pixel" if spatial is None else "geographic",
           "dataset_id": args.dataset_id, "sample_index": args.sample_index, "aoi": [0.25, 0.25, 0.75, 0.75]}
    (out / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    print(json.dumps({"output": str(out), "input_bytes": source.stat().st_size, "source_sha256": digest}))


if __name__ == "__main__":
    main()

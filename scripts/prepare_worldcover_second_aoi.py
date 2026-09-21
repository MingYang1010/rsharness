#!/usr/bin/env python3
"""Prepare a second fixed WorldCover rendered Qwen task from the local tile."""
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


SECOND_AOI = {"west": 122.65, "south": 30.75, "east": 122.75, "north": 30.85}
EXPECTED_DISTRIBUTION = {"10": 31172, "30": 3999, "40": 92, "50": 2178, "60": 2250, "80": 1400309}
EXPECTED_PIXEL_COUNT = 1_440_000


def write_json(path: Path, value: object) -> None:
    content = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if len(content) > 2 * 1024 * 1024:
        raise ValueError("task metadata exceeds bound")
    path.write_bytes(content)


def prepare(source: Path, output: Path) -> dict:
    template = source / "worldcover-grounded-vqa-1.1.0"
    for name in ("task.json", "assets.json", "scenario.json", "evaluator.json"):
        path = template / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("WorldCover template is unavailable or oversized")
    canonical = next(
        item for item in json.loads((template / "assets.json").read_text())
        if item.get("asset_id") == "asset-worldcover-n30e120-canonical"
    )
    canonical_path = source.parent / "datasets" / "worldcover-2021" / "ESA_WorldCover_10m_2021_v200_N30E120_Map.tif"
    if canonical_path.is_symlink() or not canonical_path.is_file():
        raise ValueError("canonical WorldCover class raster is unavailable")
    digest = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    if digest != canonical["sha256"]:
        raise ValueError("canonical WorldCover raster changed")

    import rasterio
    from collections import Counter
    with rasterio.open(canonical_path) as dataset:
        if dataset.crs is None or dataset.crs.to_string() != "EPSG:4326":
            raise ValueError("canonical WorldCover CRS changed")
        window = dataset.window(SECOND_AOI["west"], SECOND_AOI["south"], SECOND_AOI["east"], SECOND_AOI["north"]).round_offsets().round_lengths()
        values = dataset.read(1, window=window, masked=True)
    observed = {str(key): int(value) for key, value in sorted(Counter(values.compressed().tolist()).items(), key=lambda item: int(item[0]))}
    if observed != EXPECTED_DISTRIBUTION or int(values.compressed().size) != EXPECTED_PIXEL_COUNT:
        raise ValueError("second WorldCover AOI distribution changed")

    (output / "tasks" / "worldcover-water-qwen").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template / "assets.json", output / "tasks" / "worldcover-water-qwen" / "assets.json")
    shutil.copyfile(template / "scenario.json", output / "tasks" / "worldcover-water-qwen" / "scenario.json")
    task = json.loads((template / "task.json").read_text())
    task_identity = ["worldcover-water-qwen-v1", SECOND_AOI, EXPECTED_DISTRIBUTION, digest]
    task_id = "worldcover-water-qwen-" + hashlib.sha256(json.dumps(task_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    task.update(task_id=task_id,
                prompt=("Within the fixed AOI 122.65-122.75 E and 30.75-30.85 N, determine the dominant "
                        "ESA WorldCover 2021 class and cite both source and rendered evidence."),
                metadata={**task["metadata"], "evaluation_aoi": SECOND_AOI,
                          "acceptance": "qwen-real-interaction-rendered-profile"})
    evaluator = json.loads((template / "evaluator.json").read_text())
    evaluator["config"].update(evaluation_aoi=SECOND_AOI, expected_distribution=EXPECTED_DISTRIBUTION,
                               expected_pixel_count=EXPECTED_PIXEL_COUNT)
    write_json(output / "tasks" / "worldcover-water-qwen" / "task.json", task)
    write_json(output / "tasks" / "worldcover-water-qwen" / "evaluator.json", evaluator)
    for directory in ("state", "reports", "artifacts"):
        (output / directory).mkdir(exist_ok=True)
    job = {"task_ref": {"task_id": task_id, "task_version": "1.1.0"}, "seed": 42,
           "evaluation_aoi": SECOND_AOI, "sample_id": "water-A",
           "asset_id": "asset-worldcover-n30e120"}
    write_json(output / "job.json", job)
    return {"task_root": str((output / "tasks" / "worldcover-water-qwen").resolve()),
            "asset_id": "asset-worldcover-n30e120", "content_sha256": canonical["sha256"],
            "evaluation_aoi": SECOND_AOI, "task_id": task_id}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", args.output_name):
        raise SystemExit("output-name must be a simple runtime directory name")
    output = ROOT / "runtime" / args.output_name
    if output.exists():
        raise SystemExit("preserve existing output; choose a fresh name")
    with StorageQuota(ROOT / "runtime").hold(output, 8 * 1024 * 1024, "worldcover-second-aoi"):
        result = prepare(ROOT / "tasks", output)
    print(json.dumps({"output": str(output), "sample": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

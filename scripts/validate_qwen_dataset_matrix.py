#!/usr/bin/env python3
"""Validate the fixed Qwen real-interaction dataset matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def validate(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024:
        raise ValueError("matrix must be a bounded regular file")
    value = json.loads(path.read_text())
    if value.get("schema_version") != "qwen-dataset-matrix-v1":
        raise ValueError("unsupported matrix schema")
    datasets = value.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("matrix has no datasets")
    seen = set()
    ready = 0
    pending = 0
    for dataset in datasets:
        dataset_id = dataset.get("dataset_id")
        if not isinstance(dataset_id, str) or not ID_PATTERN.fullmatch(dataset_id):
            raise ValueError("invalid dataset_id")
        if dataset_id in seen:
            raise ValueError("duplicate dataset_id: " + dataset_id)
        seen.add(dataset_id)
        samples = dataset.get("samples", [])
        status = dataset.get("status", "ready")
        if status == "ready":
            ready += 1
            if len(samples) < 2:
                raise ValueError(dataset_id + " needs two distinct samples")
            identities = []
            for sample in samples:
                if not isinstance(sample, dict) or sample.get("sample_id") is None:
                    raise ValueError(dataset_id + " has an invalid sample")
                asset_id = sample.get("asset_id")
                asset_ids = sample.get("asset_ids")
                if not sample.get("task_root") or not sample.get("content_sha256"):
                    raise ValueError(dataset_id + " sample lacks task_root/content_sha256")
                if not asset_id and not asset_ids:
                    raise ValueError(dataset_id + " sample lacks asset identity")
                identity = (asset_id or tuple(asset_ids), sample.get("sample_kind", "standard"))
                identities.append(identity)
            if len(set(identities)) != len(identities):
                raise ValueError(dataset_id + " samples are not distinct")
            content_hashes = [sample["content_sha256"] for sample in samples]
            sample_kinds = [sample.get("sample_kind", "standard") for sample in samples]
            if len(set(zip(content_hashes, sample_kinds))) != len(content_hashes):
                raise ValueError(dataset_id + " samples are not distinct")
        elif status in {"pending_second_sample", "pending_agent_task", "pending_qwen_task_split"}:
            pending += 1
            if samples:
                raise ValueError(dataset_id + " pending entry cannot declare samples")
            if not dataset.get("reason"):
                raise ValueError(dataset_id + " pending entry lacks reason")
        else:
            raise ValueError(dataset_id + " has unsupported status")
    return {
        "schema_version": value["schema_version"],
        "datasets": len(datasets),
        "ready_datasets": ready,
        "pending_datasets": pending,
        "required_samples_per_ready_dataset": 2,
    }


def validate_runtime(path: Path, root: Path) -> dict:
    matrix = json.loads(path.read_text())
    checked = 0
    for dataset in matrix["datasets"]:
        for sample in dataset.get("samples", []):
            relative_root = sample["task_root"]
            parts = Path(relative_root).parts
            if parts and parts[0] == "runtime":
                task_root = (root.parent / Path(*parts)).resolve()
            elif parts and parts[0] == "tasks":
                task_root = (root.parent / relative_root).resolve()
            else:
                task_root = (root / relative_root).resolve()
            if not (task_root / "task.json").is_file():
                candidates = sorted(path for path in task_root.rglob("task.json")
                                    if "worldcover-grounded-vqa" not in path.parent.name)
                if len(candidates) != 1:
                    raise ValueError(dataset["dataset_id"] + " sample task is not unique")
                task_root = candidates[0].parent
            task_path = task_root / "task.json"
            assets_path = task_root / "assets.json"
            if task_root.is_symlink() or not task_path.is_file() or not assets_path.is_file():
                raise ValueError(dataset["dataset_id"] + " task files unavailable")
            task = json.loads(task_path.read_text())
            assets = json.loads(assets_path.read_text())
            expected_assets = [sample.get("asset_id")] if sample.get("asset_id") else sample.get("asset_ids")
            if set(expected_assets) != set(task["inputs"]):
                raise ValueError(dataset["dataset_id"] + " task inputs differ from matrix")
            by_id = {item["asset_id"]: item for item in assets}
            if not set(expected_assets).issubset(by_id):
                raise ValueError(dataset["dataset_id"] + " asset manifest differs from matrix")
            if sample.get("asset_id"):
                if by_id[sample["asset_id"]].get("sha256") != sample["content_sha256"]:
                    raise ValueError(dataset["dataset_id"] + " content hash differs from manifest")
            else:
                # Multi-image samples use the first public input as the stable distinct-sample identity.
                first = sample["asset_ids"][0]
                if by_id[first].get("sha256") != sample["content_sha256"]:
                    raise ValueError(dataset["dataset_id"] + " content hash differs from manifest")
            checked += 1
    return {"runtime_samples_checked": checked}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "config" / "qwen-dataset-matrix-v1.json")
    parser.add_argument("--runtime-root", type=Path, default=ROOT / "runtime")
    args = parser.parse_args()
    result = validate(args.matrix)
    if args.runtime_root.is_dir():
        result.update(validate_runtime(args.matrix, args.runtime_root))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

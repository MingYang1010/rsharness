#!/usr/bin/env python3
"""Stage three hash-pinned WHU temporal tasks without exposing evaluator labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))

from app.core.capabilities import TaskRegistry
from app.core.storage.quota import StorageQuota


MAX_FILE_BYTES = 2 * 1024 * 1024
CONFIG = Path(
    os.environ.get("EO_WHU_TEST_CONFIG", ROOT / "config" / "whu-change-samples.json")
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _write_json(path: Path, value: object) -> None:
    content = _json_bytes(value)
    if len(content) > 1024 * 1024:
        raise ValueError("generated metadata exceeds 1 MiB")
    with path.open("xb") as stream:
        stream.write(content)


def _checked_file(source: Path, root: Path, receipt: dict) -> bytes:
    if source.is_symlink() or not source.resolve().is_relative_to(root):
        raise ValueError("source file escapes the reviewed dataset root")
    if not source.is_file():
        raise ValueError("reviewed source file is missing")
    size = source.stat().st_size
    if size != receipt["size_bytes"] or size > MAX_FILE_BYTES:
        raise ValueError("reviewed source file size changed or exceeds bound")
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != receipt["sha256"]:
        raise ValueError("reviewed source file checksum changed")
    return content


def _truth(before, after, evaluator: dict) -> dict:
    import numpy

    before_mask = before == 255
    after_mask = after == 255
    new_pixels = int(numpy.count_nonzero(~before_mask & after_mask))
    demolished_pixels = int(
        numpy.count_nonzero(before_mask & ~after_mask)
    )
    changed_pixels = new_pixels + demolished_pixels
    changed_fraction = changed_pixels / int(before.size)
    if changed_pixels == 0:
        change_class, direction = "no_change", "no_change"
    else:
        change_class = (
            "minor_change"
            if changed_fraction <= evaluator["minor_change_max_fraction"]
            else "major_change"
        )
        dominance = evaluator["direction_dominance_ratio"]
        if new_pixels > demolished_pixels * dominance:
            direction = "expansion"
        elif demolished_pixels > new_pixels * dominance:
            direction = "reduction"
        else:
            direction = "mixed"
    return {
        "change_class": change_class,
        "change_direction": direction,
        "changed_pixels": changed_pixels,
        "new_pixels": new_pixels,
        "demolished_pixels": demolished_pixels,
        "changed_fraction": changed_fraction,
    }


def _inspect(path: Path, kind: str, dataset: dict):
    import numpy
    import rasterio

    with rasterio.open(path) as image:
        expected_count = (
            dataset["image_channels"] if kind == "image" else 1
        )
        if (
            image.width != dataset["width"]
            or image.height != dataset["height"]
            or image.count != expected_count
            or image.crs is not None
            or any(dtype != "uint8" for dtype in image.dtypes)
        ):
            raise ValueError("WHU split tile metadata changed")
        if kind == "label":
            values = image.read(1)
            unique = {
                int(value) for value in numpy.unique(values).tolist()
            }
            if unique - set(dataset["label_values"]):
                raise ValueError("WHU label value domain changed")
            return values
    return None


def _pixel_asset(
    *,
    asset_id: str,
    uri: str,
    receipt: dict,
    roles: list[str],
    timestamp: str,
    channels: int,
    dataset: dict,
    quality: dict | None = None,
) -> dict:
    return {
        "asset_id": asset_id,
        "uri": uri,
        "media_type": "image/tiff",
        "roles": roles,
        "sha256": receipt["sha256"],
        "size_bytes": receipt["size_bytes"],
        "spatial": None,
        "pixel": {
            "coordinate_system": "pixel",
            "width": dataset["width"],
            "height": dataset["height"],
            "channels": channels,
        },
        "temporal": {"start": timestamp, "end": timestamp},
        "platform": "aerial",
        "instrument": "optical",
        "bands": (
            ["red", "green", "blue"]
            if channels == 3
            else ["building_mask"]
        ),
        "quality": quality or {},
        "license": (
            "official research download; redistribution rights not asserted"
        ),
        "source": "WHU Building Change Detection Dataset test split",
        "source_snapshot_hash": dataset["source_archive_sha256"],
    }


def _task_files(sample: dict, config: dict, out: Path) -> tuple[dict, dict]:
    dataset = config["dataset"]
    evaluator_config = config["evaluator"]
    sample_id = sample["sample_id"]
    task_id = "whu-building-change-" + sample_id
    before_input = "asset-whu-" + sample_id + "-before"
    after_input = "asset-whu-" + sample_id + "-after"
    before_label = "asset-whu-" + sample_id + "-before-label"
    after_label = "asset-whu-" + sample_id + "-after-label"
    task_dir = out / "tasks" / ("whu-building-change-" + sample_id)
    task_dir.mkdir()

    assets = [
        _pixel_asset(
            asset_id=before_input,
            uri="local://approved-input/" + before_input + ".tif",
            receipt=sample["files"]["before_image"],
            roles=["input_image", "temporal_before"],
            timestamp="2012-01-01T00:00:00Z",
            channels=3,
            dataset=dataset,
            quality=sample.get("public_quality", {}).get("before"),
        ),
        _pixel_asset(
            asset_id=after_input,
            uri="local://approved-input/" + after_input + ".tif",
            receipt=sample["files"]["after_image"],
            roles=["input_image", "temporal_after"],
            timestamp="2016-01-01T00:00:00Z",
            channels=3,
            dataset=dataset,
            quality=sample.get("public_quality", {}).get("after"),
        ),
        _pixel_asset(
            asset_id=before_label,
            uri=(
                "local://dataset/whu-change/"
                + sample_id
                + "/before-label.tif"
            ),
            receipt=sample["files"]["before_label"],
            roles=["evaluator", "building_label", "temporal_before"],
            timestamp="2012-01-01T00:00:00Z",
            channels=1,
            dataset=dataset,
        ),
        _pixel_asset(
            asset_id=after_label,
            uri=(
                "local://dataset/whu-change/"
                + sample_id
                + "/after-label.tif"
            ),
            receipt=sample["files"]["after_label"],
            roles=["evaluator", "building_label", "temporal_after"],
            timestamp="2016-01-01T00:00:00Z",
            channels=1,
            dataset=dataset,
        ),
    ]
    weights = {
        "task.change_class_accuracy": 0.25,
        "task.direction_accuracy": 0.15,
        "task.changed_fraction_score": 0.15,
        "answer.abstention_correctness": 0.1,
        "evidence.faithfulness": 0.25,
        "process.efficiency": 0.1,
    }
    task = {
        "task_id": task_id,
        "task_version": "1.0.0",
        "family": "temporal_change",
        "prompt": (
            "Compare the 2012 before image with the 2016 after image. "
            "Report building change class, direction and changed fraction. "
            "Cite frozen full-image crop evidence from both dates."
        ),
        "inputs": [before_input, after_input],
        "scenario_profile": "whu-building-change-v1",
        "answer_schema": {
            "type": "object",
            "properties": {
                "change_class": {
                    "type": "string",
                    "enum": [
                        "no_change",
                        "minor_change",
                        "major_change",
                    ],
                },
                "change_direction": {
                    "type": "string",
                    "enum": [
                        "no_change",
                        "expansion",
                        "reduction",
                        "mixed",
                    ],
                },
                "changed_fraction": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "claims": {"type": "array"},
            },
            "required": [
                "change_class",
                "change_direction",
                "changed_fraction",
                "claims",
            ],
        },
        "evaluator": evaluator_config["evaluator_id"],
        "budget": {
            "max_steps": 12,
            "max_tool_calls": 4,
            "max_wall_time_ms": 300000,
            "max_input_bytes": 16 * 1024 * 1024,
            "max_artifact_bytes": 128 * 1024 * 1024,
        },
        "seed": 42,
        "metric_aggregation": weights,
        "metadata": {
            "observation_profile": "headless-tools-v1",
            "evaluation_profile": evaluator_config["evaluator_id"],
            "artifact_identity": "derivation-sha256-v1",
            "acceptance": "real-whu-human-label-change-evaluation",
        },
    }
    task_variant = sample.get("task_variant")
    if task_variant:
        task["task_id"] = task_id + "-" + task_variant
        task["prompt"] = (
            "Before answering, inspect the public input metadata. " + task["prompt"]
            + " If a required input has coverage below the task policy, abstain rather than infer."
        )
    if sample.get("expected_outcome") is not None:
        task["metadata"]["expected_outcome"] = sample["expected_outcome"]

    scenario = {
        "profile_id": "whu-building-change-v1",
        "domain": "building_change_detection",
        "data_cutoff": "2026-09-20T00:00:00Z",
        "freshness_max_age_seconds": None,
        "allowed_actions": [
            "tool.invoke",
            "memory.save_evidence",
            "answer.*",
        ],
        "allowed_tools": ["catalog.search", "catalog.inspect_asset", "eo_gym.crop"],
        "network_policy": "none",
        "evidence_required": True,
        "abstention_allowed": True,
        "human_review_policy": "allowed",
    }
    truth = sample["truth"]
    evaluator = {
        "evaluator_id": evaluator_config["evaluator_id"],
        "evaluator_version": evaluator_config["evaluator_version"],
        "metric_names": list(weights),
        "aggregate_weights": weights,
        "config": {
            "before_input_asset_id": before_input,
            "after_input_asset_id": after_input,
            "before_label_asset_id": before_label,
            "after_label_asset_id": after_label,
            "expected_width": dataset["width"],
            "expected_height": dataset["height"],
            "expected_pixel_count": dataset["width"] * dataset["height"],
            "expected_outcome": sample.get("expected_outcome", "submitted"),
            "expected_changed_pixels": truth["changed_pixels"],
            "expected_new_pixels": truth["new_pixels"],
            "expected_demolished_pixels": truth["demolished_pixels"],
            "minor_change_max_fraction": (
                evaluator_config["minor_change_max_fraction"]
            ),
            "direction_dominance_ratio": (
                evaluator_config["direction_dominance_ratio"]
            ),
            "changed_fraction_tolerance": (
                evaluator_config["changed_fraction_tolerance"]
            ),
            "minimum_input_coverage_fraction": sample.get(
                "minimum_input_coverage_fraction",
                evaluator_config["minimum_input_coverage_fraction"],
            ),
            "required_evidence_tool_id": (
                evaluator_config["required_evidence_tool_id"]
            ),
            "efficiency": {
                "ideal_steps": 5,
                "ideal_renderer_calls": 2,
                "wall_time_soft_limit_ms": 30000,
            },
        },
    }
    for name, value in (
        ("task.json", task),
        ("scenario.json", scenario),
        ("assets.json", assets),
        ("evaluator.json", evaluator),
    ):
        _write_json(task_dir / name, value)
    job = {
        "task_ref": {"task_id": task["task_id"], "task_version": "1.0.0"},
        "seed": 42,
        "sample_id": sample_id,
        "before_asset_id": before_input,
        "after_asset_id": after_input,
        "width": dataset["width"],
        "height": dataset["height"],
        "truth": truth,
        "expected_outcome": sample.get("expected_outcome", "submitted"),
        "minimum_input_coverage_fraction": sample.get(
            "minimum_input_coverage_fraction",
            evaluator_config["minimum_input_coverage_fraction"],
        ),
    }
    separation = {
        "public_input_asset_ids": [before_input, after_input],
        "hidden_label_asset_ids": [before_label, after_label],
    }
    return job, separation


def prepare(source: Path, out: Path, config: dict) -> None:
    dataset = config["dataset"]
    evaluator = config["evaluator"]
    if dataset["redistribution_allowed"] is not False:
        raise ValueError("WHU redistribution policy must remain fail-closed")

    for sample in config["samples"]:
        expected = sample.get("expected_outcome")
        if expected is not None and expected not in {"submitted", "abstained"}:
            raise ValueError("sample expected outcome is invalid")
        minimum = sample.get(
            "minimum_input_coverage_fraction",
            config["evaluator"]["minimum_input_coverage_fraction"],
        )
        quality = sample.get("public_quality", {})
        insufficient = any(
            item.get("coverage_fraction", 1.0) < minimum
            for item in quality.values()
        )
        if expected == "abstained" and not insufficient:
            raise ValueError(
                "expected abstention lacks insufficient public input coverage"
            )
        if expected == "submitted" and insufficient:
            raise ValueError(
                "expected submission contradicts insufficient public input coverage"
            )
    for name in (
        "inputs",
        "provider-out",
        "state",
        "reports",
        "tasks",
        "jobs",
        "evaluator-datasets",
    ):
        (out / name).mkdir(parents=True)
    provider_manifest = {}
    jobs = []
    separation = []
    source_root = source.resolve()
    for sample in config["samples"]:
        labels = {}
        for key, receipt in sample["files"].items():
            path = (source_root / receipt["relative_path"]).resolve()
            content = _checked_file(path, source_root, receipt)
            kind = "label" if key.endswith("_label") else "image"
            values = _inspect(path, kind, dataset)
            if values is not None:
                labels[key] = values
            destination_name = None
            if kind == "image":
                period = "before" if key.startswith("before") else "after"
                asset_id = (
                    "asset-whu-"
                    + sample["sample_id"]
                    + "-"
                    + period
                )
                destination_name = asset_id + ".tif"
                destination = out / "inputs" / destination_name
                provider_manifest[asset_id] = {
                    "role": "input_image",
                    "filename": destination_name,
                    "sha256": receipt["sha256"],
                }
            else:
                period = "before" if key.startswith("before") else "after"
                label_root = (
                    out
                    / "evaluator-datasets"
                    / "whu-change"
                    / sample["sample_id"]
                )
                label_root.mkdir(parents=True, exist_ok=True)
                destination = label_root / (period + "-label.tif")
            with destination.open("xb") as stream:
                stream.write(content)
        computed = _truth(
            labels["before_label"],
            labels["after_label"],
            evaluator,
        )
        for key, expected in sample["truth"].items():
            actual = computed[key]
            if (
                isinstance(expected, float)
                and not math.isclose(actual, expected, abs_tol=1e-15)
            ) or (
                not isinstance(expected, float) and actual != expected
            ):
                raise ValueError(
                    "configured truth disagrees with hidden labels"
                )
        job, boundary = _task_files(sample, config, out)
        _write_json(out / "jobs" / (sample["sample_id"] + ".json"), job)
        jobs.append(job)
        separation.append(boundary)
    _write_json(out / "inputs.json", provider_manifest)
    registry = TaskRegistry(str(out / "tasks"))
    manifests = {
        job["task_ref"]["task_id"]: registry.get(
            job["task_ref"]["task_id"], "1.0.0"
        ).task_manifest_hash
        for job in jobs
    }
    _write_json(out / "job.json", {"jobs": jobs})
    _write_json(
        out / "admission.json",
        {
            "status": "admitted",
            "config_sha256": hashlib.sha256(
                CONFIG.read_bytes()
            ).hexdigest(),
            "official_page_sha256": dataset["official_page_sha256"],
            "source_archive_sha256": dataset["source_archive_sha256"],
            "task_manifest_hashes": manifests,
            "asset_separation": separation,
            "provider_asset_count": len(provider_manifest),
            "hidden_label_count": len(jobs) * 2,
            "labels_provider_mounted": False,
            "redistribution_allowed": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", args.output_name):
        raise SystemExit("output-name must be a simple fresh directory")
    if CONFIG.is_symlink() or CONFIG.stat().st_size > 1024 * 1024:
        raise SystemExit("WHU sample config is missing or exceeds its bound")
    config = json.loads(CONFIG.read_text())
    source = args.source.resolve()
    if args.source.is_symlink() or not source.is_dir():
        raise SystemExit("source must be the reviewed WHU data directory")
    runtime_root = Path(os.environ.get("EO_WHU_TEST_ROOT", ROOT / "runtime"))
    out = runtime_root / args.output_name
    if out.exists():
        raise SystemExit("output exists; preserve it and choose a fresh name")
    with StorageQuota(runtime_root).hold(
        out, 64 * 1024 * 1024, "whu-change-smoke"
    ):
        prepare(source, out, config)
    print(
        json.dumps(
            {
                "output": str(out),
                "tasks": len(config["samples"]),
                "public_inputs": len(config["samples"]) * 2,
                "hidden_labels": len(config["samples"]) * 2,
            }
        )
    )


if __name__ == "__main__":
    main()

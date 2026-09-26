#!/usr/bin/env python3
"""Freeze reviewed XLRS grounding rows into public-image/hidden-truth tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))
from app.core.storage.quota import StorageQuota


MAX_IMAGE_BYTES = 128 * 1024 * 1024
MAX_PIXELS = 20_000_000
METRICS = {
    "task.accuracy": 0.55,
    "evidence.faithfulness": 0.35,
    "process.efficiency": 0.1,
}


def image_suffix(image: dict) -> str:
    path = image.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("grounding image path is unavailable")
    suffix = Path(path).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("grounding image container is unsupported")
    return suffix


def media_type(suffix: str) -> str:
    return "image/png" if suffix == ".png" else "image/jpeg"


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    )


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def checked_rows(root: Path) -> Iterator[dict]:
    files = sorted(root.glob("data-*.arrow"))
    if not files:
        raise ValueError("grounding Arrow shards are unavailable")
    import pyarrow as pa

    for file_index, path in enumerate(files):
        if path.is_symlink() or not path.is_file():
            raise ValueError("grounding Arrow shard is unavailable")
        with pa.memory_map(str(path), "rb") as source:
            mode = "stream"
            try:
                reader = pa.ipc.open_stream(source)
            except pa.ArrowInvalid:
                source.seek(0)
                mode = "file"
                reader = pa.ipc.open_file(source)
            required = {
                "image_width", "image_height", "question_id", "image",
                "question", "answer", "bbox", "category",
            }
            if not required.issubset(reader.schema.names):
                raise ValueError("grounding Arrow schema is missing reviewed fields")
            batches = (
                iter(reader)
                if mode == "stream"
                else (reader.get_batch(index) for index in range(reader.num_record_batches))
            )
            for batch_index, batch in enumerate(batches):
                for row_index, row in enumerate(batch.to_pylist()):
                    yield {
                        "file_index": file_index,
                        "batch_index": batch_index,
                        "row_index": row_index,
                        **row,
                    }


def normalized_bbox(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    x0, y0, x1, y1 = (float(item) for item in value)
    bbox = [x0, y0, x1, y1]
    if not finite_bbox(bbox) or not (
        0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
    ):
        return None
    return bbox


def finite_bbox(bbox: list[float]) -> bool:
    return all(abs(value) != float("inf") for value in bbox)


def decoded_image(row: dict) -> tuple[bytes, int, int, int]:
    image = row.get("image")
    if not isinstance(image, dict):
        raise ValueError("grounding image value is invalid")
    content = image.get("bytes")
    if not isinstance(content, (bytes, bytearray)) or not content:
        raise ValueError("grounding image bytes are unavailable")
    content = bytes(content)
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("grounding image exceeds provider byte limit")
    suffix = image_suffix(image)
    if suffix == ".png":
        if len(content) < 26 or content[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("grounding PNG header is invalid")
        if int.from_bytes(content[8:12], "big") != 13 or content[12:16] != b"IHDR":
            raise ValueError("grounding PNG header chunk is invalid")
        width = int.from_bytes(content[16:20], "big")
        height = int.from_bytes(content[20:24], "big")
        bit_depth, color_type = content[24], content[25]
        if bit_depth != 8 or color_type not in {2, 6}:
            raise ValueError("grounding PNG encoding is unsupported")
        channels = 3 if color_type == 2 else 4
    else:
        if len(content) < 10 or content[:2] != b"\xff\xd8":
            raise ValueError("grounding JPEG header is invalid")
        width, height, channels = _jpeg_dimensions(content)
    if width * height > MAX_PIXELS or not 1 <= channels <= 4:
        raise ValueError("grounding image exceeds provider dimensions")
    return content, width, height, channels


def _jpeg_dimensions(content: bytes) -> tuple[int, int, int]:
    offset = 2
    while offset + 9 < len(content):
        if content[offset] != 0xFF:
            offset += 1
            continue
        marker = content[offset + 1]
        if marker in {0xD8, 0x01} or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        length = int.from_bytes(content[offset + 2:offset + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in {0xC4, 0xC8, 0xCC}:
            if length < 7:
                raise ValueError("grounding JPEG frame is invalid")
            height = int.from_bytes(content[offset + 5:offset + 7], "big")
            width = int.from_bytes(content[offset + 7:offset + 9], "big")
            channels = content[offset + 9]
            return width, height, channels
        if length <= 0:
            raise ValueError("grounding JPEG segment is invalid")
        offset += 2 + length
    raise ValueError("grounding JPEG frame is unavailable")


def selected_rows(root: Path, count: int) -> list[dict]:
    selected = []
    seen_question = set()
    seen_content = set()
    for row in checked_rows(root):
        question_id = row.get("question_id")
        bbox = normalized_bbox(row.get("bbox"))
        if (
            not isinstance(question_id, str)
            or not question_id
            or question_id in seen_question
            or not isinstance(row.get("question"), str)
            or not row["question"]
            or not isinstance(row.get("answer"), str)
            or not row["answer"]
            or not isinstance(row.get("category"), str)
            or not row["category"]
            or bbox is None
        ):
            continue
        try:
            content, width, height, channels = decoded_image(row)
        except (ValueError, OSError, SyntaxError):
            continue
        digest = sha256_bytes(content)
        if digest in seen_content:
            continue
        declared_width, declared_height = row.get("image_width"), row.get("image_height")
        if (
            isinstance(declared_width, bool)
            or not isinstance(declared_width, (int, float))
            or isinstance(declared_height, bool)
            or not isinstance(declared_height, (int, float))
            or float(declared_width) != width
            or float(declared_height) != height
        ):
            continue
        seen_question.add(question_id)
        seen_content.add(digest)
        try:
            suffix = image_suffix(row["image"])
        except ValueError:
            continue
        selected.append({
            **row,
            "bbox": bbox,
            "content": content,
            "width": width,
            "height": height,
            "channels": channels,
            "sha256": digest,
            "filename": digest + suffix,
            "media_type": media_type(suffix),
        })
        if len(selected) == count:
            return selected
    raise ValueError("insufficient valid distinct XLRS grounding rows")


def task_files(row: dict) -> tuple[str, dict, dict, list[dict], dict, dict]:
    source_question = row["question"]
    question = (
        source_question.split("Description: ", 1)[1]
        if "Description: " in source_question
        else source_question
    )
    identity = [
        row["question_id"],
        row["sha256"],
        row["bbox"],
        row["width"],
        row["height"],
    ]
    suffix = hashlib.sha256(canonical(identity)).hexdigest()[:16]
    task_id = "xlrs-grounding-" + suffix
    asset_id = "asset-" + row["sha256"]
    filename = row["filename"]
    asset = {
        "asset_id": asset_id,
        "uri": "local://approved-input/" + filename,
        "media_type": row["media_type"],
        "roles": ["input_image"],
        "sha256": row["sha256"],
        "size_bytes": len(row["content"]),
        "spatial": None,
        "pixel": {
            "coordinate_system": "pixel",
            "width": row["width"],
            "height": row["height"],
            "channels": row["channels"],
        },
        "license": "existing-local-research-copy-redistribution-not-authorized",
        "source": "XLRS-Bench_visual_grounding_en/test",
    }
    task = {
        "task_id": task_id,
        "task_version": "1.0.0",
        "family": "visual_grounding",
        "prompt": (
            "Locate the target described in the public question. First crop an AOI "
            "that contains the target, save that frozen crop as evidence, then submit "
            "the target bbox in normalized image coordinates [xmin,ymin,xmax,ymax]. "
            + question
        ),
        "inputs": [asset_id],
        "scenario_profile": "xlrs-visual-grounding-v1",
        "answer_schema": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                    "minItems": 4,
                    "maxItems": 4,
                }
            },
            "required": ["bbox"],
        },
        "evaluator": "xlrs-visual-grounding-v1",
        "budget": {
            "max_steps": 12,
            "max_tool_calls": 4,
            "max_wall_time_ms": 300000,
            "max_input_bytes": 512 * 1024 * 1024,
            "max_artifact_bytes": 128 * 1024 * 1024,
        },
        "seed": 42,
        "metric_aggregation": METRICS,
        "metadata": {
            "observation_profile": "headless-tools-v1",
            "artifact_identity": "derivation-sha256-v1",
            "evaluation_profile": "xlrs-visual-grounding-v1",
            "acceptance": "xlrs-real-grounding-iou-evaluation",
            "public_question_sha256": sha256_bytes(question.encode()),
        },
    }
    scenario = {
        "profile_id": "xlrs-visual-grounding-v1",
        "domain": "visual_grounding",
        "data_cutoff": "2026-09-26T00:00:00Z",
        "freshness_max_age_seconds": None,
        "allowed_actions": ["tool.invoke", "memory.save_evidence", "answer.*"],
        "allowed_tools": ["eo_gym.crop"],
        "network_policy": "none",
        "evidence_required": True,
        "abstention_allowed": True,
        "human_review_policy": "allowed",
    }
    evaluator = {
        "evaluator_id": "xlrs-visual-grounding-v1",
        "evaluator_version": "1.0.0",
        "metric_names": list(METRICS),
        "aggregate_weights": METRICS,
        "config": {
            "input_asset_id": asset_id,
            "expected_bbox": row["bbox"],
            "minimum_iou": 0.5,
            "expected_outcome": "submitted",
            "public_question_sha256": sha256_bytes(question.encode()),
            "efficiency": {"ideal_steps": 3, "wall_time_soft_limit_ms": 120000},
        },
    }
    job = {
        "task_ref": {"task_id": task_id, "task_version": "1.0.0"},
        "seed": 42,
        "asset_id": asset_id,
        "question_id": row["question_id"],
        "source": {
            "file_index": row["file_index"],
            "batch_index": row["batch_index"],
            "row_index": row["row_index"],
        },
        "content_sha256": row["sha256"],
        "width": row["width"],
        "height": row["height"],
    }
    return task_id, task, scenario, [asset], evaluator, job


def prepare(root: Path, output: Path, count: int) -> dict:
    rows = selected_rows(root, count)
    for name in ("inputs", "tasks", "provider-out", "state", "reports", "agent"):
        (output / name).mkdir(parents=True)
    public_inputs = {}
    samples = []
    for row in rows:
        task_id, task, scenario, assets, evaluator, job = task_files(row)
        filename = assets[0]["uri"].removeprefix("local://approved-input/")
        destination = output / "inputs" / filename
        destination.write_bytes(row["content"])
        if sha256_bytes(destination.read_bytes()) != row["sha256"]:
            raise ValueError("copied grounding image changed")
        public_inputs[assets[0]["asset_id"]] = {
            "role": "input_image",
            "filename": filename,
            "sha256": row["sha256"],
        }
        task_dir = output / "tasks" / task_id
        task_dir.mkdir()
        for name, value in (
            ("task.json", task),
            ("scenario.json", scenario),
            ("assets.json", assets),
            ("evaluator.json", evaluator),
            ("job.json", job),
        ):
            write_json(task_dir / name, value)
        manifest_body = {
            "task": task,
            "scenario": scenario,
            "assets": assets,
            "evaluator": evaluator,
        }
        samples.append({
            "task_id": task_id,
            "task_manifest_hash": hashlib.sha256(canonical(manifest_body)).hexdigest(),
            "asset_id": assets[0]["asset_id"],
            "content_sha256": row["sha256"],
            "question_id": row["question_id"],
            "job_path": str((task_dir / "job.json").resolve()),
        })
    write_json(output / "inputs.json", public_inputs)
    receipt = {
        "schema_version": "xlrs-visual-grounding-v1",
        "source_root": str(root.resolve()),
        "source_shard_sha256s": {
            str(row["file_index"]): hashlib.sha256(
                sorted(root.glob("data-*.arrow"))[row["file_index"]].read_bytes()
            ).hexdigest()
            for row in rows
        },
        "sample_count": len(samples),
        "minimum_iou": 0.5,
        "bbox_convention": "normalized [xmin,ymin,xmax,ymax]",
        "separation": {
            "agent_visible": ["inputs", "inputs.json", "public task prompt/schema"],
            "evaluator_only": ["evaluator.json hidden expected_bbox"],
        },
        "samples": samples,
    }
    write_json(output / "grounding-receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("/sata/yangm/datasets/XLRS-Bench_visual_grounding_en/test"),
    )
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", args.output_name):
        raise SystemExit("output-name must be a simple runtime directory name")
    if not 1 <= args.count <= 100:
        raise SystemExit("count must be between 1 and 100")
    source = args.source.resolve()
    output = ROOT / "runtime" / args.output_name
    if output.exists():
        raise SystemExit("preserve existing runtime; choose a fresh name")
    selected = selected_rows(source, args.count)
    reserved = sum(len(row["content"]) for row in selected) + 32 * 1024 * 1024
    with StorageQuota(ROOT / "runtime").hold(
        output, reserved, "xlrs-visual-grounding"
    ):
        receipt = prepare(source, output, args.count)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

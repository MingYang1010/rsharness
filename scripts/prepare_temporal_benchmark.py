#!/usr/bin/env python3
"""Freeze four Sentinel-2 temporal tasks from reviewed local red/SCL windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))

from app.v2.capabilities import TaskRegistry
from app.v2.raster_grid import NativeSCL
from app.v2.raster_math import CLOUD_EXCLUDED_CLASSES, CLOUD_POLICY, NativeBand
from app.v2.schemas import TaskManifest
from app.v2.storage.quota import StorageQuota
from app.v2.temporal import TemporalInputProfile, TemporalSelectAlignArguments


CONFIG = ROOT / "config" / "temporal-benchmark-v1.json"
MAX_JSON_BYTES = 1024 * 1024
MAX_INPUT_BYTES = 16 * 1024 * 1024
RESERVATION_BYTES = 32 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, expected_sha256: str) -> object:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("reviewed JSON source is unavailable or unbounded")
    if _sha256(path) != expected_sha256:
        raise ValueError("reviewed JSON source checksum changed")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("reviewed JSON source is invalid") from None


def _write_json(path: Path, value: object) -> None:
    content = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    if len(content) > MAX_JSON_BYTES:
        raise ValueError("generated metadata exceeds 1 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def _copy_checked(source: Path, destination: Path, expected_sha256: str) -> int:
    if source.is_symlink() or not source.is_file():
        raise ValueError("reviewed native input is unavailable")
    size = source.stat().st_size
    if size <= 0 or size > MAX_INPUT_BYTES or _sha256(source) != expected_sha256:
        raise ValueError("reviewed native input changed or exceeds bound")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(chunk)
    if destination.stat().st_size != size or _sha256(destination) != expected_sha256:
        raise ValueError("staged native input failed checksum verification")
    return size


def _asset_by_id(assets: list[dict], asset_id: str) -> dict:
    matches = [value for value in assets if value.get("asset_id") == asset_id]
    if len(matches) != 1:
        raise ValueError("reviewed temporal asset ID is missing or ambiguous")
    return matches[0]


def _window_cloud_fraction(path: Path, profile: NativeSCL) -> float:
    import numpy
    from rasterio.io import MemoryFile

    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size <= 0
        or path.stat().st_size > MAX_INPUT_BYTES
    ):
        raise ValueError("reviewed SCL window is unavailable or unbounded")
    content = path.read_bytes()
    if (
        hashlib.sha256(content).hexdigest() != profile.sha256
        or content[:4] not in {b"II*\x00", b"MM\x00*"}
    ):
        raise ValueError("reviewed SCL window checksum or TIFF marker changed")
    with MemoryFile(content) as memory, memory.open(driver="GTiff") as image:
        if (
            (image.width, image.height, image.count, image.dtypes)
            != (profile.width, profile.height, 1, (profile.dtype,))
            or image.crs is None
            or image.crs.to_string() != profile.crs
            or list(image.transform)[:6] != profile.transform
            or image.nodata != profile.nodata
            or image.scales != (profile.scale,)
            or image.offsets != (profile.offset,)
        ):
            raise ValueError("reviewed SCL window disagrees with native profile")
        values = image.read(1)
        valid = (image.read_masks(1) > 0) & (values != 0)
    if numpy.any(values > 11):
        raise ValueError("reviewed SCL window contains an invalid class")
    valid_count = int(valid.sum())
    if valid_count <= 0:
        raise ValueError("reviewed SCL window contains no valid pixels")
    return float(
        numpy.count_nonzero(
            valid & numpy.isin(values, CLOUD_EXCLUDED_CLASSES)
        )
        / valid_count
    )


def _profiles(
    config: dict,
    assets: list[dict],
    native_inputs: dict,
    native_root: Path,
) -> tuple[list[dict], list[dict], dict]:
    profiles = []
    selected_assets = []
    selected_native = {}
    seen_assets: set[str] = set()
    for item in config["items"]:
        red_id = item["red_asset_id"]
        scl_id = item["scl_asset_id"]
        if red_id in seen_assets or scl_id in seen_assets or red_id == scl_id:
            raise ValueError("temporal asset IDs must be globally unique")
        seen_assets.update((red_id, scl_id))
        red_asset = dict(_asset_by_id(assets, red_id))
        scl_asset = dict(_asset_by_id(assets, scl_id))
        red_asset["instrument"] = "msi"
        scl_asset["instrument"] = "msi"
        red_native = NativeBand.model_validate(native_inputs[red_id]["native"])
        scl_native = NativeSCL.model_validate(native_inputs[scl_id]["native"])
        if (
            red_native.item_id != item["item_id"]
            or scl_native.item_id != item["item_id"]
            or red_asset["sha256"] != red_native.sha256
            or scl_asset["sha256"] != scl_native.sha256
            or red_asset["temporal"]["start"] != red_native.acquired
            or scl_asset["temporal"]["start"] != scl_native.acquired
            or red_asset.get("platform") != "sentinel-2"
            or scl_asset.get("platform") != "sentinel-2"
        ):
            raise ValueError("reviewed temporal profiles disagree")
        red_bbox = red_asset["spatial"]["bbox"]
        scl_bbox = scl_asset["spatial"]["bbox"]
        if red_bbox != scl_bbox:
            raise ValueError("red and SCL footprints disagree")
        source_cloud_percent = scl_asset.get("quality", {}).get(
            "cloud_cover_percent"
        )
        if not isinstance(source_cloud_percent, (int, float)):
            raise ValueError("reviewed source item cloud fraction is missing")
        cloud_fraction = _window_cloud_fraction(
            native_root / native_inputs[scl_id]["filename"], scl_native
        )
        scl_asset["quality"] = {
            **scl_asset.get("quality", {}),
            "cloud_cover_percent": cloud_fraction * 100.0,
        }
        profile = TemporalInputProfile(
            item_id=item["item_id"],
            acquired=red_native.acquired,
            platform="sentinel-2",
            instrument="msi",
            red=red_native,
            scl=scl_native,
            bbox_wgs84=[
                red_bbox["west"],
                red_bbox["south"],
                red_bbox["east"],
                red_bbox["north"],
            ],
            cloud_fraction=cloud_fraction,
        )
        profiles.append(profile.model_dump(mode="json"))
        selected_assets.extend((red_asset, scl_asset))
        selected_native[red_id] = native_inputs[red_id]
        selected_native[scl_id] = native_inputs[scl_id]
    return profiles, selected_assets, selected_native


def _task_files(
    case: dict,
    assets: list[dict],
    profiles: list[dict],
) -> tuple[dict, dict, dict, dict]:
    arguments = TemporalSelectAlignArguments.model_validate(case["arguments"])
    truth = case["truth"]
    selected = truth["expected_selection_status"] == "selected"
    weights = {
        "temporal.validity": 0.3,
        "spatial.coverage": 0.2,
        "answer.abstention_correctness": 0.2,
        "evidence.faithfulness": 0.2,
        "process.efficiency": 0.1,
    }
    task = {
        "task_id": case["task_id"],
        "task_version": "1.0.0",
        "family": "temporal_selection",
        "prompt": (
            "Use temporal.select_align@1.0.0 with exactly these reviewed "
            "criteria: "
            + json.dumps(arguments.model_dump(mode="json"), sort_keys=True)
            + ". Submit label valid_pair with full-stack evidence only when "
            "the tool selects a pair; otherwise abstain without asserting a pair."
        ),
        "inputs": [asset["asset_id"] for asset in assets],
        "scenario_profile": "sentinel-temporal-selection-v1",
        "answer_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["valid_pair"]},
                "claims": {"type": "array"},
            },
            "required": ["label", "claims"],
        },
        "evaluator": "temporal-selection-v1",
        "budget": {
            "max_steps": 6,
            "max_tool_calls": 2,
            "max_wall_time_ms": 300000,
            "max_input_bytes": 64 * 1024 * 1024,
            "max_artifact_bytes": 128 * 1024 * 1024,
        },
        "seed": 42,
        "metric_aggregation": weights,
        "metadata": {
            "observation_profile": "headless-tools-v1",
            "evaluation_profile": "temporal-selection-v1",
            "artifact_identity": "derivation-sha256-v1",
            "cloud_mask_policy": CLOUD_POLICY,
            "temporal_inputs": profiles,
            "acceptance": "operator-oracle-sentinel-temporal-v1",
        },
    }
    scenario = {
        "profile_id": "sentinel-temporal-selection-v1",
        "domain": "temporal_selection",
        "data_cutoff": "2026-09-20T00:00:00Z",
        "freshness_max_age_seconds": None,
        "allowed_actions": ["tool.invoke", "memory.save_evidence", "answer.*"],
        "allowed_tools": ["temporal.select_align"],
        "network_policy": "none",
        "evidence_required": selected,
        "abstention_allowed": True,
        "human_review_policy": "allowed",
    }
    evaluator_config = {
        **truth,
        "minimum_aligned_coverage_fraction": 0.95,
        "efficiency": {
            "ideal_steps": 3 if selected else 2,
            "ideal_tool_calls": 1,
            "expected_renderer_calls": 1 if selected else 0,
            "wall_time_soft_limit_ms": 30000,
        },
    }
    evaluator = {
        "evaluator_id": "temporal-selection-v1",
        "evaluator_version": "1.0.0",
        "metric_names": list(weights),
        "aggregate_weights": weights,
        "config": evaluator_config,
    }
    return task, scenario, evaluator, arguments.model_dump(mode="json")


def _emit_case(
    root: Path,
    run_id: str,
    case: dict,
    assets: list[dict],
    profiles: list[dict],
    native_inputs: dict,
    source_native_root: Path,
    source_receipt: dict,
    *,
    counterfactual: bool = False,
) -> dict:
    case_id = case["case_id"] + ("-false-confidence" if counterfactual else "")
    out = root / case_id
    for name in (
        "inputs",
        "native-inputs",
        "state",
        "reports",
        "credentials",
        "evaluator-datasets",
        "tasks",
    ):
        (out / name).mkdir(parents=True)
    _write_json(out / "inputs.json", {})
    staged_bytes = 0
    for value in native_inputs.values():
        filename = value["filename"]
        native = value["native"]
        staged_bytes += _copy_checked(
            source_native_root / filename,
            out / "native-inputs" / filename,
            native["sha256"],
        )
    _write_json(out / "native-inputs.json", native_inputs)
    task, scenario, evaluator, arguments = _task_files(case, assets, profiles)
    task_dir = out / "tasks" / case["task_id"]
    for name, value in (
        ("task.json", task),
        ("scenario.json", scenario),
        ("assets.json", assets),
        ("evaluator.json", evaluator),
    ):
        _write_json(task_dir / name, value)
    manifest: TaskManifest = TaskRegistry(out / "tasks").get(
        case["task_id"], "1.0.0"
    )
    expected_outcome = (
        "submitted"
        if counterfactual or case["truth"]["expected_selection_status"] == "selected"
        else "abstained"
    )
    job = {
        "run_id": run_id,
        "case_id": case_id,
        "task_ref": {"task_id": case["task_id"], "task_version": "1.0.0"},
        "task_manifest_hash": manifest.task_manifest_hash,
        "arguments": arguments,
        "truth": case["truth"],
        "expected_oracle_outcome": expected_outcome,
        "counterfactual_false_confidence": counterfactual,
    }
    _write_json(out / "job.json", job)
    _write_json(
        out / "admission.json",
        {
            **source_receipt,
            "case_id": case_id,
            "imagery_committed_to_git": False,
            "native_input_count": len(native_inputs),
            "native_input_bytes": staged_bytes,
            "task_manifest_hash": manifest.task_manifest_hash,
        },
    )
    return {
        "case_id": case_id,
        "task_id": case["task_id"],
        "task_manifest_hash": manifest.task_manifest_hash,
        "counterfactual_false_confidence": counterfactual,
    }


def prepare(
    source: Path,
    license_review_path: Path,
    out: Path,
    config: dict,
) -> dict:
    if out.exists():
        raise ValueError("output exists; preserve it and choose a fresh path")
    source = source.resolve(strict=True)
    source_config = config["source"]
    assets_value = _load_json(
        source / "tasks" / "cloud-masked-ndvi" / "assets.json",
        source_config["assets_sha256"],
    )
    native_value = _load_json(
        source / "native-inputs.json",
        source_config["native_inputs_sha256"],
    )
    license_value = _load_json(
        license_review_path.resolve(strict=True),
        source_config["license_review_sha256"],
    )
    if not isinstance(assets_value, list) or not isinstance(native_value, dict):
        raise ValueError("reviewed source manifests have unexpected shapes")
    if (
        license_value.get("attribution") != source_config["attribution"]
        or license_value.get("scope") != source_config["scope"]
        or license_value.get("not_redistribution_authorization") is not True
        or source_config.get("not_redistribution_authorization") is not True
    ):
        raise ValueError("license review does not preserve local-only scope")
    profiles, assets, native_inputs = _profiles(
        config,
        assets_value,
        native_value,
        source / "native-inputs",
    )
    out.mkdir(parents=True)
    source_receipt = {
        "attribution": source_config["attribution"],
        "license_scope": source_config["scope"],
        "not_redistribution_authorization": True,
        "source_assets_sha256": source_config["assets_sha256"],
        "source_native_inputs_sha256": source_config["native_inputs_sha256"],
        "source_license_review_sha256": source_config["license_review_sha256"],
    }
    runs = []
    for case in config["cases"]:
        runs.append(
            _emit_case(
                out,
                out.name,
                case,
                assets,
                profiles,
                native_inputs,
                source / "native-inputs",
                source_receipt,
            )
        )
    cloudy = next(
        case for case in config["cases"] if case["case_id"] == "cloudy-rejection"
    )
    runs.append(
        _emit_case(
            out,
            out.name,
            cloudy,
            assets,
            profiles,
            native_inputs,
            source / "native-inputs",
            source_receipt,
            counterfactual=True,
        )
    )
    result = {
        "schema_version": "1.0.0",
        "task_case_count": len(config["cases"]),
        "acceptance_run_count": len(runs),
        "runs": runs,
        "data_policy": {
            "scope": "local-research",
            "imagery_in_git": False,
            "redistribution_authorized": False,
        },
    }
    _write_json(out / "pack.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--license-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default=CONFIG, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output.absolute()
    runtime = (ROOT / "runtime").resolve()
    if not output.is_relative_to(runtime) or output == runtime:
        raise SystemExit("output must be a new scope under the managed runtime")
    with StorageQuota(runtime).hold(
        output,
        RESERVATION_BYTES,
        "sentinel-temporal-benchmark-pack",
    ):
        result = prepare(args.source, args.license_review, output, config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

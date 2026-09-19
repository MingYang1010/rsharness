#!/usr/bin/env python3
"""Admit a bounded CloudSEN12 label pack and validate the pinned SCL policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_api"))

from app.v2.cloud_policy_validation import (  # noqa: E402
    BENCHMARK_ID,
    CLOUD_POLICY,
    MANUAL_CLASSES,
    MANUAL_INVALID_CLASSES,
    VERSION,
    aggregate_cloud_policy,
    evaluate_cloud_policy,
)
from app.v2.raster_math import CLOUD_EXCLUDED_CLASSES  # noqa: E402
from app.v2.storage.quota import StorageQuota  # noqa: E402

MAX_MEMBER_BYTES = 1024 * 1024
OUTPUT_OVERHEAD_BYTES = 8 * 1024 * 1024


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _regular_file(path: Path) -> None:
    info = path.lstat()
    if path.is_symlink() or not path.is_file() or info.st_nlink != 1:
        raise ValueError("archive prefix must be a single regular file")


def _member_name(name: str) -> str:
    value = PurePosixPath(name)
    if not name or value.is_absolute() or ".." in value.parts or "" in value.parts:
        raise ValueError("archive contains an unsafe member path")
    normalized = str(value)
    if normalized != name or normalized.startswith("./"):
        raise ValueError("archive member paths must be canonical POSIX paths")
    return normalized


def _review_config(config: dict) -> tuple[dict[str, dict], list[float]]:
    if config.get("benchmark_id") != BENCHMARK_ID or config.get("benchmark_version") != VERSION:
        raise ValueError("benchmark identity or version changed")
    source = config.get("source", {})
    if (
        source.get("dataset_doi") != "10.57760/sciencedb.06669"
        or source.get("paper_doi") != "10.1038/s41597-022-01878-2"
        or source.get("license") != "CC BY-NC 4.0"
        or source.get("scope") != "local-research"
        or source.get("not_redistribution_authorization") is not True
    ):
        raise ValueError("reviewed source or license boundary changed")
    policy = config.get("policy", {})
    if (
        policy.get("policy_id") != CLOUD_POLICY
        or policy.get("excluded_scl_classes") != list(CLOUD_EXCLUDED_CLASSES)
        or policy.get("manual_classes") != list(MANUAL_CLASSES)
        or policy.get("manual_invalid_classes") != list(MANUAL_INVALID_CLASSES)
    ):
        raise ValueError("cloud policy contract changed")
    thresholds = policy.get("thresholds")
    if thresholds != [0.001, 0.2]:
        raise ValueError("reviewed decision thresholds changed")
    samples = config.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("at least one reviewed sample is required")
    selected: dict[str, dict] = {}
    sample_ids: set[str] = set()
    for sample in samples:
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError("sample IDs must be non-empty and unique")
        sample_ids.add(sample_id)
        profile = sample.get("profile", {})
        if (
            profile.get("width") != 509
            or profile.get("height") != 509
            or profile.get("count") != 1
            or profile.get("dtype") != "float32"
            or profile.get("crs") not in {"EPSG:32618", "EPSG:32651"}
            or not isinstance(profile.get("transform"), list)
            or len(profile["transform"]) != 6
            or not isinstance(profile.get("nodata"), (int, float))
        ):
            raise ValueError("reviewed sample profile is invalid")
        for role in ("manual", "scl"):
            member = sample.get(role, {})
            path = member.get("path")
            size = member.get("size_bytes")
            digest = member.get("sha256")
            if (
                not isinstance(path, str)
                or _member_name(path) != path
                or path in selected
                or type(size) is not int
                or not 0 < size <= MAX_MEMBER_BYTES
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("reviewed archive member contract is invalid")
            expected_suffix = "/labels/manual_hq.tif" if role == "manual" else "/labels/sen2cor.tif"
            if not path.endswith(expected_suffix):
                raise ValueError("reviewed archive label role is invalid")
            selected[path] = {
                **member,
                "role": role,
                "sample_id": sample_id,
                "profile": profile,
            }
    return selected, [float(value) for value in thresholds]


def _read_selected_members(
    archive_prefix: Path,
    selected: dict[str, dict],
) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    try:
        with archive_prefix.open("rb") as raw, tarfile.open(fileobj=raw, mode="r|gz") as archive:
            for member in archive:
                name = _member_name(member.name)
                if member.issym() or member.islnk():
                    raise ValueError("archive links are forbidden")
                if name not in selected:
                    continue
                if name in found:
                    raise ValueError("archive contains a duplicate reviewed member")
                expected = selected[name]
                if not member.isfile() or member.size != expected["size_bytes"] or member.size > MAX_MEMBER_BYTES:
                    raise ValueError("reviewed archive member type or size changed")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError("reviewed archive member is unreadable")
                content = handle.read(member.size + 1)
                if len(content) != member.size:
                    raise ValueError("reviewed archive member is truncated")
                if hashlib.sha256(content).hexdigest() != expected["sha256"]:
                    raise ValueError("reviewed archive member checksum changed")
                found[name] = content
                if len(found) == len(selected):
                    break
    except (tarfile.TarError, EOFError, OSError) as error:
        raise ValueError("archive prefix cannot provide the reviewed members") from error
    missing = sorted(set(selected) - set(found))
    if missing:
        raise ValueError("archive prefix is missing reviewed members")
    return found


def _read_label(content: bytes, expected: dict):
    import numpy
    import rasterio
    from rasterio.io import MemoryFile

    if content[:4] not in {b"II*\x00", b"MM\x00*"}:
        raise ValueError("reviewed label is not a classic TIFF")
    profile = expected["profile"]
    with rasterio.Env(
        GDAL_PAM_ENABLED="NO",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    ), MemoryFile(content) as memory, memory.open(driver="GTiff") as image:
        if (
            (image.width, image.height, image.count, image.dtypes)
            != (
                profile["width"],
                profile["height"],
                profile["count"],
                (profile["dtype"],),
            )
            or image.crs is None
            or image.crs.to_string() != profile["crs"]
            or list(image.transform)[:6] != profile["transform"]
            or image.nodata != profile["nodata"]
        ):
            raise ValueError("reviewed label TIFF profile changed")
        values = image.read(1)
        valid = image.read_masks(1) > 0
    if values.shape != (profile["height"], profile["width"]) or valid.dtype != numpy.bool_:
        raise ValueError("reviewed label array changed")
    return values, valid


def _evaluate(config: dict, selected: dict[str, dict], content: dict[str, bytes], thresholds: list[float]) -> tuple[list[dict], dict]:
    windows = []
    for sample in config["samples"]:
        manual_path = sample["manual"]["path"]
        scl_path = sample["scl"]["path"]
        manual, manual_valid = _read_label(content[manual_path], selected[manual_path])
        scl, scl_valid = _read_label(content[scl_path], selected[scl_path])
        evaluation = evaluate_cloud_policy(
            manual,
            scl,
            manual_valid=manual_valid,
            scl_valid=scl_valid,
            thresholds=thresholds,
        )
        windows.append(
            {
                "sample_id": sample["sample_id"],
                "manual_member": manual_path,
                "scl_member": scl_path,
                "evaluation": evaluation,
            }
        )
    return windows, aggregate_cloud_policy(windows, thresholds=thresholds)


def prepare(
    archive_prefix: Path,
    output: Path,
    config: dict,
    *,
    runtime_root: Path,
) -> dict:
    archive_prefix = Path(archive_prefix)
    output = Path(output)
    runtime_root = Path(runtime_root)
    selected, thresholds = _review_config(config)
    archive = config.get("archive", {})
    prefix = archive.get("prefix", {})
    _regular_file(archive_prefix)
    if (
        archive_prefix.stat().st_size != prefix.get("size_bytes")
        or _sha256_path(archive_prefix) != prefix.get("sha256")
    ):
        raise ValueError("archive prefix size or checksum changed")
    if output.exists():
        raise ValueError("output already exists")

    content = _read_selected_members(archive_prefix, selected)
    windows, aggregate = _evaluate(config, selected, content, thresholds)
    selected_bytes = sum(len(value) for value in content.values())
    report = {
        "schema_version": "1.0.0",
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": VERSION,
        "source": config["source"],
        "archive": {
            **archive,
            "prefix_verified": True,
            "full_archive_downloaded": False,
        },
        "policy": config["policy"],
        "selected_sample_count": len(windows),
        "selected_member_count": len(content),
        "selected_member_bytes": selected_bytes,
        "windows": windows,
        "aggregate": aggregate,
        "scope_limits": {
            "roi_count": len({sample["roi"] for sample in config["samples"]}),
            "global_accuracy_claim": False,
            "human_labels_infallible_claim": False,
        },
        "data_policy": {
            "labels_committed_to_git": False,
            "archive_prefix_committed_to_git": False,
            "redistribution_authorized": False,
        },
    }
    report_bytes = _json_bytes(report)
    capacity = selected_bytes + len(report_bytes) + OUTPUT_OVERHEAD_BYTES
    with StorageQuota(runtime_root).hold(output, capacity, "cloud-policy-validation-pack"):
        labels = output / "labels"
        labels.mkdir(parents=True, exist_ok=False)
        for name in sorted(content):
            destination = labels.joinpath(*PurePosixPath(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content[name])
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                raise
        report_path = output / "cloud-policy-validation.json"
        descriptor = os.open(
            report_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(report_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "cloud-policy-benchmark-v1.json",
    )
    parser.add_argument("--runtime-root", type=Path, default=ROOT / "runtime")
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    report = prepare(
        arguments.archive_prefix,
        arguments.output,
        config,
        runtime_root=arguments.runtime_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

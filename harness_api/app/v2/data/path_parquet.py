"""Bounded admission of reviewed local image paths from Parquet columns."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from app.v2.data.packed import (
    MAX_IMAGE_BYTES,
    MAX_RECEIPT_BYTES,
    AdmissionError,
    checked_shard,
    digest_file,
    probe_bytes,
    projected_rows,
    validate_column,
)


MAX_PATH_CHARACTERS = 4096


def reviewed_relative_path(value: object) -> PurePosixPath:
    """Accept one normalized POSIX relative path, never a URL or host path."""
    if not isinstance(value, str) or not value or len(value) > MAX_PATH_CHARACTERS:
        raise AdmissionError("invalid_image_path")
    if "\x00" in value or "\\" in value or "://" in value:
        raise AdmissionError("external_image_reference_not_allowed")
    path = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if path.is_absolute() or windows.is_absolute() or windows.drive:
        raise AdmissionError("external_image_reference_not_allowed")
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise AdmissionError("image_path_not_normalized")
    if str(path) != value:
        raise AdmissionError("image_path_not_normalized")
    return path


def _component_prefix(path: PurePosixPath, prefix: PurePosixPath) -> bool:
    return path.parts[:len(prefix.parts)] == prefix.parts


def _walk_without_symlinks(root: Path, relative: PurePosixPath) -> Path:
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise AdmissionError("symlink_image_path")
    return candidate


def checked_image_root(dataset_root: Path, relative: PurePosixPath) -> Path:
    if dataset_root.is_symlink() or not dataset_root.is_absolute():
        raise AdmissionError("invalid_dataset_root")
    root = dataset_root.resolve(strict=True)
    if not root.is_dir():
        raise AdmissionError("invalid_dataset_root")
    candidate = _walk_without_symlinks(root, relative)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise AdmissionError("invalid_image_root") from None
    if not resolved.is_relative_to(root) or not resolved.is_dir():
        raise AdmissionError("invalid_image_root")
    return resolved


def checked_local_image(root: Path, relative: PurePosixPath) -> Path:
    candidate = _walk_without_symlinks(root, relative)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise AdmissionError("image_path_missing") from None
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise AdmissionError("image_path_outside_root")
    if resolved.stat().st_size > MAX_IMAGE_BYTES:
        raise AdmissionError("image_size_limit")
    return resolved


def _unchanged(before, path: Path, code: str) -> None:
    try:
        after = path.stat()
    except OSError:
        raise AdmissionError(code) from None
    identity = lambda value: (value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(after):
        raise AdmissionError(code)


def extract_path_samples(
    spec: dict,
    output: Path,
    sample_count: int = 3,
    max_rows: int = 128,
    max_output_bytes: int = 512 * 1024 * 1024,
) -> dict:
    """Copy bounded, reviewed path-column images into an immutable runtime sample."""
    import pyarrow as pa

    if not 1 <= sample_count <= 100 or not 1 <= max_rows <= 10000 or max_output_bytes <= 0:
        raise AdmissionError("invalid_limits")
    column = spec["image_column"]
    validate_column(column)
    if spec.get("reviewed_role") != "input_image" or not spec.get("review_id"):
        raise AdmissionError("image_column_review_required")

    dataset_root = Path(spec["root"])
    if dataset_root.is_symlink() or not dataset_root.is_absolute() or not dataset_root.is_dir():
        raise AdmissionError("invalid_dataset_root")
    dataset_root = dataset_root.resolve()
    parquet_relative = reviewed_relative_path(spec["parquet"])
    if PurePosixPath(parquet_relative).suffix.lower() != ".parquet":
        raise AdmissionError("parquet_required")
    parquet = checked_shard(dataset_root, dataset_root.joinpath(*parquet_relative.parts))
    image_root_relative = reviewed_relative_path(spec["image_root"])
    image_root = checked_image_root(dataset_root, image_root_relative)

    raw_prefixes = spec.get("allowed_path_prefixes")
    if not isinstance(raw_prefixes, list) or not raw_prefixes:
        raise AdmissionError("allowed_path_prefixes_required")
    prefixes = [reviewed_relative_path(value) for value in raw_prefixes]
    if output.exists():
        raise AdmissionError("output_exists_preserve_previous_run")
    if output.resolve().is_relative_to(dataset_root):
        raise AdmissionError("output_must_not_modify_source_root")

    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "private").mkdir(mode=0o700)
    samples: list[dict] = []
    origins: list[dict] = []
    failures: list[dict] = []
    seen: set[str] = set()
    duplicates = inspected = used = 0
    parquet_before = parquet.stat()
    parquet_hash = digest_file(parquet)
    rows = projected_rows(parquet, column, max_rows)
    try:
        for row, cell in rows:
            inspected += 1
            try:
                relative = reviewed_relative_path(cell)
                if not any(_component_prefix(relative, prefix) for prefix in prefixes):
                    raise AdmissionError("image_path_prefix_forbidden")
                image = checked_local_image(image_root, relative)
                image_before = image.stat()
                content = image.read_bytes()
                _unchanged(image_before, image, "image_changed_during_read")
                digest = hashlib.sha256(content).hexdigest()
                origin = {
                    "asset_id": "asset-" + digest,
                    "parquet_sha256": parquet_hash,
                    "parquet_row": row,
                    "image_column": column,
                    "image_relative_path": str(relative),
                    "image_sha256": digest,
                }
                if digest in seen:
                    duplicates += 1
                    origins.append(origin)
                    continue
                probe = probe_bytes(content)
                if used + len(content) > max_output_bytes:
                    raise AdmissionError("output_byte_limit")
            except (AdmissionError, OSError) as error:
                code = str(error) if isinstance(error, AdmissionError) else "image_read_failed"
                failures.append({"row": row, "code": code})
                if code == "image_changed_during_read":
                    raise
                continue
            filename = digest + probe.pop("extension")
            staged = output / "inputs" / filename
            with staged.open("xb") as stream:
                stream.write(content)
            if digest_file(staged) != digest:
                raise AdmissionError("staged_checksum_mismatch")
            used += len(content)
            seen.add(digest)
            origins.append(origin)
            samples.append({
                **probe,
                "asset_id": "asset-" + digest,
                "sha256": digest,
                "relative_path": filename,
                "size_bytes": len(content),
                "source_snapshot_hash": parquet_hash,
                "status": "readable_sample",
            })
            if len(samples) >= sample_count:
                break
    except pa.ArrowException as error:
        failures.append({"code": type(error).__name__})
    finally:
        rows.close()
        _unchanged(parquet_before, parquet, "source_changed_during_read")

    report = {
        "schema_version": "path-parquet-image-samples-1",
        "scope": "reviewed-local-relative-image-path-samples-only",
        "datasets": [{
            "dataset_id": spec["dataset_id"],
            "root": str((output / "inputs").resolve()),
            "samples": samples,
            "status": "sampled_readable" if len(samples) == sample_count else "partial",
        }],
        "sampling": "ordered Parquet rows, distinct content; not a representative benchmark sample",
        "unique_images": len(samples),
        "duplicate_images": duplicates,
        "inspected_rows": inspected,
        "output_bytes": used,
        "failures": failures,
        "label_columns_exported": False,
    }
    receipt = {
        "dataset_id": spec["dataset_id"],
        "dataset_root": str(dataset_root),
        "image_root": str(image_root),
        "review_id": spec["review_id"],
        "image_column": column,
        "allowed_path_prefixes": [str(value) for value in prefixes],
        "parquet": {
            "relative_path": str(parquet_relative),
            "sha256": parquet_hash,
            "size_bytes": parquet_before.st_size,
        },
        "origins": origins,
        "pyarrow_version": pa.__version__,
        "license_status": spec["license"],
    }
    manifest = {
        sample["asset_id"]: {
            "role": "input_image",
            "filename": sample["relative_path"],
            "sha256": sample["sha256"],
        }
        for sample in samples
    }
    metadata = [
        (output / "private/receipt.json", receipt),
        (output / "coverage.json", report),
        (output / "inputs.json", manifest),
    ]
    payloads = [(path, (json.dumps(value, indent=2) + "\n").encode()) for path, value in metadata]
    if sum(len(value) for _, value in payloads) > MAX_RECEIPT_BYTES:
        raise AdmissionError("receipt_size_limit")
    for path, value in payloads:
        path.write_bytes(value)
    return report

"""Dataset-wide, metadata-only coverage audit for reviewed Parquet image paths."""
from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from app.core.data.packed import (
    MAX_IMAGE_PIXELS,
    AdmissionError,
    checked_shard,
    digest_file,
    validate_column,
)
from app.core.data.path_parquet import (
    _component_prefix,
    checked_image_root,
    checked_local_image,
    reviewed_relative_path,
)


MAX_COVERAGE_ROWS = 100_000
ALLOWED_DRIVERS = {"GTiff", "JPEG", "PNG"}


def _unchanged(before, path: Path, code: str) -> None:
    try:
        after = path.stat()
    except OSError:
        raise AdmissionError(code) from None
    identity = lambda value: (value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(after):
        raise AdmissionError(code)


def _header_signature(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
    except OSError:
        raise AdmissionError("image_header_read_failed") from None
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if prefix.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if prefix[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "GTiff"
    raise AdmissionError("unsupported_image_encoding")


def probe_image_header(path: Path) -> dict:
    """Inspect only a local image header; never decode pixel blocks."""
    import rasterio
    from rasterio.warp import transform_bounds

    before = path.stat()
    expected_driver = _header_signature(path)
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_PAM_ENABLED="NO"):
            with rasterio.open(path, sharing=False) as image:
                if image.driver not in ALLOWED_DRIVERS or image.driver != expected_driver:
                    raise AdmissionError("unsupported_image_encoding")
                pixels = image.width * image.height
                if not 1 <= image.count <= 4:
                    raise AdmissionError("provider_image_dimensions_limit")
                reference_kind = "pixel"
                if image.crs:
                    bbox = transform_bounds(image.crs, "EPSG:4326", *image.bounds)
                    if not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
                        raise AdmissionError("invalid_geographic_extent")
                    reference_kind = "geographic"
                outcome = (
                    "reviewed_window_header_candidate"
                    if pixels > MAX_IMAGE_PIXELS
                    else "whole_image_header_eligible"
                )
                result = {
                    "outcome": outcome,
                    "driver": image.driver,
                    "reference_kind": reference_kind,
                    "width": image.width,
                    "height": image.height,
                    "bands": image.count,
                    "pixels": pixels,
                    "size_bytes": before.st_size,
                }
    except AdmissionError:
        raise
    except (OSError, ValueError, rasterio.errors.RasterioError):
        raise AdmissionError("image_header_decode_failed") from None
    _unchanged(before, path, "image_changed_during_header_audit")
    return result


def audit_path_coverage(
    spec: dict,
    max_rows: int = MAX_COVERAGE_ROWS,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict, dict]:
    """Audit every bounded path row without reading labels, pixels, or raw paths out."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import rasterio

    if not 1 <= max_rows <= MAX_COVERAGE_ROWS:
        raise AdmissionError("coverage_row_limit_invalid")
    column = spec["image_column"]
    validate_column(column)
    if spec.get("reviewed_role") != "input_image" or not spec.get("review_id"):
        raise AdmissionError("image_column_review_required")

    dataset_root = Path(spec["root"])
    if dataset_root.is_symlink() or not dataset_root.is_absolute() or not dataset_root.is_dir():
        raise AdmissionError("invalid_dataset_root")
    dataset_root = dataset_root.resolve()
    parquet_relative = reviewed_relative_path(spec["parquet"])
    if parquet_relative.suffix.lower() != ".parquet":
        raise AdmissionError("parquet_required")
    parquet = checked_shard(dataset_root, dataset_root.joinpath(*parquet_relative.parts))
    image_root_relative = reviewed_relative_path(spec["image_root"])
    image_root = checked_image_root(dataset_root, image_root_relative)
    raw_prefixes = spec.get("allowed_path_prefixes")
    if not isinstance(raw_prefixes, list) or not raw_prefixes:
        raise AdmissionError("allowed_path_prefixes_required")
    prefixes = [reviewed_relative_path(value) for value in raw_prefixes]

    parquet_before = parquet.stat()
    parquet_hash = digest_file(parquet)
    outcomes: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    drivers: Counter[str] = Counter()
    references: Counter[str] = Counter()
    seen_path_hashes: set[bytes] = set()
    rows_scanned = duplicate_rows = unique_file_bytes = 0

    try:
        with pa.memory_map(str(parquet), "r") as source:
            reader = pq.ParquetFile(source)
            if column not in reader.schema_arrow.names:
                raise AdmissionError("image_column_missing")
            rows_total = reader.metadata.num_rows
            scan_limit = min(rows_total, max_rows)
            batches = reader.iter_batches(batch_size=256, columns=[column], use_threads=False)
            for batch in batches:
                if batch.schema.names != [column]:
                    raise AdmissionError("column_projection_failed")
                for index in range(batch.num_rows):
                    if rows_scanned >= scan_limit:
                        break
                    rows_scanned += 1
                    try:
                        relative = reviewed_relative_path(batch.column(0)[index].as_py())
                        if not any(_component_prefix(relative, prefix) for prefix in prefixes):
                            raise AdmissionError("image_path_prefix_forbidden")
                        path_hash = hashlib.sha256(str(relative).encode("utf-8")).digest()
                        if path_hash in seen_path_hashes:
                            duplicate_rows += 1
                            if progress is not None:
                                progress(rows_scanned, scan_limit)
                            continue
                        seen_path_hashes.add(path_hash)
                        image = checked_local_image(image_root, relative)
                        probe = probe_image_header(image)
                        outcomes[probe["outcome"]] += 1
                        drivers[probe["driver"]] += 1
                        references[probe["reference_kind"]] += 1
                        unique_file_bytes += probe["size_bytes"]
                    except AdmissionError as error:
                        code = str(error)
                        if code == "image_changed_during_header_audit":
                            raise
                        failures[code] += 1
                    if progress is not None:
                        progress(rows_scanned, scan_limit)
                if rows_scanned >= scan_limit:
                    break
    except pa.ArrowException:
        raise AdmissionError("parquet_read_failed") from None
    finally:
        _unchanged(parquet_before, parquet, "source_changed_during_coverage_audit")

    accounted = sum(outcomes.values()) + sum(failures.values()) + duplicate_rows
    if accounted != rows_scanned:
        raise AdmissionError("coverage_accounting_mismatch")
    report = {
        "schema_version": "path-parquet-header-coverage-1",
        "scope": "reviewed-path-column header coverage only; not image admission",
        "dataset_id": spec["dataset_id"],
        "rows_total": rows_total,
        "rows_scanned": rows_scanned,
        "scan_complete": rows_scanned == rows_total,
        "distinct_reviewed_path_values": len(seen_path_hashes),
        "duplicate_path_rows": duplicate_rows,
        "unique_files_by_outcome": dict(sorted(outcomes.items())),
        "failures_by_code": dict(sorted(failures.items())),
        "drivers": dict(sorted(drivers.items())),
        "reference_kinds": dict(sorted(references.items())),
        "unique_file_bytes": unique_file_bytes,
        "label_columns_read": False,
        "raw_paths_exported": False,
        "pixel_content_read": False,
        "image_content_hashes_verified": False,
        "admission_status": "coverage_only",
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
        "max_rows": max_rows,
        "pyarrow_version": pa.__version__,
        "rasterio_version": rasterio.__version__,
        "license_status": spec["license"],
        "row_values_retained": False,
        "image_content_hashes_verified": False,
    }
    return report, receipt


def validate_coverage_output(output: Path, dataset_root: Path) -> None:
    """Keep audit products outside the immutable source dataset tree."""
    if output.exists():
        raise AdmissionError("output_exists_preserve_previous_run")
    try:
        source = dataset_root.resolve(strict=True)
    except OSError:
        raise AdmissionError("invalid_dataset_root") from None
    if output.resolve().is_relative_to(source):
        raise AdmissionError("output_must_not_modify_source_root")

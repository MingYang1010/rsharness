"""Deterministic, bounded window admission for reviewed local rasters."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from app.v2.data.packed import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_RECEIPT_BYTES,
    AdmissionError,
    checked_shard,
    digest_file,
    probe_bytes,
    projected_rows,
    validate_column,
)
from app.v2.data.path_parquet import (
    _component_prefix,
    _unchanged,
    checked_image_root,
    checked_local_image,
    reviewed_relative_path,
)


POLICY_ID = "reviewed-raster-window-v1"
OUTPUT_PROFILE_ID = "gtiff-window-lossless-v1"
TILE_SIZE = 2048
MAX_DECODED_WINDOW_BYTES = 128 * 1024 * 1024
MAX_SOURCE_DIMENSION = 1_000_000
ALLOWED_SOURCE_DRIVERS = {"GTiff", "PNG", "JPEG"}


@dataclass
class _Source:
    path: Path
    before: object
    sha256: str
    relative_path: str
    rows: list[int] = field(default_factory=list)
    header: dict = field(default_factory=dict)

    @property
    def tiles_x(self) -> int:
        return math.ceil(self.header["width"] / TILE_SIZE)

    @property
    def tiles_y(self) -> int:
        return math.ceil(self.header["height"] / TILE_SIZE)

    @property
    def tile_count(self) -> int:
        return self.tiles_x * self.tiles_y


def iter_grid_windows(width: int, height: int, tile_size: int = TILE_SIZE) -> Iterator[tuple[int, int, int, int]]:
    """Yield clipped half-open windows in row-major order."""
    if width <= 0 or height <= 0 or tile_size <= 0:
        raise AdmissionError("invalid_window_grid")
    for row_off in range(0, height, tile_size):
        for col_off in range(0, width, tile_size):
            yield (
                col_off,
                row_off,
                min(tile_size, width - col_off),
                min(tile_size, height - row_off),
            )


def _window_at(source: _Source, index: int) -> tuple[int, int, int, int]:
    if index < 0 or index >= source.tile_count:
        raise AdmissionError("window_index_out_of_range")
    tile_row, tile_col = divmod(index, source.tiles_x)
    col_off = tile_col * TILE_SIZE
    row_off = tile_row * TILE_SIZE
    return (
        col_off,
        row_off,
        min(TILE_SIZE, source.header["width"] - col_off),
        min(TILE_SIZE, source.header["height"] - row_off),
    )


def _json_number(value):
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise AdmissionError("non_finite_nodata")
    return int(number) if number.is_integer() else number


def _read_header(path: Path) -> tuple[object, dict]:
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds

    before = path.stat()
    try:
        with rasterio.open(path) as source:
            if source.driver not in ALLOWED_SOURCE_DRIVERS:
                raise AdmissionError("unsupported_window_source_driver")
            if not (1 <= source.width <= MAX_SOURCE_DIMENSION and 1 <= source.height <= MAX_SOURCE_DIMENSION):
                raise AdmissionError("source_dimensions_limit")
            if not 1 <= source.count <= 4:
                raise AdmissionError("provider_image_dimensions_limit")
            if len(set(source.dtypes)) != 1:
                raise AdmissionError("mixed_source_dtypes_not_supported")
            dtype = np.dtype(source.dtypes[0])
            if dtype.hasobject:
                raise AdmissionError("unsupported_source_dtype")
            nodata = _json_number(source.nodata)
            transform = list(source.transform)[:6]
            crs_wkt = source.crs.to_wkt() if source.crs else None
            spatial = None
            if source.crs:
                bbox = list(transform_bounds(source.crs, "EPSG:4326", *source.bounds))
                if not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
                    raise AdmissionError("invalid_geographic_extent")
                spatial = {"native_crs": str(source.crs), "bbox_wgs84": bbox}
            header = {
                "driver": source.driver,
                "width": source.width,
                "height": source.height,
                "count": source.count,
                "dtype": dtype.name,
                "crs_wkt": crs_wkt,
                "transform": transform,
                "nodata": nodata,
                "colorinterp": [value.name for value in source.colorinterp],
                "mask_flags": [[flag.name for flag in values] for values in source.mask_flag_enums],
                "spatial": spatial,
            }
    except rasterio.errors.RasterioError:
        raise AdmissionError("image_decode_failed") from None
    _unchanged(before, path, "source_changed_during_header_read")
    return before, header


def _derivation(source: _Source, window: tuple[int, int, int, int], parquet_hash: str) -> tuple[str, dict]:
    document = {
        "policy_id": POLICY_ID,
        "output_profile_id": OUTPUT_PROFILE_ID,
        "parquet_sha256": parquet_hash,
        "source_sha256": source.sha256,
        "source": {
            key: source.header[key]
            for key in (
                "width", "height", "count", "dtype", "crs_wkt", "transform",
                "nodata", "colorinterp", "mask_flags",
            )
        },
        "window": {
            "col_off": window[0],
            "row_off": window[1],
            "width": window[2],
            "height": window[3],
        },
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest(), document


def _materialize_window(source: _Source, window: tuple[int, int, int, int], working: Path) -> dict:
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    col_off, row_off, width, height = window
    decoded_bytes = width * height * (source.header["count"] * np.dtype(source.header["dtype"]).itemsize + 1)
    if decoded_bytes > MAX_DECODED_WINDOW_BYTES:
        raise AdmissionError("decoded_window_byte_limit")
    raster_window = Window(col_off, row_off, width, height)
    try:
        with rasterio.open(source.path) as input_image:
            pixels = input_image.read(window=raster_window)
            masks = input_image.read_masks(window=raster_window)
            if any(not np.array_equal(masks[0], masks[index]) for index in range(1, input_image.count)):
                raise AdmissionError("band_masks_not_losslessly_representable")
            transform = input_image.window_transform(raster_window)
            colorinterp = input_image.colorinterp
            crs = input_image.crs
            nodata = input_image.nodata
        _unchanged(source.before, source.path, "source_changed_during_window_read")
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK="YES"):
            with rasterio.open(
                working,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=source.header["count"],
                dtype=source.header["dtype"],
                crs=crs,
                transform=transform,
                nodata=nodata,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                interleave="band",
                BIGTIFF="NO",
            ) as output_image:
                output_image.write(pixels)
                output_image.write_mask(masks[0])
                output_image.colorinterp = colorinterp
        if Path(str(working) + ".msk").exists() or Path(str(working) + ".aux.xml").exists():
            raise AdmissionError("sidecar_output_not_allowed")
        with working.open("rb") as stream:
            marker = stream.read(4)
        if marker not in {b"II*\x00", b"MM\x00*"}:
            raise AdmissionError("classic_tiff_required")
        with rasterio.open(working) as output_image:
            if (
                output_image.driver != "GTiff"
                or (output_image.width, output_image.height, output_image.count) != (width, height, source.header["count"])
                or tuple(output_image.dtypes) != (source.header["dtype"],) * source.header["count"]
                or output_image.crs != crs
                or output_image.transform != transform
                or output_image.nodata != nodata
                or output_image.colorinterp != colorinterp
                or not np.array_equal(output_image.read(), pixels)
                or not np.array_equal(output_image.read_masks(1), masks[0])
            ):
                raise AdmissionError("staged_window_verification_failed")
    except rasterio.errors.RasterioError:
        raise AdmissionError("window_io_failed") from None
    return {
        "transform": list(transform)[:6],
        "decoded_bytes": decoded_bytes,
    }


def extract_raster_windows(
    spec: dict,
    output: Path,
    sample_count: int = 3,
    max_rows: int = 128,
    max_output_bytes: int = 512 * 1024 * 1024,
) -> dict:
    """Admit deterministic lossless windows from reviewed oversized raster paths."""
    import pyarrow as pa

    if not 1 <= sample_count <= 100 or not 1 <= max_rows <= 10000 or max_output_bytes <= 0:
        raise AdmissionError("invalid_limits")
    if spec.get("window_policy") != POLICY_ID:
        raise AdmissionError("window_policy_review_required")
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
    image_root = checked_image_root(dataset_root, reviewed_relative_path(spec["image_root"]))
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
    (output / "private/staging").mkdir(parents=True, mode=0o700)
    samples: list[dict] = []
    private_windows: list[dict] = []
    failures: list[dict] = []
    sources: list[_Source] = []
    source_by_hash: dict[str, _Source] = {}
    duplicate_rows = ineligible = inspected = used = 0
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
                before, header = _read_header(image)
                if header["width"] * header["height"] <= MAX_IMAGE_PIXELS:
                    ineligible += 1
                    continue
                source_hash = digest_file(image)
                _unchanged(before, image, "source_changed_during_hash")
                if source_hash in source_by_hash:
                    duplicate_rows += 1
                    source_by_hash[source_hash].rows.append(row)
                    continue
                source = _Source(
                    path=image,
                    before=before,
                    sha256=source_hash,
                    relative_path=str(relative),
                    rows=[row],
                    header=header,
                )
                source_by_hash[source_hash] = source
                sources.append(source)
                if len(sources) >= sample_count:
                    break
            except (AdmissionError, OSError) as error:
                code = str(error) if isinstance(error, AdmissionError) else "image_read_failed"
                failures.append({"row": row, "code": code})
                if code.startswith("source_changed_during"):
                    raise AdmissionError(code)

        round_index = 0
        while len(samples) < sample_count:
            progressed = False
            for source_index, source in enumerate(sources):
                if round_index >= source.tile_count or len(samples) >= sample_count:
                    continue
                progressed = True
                window = _window_at(source, round_index)
                derivation_hash, derivation = _derivation(source, window, parquet_hash)
                working = output / "private/staging" / f"{derivation_hash}.tif"
                try:
                    materialized = _materialize_window(source, window, working)
                    content = working.read_bytes()
                    if len(content) > MAX_IMAGE_BYTES:
                        raise AdmissionError("image_size_limit")
                    if used + len(content) > max_output_bytes:
                        raise AdmissionError("output_byte_limit")
                    probe = probe_bytes(content)
                    content_hash = hashlib.sha256(content).hexdigest()
                    filename = f"{derivation_hash}.tif"
                    staged = output / "inputs" / filename
                    if staged.exists():
                        raise AdmissionError("derived_window_collision")
                    working.rename(staged)
                    if digest_file(staged) != content_hash:
                        raise AdmissionError("staged_checksum_mismatch")
                except (AdmissionError, OSError) as error:
                    code = str(error) if isinstance(error, AdmissionError) else "window_write_failed"
                    failures.append({"source_index": source_index, "window_index": round_index, "code": code})
                    if code.startswith("source_changed_during"):
                        raise AdmissionError(code)
                    continue
                used += len(content)
                samples.append({
                    **probe,
                    "asset_id": "asset-" + derivation_hash,
                    "sha256": content_hash,
                    "relative_path": filename,
                    "size_bytes": len(content),
                    "source_snapshot_hash": derivation_hash,
                    "status": "readable_sample",
                    "admission_profile": POLICY_ID,
                    "output_profile": OUTPUT_PROFILE_ID,
                    "pixel_window": derivation["window"],
                    "window_transform": materialized["transform"],
                })
                private_windows.append({
                    "asset_id": "asset-" + derivation_hash,
                    "source_sha256": source.sha256,
                    "source_snapshot_hash": derivation_hash,
                    "sha256": content_hash,
                    "derivation": derivation,
                })
            if not progressed:
                break
            round_index += 1

        for source in sources:
            _unchanged(source.before, source.path, "source_changed_before_publish")
            if digest_file(source.path) != source.sha256:
                raise AdmissionError("source_checksum_changed_before_publish")
    except pa.ArrowException as error:
        failures.append({"code": type(error).__name__})
    finally:
        rows.close()
        _unchanged(parquet_before, parquet, "source_changed_during_read")

    report = {
        "schema_version": "reviewed-raster-window-samples-1",
        "scope": "reviewed-local-oversized-raster-windows-only",
        "policy_id": POLICY_ID,
        "output_profile_id": OUTPUT_PROFILE_ID,
        "datasets": [{
            "dataset_id": spec["dataset_id"],
            "root": str((output / "inputs").resolve()),
            "samples": samples,
            "status": "sampled_readable" if len(samples) == sample_count else "partial",
        }],
        "sampling": "reviewed rows, distinct source checksum, round-robin row-major 2048-pixel grid",
        "unique_windows": len(samples),
        "unique_sources": len(sources),
        "duplicate_source_rows": duplicate_rows,
        "ineligible_whole_images": ineligible,
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
        "policy_id": POLICY_ID,
        "output_profile_id": OUTPUT_PROFILE_ID,
        "parquet": {
            "relative_path": str(parquet_relative),
            "sha256": parquet_hash,
            "size_bytes": parquet_before.st_size,
        },
        "sources": [{
            "source_sha256": source.sha256,
            "image_relative_path": source.relative_path,
            "parquet_rows": source.rows,
            "header": source.header,
        } for source in sources],
        "windows": private_windows,
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


def _read_json_bounded(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECEIPT_BYTES:
        raise AdmissionError("invalid_admission_metadata")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        raise AdmissionError("invalid_admission_metadata") from None
    if not isinstance(value, dict):
        raise AdmissionError("invalid_admission_metadata")
    return value


def verify_raster_window_output(output: Path) -> dict:
    """Independently verify admitted windows against immutable local sources."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    if output.is_symlink() or not output.is_absolute():
        raise AdmissionError("invalid_admission_output")
    root = output.resolve(strict=True)
    coverage = _read_json_bounded(root / "coverage.json")
    receipt = _read_json_bounded(root / "private/receipt.json")
    manifest = _read_json_bounded(root / "inputs.json")
    if (
        coverage.get("policy_id") != POLICY_ID
        or receipt.get("policy_id") != POLICY_ID
        or coverage.get("output_profile_id") != OUTPUT_PROFILE_ID
        or receipt.get("output_profile_id") != OUTPUT_PROFILE_ID
        or coverage.get("label_columns_exported") is not False
    ):
        raise AdmissionError("admission_policy_mismatch")
    datasets = coverage.get("datasets")
    windows = receipt.get("windows")
    sources_value = receipt.get("sources")
    if (
        not isinstance(datasets, list)
        or len(datasets) != 1
        or not isinstance(windows, list)
        or not 1 <= len(windows) <= 100
        or not isinstance(sources_value, list)
        or not 1 <= len(sources_value) <= 100
    ):
        raise AdmissionError("invalid_admission_metadata")
    samples_value = datasets[0].get("samples")
    if not isinstance(samples_value, list) or len(samples_value) != len(windows):
        raise AdmissionError("admission_window_count_mismatch")
    samples = {sample.get("asset_id"): sample for sample in samples_value if isinstance(sample, dict)}
    if len(samples) != len(samples_value):
        raise AdmissionError("duplicate_admission_asset")

    image_root = Path(receipt.get("image_root", ""))
    if image_root.is_symlink() or not image_root.is_absolute() or not image_root.is_dir():
        raise AdmissionError("invalid_image_root")
    image_root = image_root.resolve(strict=True)
    sources: dict[str, Path] = {}
    for value in sources_value:
        if not isinstance(value, dict):
            raise AdmissionError("invalid_admission_metadata")
        source_hash = value.get("source_sha256")
        relative = reviewed_relative_path(value.get("image_relative_path"))
        source_path = checked_local_image(image_root, relative)
        if not isinstance(source_hash, str) or digest_file(source_path) != source_hash:
            raise AdmissionError("source_checksum_mismatch")
        if source_hash in sources and sources[source_hash] != source_path:
            raise AdmissionError("source_identity_conflict")
        sources[source_hash] = source_path

    inputs_root = (root / "inputs").resolve(strict=True)
    verified = 0
    for private in windows:
        if not isinstance(private, dict) or not isinstance(private.get("derivation"), dict):
            raise AdmissionError("invalid_admission_metadata")
        derivation = private["derivation"]
        canonical = json.dumps(derivation, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        derivation_hash = hashlib.sha256(canonical).hexdigest()
        asset_id = private.get("asset_id")
        source_hash = private.get("source_sha256")
        sample = samples.get(asset_id)
        entry = manifest.get(asset_id)
        if (
            derivation_hash != private.get("source_snapshot_hash")
            or derivation_hash != str(asset_id).removeprefix("asset-")
            or not isinstance(sample, dict)
            or not isinstance(entry, dict)
            or sample.get("source_snapshot_hash") != derivation_hash
            or derivation.get("source_sha256") != source_hash
            or source_hash not in sources
        ):
            raise AdmissionError("derivation_identity_mismatch")
        filename = reviewed_relative_path(entry.get("filename"))
        if len(filename.parts) != 1:
            raise AdmissionError("invalid_staged_window_path")
        staged = inputs_root / filename.name
        if staged.is_symlink() or not staged.is_file() or not staged.resolve().is_relative_to(inputs_root):
            raise AdmissionError("invalid_staged_window_path")
        if staged.stat().st_size > MAX_IMAGE_BYTES:
            raise AdmissionError("image_size_limit")
        if digest_file(staged) != entry.get("sha256") or entry.get("sha256") != sample.get("sha256"):
            raise AdmissionError("staged_window_checksum_mismatch")
        with staged.open("rb") as stream:
            if stream.read(4) not in {b"II*\x00", b"MM\x00*"}:
                raise AdmissionError("classic_tiff_required")
        if Path(str(staged) + ".msk").exists() or Path(str(staged) + ".aux.xml").exists():
            raise AdmissionError("sidecar_output_not_allowed")

        window_value = derivation.get("window")
        if not isinstance(window_value, dict):
            raise AdmissionError("invalid_window_metadata")
        try:
            values = tuple(window_value[key] for key in ("col_off", "row_off", "width", "height"))
        except KeyError:
            raise AdmissionError("invalid_window_metadata") from None
        if (
            any(not isinstance(value, int) for value in values)
            or values[0] < 0
            or values[1] < 0
            or values[2] <= 0
            or values[3] <= 0
        ):
            raise AdmissionError("invalid_window_metadata")
        raster_window = Window(*values)
        try:
            with rasterio.open(sources[source_hash]) as source, rasterio.open(staged) as derived:
                if values[0] + values[2] > source.width or values[1] + values[3] > source.height:
                    raise AdmissionError("window_outside_source")
                pixels = source.read(window=raster_window)
                masks = source.read_masks(window=raster_window)
                if any(not np.array_equal(masks[0], masks[index]) for index in range(1, source.count)):
                    raise AdmissionError("band_masks_not_losslessly_representable")
                expected_transform = source.window_transform(raster_window)
                if (
                    (derived.width, derived.height, derived.count) != (values[2], values[3], source.count)
                    or derived.dtypes != source.dtypes
                    or derived.crs != source.crs
                    or derived.transform != expected_transform
                    or derived.nodata != source.nodata
                    or derived.colorinterp != source.colorinterp
                    or not np.array_equal(derived.read(), pixels)
                    or not np.array_equal(derived.read_masks(1), masks[0])
                    or sample.get("window_transform") != list(expected_transform)[:6]
                ):
                    raise AdmissionError("admitted_window_reference_mismatch")
        except rasterio.errors.RasterioError:
            raise AdmissionError("window_io_failed") from None
        verified += 1
    return {
        "status": "passed",
        "policy_id": POLICY_ID,
        "verified_windows": verified,
        "verified_sources": len(sources),
    }

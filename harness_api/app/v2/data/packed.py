"""Bounded Arrow/Parquet image-column projection and private provenance receipts."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterator


MAX_IMAGE_BYTES = 128 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_SHARD_BYTES = 2 * 1024 * 1024 * 1024
MAX_RECEIPT_BYTES = 8 * 1024 * 1024
DENIED_COLUMNS = {"label", "labels", "answer", "answers", "mask", "masks", "annotation", "annotations", "bbox", "gt", "groundtruth"}


class AdmissionError(ValueError):
    """Errors expose stable codes, never row values or private labels."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_shard(root: Path, path: Path) -> Path:
    root = root.resolve(strict=True)
    if path.is_symlink():
        raise AdmissionError("symlink_shard")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise AdmissionError("shard_outside_root")
    current = path.absolute()
    while current != root and current != current.parent:
        if current.is_symlink():
            raise AdmissionError("symlink_shard")
        current = current.parent
    if resolved.stat().st_size > MAX_SHARD_BYTES:
        raise AdmissionError("shard_size_limit")
    return resolved


def validate_column(column: str) -> None:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", column):
        raise AdmissionError("invalid_image_column")
    tokens = set(re.split(r"[^a-z0-9]+", column.lower()))
    if tokens & DENIED_COLUMNS:
        raise AdmissionError("label_column_forbidden")


def projected_rows(path: Path, column: str, max_rows: int) -> Iterator[tuple[int, object]]:
    """Only one reviewed image column is decoded; row values never include labels."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    validate_column(column)
    if not 1 <= max_rows <= 10000:
        raise AdmissionError("row_limit_invalid")
    with pa.memory_map(str(path), "r") as source:
        if path.suffix.lower() == ".parquet":
            reader = pq.ParquetFile(source)
            if column not in reader.schema_arrow.names:
                raise AdmissionError("image_column_missing")
            batches = reader.iter_batches(batch_size=1, columns=[column], use_threads=False)
        elif path.suffix.lower() == ".arrow":
            try:
                initial = pa.ipc.open_file(source)
                mode = "file"
            except pa.ArrowInvalid:
                source.seek(0)
                initial = pa.ipc.open_stream(source)
                mode = "stream"
            index = initial.schema.get_field_index(column)
            if index < 0:
                raise AdmissionError("image_column_missing")
            source.seek(0)
            options = pa.ipc.IpcReadOptions(included_fields=[index], use_threads=False)
            if mode == "file":
                reader = pa.ipc.open_file(source, options=options)
                batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
            else:
                reader = pa.ipc.open_stream(source, options=options)
                batches = iter(reader)
        else:
            raise AdmissionError("unsupported_container")
        row_index = 0
        for batch in batches:
            if batch.schema.names != [column]:
                raise AdmissionError("column_projection_failed")
            for row in range(min(batch.num_rows, max_rows - row_index)):
                # Convert the selected scalar, never an entire table/record.
                yield row_index, batch.column(0)[row].as_py()
                row_index += 1
            if row_index >= max_rows:
                return


def embedded_images(value: object, depth: int = 0) -> list[bytes]:
    if depth > 3:
        raise AdmissionError("image_nesting_limit")
    if isinstance(value, list):
        if len(value) > 16:
            raise AdmissionError("image_sequence_limit")
        images = []
        for item in value:
            images.extend(embedded_images(item, depth + 1))
            if len(images) > 16:
                raise AdmissionError("image_sequence_limit")
        return images
    if isinstance(value, dict):
        # HF Image may contain a local/remote path. Never dereference it.
        value = value.get("bytes")
        if value is None:
            raise AdmissionError("external_image_reference_not_allowed")
        return embedded_images(value, depth + 1)
    if not isinstance(value, bytes) or not value:
        raise AdmissionError("missing_embedded_image")
    if len(value) > MAX_IMAGE_BYTES:
        raise AdmissionError("image_size_limit")
    return [value]


def probe_bytes(content: bytes) -> dict:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        extension, driver, media_type = ".png", "PNG", "image/png"
    elif content.startswith(b"\xff\xd8\xff"):
        extension, driver, media_type = ".jpg", "JPEG", "image/jpeg"
    elif content[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        extension, driver, media_type = ".tif", "GTiff", "image/tiff"
    else:
        raise AdmissionError("unsupported_image_encoding")
    try:
        with MemoryFile(content) as memory, memory.open(driver=driver) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS or not 1 <= image.count <= 4:
                raise AdmissionError("provider_image_dimensions_limit")
            spatial = None
            if image.crs:
                bbox = list(transform_bounds(image.crs, "EPSG:4326", *image.bounds))
                if not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
                    raise AdmissionError("invalid_geographic_extent")
                spatial = {"native_crs": str(image.crs), "bbox_wgs84": bbox}
            # Bound peak array allocation while checking all image pixels.
            for band in range(1, image.count + 1):
                image.read(band)
            return {"width": image.width, "height": image.height, "bands": image.count,
                    "spatial": spatial, "reference_kind": "geographic" if spatial else "pixel",
                    "extension": extension, "media_type": media_type}
    except rasterio.errors.RasterioError:
        raise AdmissionError("image_decode_failed") from None


def extract_samples(spec: dict, output: Path, sample_count: int = 3,
                    max_shards: int = 4, max_rows: int = 128,
                    max_output_bytes: int = 512 * 1024 * 1024) -> dict:
    """Operator-controlled sample admission, not dataset-wide or license approval."""
    import pyarrow as pa
    if (not 1 <= sample_count <= 100 or not 1 <= max_shards <= 100
            or not 1 <= max_rows <= 10000 or max_output_bytes <= 0):
        raise AdmissionError("invalid_limits")
    column = spec["image_column"]
    validate_column(column)
    if spec.get("reviewed_role") != "input_image" or not spec.get("review_id"):
        raise AdmissionError("image_column_review_required")
    root = Path(spec["root"])
    if root.is_symlink() or not root.is_dir():
        raise AdmissionError("invalid_dataset_root")
    root = root.resolve()
    if output.exists():
        raise AdmissionError("output_exists_preserve_previous_run")
    if output.resolve().is_relative_to(root):
        raise AdmissionError("output_must_not_modify_source_root")
    shards = sorted([*root.rglob("*.arrow"), *root.rglob("*.parquet")])
    if not shards:
        raise AdmissionError("no_packed_shards")
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "private").mkdir(mode=0o700)
    samples, origins, failures, sources = [], [], [], []
    seen = set()
    used = duplicates = inspected = 0
    for candidate in shards[:max_shards]:
        if len(samples) >= sample_count:
            break
        try:
            path = checked_shard(root, candidate)
            before = path.stat()
            source_hash = digest_file(path)
            sources.append({"relative_path": str(path.relative_to(root)), "sha256": source_hash,
                            "size_bytes": before.st_size})
            rows = projected_rows(path, column, max_rows)
            try:
                for row, cell in rows:
                    inspected += 1
                    try:
                        values = embedded_images(cell)
                    except AdmissionError as error:
                        failures.append({"shard": str(path.relative_to(root)), "row": row, "code": str(error)})
                        continue
                    for image_index, content in enumerate(values):
                        digest = hashlib.sha256(content).hexdigest()
                        origin = {"asset_id": "asset-" + digest, "source_sha256": source_hash,
                                  "shard": str(path.relative_to(root)), "column": column,
                                  "row": row, "image_index": image_index}
                        if digest in seen:
                            duplicates += 1
                            origins.append(origin)
                            continue
                        try:
                            probe = probe_bytes(content)
                            if used + len(content) > max_output_bytes:
                                raise AdmissionError("output_byte_limit")
                        except AdmissionError as error:
                            failures.append({**origin, "code": str(error)})
                            continue
                        filename = digest + probe.pop("extension")
                        with (output / "inputs" / filename).open("xb") as stream:
                            stream.write(content)
                        if digest_file(output / "inputs" / filename) != digest:
                            raise AdmissionError("staged_checksum_mismatch")
                        used += len(content)
                        seen.add(digest)
                        origins.append(origin)
                        samples.append({**probe, "asset_id": "asset-" + digest, "sha256": digest,
                            "relative_path": filename, "size_bytes": len(content),
                            "source_snapshot_hash": source_hash, "status": "readable_sample"})
                        if len(samples) >= sample_count:
                            break
                    if len(samples) >= sample_count:
                        break
            finally:
                rows.close()
                # Also fail closed if parsing fails after some images were staged.
                try:
                    after = path.stat()
                except OSError:
                    raise AdmissionError("source_changed_during_read") from None
                if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                    raise AdmissionError("source_changed_during_read")
        except (pa.ArrowException, OSError, AdmissionError) as error:
            # Do not copy parser messages, which may contain private row data.
            code = str(error) if isinstance(error, AdmissionError) else type(error).__name__
            failures.append({"shard": str(candidate.relative_to(root)), "code": code})
            if code in {"source_changed_during_read", "staged_checksum_mismatch"}:
                raise AdmissionError(code) from None
    report = {"schema_version": "packed-image-samples-1", "scope": "reviewed-local-input-samples-only",
        "datasets": [{"dataset_id": spec["dataset_id"], "root": str((output / "inputs").resolve()),
                      "samples": samples, "status": "sampled_readable" if len(samples) == sample_count else "partial"}],
        "sampling": "ordered shards and rows, distinct content; not a representative benchmark sample",
        "unique_images": len(samples), "duplicate_images": duplicates, "inspected_rows": inspected,
        "output_bytes": used, "shards_total": len(shards), "shards_read": len(sources),
        "failures": failures, "label_columns_exported": False}
    receipt = {"dataset_id": spec["dataset_id"], "source_root": str(root), "review_id": spec["review_id"],
               "image_column": column, "sources": sources, "origins": origins,
               "pyarrow_version": pa.__version__, "license_status": spec["license"]}
    manifest = {s["asset_id"]: {"role": "input_image", "filename": s["relative_path"], "sha256": s["sha256"]} for s in samples}
    metadata = [(output / "private/receipt.json", receipt), (output / "coverage.json", report),
                (output / "inputs.json", manifest)]
    payloads = [(path, (json.dumps(value, indent=2) + "\n").encode()) for path, value in metadata]
    if sum(len(value) for _, value in payloads) > MAX_RECEIPT_BYTES:
        raise AdmissionError("receipt_size_limit")
    for path, value in payloads:
        path.write_bytes(value)
    return report

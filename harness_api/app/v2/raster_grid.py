"""Operator-only SCL nearest-neighbor kernel; not yet an Agent tool.

Both inputs are reviewed assets, never user-supplied grids or URLs. The
reference supplies geometry only, not its radiometric validity mask.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .raster_math import MAX_INPUT, MAX_OUTPUT, NativeBand
from .schemas import Identifier, Sha256, UtcTimestamp, V2RequestModel

VERSION = "1.0.0"
TOOL_ID = "raster.resample"
NODATA = 255


class GridArguments(V2RequestModel):
    source_asset_id: Identifier
    reference_asset_id: Identifier
    method: Literal["nearest"]

    @model_validator(mode="after")
    def distinct(self):
        if self.source_asset_id == self.reference_asset_id:
            raise ValueError("distinct source and reference required")
        return self


class NativeSCL(NativeBand):
    band: Literal["scl"]
    dtype: Literal["uint8"]
    scale: Literal[1.0]
    offset: Literal[0.0]
    nodata: Literal[0.0]


class GridResult(V2RequestModel):
    operation: Literal["scl-to-reference-grid"]
    method: Literal["nearest"]
    input_asset_ids: list[Identifier] = Field(min_length=2, max_length=2)
    input_sha256: list[Sha256] = Field(min_length=2, max_length=2)
    acquired: UtcTimestamp
    crs: str
    transform: list[float] = Field(min_length=6, max_length=6)
    bbox_wgs84: list[float] = Field(min_length=4, max_length=4)
    width: int = Field(gt=0, le=1024)
    height: int = Field(gt=0, le=1024)
    dtype: Literal["uint8"]
    nodata: Literal[255]
    valid_pixels: int = Field(ge=0)
    total_pixels: int = Field(gt=0, le=1024 * 1024)
    valid_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    class_counts: list[int] = Field(min_length=12, max_length=12)
    invalid_policy: Literal["source-mask-or-scl-zero-or-outside-source"]
    reference_policy: Literal["geometry-only-ignore-reference-values-and-mask"]
    cloud_mask_applied: Literal[False]

    @model_validator(mode="after")
    def consistent(self):
        if len(set(self.input_asset_ids)) != 2:
            raise ValueError("distinct source and reference required")
        if self.total_pixels != self.width * self.height or self.valid_pixels > self.total_pixels:
            raise ValueError("pixel counts disagree")
        if (self.class_counts[0] != 0 or any(v < 0 for v in self.class_counts)
                or sum(self.class_counts) != self.valid_pixels
                or self.valid_fraction != self.valid_pixels / self.total_pixels):
            raise ValueError("class counts or coverage disagree")
        if not all(math.isfinite(v) for v in self.transform + self.bbox_wgs84):
            raise ValueError("finite grid required")
        a, b, _, d, e, _ = self.transform
        if a <= 0 or e >= 0 or b != 0 or d != 0:
            raise ValueError("north-up grid required")
        if (self.crs[:8] not in {"EPSG:326", "EPSG:327"} or len(self.crs) != 10
                or not self.crs[-2:].isdigit() or not 1 <= int(self.crs[-2:]) <= 60):
            raise ValueError("reviewed UTM CRS required")
        west, south, east, north = self.bbox_wgs84
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("invalid geographic footprint")
        return self


def checked_grids(source: NativeSCL, reference: NativeBand) -> tuple[NativeSCL, NativeBand]:
    # Revalidate also model_copy/update inputs; callers cannot bypass admission.
    source = NativeSCL.model_validate(source.model_dump(mode="json"))
    reference = NativeBand.model_validate(reference.model_dump(mode="json"))
    if source.asset_id == reference.asset_id:
        raise ValueError("distinct assets required")
    if source.item_id != reference.item_id or source.acquired != reference.acquired:
        raise ValueError("same scene and acquisition required; no temporal alignment")
    return source, reference


def _read_checked(path: Path, profile: NativeBand, *, pixels: bool):
    from rasterio.io import MemoryFile

    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT:
        raise ValueError("bounded regular input required")
    with path.open("rb") as stream:
        content = stream.read(MAX_INPUT + 1)
    if (len(content) > MAX_INPUT or hashlib.sha256(content).hexdigest() != profile.sha256
            or content[:4] not in {b"II*\x00", b"MM\x00*"}):
        raise ValueError("input checksum or TIFF mismatch")
    with MemoryFile(content) as memory, memory.open(driver="GTiff") as image:
        if ((image.width, image.height, image.count, image.dtypes) != (profile.width, profile.height, 1, (profile.dtype,))
                or image.crs is None or image.crs.to_string() != profile.crs
                or list(image.transform)[:6] != profile.transform or image.nodata != profile.nodata
                or image.scales != (profile.scale,) or image.offsets != (profile.offset,)):
            raise ValueError("input differs from reviewed native metadata")
        return (image.read(1), image.read_masks(1) > 0) if pixels else None


def class_summary(values, valid) -> dict:
    import numpy as np

    return {"valid_pixels": int(valid.sum()), "total_pixels": int(valid.size),
            "valid_fraction": int(valid.sum()) / int(valid.size),
            "class_counts": np.bincount(values[valid], minlength=12).tolist()}


def validate_grid(content: bytes, result: GridResult) -> None:
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds

    result = GridResult.model_validate(result.model_dump(mode="json"))
    if len(content) > MAX_OUTPUT or content[:4] not in {b"II*\x00", b"MM\x00*"}:
        raise ValueError("bounded classic TIFF required")
    with rasterio.Env(GDAL_PAM_ENABLED="NO", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), MemoryFile(content) as memory:
        with memory.open(driver="GTiff") as image:
            if ((image.width, image.height, image.count, image.dtypes) != (result.width, result.height, 1, ("uint8",))
                    or image.crs is None or image.crs.to_string() != result.crs
                    or list(image.transform)[:6] != result.transform or image.nodata != NODATA
                    or image.scales != (1.,) or image.offsets != (0.,)):
                raise ValueError("categorical output profile mismatch")
            bbox = list(transform_bounds(image.crs, "EPSG:4326", *image.bounds, densify_pts=21))
            if not np.allclose(bbox, result.bbox_wgs84, rtol=0, atol=1e-10):
                raise ValueError("categorical output footprint mismatch")
            values, valid = image.read(1), image.read_masks(1) > 0
            if np.any(values[~valid] != NODATA) or np.any((values[valid] < 1) | (values[valid] > 11)):
                raise ValueError("categorical class or nodata mismatch")
            if class_summary(values, valid) != {key: getattr(result, key) for key in class_summary(values, valid)}:
                raise ValueError("class histogram differs from pixels")


def compute_grid(source_path: Path, reference_path: Path, source: NativeSCL,
                 reference: NativeBand) -> tuple[bytes, GridResult]:
    """Explicit nearest warp to a pinned asset grid, preserving SCL class codes."""
    import numpy as np
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.transform import array_bounds
    from rasterio.warp import reproject, transform_bounds

    source, reference = checked_grids(source, reference)
    with rasterio.Env(GDAL_PAM_ENABLED="NO", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      GDAL_NUM_THREADS="1", GDAL_TIFF_INTERNAL_MASK=True):
        values, mask = _read_checked(source_path, source, pixels=True)
        _read_checked(reference_path, reference, pixels=False)
        if np.any(values > 11):
            raise ValueError("SCL outside reviewed class domain 0..11")
        valid_source = mask & (values != 0)
        source_values = np.where(valid_source, values, NODATA).astype("uint8")
        output_values = np.full((reference.height, reference.width), NODATA, dtype="uint8")
        target_transform = Affine(*reference.transform)
        reproject(source=source_values, destination=output_values,
                  src_transform=Affine(*source.transform), src_crs=source.crs, src_nodata=NODATA,
                  dst_transform=target_transform, dst_crs=reference.crs, dst_nodata=NODATA,
                  resampling=Resampling.nearest, num_threads=1, warp_mem_limit=32,
                  init_dest_nodata=True, ERROR_THRESHOLD=0.0)
        valid = output_values != NODATA
        bbox = list(transform_bounds(reference.crs, "EPSG:4326",
                    *array_bounds(reference.height, reference.width, target_transform), densify_pts=21))
        result = GridResult(operation="scl-to-reference-grid", method="nearest",
            input_asset_ids=[source.asset_id, reference.asset_id], input_sha256=[source.sha256, reference.sha256],
            acquired=source.acquired, crs=reference.crs, transform=reference.transform, bbox_wgs84=bbox,
            width=reference.width, height=reference.height, dtype="uint8", nodata=NODATA,
            invalid_policy="source-mask-or-scl-zero-or-outside-source",
            reference_policy="geometry-only-ignore-reference-values-and-mask", cloud_mask_applied=False,
            **class_summary(output_values, valid))
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=reference.width, height=reference.height, count=1,
                             dtype="uint8", crs=reference.crs, transform=target_transform, nodata=NODATA,
                             compress="deflate") as output:
                output.write(output_values, 1)
                output.write_mask(valid.astype("uint8") * 255)
                output.update_tags(operation=result.operation, method="nearest", version=VERSION,
                    cloud_mask_applied="false", reference_policy=result.reference_policy,
                    attribution=f"Contains modified Copernicus Sentinel data ({source.acquired[:4]})")
            content = memory.read()
    validate_grid(content, result)
    return content, result

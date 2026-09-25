"""Deterministic temporal pair selection over operator-reviewed EO inputs."""
from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .raster_grid import NativeSCL
from .raster_math import CLOUD_EXCLUDED_CLASSES, CLOUD_POLICY, NativeBand
from .schemas import (
    Sha256,
    Identifier,
    SpatialBoundingBox,
    TemporalExtent,
    UtcTimestamp,
    V2RequestModel,
)


TOOL_ID = "temporal.select_align"
VERSION = "1.0.0"
MEDIA_TYPE = "image/tiff"
MAX_OUTPUT = 64 * 1024 * 1024
MAX_CANDIDATES = 32
STACK_NODATA = 65535
STACK_BANDS = ["before_red", "before_scl", "after_red", "after_scl"]


class TemporalSelectAlignArguments(V2RequestModel):
    operation: Literal["select_align"]
    before: TemporalExtent
    after: TemporalExtent
    aoi: SpatialBoundingBox
    band: Literal["red"]
    minimum_coverage_fraction: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    maximum_cloud_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    cloud_policy: Literal[
        "sentinel-2-scl-cloud-shadow-cirrus-snow-saturation-v1"
    ]

    @model_validator(mode="after")
    def ordered_windows(self) -> "TemporalSelectAlignArguments":
        if _instant(self.before.end) >= _instant(self.after.start):
            raise ValueError("before window must end before after window starts")
        return self


class TemporalInputProfile(V2RequestModel):
    item_id: Identifier
    acquired: UtcTimestamp
    platform: Identifier
    instrument: Identifier
    red: NativeBand
    scl: NativeSCL
    bbox_wgs84: list[float] = Field(min_length=4, max_length=4)
    cloud_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def consistent(self) -> "TemporalInputProfile":
        west, south, east, north = self.bbox_wgs84
        if not all(math.isfinite(value) for value in self.bbox_wgs84):
            raise ValueError("candidate footprint must be finite")
        if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
            raise ValueError("candidate footprint is invalid")
        if (
            self.red.item_id != self.item_id
            or self.scl.item_id != self.item_id
            or self.red.acquired != self.acquired
            or self.scl.acquired != self.acquired
            or self.red.band != "red"
            or self.scl.band != "scl"
        ):
            raise ValueError("candidate native profiles disagree")
        return self


class TemporalSelectedInput(V2RequestModel):
    item_id: Identifier
    acquired: UtcTimestamp
    platform: Identifier
    instrument: Identifier
    red_asset_id: Identifier
    scl_asset_id: Identifier
    coverage_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    cloud_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class TemporalSelectionResult(V2RequestModel):
    operation: Literal["select_align"]
    status: Literal["selected", "rejected"]
    reason: Literal[
        "selected",
        "wrong_date",
        "insufficient_coverage",
        "cloudy",
        "sensor_mismatch",
    ]
    before: TemporalSelectedInput | None = None
    after: TemporalSelectedInput | None = None
    considered_item_ids: list[Identifier] = Field(max_length=MAX_CANDIDATES)
    minimum_coverage_fraction: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    maximum_cloud_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    cloud_policy: Literal[
        "sentinel-2-scl-cloud-shadow-cirrus-snow-saturation-v1"
    ]

    @model_validator(mode="after")
    def outcome_consistent(self) -> "TemporalSelectionResult":
        selected = self.before is not None and self.after is not None
        if (self.status == "selected") != selected:
            raise ValueError("selected outcome requires both temporal inputs")
        if (self.reason == "selected") != selected:
            raise ValueError("selected reason disagrees with temporal inputs")
        return self


class TemporalAlignRequest(V2RequestModel):
    before_red_asset_id: Identifier
    before_scl_asset_id: Identifier
    after_red_asset_id: Identifier
    after_scl_asset_id: Identifier
    cloud_policy: Literal[
        "sentinel-2-scl-cloud-shadow-cirrus-snow-saturation-v1"
    ]

    @model_validator(mode="after")
    def distinct(self) -> "TemporalAlignRequest":
        values = [
            self.before_red_asset_id,
            self.before_scl_asset_id,
            self.after_red_asset_id,
            self.after_scl_asset_id,
        ]
        if len(set(values)) != 4:
            raise ValueError("four distinct temporal input assets required")
        return self


class TemporalStackResult(V2RequestModel):
    operation: Literal["two-date-red-scl-stack"]
    input_asset_ids: list[Identifier] = Field(min_length=4, max_length=4)
    input_sha256: list[Sha256] = Field(min_length=4, max_length=4)
    before_item_id: Identifier
    after_item_id: Identifier
    before_acquired: UtcTimestamp
    after_acquired: UtcTimestamp
    crs: str
    transform: list[float] = Field(min_length=6, max_length=6)
    bbox_wgs84: list[float] = Field(min_length=4, max_length=4)
    width: int = Field(gt=0, le=1024)
    height: int = Field(gt=0, le=1024)
    count: Literal[4]
    dtype: Literal["uint16"]
    nodata: Literal[65535]
    band_order: list[str] = Field(min_length=4, max_length=4)
    scales: list[float] = Field(min_length=4, max_length=4)
    offsets: list[float] = Field(min_length=4, max_length=4)
    before_valid_pixels: int = Field(ge=0)
    after_valid_pixels: int = Field(ge=0)
    aligned_valid_pixels: int = Field(ge=0)
    total_pixels: int = Field(gt=0, le=1024 * 1024)
    before_coverage_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    after_coverage_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    aligned_coverage_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    before_cloud_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    after_cloud_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    cloud_policy: Literal[
        "sentinel-2-scl-cloud-shadow-cirrus-snow-saturation-v1"
    ]
    alignment_method: Literal["exact-red-grid-and-nearest-scl"]
    invalid_policy: Literal[
        "source-mask-or-nodata-or-scl-zero-or-outside-source-or-cross-date-gap"
    ]

    @model_validator(mode="after")
    def consistent(self) -> "TemporalStackResult":
        if self.band_order != STACK_BANDS:
            raise ValueError("temporal stack band order is fixed")
        if len(set(self.input_asset_ids)) != 4:
            raise ValueError("temporal stack inputs must be distinct")
        if _instant(self.before_acquired) >= _instant(self.after_acquired):
            raise ValueError("temporal stack acquisitions are not ordered")
        if self.total_pixels != self.width * self.height:
            raise ValueError("temporal stack pixel count disagrees")
        if not (
            self.aligned_valid_pixels <= self.before_valid_pixels <= self.total_pixels
            and self.aligned_valid_pixels <= self.after_valid_pixels <= self.total_pixels
        ):
            raise ValueError("temporal stack valid counts disagree")
        expected = (
            self.before_valid_pixels / self.total_pixels,
            self.after_valid_pixels / self.total_pixels,
            self.aligned_valid_pixels / self.total_pixels,
        )
        actual = (
            self.before_coverage_fraction,
            self.after_coverage_fraction,
            self.aligned_coverage_fraction,
        )
        if expected != actual:
            raise ValueError("temporal stack coverage fractions disagree")
        if not all(math.isfinite(value) for value in self.transform + self.bbox_wgs84):
            raise ValueError("temporal stack grid must be finite")
        return self


class TemporalToolResult(V2RequestModel):
    selection: TemporalSelectionResult
    stack: TemporalStackResult | None = None

    @model_validator(mode="after")
    def outcome_consistent(self) -> "TemporalToolResult":
        if (self.selection.status == "selected") != (self.stack is not None):
            raise ValueError("selected temporal result requires stack metadata")
        return self


def _instant(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")


def _within(value: str, extent: TemporalExtent) -> bool:
    instant = _instant(value)
    return _instant(extent.start) <= instant <= _instant(extent.end)


def coverage_fraction(candidate: list[float], requested: SpatialBoundingBox) -> float:
    """Spherical lon/lat rectangle coverage of the requested AOI."""
    west, south, east, north = candidate
    intersection_west = max(west, requested.west)
    intersection_south = max(south, requested.south)
    intersection_east = min(east, requested.east)
    intersection_north = min(north, requested.north)
    if intersection_west >= intersection_east or intersection_south >= intersection_north:
        return 0.0

    def area(left: float, bottom: float, right: float, top: float) -> float:
        return math.radians(right - left) * (
            math.sin(math.radians(top)) - math.sin(math.radians(bottom))
        )

    requested_area = area(
        requested.west,
        requested.south,
        requested.east,
        requested.north,
    )
    intersection_area = area(
        intersection_west,
        intersection_south,
        intersection_east,
        intersection_north,
    )
    return min(1.0, max(0.0, intersection_area / requested_area))


def _selected(candidate: TemporalInputProfile, coverage: float) -> TemporalSelectedInput:
    return TemporalSelectedInput(
        item_id=candidate.item_id,
        acquired=candidate.acquired,
        platform=candidate.platform,
        instrument=candidate.instrument,
        red_asset_id=candidate.red.asset_id,
        scl_asset_id=candidate.scl.asset_id,
        coverage_fraction=coverage,
        cloud_fraction=candidate.cloud_fraction,
    )


def select_temporal_pair(
    arguments: TemporalSelectAlignArguments,
    candidates: list[TemporalInputProfile],
) -> TemporalSelectionResult:
    arguments = TemporalSelectAlignArguments.model_validate(
        arguments.model_dump(mode="json")
    )
    candidates = [
        TemporalInputProfile.model_validate(candidate.model_dump(mode="json"))
        for candidate in candidates
    ]
    if not candidates or len(candidates) > MAX_CANDIDATES:
        raise ValueError("one to 32 reviewed temporal candidates required")
    if len({candidate.item_id for candidate in candidates}) != len(candidates):
        raise ValueError("temporal candidate item IDs must be unique")
    input_ids = [
        asset_id
        for candidate in candidates
        for asset_id in (candidate.red.asset_id, candidate.scl.asset_id)
    ]
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("temporal candidate asset IDs must be unique")

    considered = sorted(candidate.item_id for candidate in candidates)
    common = {
        "operation": "select_align",
        "considered_item_ids": considered,
        "minimum_coverage_fraction": arguments.minimum_coverage_fraction,
        "maximum_cloud_fraction": arguments.maximum_cloud_fraction,
        "cloud_policy": arguments.cloud_policy,
    }

    before_by_date = [candidate for candidate in candidates if _within(candidate.acquired, arguments.before)]
    after_by_date = [candidate for candidate in candidates if _within(candidate.acquired, arguments.after)]
    if not before_by_date or not after_by_date:
        return TemporalSelectionResult(status="rejected", reason="wrong_date", **common)

    def with_coverage(values: list[TemporalInputProfile]):
        return [
            (candidate, coverage_fraction(candidate.bbox_wgs84, arguments.aoi))
            for candidate in values
        ]

    before_coverage = with_coverage(before_by_date)
    after_coverage = with_coverage(after_by_date)
    before_covered = [
        value for value in before_coverage
        if value[1] >= arguments.minimum_coverage_fraction
    ]
    after_covered = [
        value for value in after_coverage
        if value[1] >= arguments.minimum_coverage_fraction
    ]
    if not before_covered or not after_covered:
        return TemporalSelectionResult(
            status="rejected", reason="insufficient_coverage", **common
        )

    before_clear = [
        value for value in before_covered
        if value[0].cloud_fraction <= arguments.maximum_cloud_fraction
    ]
    after_clear = [
        value for value in after_covered
        if value[0].cloud_fraction <= arguments.maximum_cloud_fraction
    ]
    if not before_clear or not after_clear:
        return TemporalSelectionResult(status="rejected", reason="cloudy", **common)

    pairs = [
        (before, after)
        for before in before_clear
        for after in after_clear
        if before[0].platform == after[0].platform
        and before[0].instrument == after[0].instrument
    ]
    if not pairs:
        return TemporalSelectionResult(
            status="rejected", reason="sensor_mismatch", **common
        )
    before, after = min(
        pairs,
        key=lambda pair: (
            pair[0][0].cloud_fraction + pair[1][0].cloud_fraction,
            -min(pair[0][1], pair[1][1]),
            pair[0][0].acquired,
            pair[1][0].acquired,
            pair[0][0].item_id,
            pair[1][0].item_id,
        ),
    )
    return TemporalSelectionResult(
        status="selected",
        reason="selected",
        before=_selected(*before),
        after=_selected(*after),
        **common,
    )


def checked_temporal_inputs(
    before_red: NativeBand,
    before_scl: NativeSCL,
    after_red: NativeBand,
    after_scl: NativeSCL,
) -> tuple[NativeBand, NativeSCL, NativeBand, NativeSCL]:
    from .raster_grid import checked_grids

    before_red = NativeBand.model_validate(before_red.model_dump(mode="json"))
    before_scl = NativeSCL.model_validate(before_scl.model_dump(mode="json"))
    after_red = NativeBand.model_validate(after_red.model_dump(mode="json"))
    after_scl = NativeSCL.model_validate(after_scl.model_dump(mode="json"))
    checked_grids(before_scl, before_red)
    checked_grids(after_scl, after_red)
    for field in ("crs", "transform", "width", "height"):
        if getattr(before_red, field) != getattr(after_red, field):
            raise ValueError("two-date reflectance inputs require one exact grid")
    if _instant(before_red.acquired) >= _instant(after_red.acquired):
        raise ValueError("before acquisition must precede after acquisition")
    return before_red, before_scl, after_red, after_scl


def _read_native(path: Path, profile: NativeBand):
    import numpy as np
    from rasterio.io import MemoryFile

    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("bounded regular temporal input required")
    content = path.read_bytes()
    if (
        hashlib.sha256(content).hexdigest() != profile.sha256
        or content[:4] not in {b"II*\x00", b"MM\x00*"}
    ):
        raise ValueError("temporal input checksum or TIFF type mismatch")
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
            raise ValueError("temporal input differs from reviewed profile")
        return image.read(1), image.read_masks(1) > 0


def _align_scl(path: Path, source: NativeSCL, reference: NativeBand):
    import numpy as np
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    values, valid = _read_native(path, source)
    if np.any(values > 11):
        raise ValueError("SCL outside reviewed class domain")
    source_values = np.where(valid & (values != 0), values, 255).astype("uint8")
    output = np.full((reference.height, reference.width), 255, dtype="uint8")
    reproject(
        source=source_values,
        destination=output,
        src_transform=Affine(*source.transform),
        src_crs=source.crs,
        src_nodata=255,
        dst_transform=Affine(*reference.transform),
        dst_crs=reference.crs,
        dst_nodata=255,
        resampling=Resampling.nearest,
        num_threads=1,
        warp_mem_limit=32,
        init_dest_nodata=True,
        ERROR_THRESHOLD=0.0,
    )
    return output, output != 255


def _cloud_fraction(values, valid) -> float:
    import numpy as np

    count = int(valid.sum())
    return (
        int(np.count_nonzero(valid & np.isin(values, CLOUD_EXCLUDED_CLASSES))) / count
        if count
        else 0.0
    )


def compute_temporal_stack(
    paths: tuple[Path, Path, Path, Path],
    profiles: tuple[NativeBand, NativeSCL, NativeBand, NativeSCL],
) -> tuple[bytes, TemporalStackResult]:
    import numpy as np
    from affine import Affine
    from rasterio.io import MemoryFile
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds

    before_red, before_scl, after_red, after_scl = checked_temporal_inputs(*profiles)
    before_red_values, before_red_valid = _read_native(paths[0], before_red)
    before_scl_values, before_scl_valid = _align_scl(paths[1], before_scl, before_red)
    after_red_values, after_red_valid = _read_native(paths[2], after_red)
    after_scl_values, after_scl_valid = _align_scl(paths[3], after_scl, after_red)
    before_valid = before_red_valid & before_scl_valid
    after_valid = after_red_valid & after_scl_valid
    aligned_valid = before_valid & after_valid
    arrays = [
        before_red_values.astype("uint16", copy=False),
        before_scl_values.astype("uint16", copy=False),
        after_red_values.astype("uint16", copy=False),
        after_scl_values.astype("uint16", copy=False),
    ]
    stacked = np.stack(
        [np.where(aligned_valid, array, STACK_NODATA) for array in arrays]
    ).astype("uint16")
    total = int(aligned_valid.size)
    transform = Affine(*before_red.transform)
    bbox = list(
        transform_bounds(
            before_red.crs,
            "EPSG:4326",
            *array_bounds(before_red.height, before_red.width, transform),
            densify_pts=21,
        )
    )
    result = TemporalStackResult(
        operation="two-date-red-scl-stack",
        input_asset_ids=[
            before_red.asset_id,
            before_scl.asset_id,
            after_red.asset_id,
            after_scl.asset_id,
        ],
        input_sha256=[
            before_red.sha256,
            before_scl.sha256,
            after_red.sha256,
            after_scl.sha256,
        ],
        before_item_id=before_red.item_id,
        after_item_id=after_red.item_id,
        before_acquired=before_red.acquired,
        after_acquired=after_red.acquired,
        crs=before_red.crs,
        transform=before_red.transform,
        bbox_wgs84=bbox,
        width=before_red.width,
        height=before_red.height,
        count=4,
        dtype="uint16",
        nodata=STACK_NODATA,
        band_order=STACK_BANDS,
        scales=[before_red.scale, 1.0, after_red.scale, 1.0],
        offsets=[before_red.offset, 0.0, after_red.offset, 0.0],
        before_valid_pixels=int(before_valid.sum()),
        after_valid_pixels=int(after_valid.sum()),
        aligned_valid_pixels=int(aligned_valid.sum()),
        total_pixels=total,
        before_coverage_fraction=int(before_valid.sum()) / total,
        after_coverage_fraction=int(after_valid.sum()) / total,
        aligned_coverage_fraction=int(aligned_valid.sum()) / total,
        before_cloud_fraction=_cloud_fraction(before_scl_values, before_scl_valid),
        after_cloud_fraction=_cloud_fraction(after_scl_values, after_scl_valid),
        cloud_policy=CLOUD_POLICY,
        alignment_method="exact-red-grid-and-nearest-scl",
        invalid_policy=(
            "source-mask-or-nodata-or-scl-zero-or-outside-source-or-cross-date-gap"
        ),
    )
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=result.width,
            height=result.height,
            count=4,
            dtype="uint16",
            crs=result.crs,
            transform=transform,
            nodata=STACK_NODATA,
            compress="deflate",
        ) as output:
            output.write(stacked)
            output.write_mask(aligned_valid.astype("uint8") * 255)
            output.scales = tuple(result.scales)
            output.offsets = tuple(result.offsets)
            for index, name in enumerate(STACK_BANDS, start=1):
                output.set_band_description(index, name)
            output.update_tags(
                operation=result.operation,
                version=VERSION,
                before_acquired=result.before_acquired,
                after_acquired=result.after_acquired,
                cloud_policy=CLOUD_POLICY,
                alignment_method=result.alignment_method,
                attribution="Contains modified Copernicus Sentinel data (2024)",
            )
        content = memory.read()
    validate_temporal_stack(content, result)
    return content, result


def validate_temporal_stack(content: bytes, result: TemporalStackResult) -> None:
    import numpy as np
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds

    result = TemporalStackResult.model_validate(result.model_dump(mode="json"))
    if len(content) > MAX_OUTPUT or content[:4] not in {b"II*\x00", b"MM\x00*"}:
        raise ValueError("bounded classic temporal TIFF required")
    with MemoryFile(content) as memory, memory.open(driver="GTiff") as image:
        if (
            (image.width, image.height, image.count, image.dtypes)
            != (result.width, result.height, 4, ("uint16",) * 4)
            or image.crs is None
            or image.crs.to_string() != result.crs
            or list(image.transform)[:6] != result.transform
            or image.nodata != STACK_NODATA
            or list(image.scales) != result.scales
            or list(image.offsets) != result.offsets
            or list(image.descriptions) != result.band_order
        ):
            raise ValueError("temporal stack profile mismatch")
        bbox = list(
            transform_bounds(image.crs, "EPSG:4326", *image.bounds, densify_pts=21)
        )
        if not np.allclose(bbox, result.bbox_wgs84, rtol=0, atol=1e-10):
            raise ValueError("temporal stack footprint mismatch")
        values = image.read()
        valid = image.dataset_mask() > 0
        if (
            np.any(values[:, ~valid] != STACK_NODATA)
            or np.any(values[:, valid] == STACK_NODATA)
            or np.any((values[1, valid] < 1) | (values[1, valid] > 11))
            or np.any((values[3, valid] < 1) | (values[3, valid] > 11))
            or int(valid.sum()) != result.aligned_valid_pixels
        ):
            raise ValueError("temporal stack pixels or mask mismatch")
        tags = image.tags()
        expected_tags = {
            "operation": result.operation,
            "version": VERSION,
            "before_acquired": result.before_acquired,
            "after_acquired": result.after_acquired,
            "cloud_policy": result.cloud_policy,
            "alignment_method": result.alignment_method,
        }
        if any(tags.get(key) != value for key, value in expected_tags.items()):
            raise ValueError("temporal stack tags mismatch")

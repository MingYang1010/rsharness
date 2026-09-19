"""Deterministic temporal pair selection over operator-reviewed EO inputs."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from .raster_grid import NativeSCL
from .raster_math import CLOUD_POLICY, NativeBand
from .schemas import (
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

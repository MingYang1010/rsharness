"""Bounded zonal statistics over one reviewed episode-local raster artifact."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .raster_grid import CONTINUOUS_NODATA
from .raster_math import (NDMI_FORMULA, NDMI_INVALID_POLICY,
                          NDMI_VERSION as BAND_MATH_NDMI_VERSION, MAX_OUTPUT)
from .schemas import ArtifactId, Identifier, Sha256, V2RequestModel

TOOL_ID = "raster.zonal_stats"
VERSION = "1.0.0"
NDMI_VERSION = "1.1.0"
INCLUSION_POLICY = "pixel-centre-in-polygon-boundary-inclusive"
VALIDITY_POLICY = "source-mask-and-finite-and-not-nodata"
MAX_ZONE_VERTICES = 64


def _utm(crs: str) -> bool:
    return (len(crs) == 10 and crs[:8] in {"EPSG:326", "EPSG:327"}
            and crs[-2:].isdigit() and 1 <= int(crs[-2:]) <= 60)


def _orientation(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _between(a: float, b: float, value: float) -> bool:
    return min(a, b) <= value <= max(a, b)


def _segments_intersect(a, b, c, d) -> bool:
    values = (_orientation(a, b, c), _orientation(a, b, d),
              _orientation(c, d, a), _orientation(c, d, b))
    if values[0] * values[1] < 0 and values[2] * values[3] < 0:
        return True
    for value, point, start, end in (
            (values[0], c, a, b), (values[1], d, a, b),
            (values[2], a, c, d), (values[3], b, c, d)):
        if value == 0 and _between(start[0], end[0], point[0]) and _between(
                start[1], end[1], point[1]):
            return True
    return False


class ZoneSpec(V2RequestModel):
    zone_id: Identifier
    crs: str
    coordinates: list[list[float]] = Field(min_length=4,
                                            max_length=MAX_ZONE_VERTICES + 1)
    bbox_wgs84: list[float] = Field(min_length=4, max_length=4)
    inclusion_policy: Literal[INCLUSION_POLICY]

    @model_validator(mode="after")
    def valid_polygon(self):
        if not _utm(self.crs):
            raise ValueError("reviewed UTM zone CRS required")
        points = self.coordinates
        if any(len(point) != 2 or not all(math.isfinite(v) for v in point)
               for point in points):
            raise ValueError("finite two-dimensional zone coordinates required")
        if points[0] != points[-1] or len({tuple(point) for point in points[:-1]}) < 3:
            raise ValueError("closed polygon with three distinct vertices required")
        if any(points[index] == points[index + 1]
               for index in range(len(points) - 1)):
            raise ValueError("duplicate consecutive polygon vertices")
        area = sum(points[index][0] * points[index + 1][1]
                   - points[index + 1][0] * points[index][1]
                   for index in range(len(points) - 1)) / 2
        if not math.isfinite(area) or abs(area) < 1e-6:
            raise ValueError("nonzero finite polygon area required")
        edges = list(zip(points[:-1], points[1:]))
        for left, first in enumerate(edges):
            for right, second in enumerate(edges):
                if right <= left or right in {left - 1, left + 1} or {
                        left, right} == {0, len(edges) - 1}:
                    continue
                if _segments_intersect(*first, *second):
                    raise ValueError("self-intersecting polygon refused")
        west, south, east, north = self.bbox_wgs84
        if (not all(math.isfinite(v) for v in self.bbox_wgs84)
                or not -180 <= west < east <= 180
                or not -90 <= south < north <= 90):
            raise ValueError("valid WGS84 zone bounds required")
        return self


class ZonalArguments(V2RequestModel):
    raster_artifact_id: ArtifactId
    zone_id: Identifier


class ZonalSource(V2RequestModel):
    artifact_id: ArtifactId
    sha256: Sha256
    size_bytes: int = Field(gt=0, le=MAX_OUTPUT)
    crs: str
    transform: list[float] = Field(min_length=6, max_length=6)
    width: int = Field(gt=0, le=1024)
    height: int = Field(gt=0, le=1024)
    dtype: Literal["float32"]
    nodata: Literal[-9999.0]
    lineage_parameters_hash: Sha256
    source_operation: Literal["continuous-to-reference-grid", "ndmi"] = (
        "continuous-to-reference-grid"
    )

    @model_validator(mode="after")
    def valid_grid(self):
        if not _utm(self.crs) or not all(math.isfinite(v) for v in self.transform):
            raise ValueError("reviewed finite UTM grid required")
        a, b, _, d, e, _ = self.transform
        if a <= 0 or e >= 0 or b != 0 or d != 0:
            raise ValueError("north-up grid required")
        return self


class ZonalRequest(V2RequestModel):
    arguments: ZonalArguments
    source: ZonalSource
    zone: ZoneSpec

    @model_validator(mode="after")
    def consistent(self):
        if (self.arguments.raster_artifact_id != self.source.artifact_id
                or self.arguments.zone_id != self.zone.zone_id
                or self.zone.crs != self.source.crs):
            raise ValueError("zonal request identities or CRS disagree")
        left, top = self.source.transform[2], self.source.transform[5]
        right = left + self.source.transform[0] * self.source.width
        bottom = top + self.source.transform[4] * self.source.height
        if any(not left <= point[0] <= right or not bottom <= point[1] <= top
               for point in self.zone.coordinates):
            raise ValueError("zone exceeds source grid")
        return self


class ZonalResult(V2RequestModel):
    operation: Literal["zonal-statistics"]
    source_artifact_id: ArtifactId
    source_sha256: Sha256
    source_lineage_parameters_hash: Sha256
    zone_id: Identifier
    zone_crs: str
    zone_bbox_wgs84: list[float] = Field(min_length=4, max_length=4)
    inclusion_policy: Literal[INCLUSION_POLICY]
    validity_policy: Literal[VALIDITY_POLICY]
    zone_pixels: int = Field(gt=0, le=1024 * 1024)
    valid_pixels: int = Field(ge=0)
    invalid_pixels: int = Field(ge=0)
    valid_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    minimum: float | None = Field(default=None, allow_inf_nan=False)
    maximum: float | None = Field(default=None, allow_inf_nan=False)
    mean: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def consistent(self):
        if (self.valid_pixels + self.invalid_pixels != self.zone_pixels
                or self.valid_fraction != self.valid_pixels / self.zone_pixels):
            raise ValueError("zonal pixel counts disagree")
        values = (self.minimum, self.maximum, self.mean)
        if self.valid_pixels:
            if any(value is None for value in values) or not self.minimum <= self.mean <= self.maximum:
                raise ValueError("valid zone requires ordered statistics")
        elif any(value is not None for value in values):
            raise ValueError("empty valid zone requires null statistics")
        west, south, east, north = self.zone_bbox_wgs84
        if (not _utm(self.zone_crs)
                or not -180 <= west < east <= 180
                or not -90 <= south < north <= 90):
            raise ValueError("invalid zonal geometry metadata")
        return self


def _content(path: Path, source: ZonalSource) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != source.size_bytes:
        raise ValueError("bounded regular zonal source required")
    content = path.read_bytes()
    if (len(content) > MAX_OUTPUT
            or hashlib.sha256(content).hexdigest() != source.sha256
            or content[:4] not in {b"II*\x00", b"MM\x00*"}):
        raise ValueError("zonal source checksum or TIFF mismatch")
    return content


def validate_zonal_source(content: bytes, source: ZonalSource) -> None:
    from rasterio.io import MemoryFile

    if (len(content) != source.size_bytes
            or hashlib.sha256(content).hexdigest() != source.sha256
            or content[:4] not in {b"II*\x00", b"MM\x00*"}):
        raise ValueError("zonal source checksum or TIFF mismatch")
    with MemoryFile(content) as memory, memory.open(driver="GTiff") as image:
        if ((image.width, image.height, image.count, image.dtypes) !=
                (source.width, source.height, 1, (source.dtype,))
                or image.crs is None or image.crs.to_string() != source.crs
                or list(image.transform)[:6] != source.transform
                or image.nodata != source.nodata
                or image.scales != (1.,) or image.offsets != (0.,)):
            raise ValueError("zonal source profile mismatch")
        if source.source_operation == "ndmi":
            tags = image.tags()
            if (tags.get("operation") != "ndmi"
                    or tags.get("version") != BAND_MATH_NDMI_VERSION
                    or tags.get("formula_id") != NDMI_FORMULA
                    or tags.get("invalid_policy") != NDMI_INVALID_POLICY):
                raise ValueError("zonal NDMI provenance tags mismatch")


def validate_zone_bounds(zone: ZoneSpec) -> None:
    import numpy as np
    from rasterio.warp import transform_bounds

    xs = [point[0] for point in zone.coordinates]
    ys = [point[1] for point in zone.coordinates]
    bounds = transform_bounds(
        zone.crs, "EPSG:4326", min(xs), min(ys), max(xs), max(ys),
        densify_pts=21)
    if not np.allclose(bounds, zone.bbox_wgs84, rtol=0, atol=1e-10):
        raise ValueError("zone WGS84 bounds differ from polygon")


def _inside_polygon(xs, ys, coordinates):
    import numpy as np

    inside = np.zeros(xs.shape, dtype=bool)
    boundary = np.zeros(xs.shape, dtype=bool)
    scale = max(1., max(abs(value) for point in coordinates for value in point))
    tolerance = scale * 1e-12
    for first, second in zip(coordinates[:-1], coordinates[1:]):
        x1, y1 = first
        x2, y2 = second
        cross = (xs - x1) * (y2 - y1) - (ys - y1) * (x2 - x1)
        boundary |= ((np.abs(cross) <= tolerance)
                     & (xs >= min(x1, x2) - tolerance)
                     & (xs <= max(x1, x2) + tolerance)
                     & (ys >= min(y1, y2) - tolerance)
                     & (ys <= max(y1, y2) + tolerance))
        crosses = (y1 > ys) != (y2 > ys)
        denominator = y2 - y1
        intersection = x1 + (ys - y1) * (x2 - x1) / (
            denominator if denominator else 1.)
        inside ^= crosses & (xs < intersection)
    return inside | boundary


def compute_zonal_stats(path: Path, request: ZonalRequest) -> ZonalResult:
    import numpy as np
    from rasterio.io import MemoryFile

    content = _content(path, request.source)
    validate_zonal_source(content, request.source)
    validate_zone_bounds(request.zone)
    with MemoryFile(content) as memory, memory.open(driver="GTiff") as image:
        values = image.read(1)
        valid_source = image.read_masks(1) > 0
        rows, columns = np.indices(values.shape, dtype="float64")
        xs = (image.transform.c + image.transform.a * (columns + .5)
              + image.transform.b * (rows + .5))
        ys = (image.transform.f + image.transform.d * (columns + .5)
              + image.transform.e * (rows + .5))
        zone = _inside_polygon(xs, ys, request.zone.coordinates)
        zone_pixels = int(zone.sum())
        if not zone_pixels:
            raise ValueError("zone contains no source pixel centres")
        valid = zone & valid_source & np.isfinite(values)
        if image.nodata is not None:
            valid &= values != image.nodata
        selected = values[valid].astype("float64")
    count = int(selected.size)
    return ZonalResult(
        operation="zonal-statistics",
        source_artifact_id=request.source.artifact_id,
        source_sha256=request.source.sha256,
        source_lineage_parameters_hash=request.source.lineage_parameters_hash,
        zone_id=request.zone.zone_id,
        zone_crs=request.zone.crs,
        zone_bbox_wgs84=request.zone.bbox_wgs84,
        inclusion_policy=INCLUSION_POLICY,
        validity_policy=VALIDITY_POLICY,
        zone_pixels=zone_pixels,
        valid_pixels=count,
        invalid_pixels=zone_pixels - count,
        valid_fraction=count / zone_pixels,
        minimum=float(selected.min()) if count else None,
        maximum=float(selected.max()) if count else None,
        mean=float(selected.mean()) if count else None,
    )

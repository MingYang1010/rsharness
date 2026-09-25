"""Deterministic local metadata queries over reviewed episode inputs only.

No filesystem/network access, inventory scan, URI dereference or label projection.
Task manifests and their public metadata remain operator-reviewed trusted input.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import ConfigDict, Field, ValidationError, model_validator

from ..domain import V2DomainError
from ..events import canonical_json
from ..schemas import (Identifier, PixelAssetRef, SpatialBoundingBox, TaskAsset,
                       TaskManifest, ToolInvokeAction, UtcTimestamp, V2RequestModel)
from .runtime import PreparedTool, ToolOutput

MAX_RECORD_BYTES = 8192
MAX_RESULT_BYTES = 65536
PRIVATE_ROLES = {"label", "labels", "gold", "ground_truth", "ground-truth",
                 "annotation", "annotations", "target", "targets", "mask", "answer"}


class TimeFilter(V2RequestModel):
    start: UtcTimestamp
    end: UtcTimestamp

    @model_validator(mode="after")
    def ordered(self) -> "TimeFilter":
        if datetime.fromisoformat(self.start) > datetime.fromisoformat(self.end):
            raise ValueError("start must not exceed end")
        return self


class QueryBoundingBox(SpatialBoundingBox):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchArguments(V2RequestModel):
    bbox: QueryBoundingBox | None = None
    time_range: TimeFilter | None = None
    platform: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    bands: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(default_factory=list, max_length=32)
    max_cloud_cover_percent: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    limit: int = Field(default=20, ge=1, le=50)
    offset: int = Field(default=0, ge=0, le=1000)


class InspectArguments(V2RequestModel):
    asset_id: Identifier


def _is_public_image(asset: TaskAsset) -> bool:
    roles = {r.strip().lower() for r in asset.roles}
    return asset.media_type.startswith("image/") and not roles.intersection(PRIVATE_ROLES)


def public_metadata(asset: TaskAsset) -> dict[str, Any]:
    # Do not model_dump the whole asset: source/uri/geometry/roles are private.
    result = {name: getattr(asset, name) for name in (
        "asset_id", "media_type", "sha256", "size_bytes", "platform", "instrument",
        "bands", "polarizations", "license", "source_snapshot_hash")}
    result["temporal"] = asset.temporal.model_dump(mode="json") if asset.temporal else None
    result["quality"] = asset.quality.model_dump(mode="json")
    result["spatial"] = ({"crs": asset.spatial.crs, "bbox": asset.spatial.bbox.model_dump(),
                          "gsd_meters": asset.spatial.gsd_meters, "shape": asset.spatial.shape}
                         if asset.spatial else None)
    result["pixel"] = asset.pixel.model_dump() if isinstance(asset, PixelAssetRef) else None
    if len(canonical_json(result).encode("utf-8")) > MAX_RECORD_BYTES:
        raise V2DomainError("catalog_metadata_too_large", "reviewed metadata record exceeds catalog limit", phase="policy")
    return result


def _matches(asset: TaskAsset, query: SearchArguments) -> bool:
    if query.bbox is not None:
        # Current schema uses bounded WGS84 boxes; refuse to reinterpret other CRS.
        if asset.spatial is None or asset.spatial.crs.upper() not in {"EPSG:4326", "OGC:CRS84"}:
            return False
        a, b = asset.spatial.bbox, query.bbox
        if a.east < b.west or a.west > b.east or a.north < b.south or a.south > b.north:
            return False
    if query.time_range is not None:
        if asset.temporal is None:
            return False
        try:
            start, end = datetime.fromisoformat(asset.temporal.start), datetime.fromisoformat(asset.temporal.end)
        except ValueError:
            return False
        if start > end or end < datetime.fromisoformat(query.time_range.start) or start > datetime.fromisoformat(query.time_range.end):
            return False
    if query.platform is not None and asset.platform != query.platform:
        return False
    if not set(query.bands).issubset(asset.bands):
        return False
    if query.max_cloud_cover_percent is not None:
        cloud = asset.quality.cloud_cover_percent
        if cloud is None or cloud > query.max_cloud_cover_percent:
            return False
    return True


class CatalogExecutor:
    tool_ids = ["catalog.search", "catalog.inspect_asset"]
    tool_version = "1.0.0"

    def plan(self, action: ToolInvokeAction, manifest: TaskManifest,
             accessible_asset_refs: list[str]) -> PreparedTool:
        if action.tool_id not in self.tool_ids or action.tool_id not in manifest.scenario.allowed_tools:
            raise V2DomainError("policy_rejected", "tool is not allowed by this task", 403, phase="policy")
        try:
            query = (SearchArguments if action.tool_id == "catalog.search" else InspectArguments).model_validate(action.arguments)
        except (ValidationError, ValueError):
            raise V2DomainError("invalid_tool_arguments", "invalid catalog query; use the documented fields and bounds", phase="request") from None
        allowed = set(manifest.task.inputs).intersection(accessible_asset_refs)
        assets = sorted((a for a in manifest.assets if a.asset_id in allowed and _is_public_image(a)), key=lambda a: a.asset_id)
        if isinstance(query, InspectArguments):
            assets = [a for a in assets if a.asset_id == query.asset_id]
            if not assets:
                # Same error for nonexistent, hidden and label IDs; no enumeration.
                raise V2DomainError("policy_rejected", "asset is not an accessible image input", 403, phase="policy")
        records = [(a, public_metadata(a)) for a in assets]
        scan_bytes = sum(len(canonical_json(record).encode("utf-8")) for _, record in records)
        cost = {"model": "logical-public-metadata-v1", "records_scanned": len(records), "input_bytes": scan_bytes}
        if isinstance(query, SearchArguments):
            matches = [record for a, record in records if _matches(a, query)]
            end = query.offset + query.limit
            metadata = {"assets": matches[query.offset:end], "matched_count": len(matches),
                        "next_offset": end if end < len(matches) else None, "cost": cost}
        else:
            metadata = {"asset": records[0][1], "cost": cost}
        if len(canonical_json(metadata).encode("utf-8")) > MAX_RESULT_BYTES:
            raise V2DomainError("catalog_result_too_large", "catalog result exceeds 64 KiB; reduce page limit", phase="policy")
        output = ToolOutput(None, metadata, scan_bytes)
        return PreparedTool(self.tool_version, scan_bytes, 0, lambda: output, metadata_only=True)

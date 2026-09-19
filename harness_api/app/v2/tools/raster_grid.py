"""Harness policy and lineage for explicit categorical grid alignment."""
from __future__ import annotations

import hashlib

import httpx

from ..artifacts import ArtifactStoreError
from ..domain import V2DomainError
from ..events import sha256_json
from ..raster_grid import (GridArguments, GridResult, NativeSCL, TOOL_ID, VERSION,
                           checked_grids, validate_grid)
from ..raster_math import MAX_INPUT, MAX_OUTPUT, MEDIA_TYPE, NativeBand
from ..schemas import ArtifactLineage, SpatialBoundingBox, SpatialExtent, TemporalExtent
from .catalog import _is_public_image
from .runtime import PreparedTool, ToolOutput


class RasterGridExecutor:
    tool_id = TOOL_ID
    tool_version = VERSION

    def __init__(self, base_url, artifacts, transport=None):
        self.base_url, self.artifacts, self.transport = base_url, artifacts, transport

    def plan(self, action, manifest, accessible_asset_refs):
        if action.tool_id != TOOL_ID or TOOL_ID not in manifest.scenario.allowed_tools:
            raise V2DomainError("policy_rejected", "grid tool not allowed", 403, phase="policy")
        try:
            args = GridArguments.model_validate(action.arguments)
            ids = [args.source_asset_id, args.reference_asset_id]
            if not set(ids).issubset(set(manifest.task.inputs).intersection(accessible_asset_refs)):
                raise V2DomainError("policy_rejected", "grid input not accessible", 403, phase="policy")
            raw = manifest.task.metadata["grid_inputs"]
            source = NativeSCL.model_validate(raw[args.source_asset_id])
            reference = NativeBand.model_validate(raw[args.reference_asset_id])
            checked_grids(source, reference)
            assets = [next(a for a in manifest.assets if a.asset_id == key) for key in ids]
            source_asset, reference_asset = assets
            for asset, profile in zip(assets, (source, reference)):
                if (not _is_public_image(asset) or asset.sha256 != profile.sha256
                        or asset.asset_id != profile.asset_id or asset.size_bytes > MAX_INPUT
                        or asset.spatial is None or asset.temporal is None
                        or asset.temporal.start != profile.acquired or asset.temporal.end != profile.acquired):
                    raise ValueError("profile/asset mismatch")
            if (source_asset.bands != ["scl"] or "scene_classification" not in source_asset.roles
                    or reference_asset.bands != [reference.band]
                    or "reflectance" not in reference_asset.roles):
                raise ValueError("reviewed source/reference roles required")
        except V2DomainError:
            raise
        except (ValueError, KeyError, TypeError, StopIteration):
            raise V2DomainError("invalid_tool_arguments",
                "reviewed same-scene SCL and reference-grid assets required", phase="request") from None
        size = sum(a.size_bytes for a in assets)
        return PreparedTool(VERSION, size, MAX_OUTPUT,
                            lambda: self.invoke(args, source, reference, size))

    def invoke(self, args, source, reference, size):
        try:
            with httpx.Client(base_url=self.base_url, timeout=30, transport=self.transport,
                              trust_env=False, follow_redirects=False) as client:
                with client.stream("POST", "/resample-grid", json=args.model_dump()) as response:
                    response.raise_for_status()
                    if response.status_code != 200 or response.headers.get("X-Raster-Grid-Version") != VERSION:
                        raise ValueError("unsupported raster grid provider")
                    raw = response.headers.get("X-Raster-Grid-Metadata", "")
                    if len(raw) > 16384:
                        raise ValueError("oversized metadata")
                    result = GridResult.model_validate_json(raw)
                    content = bytearray()
                    for block in response.iter_bytes():
                        if len(content) + len(block) > MAX_OUTPUT:
                            raise ValueError("oversized output")
                        content.extend(block)
                    content = bytes(content)
                    if hashlib.sha256(content).hexdigest() != response.headers.get("X-Content-SHA256"):
                        raise ValueError("raster content checksum mismatch")
            if (result.input_asset_ids != [source.asset_id, reference.asset_id]
                    or result.input_sha256 != [source.sha256, reference.sha256]
                    or result.acquired != source.acquired or result.crs != reference.crs
                    or result.transform != reference.transform or result.width != reference.width
                    or result.height != reference.height):
                raise ValueError("grid source or target geometry mismatch")
            validate_grid(content, result)
        except httpx.TimeoutException:
            raise V2DomainError("tool_timeout", "raster provider timed out", 504, True, "tool") from None
        except httpx.HTTPError:
            raise V2DomainError("tool_unavailable", "raster provider failed", 502, True, "tool") from None
        except (ValueError, KeyError, TypeError):
            raise V2DomainError("invalid_tool_output", "categorical grid validation failed", 502,
                                phase="tool") from None
        west, south, east, north = result.bbox_wgs84
        spatial = SpatialExtent(crs="EPSG:4326",
            bbox=SpatialBoundingBox(west=west, south=south, east=east, north=north),
            gsd_meters=reference.transform[0], shape=[reference.height, reference.width, 1])
        try:
            artifact = self.artifacts.put_bytes(content, kind="raster", media_type=MEDIA_TYPE,
                spatial=spatial, temporal=TemporalExtent(start=source.acquired, end=source.acquired),
                lineage=ArtifactLineage(tool_id=TOOL_ID, tool_version=VERSION,
                    input_refs=result.input_asset_ids,
                    parameters_hash=sha256_json({"arguments": args.model_dump(),
                        "grid_inputs": [source.model_dump(mode="json"), reference.model_dump(mode="json")],
                        "invalid_policy": result.invalid_policy,
                        "reference_policy": result.reference_policy})))
        except ArtifactStoreError as error:
            raise V2DomainError(error.code, "categorical artifact storage failed", 503, True,
                                "artifact") from None
        return ToolOutput(artifact, result.model_dump(mode="json"), size)

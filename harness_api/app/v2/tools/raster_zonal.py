"""Episode-local policy and isolated execution for zonal statistics."""
from __future__ import annotations

import httpx

from ..artifact_identity import DERIVATION_SCHEME, validate_derivation_metadata
from ..artifacts import ArtifactStoreError
from ..domain import V2DomainError
from ..events import sha256_json
from ..raster_grid import (CONTINUOUS_VERSION, ContinuousBand, GridArguments,
                           TOOL_ID as GRID_TOOL, checked_continuous_grids)
from ..raster_zonal import (TOOL_ID, VERSION, ZoneSpec, ZonalArguments,
                            ZonalRequest, ZonalResult, ZonalSource,
                            validate_zonal_source, validate_zone_bounds)
from ..raster_math import MAX_OUTPUT, MEDIA_TYPE
from .runtime import PreparedTool, ToolOutput


class RasterZonalExecutor:
    tool_id = TOOL_ID
    tool_version = VERSION
    tool_versions = {VERSION}

    def __init__(self, base_url, artifacts, transport=None):
        self.base_url, self.artifacts, self.transport = base_url, artifacts, transport

    def plan(self, action, manifest, accessible_asset_refs, episode_artifacts=None):
        if action.tool_id != TOOL_ID or TOOL_ID not in manifest.scenario.allowed_tools:
            raise V2DomainError("policy_rejected", "zonal tool not allowed", 403,
                                phase="policy")
        try:
            args = ZonalArguments.model_validate(action.arguments)
            if manifest.task.metadata.get("artifact_identity") != DERIVATION_SCHEME:
                raise ValueError("task does not use derivation artifacts")
            zone = ZoneSpec.model_validate(
                manifest.task.metadata["zonal_inputs"][args.zone_id])
            validate_zone_bounds(zone)
            artifact = (episode_artifacts or {}).get(args.raster_artifact_id)
            if artifact is None:
                raise V2DomainError(
                    "policy_rejected",
                    "raster artifact is not accessible in this episode",
                    403,
                    phase="policy",
                )
            validate_derivation_metadata(artifact.model_dump(mode="json"))
            if (artifact.kind != "raster" or artifact.media_type != MEDIA_TYPE
                    or artifact.size_bytes > MAX_OUTPUT
                    or artifact.spatial is None or artifact.temporal is None
                    or artifact.spatial.crs != "EPSG:4326"
                    or artifact.lineage.tool_id != GRID_TOOL
                    or artifact.lineage.tool_version != CONTINUOUS_VERSION
                    or len(artifact.lineage.input_refs) != 2):
                raise ValueError("continuous raster artifact required")
            source_id, reference_id = artifact.lineage.input_refs
            if not {source_id, reference_id}.issubset(
                    set(manifest.task.inputs).intersection(accessible_asset_refs)):
                raise ValueError("artifact inputs are outside task scope")
            raw = manifest.task.metadata["grid_inputs"]
            source = ContinuousBand.model_validate(raw[source_id])
            reference = ContinuousBand.model_validate(raw[reference_id])
            checked_continuous_grids(source, reference)
            expected_lineage = sha256_json({
                "arguments": GridArguments(
                    source_asset_id=source_id,
                    reference_asset_id=reference_id,
                    method="bilinear",
                ).model_dump(),
                "grid_inputs": [source.model_dump(mode="json"),
                                reference.model_dump(mode="json")],
                "invalid_policy": (
                    "source-mask-or-nodata-or-nonfinite-or-outside-source-"
                    "or-incomplete-bilinear-neighborhood"
                ),
                "reference_policy": "geometry-only-ignore-reference-values-and-mask",
            })
            if artifact.lineage.parameters_hash != expected_lineage:
                raise ValueError("continuous artifact lineage policy mismatch")
            west = artifact.spatial.bbox.west
            south = artifact.spatial.bbox.south
            east = artifact.spatial.bbox.east
            north = artifact.spatial.bbox.north
            zone_west, zone_south, zone_east, zone_north = zone.bbox_wgs84
            if (artifact.spatial.shape != [reference.height, reference.width, 1]
                    or artifact.spatial.gsd_meters != reference.transform[0]
                    or artifact.temporal.start != source.acquired
                    or artifact.temporal.end != source.acquired
                    or not west <= zone_west < zone_east <= east
                    or not south <= zone_south < zone_north <= north):
                raise ValueError("zone or artifact metadata differs from task grid")
            zonal_source = ZonalSource(
                artifact_id=artifact.artifact_id,
                sha256=artifact.sha256,
                size_bytes=artifact.size_bytes,
                crs=reference.crs,
                transform=reference.transform,
                width=reference.width,
                height=reference.height,
                dtype="float32",
                nodata=-9999.,
                lineage_parameters_hash=expected_lineage,
            )
            request = ZonalRequest(arguments=args, source=zonal_source, zone=zone)
        except V2DomainError:
            raise
        except (ValueError, KeyError, TypeError):
            raise V2DomainError(
                "invalid_tool_arguments",
                "reviewed episode-local continuous raster and pinned zone required",
                phase="request",
            ) from None
        return PreparedTool(
            VERSION,
            artifact.size_bytes,
            0,
            lambda: self.invoke(request, artifact),
            metadata_only=True,
        )

    def invoke(self, request: ZonalRequest, artifact):
        try:
            try:
                content = self.artifacts.read_content(artifact).content
            except ArtifactStoreError as error:
                raise V2DomainError(
                    error.code, "zonal source artifact unavailable", 409,
                    phase="artifact") from None
            validate_zonal_source(content, request.source)
            with httpx.Client(
                    base_url=self.base_url,
                    timeout=30,
                    transport=self.transport,
                    trust_env=False,
                    follow_redirects=False) as client:
                with client.stream(
                        "POST",
                        "/zonal-stats",
                        content=content,
                        headers={"Content-Type": MEDIA_TYPE,
                                 "X-Raster-Zonal-Request": request.model_dump_json()},
                ) as response:
                    response.raise_for_status()
                    if (response.status_code != 200
                            or response.headers.get("X-Raster-Zonal-Version") != VERSION):
                        raise ValueError("unsupported zonal provider response")
                    raw = bytearray()
                    for block in response.iter_bytes():
                        if len(raw) + len(block) > 16384:
                            raise ValueError("oversized zonal provider response")
                        raw.extend(block)
                    result = ZonalResult.model_validate_json(bytes(raw))
            if (result.source_artifact_id != request.source.artifact_id
                    or result.source_sha256 != request.source.sha256
                    or result.source_lineage_parameters_hash !=
                    request.source.lineage_parameters_hash
                    or result.zone_id != request.zone.zone_id
                    or result.zone_crs != request.zone.crs
                    or result.zone_bbox_wgs84 != request.zone.bbox_wgs84
                    or result.inclusion_policy != request.zone.inclusion_policy):
                raise ValueError("zonal provider identity mismatch")
        except httpx.TimeoutException:
            raise V2DomainError("tool_timeout", "raster provider timed out", 504,
                                True, "tool") from None
        except httpx.HTTPError:
            raise V2DomainError("tool_unavailable", "raster provider failed", 502,
                                True, "tool") from None
        except (ValueError, KeyError, TypeError):
            raise V2DomainError("invalid_tool_output", "zonal output validation failed",
                                502, phase="tool") from None
        return ToolOutput(None, result.model_dump(mode="json"),
                          artifact.size_bytes)

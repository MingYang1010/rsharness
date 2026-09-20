"""Episode-local policy and isolated execution for zonal statistics."""
from __future__ import annotations

import httpx

from ..artifact_identity import DERIVATION_SCHEME, validate_derivation_metadata
from ..artifacts import ArtifactStoreError
from ..domain import V2DomainError
from ..events import sha256_json
from ..raster_grid import (CONTINUOUS_VERSION, ContinuousBand, GridArguments,
                           TOOL_ID as GRID_TOOL, checked_continuous_grids)
from ..raster_zonal import (NDMI_VERSION, TOOL_ID, VERSION, ZoneSpec,
                            ZonalArguments, ZonalRequest, ZonalResult,
                            ZonalSource, validate_zonal_source,
                            validate_zone_bounds)
from ..raster_math import (NDMI_FORMULA, NDMI_VERSION as BAND_MATH_NDMI_VERSION,
                           AlignedSWIRInput, BandMathArguments, MAX_OUTPUT,
                           MEDIA_TYPE, TOOL_ID as BAND_MATH_TOOL,
                           ndmi_lineage_payload)
from .runtime import PreparedTool, ToolOutput


class RasterZonalExecutor:
    tool_id = TOOL_ID
    tool_version = VERSION
    tool_versions = {VERSION, NDMI_VERSION}

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
                    or artifact.spatial.crs != "EPSG:4326"):
                raise ValueError("reviewed raster artifact required")
            allowed = set(manifest.task.inputs).intersection(accessible_asset_refs)
            raw = manifest.task.metadata["grid_inputs"]
            if (artifact.lineage.tool_id == GRID_TOOL
                    and artifact.lineage.tool_version == CONTINUOUS_VERSION
                    and len(artifact.lineage.input_refs) == 2):
                source_id, reference_id = artifact.lineage.input_refs
                if not {source_id, reference_id}.issubset(allowed):
                    raise ValueError("artifact inputs are outside task scope")
                source = ContinuousBand.model_validate(raw[source_id])
                reference = ContinuousBand.model_validate(raw[reference_id])
                checked_continuous_grids(source, reference)
                expected_lineage = self.grid_lineage(source, reference)
                if artifact.lineage.parameters_hash != expected_lineage:
                    raise ValueError("continuous artifact lineage policy mismatch")
                zonal_version = VERSION
                source_operation = "continuous-to-reference-grid"
            elif (artifact.lineage.tool_id == BAND_MATH_TOOL
                  and artifact.lineage.tool_version == BAND_MATH_NDMI_VERSION
                  and len(artifact.lineage.input_refs) == 2
                  and manifest.task.metadata.get("band_math_formula")
                  == NDMI_FORMULA):
                nir_id, swir_artifact_id = artifact.lineage.input_refs
                swir_artifact = (episode_artifacts or {}).get(swir_artifact_id)
                if swir_artifact is None:
                    raise V2DomainError(
                        "policy_rejected",
                        "NDMI parent artifact is not accessible in this episode",
                        403,
                        phase="policy",
                    )
                validate_derivation_metadata(
                    swir_artifact.model_dump(mode="json"))
                if (swir_artifact.kind != "raster"
                        or swir_artifact.media_type != MEDIA_TYPE
                        or swir_artifact.size_bytes > MAX_OUTPUT
                        or swir_artifact.spatial is None
                        or swir_artifact.temporal is None
                        or swir_artifact.lineage.tool_id != GRID_TOOL
                        or swir_artifact.lineage.tool_version != CONTINUOUS_VERSION
                        or len(swir_artifact.lineage.input_refs) != 2):
                    raise ValueError("reviewed aligned SWIR parent required")
                source_id, reference_id = swir_artifact.lineage.input_refs
                if (nir_id != reference_id
                        or not {source_id, reference_id}.issubset(allowed)):
                    raise ValueError("NDMI lineage is outside task scope")
                source = ContinuousBand.model_validate(raw[source_id])
                reference = ContinuousBand.model_validate(raw[reference_id])
                checked_continuous_grids(source, reference)
                if source.band != "swir16" or reference.band != "nir":
                    raise ValueError("fixed B08/B11 NDMI lineage required")
                if (swir_artifact.spatial.crs != "EPSG:4326"
                        or swir_artifact.spatial.shape
                        != [reference.height, reference.width, 1]
                        or swir_artifact.spatial.gsd_meters
                        != reference.transform[0]
                        or swir_artifact.temporal.start != source.acquired
                        or swir_artifact.temporal.end != source.acquired):
                    raise ValueError("aligned SWIR parent metadata mismatch")
                expected_grid = self.grid_lineage(source, reference)
                if swir_artifact.lineage.parameters_hash != expected_grid:
                    raise ValueError("aligned SWIR parent lineage mismatch")
                swir = AlignedSWIRInput(
                    artifact_id=swir_artifact.artifact_id,
                    sha256=swir_artifact.sha256,
                    size_bytes=swir_artifact.size_bytes,
                    source_asset_id=source.asset_id,
                    source_sha256=source.sha256,
                    reference_asset_id=reference.asset_id,
                    reference_sha256=reference.sha256,
                    acquired=reference.acquired,
                    crs=reference.crs,
                    transform=reference.transform,
                    width=reference.width,
                    height=reference.height,
                    dtype="float32",
                    nodata=-9999.,
                    lineage_parameters_hash=expected_grid,
                )
                formula_args = BandMathArguments(
                    operation="ndmi", nir_asset_id=reference.asset_id,
                    swir_artifact_id=swir_artifact.artifact_id)
                expected_lineage = sha256_json(
                    ndmi_lineage_payload(formula_args, reference, swir))
                if artifact.lineage.parameters_hash != expected_lineage:
                    raise ValueError("NDMI artifact lineage policy mismatch")
                zonal_version = NDMI_VERSION
                source_operation = "ndmi"
            else:
                raise ValueError("unsupported raster artifact lineage")
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
                source_operation=source_operation,
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
            zonal_version,
            artifact.size_bytes,
            0,
            lambda: self.invoke(request, artifact, zonal_version),
            metadata_only=True,
        )

    @staticmethod
    def grid_lineage(source, reference):
        return sha256_json({
            "arguments": GridArguments(
                source_asset_id=source.asset_id,
                reference_asset_id=reference.asset_id,
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

    def invoke(self, request: ZonalRequest, artifact, zonal_version=VERSION):
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
                            or response.headers.get("X-Raster-Zonal-Version")
                            != zonal_version):
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

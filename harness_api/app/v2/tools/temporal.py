"""Harness policy, typed artifact and provider client for two-date alignment."""
from __future__ import annotations

import hashlib

import httpx
from pydantic import ValidationError

from ..artifact_identity import DERIVATION_SCHEME
from ..artifacts import ArtifactStore, ArtifactStoreError
from ..domain import V2DomainError
from ..events import sha256_json
from ..raster_math import CLOUD_POLICY
from ..schemas import (
    ArtifactLineage,
    AssetRef,
    SpatialBoundingBox,
    SpatialExtent,
    TemporalExtent,
    TemporalStackDescriptor,
    TemporalStackMember,
)
from ..temporal import (
    MAX_OUTPUT,
    MEDIA_TYPE,
    TOOL_ID,
    VERSION,
    TemporalAlignRequest,
    TemporalInputProfile,
    TemporalSelectAlignArguments,
    TemporalSelectionResult,
    TemporalStackResult,
    TemporalToolResult,
    select_temporal_pair,
    validate_temporal_stack,
)
from .catalog import _is_public_image
from .runtime import PreparedTool, ToolOutput


def _same_bbox(asset: AssetRef, candidate: TemporalInputProfile) -> bool:
    if asset.spatial is None:
        return False
    actual = asset.spatial.bbox
    return all(
        abs(left - right) <= 1e-9
        for left, right in zip(
            [actual.west, actual.south, actual.east, actual.north],
            candidate.bbox_wgs84,
        )
    )


class TemporalExecutor:
    tool_id = TOOL_ID
    tool_version = VERSION
    max_output_bytes = MAX_OUTPUT

    def __init__(self, base_url: str, artifacts: ArtifactStore, transport=None):
        self.base_url = base_url.rstrip("/")
        self.artifacts = artifacts
        self.transport = transport

    def _candidates(self, manifest, accessible_asset_refs) -> list[TemporalInputProfile]:
        try:
            if (
                manifest.task.metadata.get("artifact_identity") != DERIVATION_SCHEME
                or manifest.task.metadata.get("cloud_mask_policy") != CLOUD_POLICY
            ):
                raise ValueError("task does not opt into reviewed temporal artifacts")
            raw = manifest.task.metadata["temporal_inputs"]
            if not isinstance(raw, list):
                raise ValueError("temporal inputs must be a list")
            candidates = [TemporalInputProfile.model_validate(value) for value in raw]
            assets = {asset.asset_id: asset for asset in manifest.assets}
            allowed = set(manifest.task.inputs).intersection(accessible_asset_refs)
            for candidate in candidates:
                for profile, band, role in (
                    (candidate.red, "red", "reflectance"),
                    (candidate.scl, "scl", "scene_classification"),
                ):
                    asset = assets[profile.asset_id]
                    if (
                        profile.asset_id not in allowed
                        or not _is_public_image(asset)
                        or asset.media_type != MEDIA_TYPE
                        or asset.bands != [band]
                        or role not in asset.roles
                        or asset.sha256 != profile.sha256
                        or asset.temporal is None
                        or asset.temporal.start != profile.acquired
                        or asset.temporal.end != profile.acquired
                        or asset.platform != candidate.platform
                        or asset.instrument != candidate.instrument
                        or not _same_bbox(asset, candidate)
                    ):
                        raise ValueError("temporal candidate asset disagrees with reviewed profile")
                cloud = assets[candidate.scl.asset_id].quality.cloud_cover_percent
                if cloud is None or abs(cloud / 100.0 - candidate.cloud_fraction) > 1e-12:
                    raise ValueError("window cloud fraction disagrees with reviewed SCL profile")
            return candidates
        except (KeyError, TypeError, ValueError, ValidationError):
            raise V2DomainError(
                "invalid_tool_arguments",
                "reviewed temporal red/SCL candidates are required",
                phase="request",
            ) from None

    def plan(self, action, manifest, accessible_asset_refs):
        if action.tool_id != TOOL_ID or TOOL_ID not in manifest.scenario.allowed_tools:
            raise V2DomainError(
                "policy_rejected", "temporal tool not allowed", 403, phase="policy"
            )
        try:
            arguments = TemporalSelectAlignArguments.model_validate(action.arguments)
        except ValidationError:
            raise V2DomainError(
                "invalid_tool_arguments",
                "versioned temporal windows, AOI and thresholds required",
                phase="request",
            ) from None
        candidates = self._candidates(manifest, accessible_asset_refs)
        try:
            selection = select_temporal_pair(arguments, candidates)
        except ValueError:
            raise V2DomainError(
                "invalid_tool_arguments", "temporal candidates are ambiguous", phase="request"
            ) from None
        if selection.status == "rejected":
            metadata = TemporalToolResult(selection=selection).model_dump(mode="json")
            return PreparedTool(
                VERSION,
                0,
                0,
                lambda: ToolOutput(None, metadata, 0),
                metadata_only=True,
            )
        selected_ids = {
            selection.before.red_asset_id,
            selection.before.scl_asset_id,
            selection.after.red_asset_id,
            selection.after.scl_asset_id,
        }
        assets = [asset for asset in manifest.assets if asset.asset_id in selected_ids]
        if len(assets) != 4:
            raise V2DomainError(
                "invalid_tool_arguments", "selected temporal assets are unavailable", phase="request"
            )
        size = sum(asset.size_bytes for asset in assets)
        return PreparedTool(
            VERSION,
            size,
            MAX_OUTPUT,
            lambda: self.invoke(arguments, candidates, selection, size),
        )

    def invoke(self, arguments, candidates, selection, input_bytes):
        by_item = {candidate.item_id: candidate for candidate in candidates}
        before = by_item[selection.before.item_id]
        after = by_item[selection.after.item_id]
        request = TemporalAlignRequest(
            before_red_asset_id=before.red.asset_id,
            before_scl_asset_id=before.scl.asset_id,
            after_red_asset_id=after.red.asset_id,
            after_scl_asset_id=after.scl.asset_id,
            cloud_policy=arguments.cloud_policy,
        )
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=40,
                transport=self.transport,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                with client.stream(
                    "POST", "/temporal-align", json=request.model_dump(mode="json")
                ) as response:
                    response.raise_for_status()
                    if (
                        response.status_code != 200
                        or response.headers.get("X-Temporal-Version") != VERSION
                    ):
                        raise ValueError("unsupported temporal provider")
                    raw = response.headers.get("X-Temporal-Metadata", "")
                    if len(raw) > 32768:
                        raise ValueError("oversized temporal metadata")
                    result = TemporalStackResult.model_validate_json(raw)
                    content = bytearray()
                    for block in response.iter_bytes():
                        if len(content) + len(block) > MAX_OUTPUT:
                            raise ValueError("oversized temporal output")
                        content.extend(block)
                    content = bytes(content)
                    if hashlib.sha256(content).hexdigest() != response.headers.get(
                        "X-Content-SHA256"
                    ):
                        raise ValueError("temporal content checksum mismatch")
            expected_ids = [
                before.red.asset_id,
                before.scl.asset_id,
                after.red.asset_id,
                after.scl.asset_id,
            ]
            expected_hashes = [
                before.red.sha256,
                before.scl.sha256,
                after.red.sha256,
                after.scl.sha256,
            ]
            if (
                result.input_asset_ids != expected_ids
                or result.input_sha256 != expected_hashes
                or result.before_acquired != before.acquired
                or result.after_acquired != after.acquired
                or result.before_cloud_fraction != before.cloud_fraction
                or result.after_cloud_fraction != after.cloud_fraction
                or result.before_coverage_fraction < arguments.minimum_coverage_fraction
                or result.after_coverage_fraction < arguments.minimum_coverage_fraction
                or result.before_cloud_fraction > arguments.maximum_cloud_fraction
                or result.after_cloud_fraction > arguments.maximum_cloud_fraction
            ):
                raise ValueError("temporal provider result disagrees with selection")
            validate_temporal_stack(content, result)
        except httpx.TimeoutException:
            raise V2DomainError(
                "tool_timeout", "temporal provider timed out", 504, True, "tool"
            ) from None
        except httpx.HTTPError:
            raise V2DomainError(
                "tool_unavailable", "temporal provider failed", 502, True, "tool"
            ) from None
        except (KeyError, TypeError, ValueError, ValidationError):
            raise V2DomainError(
                "invalid_tool_output",
                "temporal provider output failed validation",
                502,
                phase="tool",
            ) from None

        west, south, east, north = result.bbox_wgs84
        spatial = SpatialExtent(
            crs="EPSG:4326",
            bbox=SpatialBoundingBox(west=west, south=south, east=east, north=north),
            gsd_meters=result.transform[0],
            shape=[result.height, result.width, 4],
        )
        descriptor = TemporalStackDescriptor(
            before=TemporalStackMember(
                item_id=before.item_id,
                acquired=before.acquired,
                platform=before.platform,
                instrument=before.instrument,
                red_asset_id=before.red.asset_id,
                scl_asset_id=before.scl.asset_id,
                bands=["red", "scl"],
                coverage_fraction=selection.before.coverage_fraction,
                cloud_fraction=result.before_cloud_fraction,
            ),
            after=TemporalStackMember(
                item_id=after.item_id,
                acquired=after.acquired,
                platform=after.platform,
                instrument=after.instrument,
                red_asset_id=after.red.asset_id,
                scl_asset_id=after.scl.asset_id,
                bands=["red", "scl"],
                coverage_fraction=selection.after.coverage_fraction,
                cloud_fraction=result.after_cloud_fraction,
            ),
            grid_crs=result.crs,
            grid_transform=result.transform,
            width=result.width,
            height=result.height,
            band_order=result.band_order,
            cloud_policy=result.cloud_policy,
            alignment_method=result.alignment_method,
        )
        try:
            artifact = self.artifacts.put_bytes(
                content,
                kind="raster",
                media_type=MEDIA_TYPE,
                spatial=spatial,
                temporal=TemporalExtent(
                    start=result.before_acquired, end=result.after_acquired
                ),
                temporal_stack=descriptor,
                lineage=ArtifactLineage(
                    tool_id=TOOL_ID,
                    tool_version=VERSION,
                    input_refs=result.input_asset_ids,
                    parameters_hash=sha256_json(
                        {
                            "arguments": arguments.model_dump(mode="json"),
                            "selection": selection.model_dump(mode="json"),
                            "temporal_inputs": [
                                before.model_dump(mode="json"),
                                after.model_dump(mode="json"),
                            ],
                            "output_policy": {
                                "band_order": result.band_order,
                                "alignment_method": result.alignment_method,
                                "invalid_policy": result.invalid_policy,
                                "cloud_policy": result.cloud_policy,
                            },
                        }
                    ),
                ),
            )
        except ArtifactStoreError as error:
            raise V2DomainError(
                error.code,
                "temporal artifact storage failed",
                503,
                True,
                "artifact",
            ) from None
        metadata = TemporalToolResult(selection=selection, stack=result).model_dump(
            mode="json"
        )
        return ToolOutput(
            artifact,
            metadata,
            input_bytes,
            artifact_observation_type="temporal_stack",
        )

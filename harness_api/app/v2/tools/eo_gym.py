"""Typed client for the isolated EO-Gym provider, never a second agent loop."""
from __future__ import annotations

import hashlib

import httpx
from pydantic import ValidationError

from ...eo_gym_bridge import CropArguments, REVISION
from ..artifacts import ArtifactStore, ArtifactStoreError
from ..domain import V2DomainError
from ..events import sha256_json
from ..schemas import ArtifactLineage, Artifact, PixelExtent, PixelAssetRef, TaskAsset, TaskManifest, ToolInvokeAction
from .runtime import ToolOutput


class EOGymExecutor:
    tool_id = "eo_gym.crop"
    tool_version = "1.1.0"
    max_output_bytes = 64 * 1024 * 1024

    def __init__(self, base_url: str, artifacts: ArtifactStore, transport=None):
        self.base_url = base_url.rstrip("/")
        self.artifacts = artifacts
        self.transport = transport

    def prepare(self, action: ToolInvokeAction, manifest: TaskManifest) -> tuple[CropArguments, TaskAsset]:
        if action.tool_id != self.tool_id or action.tool_id not in manifest.scenario.allowed_tools:
            raise V2DomainError("policy_rejected", "tool is not allowed by this task", 403, phase="policy")
        try:
            arguments = CropArguments.model_validate(action.arguments)
        except ValidationError:
            raise V2DomainError("invalid_tool_arguments", "expected asset_id and normalized aoi", phase="request") from None
        if arguments.asset_id not in manifest.task.inputs:
            raise V2DomainError("policy_rejected", "asset is not a task input", 403, phase="policy")
        asset = next((a for a in manifest.assets if a.asset_id == arguments.asset_id), None)
        if asset is None or any(r in {"label", "labels", "gold", "ground_truth", "annotation"} for r in asset.roles):
            raise V2DomainError("policy_rejected", "asset is not an image input", 403, phase="policy")
        return arguments, asset

    @staticmethod
    def _bounded_response(client: httpx.Client, method: str, url: str, limit: int, **kwargs) -> bytes:
        with client.stream(method, url, **kwargs) as response:
            if response.status_code == 507:
                raise V2DomainError("provider_storage_capacity", "provider temporary cache is full", 507, False, "tool")
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_bytes():
                if len(content) + len(chunk) > limit:
                    raise V2DomainError("tool_output_too_large", "provider response exceeds size limit", phase="tool")
                content.extend(chunk)
            return bytes(content)

    def invoke(self, action: ToolInvokeAction, manifest: TaskManifest) -> ToolOutput:
        import json
        arguments, asset = self.prepare(action, manifest)
        try:
            with httpx.Client(base_url=self.base_url, timeout=30, trust_env=False, follow_redirects=False,
                              transport=self.transport) as client:
                body = self._bounded_response(client, "POST", "/execute", 65536, json={
                    "tool_name": "crop_optical_or_sar_image", "arguments": arguments.model_dump(),
                })
                result = json.loads(body)["output"]
                if result["provider"] != "eo-gym" or result["upstream_revision"] != REVISION or result["simulation"] is not False:
                    raise ValueError("unexpected provider or simulation")
                if result["input_sha256"] != asset.sha256 or result["input_asset_id"] != asset.asset_id:
                    raise ValueError("provider input mismatch")
                if result["aoi_norm"] != list(arguments.aoi):
                    raise ValueError("provider AOI mismatch")
                if isinstance(asset, PixelAssetRef):
                    expected_bbox = [round(v * dimension) for v, dimension in zip(
                        arguments.aoi, (asset.pixel.width, asset.pixel.height, asset.pixel.width, asset.pixel.height))]
                    if result["bbox_px"] != expected_bbox:
                        raise ValueError("provider crop disagrees with declared pixel dimensions")
                    if (result["width"], result["height"]) != (expected_bbox[2] - expected_bbox[0], expected_bbox[3] - expected_bbox[1]):
                        raise ValueError("provider crop shape disagrees with pixel window")
                digest = result["sha256"]
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("invalid artifact hash")
                content = self._bounded_response(client, "GET", "/artifacts/" + digest, self.max_output_bytes)
            if hashlib.sha256(content).hexdigest() != digest or len(content) != result["size_bytes"]:
                raise ValueError("provider artifact checksum or size mismatch")
            if result["media_type"] != "image/png" or not content.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("provider artifact is not PNG")
            from rasterio.io import MemoryFile
            with MemoryFile(content) as memory, memory.open() as image:
                if image.width != result["width"] or image.height != result["height"]:
                    raise ValueError("provider image dimensions mismatch")
                pixel = PixelExtent(coordinate_system="pixel", width=image.width, height=image.height, channels=image.count)
        except httpx.TimeoutException:
            raise V2DomainError("tool_timeout", "EO-Gym provider timed out", 504, True, "tool") from None
        except httpx.HTTPError:
            raise V2DomainError("tool_unavailable", "EO-Gym provider request failed", 502, True, "tool") from None
        except (ValueError, KeyError, TypeError):
            raise V2DomainError("invalid_tool_output", "EO-Gym result failed validation", 502, phase="tool") from None
        try:
            artifact = self.artifacts.put_bytes(content, kind="image", media_type="image/png", pixel=pixel, lineage=ArtifactLineage(
                tool_id=self.tool_id, tool_version=self.tool_version, input_refs=[asset.asset_id],
                parameters_hash=sha256_json({"arguments": arguments.model_dump(), "input_sha256": asset.sha256,
                                            "upstream_revision": REVISION}),
            ))
        except ArtifactStoreError as error:
            if error.code in {"artifact_storage_capacity", "artifact_store_unavailable"}:
                raise V2DomainError(error.code, "artifact storage cannot accept output",
                                    507 if error.code == "artifact_storage_capacity" else 503,
                                    error.code == "artifact_store_unavailable", "artifact") from None
            raise V2DomainError("invalid_tool_output", "EO-Gym image payload failed validation", 502, phase="tool") from None
        # Do not infer crop georeferencing from a PNG. Pixel metadata and lineage
        # are sufficient until a separately verified raster transform is available.
        metadata = {key: result[key] for key in ("width", "height", "bbox_px", "aoi_norm", "input_asset_id", "input_sha256", "upstream_revision")}
        return ToolOutput(artifact=artifact, metadata=metadata, input_bytes=asset.size_bytes)

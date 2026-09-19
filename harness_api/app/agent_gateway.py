"""Separate, fail-closed Agent surface for one operator-provisioned episode.

Deploy on separate front/back networks. Never expose the operator API alongside it.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator
from starlette.types import ASGIApp, Receive, Scope, Send

from .eo_gym_bridge import CropArguments
from .v2.raster_math import (BandMathArguments, MASKED_VERSION as RASTER_MASKED_VERSION,
                             MaskedNDVIResult, NDVIResult, TOOL_ID as RASTER_TOOL,
                             VERSION as RASTER_VERSION, MAX_OUTPUT as MAX_RASTER)
from .v2.raster_grid import (GridArguments, GridResult, TOOL_ID as GRID_TOOL,
                             VERSION as GRID_VERSION)
from .v2.domain import _action_allowed
from .v2.artifact_identity import DERIVATION_SCHEME, LEGACY_SCHEME, validate_derivation_metadata
from .v2.schemas import (Artifact, ArtifactId, AssetQuality, BudgetSpec, EpisodeId,
                         Identifier, MapState, Observation, ObservationId, PixelExtent,
                         SemanticVersion, Sha256, SpatialBoundingBox, StepRequest,
                         TaskManifest, TemporalExtent, V2EpisodeState, V2RequestModel)
from .v2.tools.catalog import InspectArguments, SearchArguments, _is_public_image

MAX_REQUEST = 128 * 1024
MAX_JSON = 2 * 1024 * 1024
MAX_IMAGE = 64 * 1024 * 1024
TOOL_ARGUMENTS = {"catalog.search": SearchArguments, "catalog.inspect_asset": InspectArguments,
                  "eo_gym.crop": CropArguments, RASTER_TOOL: BandMathArguments,
                  GRID_TOOL: GridArguments}
SAFE_CODES = {"state_version_conflict", "episode_closed", "idempotency_conflict", "tool_in_progress",
              "tool_interrupted", "tool_budget_exceeded", "policy_rejected", "invalid_tool_arguments",
              "invalid_evidence", "invalid_answer", "tool_timeout", "tool_unavailable", "tool_failed",
              "provider_storage_capacity", "artifact_storage_capacity", "artifact_store_unavailable",
              "artifact_metadata_conflict", "artifact_content_corrupt", "artifact_content_missing",
              "catalog_metadata_too_large", "catalog_result_too_large"}
ACTION_NAMES = ["tool.invoke", "memory.save_evidence", "answer.submit", "answer.abstain",
                "answer.request_human_review", "map.set_view", "map.pan", "map.zoom",
                "map.layer.set_visibility", "map.layer.set_opacity", "map.time.set_range"]


class PublicTask(V2RequestModel):
    task_id: Identifier
    task_version: SemanticVersion
    prompt: str = Field(min_length=1, max_length=50000)
    answer_schema: dict[str, Any]
    budget: BudgetSpec
    input_asset_refs: list[Identifier] = Field(min_length=1, max_length=1000)
    allowed_actions: list[str]
    allowed_tools: list[str]
    artifact_identity: Literal["content-sha256-v1", "derivation-sha256-v1"] = LEGACY_SCHEME


class AgentBinding(V2RequestModel):
    episode_id: EpisodeId
    task_manifest_hash: Sha256
    token_sha256: Sha256
    task: PublicTask

    @model_validator(mode="after")
    def supported_surface(self):
        if not set(self.task.allowed_tools).issubset(TOOL_ARGUMENTS) or not set(self.task.allowed_actions).issubset(ACTION_NAMES):
            raise ValueError("binding requests unaudited actions or tools")
        return self


def build_binding(manifest: TaskManifest, state: V2EpisodeState, capabilities: dict,
                  token_sha256: str) -> AgentBinding:
    """Trusted operator only. Prompt/schema/public descriptive strings need review."""
    inputs = [a for a in manifest.assets if a.asset_id in manifest.task.inputs]
    if any(not _is_public_image(a) for a in inputs):
        raise ValueError("task inputs contain a non-public image or label role")
    if (state.task_manifest_hash != manifest.task_manifest_hash or
            state.task_ref.task_id != manifest.task.task_id or state.task_ref.task_version != manifest.task.task_version):
        raise ValueError("episode task pin mismatch")
    return AgentBinding(episode_id=state.episode_id, task_manifest_hash=manifest.task_manifest_hash,
        token_sha256=token_sha256, task=PublicTask(
            task_id=manifest.task.task_id, task_version=manifest.task.task_version,
            prompt=manifest.task.prompt, answer_schema=manifest.task.answer_schema, budget=manifest.task.budget,
            input_asset_refs=manifest.task.inputs,
            allowed_actions=[a for a in ACTION_NAMES if a in capabilities["actions"] and _action_allowed(a, manifest.scenario.allowed_actions)],
            allowed_tools=[t for t in TOOL_ARGUMENTS if t in capabilities["tools"] and t in manifest.scenario.allowed_tools],
            artifact_identity=manifest.task.metadata.get("artifact_identity", LEGACY_SCHEME)))


class PublicSpatial(BaseModel):
    crs: str
    bbox: SpatialBoundingBox
    gsd_meters: float | None = None
    shape: list[int] | None = None


class PublicAsset(BaseModel):
    asset_id: Identifier
    media_type: str
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    platform: str | None = None
    instrument: str | None = None
    bands: list[str] = Field(default_factory=list)
    polarizations: list[str] = Field(default_factory=list)
    license: str
    source_snapshot_hash: Sha256 | None = None
    temporal: TemporalExtent | None = None
    quality: AssetQuality
    spatial: PublicSpatial | None = None
    pixel: PixelExtent | None = None


class CatalogCost(BaseModel):
    model: str = Field(pattern=r"^logical-public-metadata-v1$")
    records_scanned: int = Field(ge=0, le=1000)
    input_bytes: int = Field(ge=0, le=8192000)


class CropResult(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    bbox_px: list[int] = Field(min_length=4, max_length=4)
    aoi_norm: list[float] = Field(min_length=4, max_length=4)
    input_asset_id: Identifier
    input_sha256: Sha256
    upstream_revision: str = Field(pattern=r"^[a-f0-9]{40}$")


class GatewayError(Exception):
    def __init__(self, code: str, status: int = 502, retryable: bool = False):
        self.code, self.status, self.retryable = code, status, retryable


class AgentGuard:
    """Hold concurrency/time bounds through the final response byte, not headers."""

    def __init__(self, app: ASGIApp, token_sha256: str):
        self.app, self.token_sha256 = app, token_sha256
        self.slots = asyncio.Semaphore(4)

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or (scope["path"] == "/healthz" and scope["method"] == "GET"):
            return await self.app(scope, receive, send)
        request = Request(scope, receive)
        authorization = request.headers.get("authorization", "")
        token = authorization[7:] if authorization.startswith("Bearer ") else ""
        if len(token) != 64 or not hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), self.token_sha256):
            return await JSONResponse({"error": {"code": "unauthorized"}}, status_code=401)(scope, receive, send)
        if scope.get("query_string"):
            return await JSONResponse({"error": {"code": "query_parameters_not_allowed"}}, status_code=422)(scope, receive, send)
        started = False

        async def protected_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                message["headers"] = [(k, v) for k, v in message.get("headers", []) if k.lower() not in {b"cache-control", b"x-content-type-options"}]
                message["headers"] += [(b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff")]
            await send(message)

        try:
            async with asyncio.timeout(50), self.slots:
                await self.app(scope, receive, protected_send)
        except Exception as error:
            if started:
                # Response headers cannot be replaced; terminate this connection.
                raise RuntimeError("Agent response interrupted") from None
            code, status = ("gateway_timeout", 504) if isinstance(error, TimeoutError) else ("invalid_upstream_response", 502)
            await JSONResponse({"error": {"code": code, "retryable": status == 504}}, status_code=status)(scope, receive, send)


def public_map(value: MapState | None, allowed: set[str]):
    if value is None:
        return None
    result = value.model_dump(mode="json")
    result["layers"] = {key: {**layer, "name": layer["asset_id"]} for key, layer in result["layers"].items()
                        if layer["asset_id"] in allowed}
    return result


def public_state(value: dict, binding: AgentBinding) -> dict:
    state = V2EpisodeState.model_validate(value)
    if (state.episode_id != binding.episode_id or state.task_manifest_hash != binding.task_manifest_hash or
            (state.task_ref.task_id, state.task_ref.task_version) != (binding.task.task_id, binding.task.task_version)):
        raise GatewayError("session_pin_mismatch", 409)
    result = state.model_dump(mode="json", exclude={"evaluation", "map"})
    allowed = set(binding.task.input_asset_refs)
    result["accessible_asset_refs"] = [a for a in state.accessible_asset_refs if a in allowed]
    result["map"] = public_map(state.map, allowed)
    return result


def public_observation(value: dict, binding: AgentBinding) -> dict:
    observation = Observation.model_validate(value)
    allowed = set(binding.task.input_asset_refs)
    items = []
    for item in observation.items:
        if item.type == "asset_metadata" and item.asset_refs:
            items.append({"type": item.type, "asset_refs": [a for a in item.asset_refs if a in allowed]})
        elif item.type == "map_state" and item.inline is not None:
            items.append({"type": item.type, "inline": public_map(MapState.model_validate(item.inline), allowed)})
        elif item.type in {"rendered_view", "raster_chip"} and item.artifact_ref:
            items.append({"type": item.type, "artifact_ref": item.artifact_ref})
        elif item.type == "tool_result" and item.inline is not None:
            raw = item.inline
            tool_id = raw.get("tool_id")
            if tool_id not in binding.task.allowed_tools or tool_id not in TOOL_ARGUMENTS:
                raise GatewayError("unsupported_public_observation")
            common = {"tool_id": tool_id, "tool_version": TypeAdapter(SemanticVersion).validate_python(raw["tool_version"]),
                      "status": raw["status"]}
            if raw["status"] == "failed":
                common["error_code"] = raw.get("error_code") if raw.get("error_code") in SAFE_CODES else "tool_failed"
            elif raw["status"] == "completed":
                if tool_id.startswith("catalog."):
                    def asset(record):
                        parsed = PublicAsset.model_validate(record)
                        if parsed.asset_id not in allowed:
                            raise GatewayError("upstream_scope_mismatch")
                        return parsed.model_dump(mode="json")
                    common["cost"] = CatalogCost.model_validate(raw["cost"]).model_dump()
                    if tool_id == "catalog.inspect_asset":
                        common["asset"] = asset(raw["asset"])
                    else:
                        if not isinstance(raw["assets"], list) or len(raw["assets"]) > 50:
                            raise GatewayError("invalid_upstream_response")
                        common["assets"] = [asset(a) for a in raw["assets"]]
                        for key in ("matched_count", "next_offset"):
                            number = raw[key]
                            if key == "next_offset" and number is None:
                                common[key] = None
                            elif type(number) is int and 0 <= number <= 1000:
                                common[key] = number
                            else:
                                raise GatewayError("invalid_upstream_response")
                elif tool_id == RASTER_TOOL:
                    result_type = MaskedNDVIResult if raw.get("cloud_mask_applied") is True else NDVIResult
                    science = result_type.model_validate({k: raw[k] for k in result_type.model_fields})
                    if not set(science.input_asset_ids).issubset(allowed):
                        raise GatewayError("upstream_scope_mismatch")
                    common.update(science.model_dump(mode="json"))
                elif tool_id == GRID_TOOL:
                    science = GridResult.model_validate({k: raw[k] for k in GridResult.model_fields})
                    if not set(science.input_asset_ids).issubset(allowed):
                        raise GatewayError("upstream_scope_mismatch")
                    common.update(science.model_dump(mode="json"))
                else:
                    crop = CropResult.model_validate(raw)
                    if crop.input_asset_id not in allowed:
                        raise GatewayError("upstream_scope_mismatch")
                    common.update(crop.model_dump(mode="json"))
            else:
                raise GatewayError("invalid_upstream_response")
            items.append({"type": item.type, "inline": common})
        else:
            raise GatewayError("unsupported_public_observation")
    return {"observation_id": observation.observation_id, "sequence": observation.sequence,
            "primary_type": observation.primary_type, "items": items,
            "state_hash": observation.state_hash, "semantic_state_hash": observation.semantic_state_hash,
            "task_manifest_hash": binding.task_manifest_hash}


def create_app(binding: AgentBinding | None = None, base_url: str | None = None, transport=None) -> FastAPI:
    if binding is None:
        path = Path(os.environ["EO_AGENT_BINDING_FILE"])
        if path.stat().st_size > 256 * 1024:
            raise ValueError("binding file exceeds limit")
        binding = AgentBinding.model_validate_json(path.read_bytes())
    base_url = base_url or os.environ["EO_AGENT_BACKEND_URL"]
    url = urlsplit(base_url)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}:
        raise ValueError("backend must be an operator-configured HTTP origin")
    app = FastAPI(title="EO Harness scoped Agent API", docs_url=None, redoc_url=None,
                  openapi_url=None, redirect_slashes=False)
    app.add_middleware(AgentGuard, token_sha256=binding.token_sha256)

    @app.exception_handler(GatewayError)
    async def gateway_error(request, error):
        return JSONResponse({"error": {"code": error.code, "message": "Agent request could not be completed",
                                       "retryable": error.retryable}}, status_code=error.status)

    async def upstream(method: str, path: str, body=None, image=False):
        limit = MAX_IMAGE if image else MAX_JSON
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=40, trust_env=False,
                                         follow_redirects=False, transport=transport) as client:
                async with client.stream(method, path, json=body) as response:
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(content) + len(chunk) > limit:
                            raise GatewayError("upstream_response_too_large")
                        content.extend(chunk)
                    if response.status_code >= 400:
                        try:
                            code = json.loads(content).get("error", {}).get("code")
                        except (ValueError, AttributeError):
                            code = None
                        raise GatewayError(code if code in SAFE_CODES else "upstream_request_rejected",
                                           response.status_code if response.status_code in {403, 404, 409, 422, 429, 503, 504} else 502,
                                           response.status_code in {429, 503, 504})
                    if response.status_code != 200:
                        raise GatewayError("invalid_upstream_status")
            return bytes(content) if image else json.loads(content)["data"]
        except httpx.HTTPError:
            raise GatewayError("upstream_unavailable", 503, True) from None
        except (ValueError, KeyError, TypeError):
            raise GatewayError("invalid_upstream_response") from None

    async def state():
        value = await upstream("GET", f"/v2/episodes/{binding.episode_id}/state")
        if value["episode_id"] != binding.episode_id:
            raise GatewayError("upstream_scope_mismatch")
        return public_state(value["state"], binding)

    async def observation(observation_id):
        TypeAdapter(ObservationId).validate_python(observation_id)
        value = await upstream("GET", f"/v2/episodes/{binding.episode_id}/observations/{observation_id}")
        if value["episode_id"] != binding.episode_id or value["observation"]["observation_id"] != observation_id:
            raise GatewayError("upstream_scope_mismatch")
        return public_observation(value["observation"], binding)

    @app.exception_handler(ValidationError)
    async def invalid_upstream(request, error):
        return JSONResponse({"error": {"code": "invalid_upstream_response"}}, status_code=502)

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/agent/session")
    async def session():
        current = await state()
        latest = await observation(current["observation_refs"][-1])
        return {"task": binding.task.model_dump(mode="json"), "state": current, "observation": latest,
                "tool_schemas": {name: TOOL_ARGUMENTS[name].model_json_schema() for name in binding.task.allowed_tools}}

    @app.get("/agent/state")
    async def get_state():
        return {"state": await state()}

    @app.get("/agent/observations/{observation_id}")
    async def get_observation(observation_id: str):
        try:
            TypeAdapter(ObservationId).validate_python(observation_id)
        except ValidationError:
            raise GatewayError("invalid_observation_id", 422) from None
        await state()
        return {"observation": await observation(observation_id)}

    @app.post("/agent/step")
    async def step(request: Request):
        try:
            length = int(request.headers.get("content-length", "-1"))
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_REQUEST:
            raise GatewayError("bounded_content_length_required", 413)
        content = bytearray()
        async with asyncio.timeout(10):
            async for chunk in request.stream():
                if len(content) + len(chunk) > length:
                    raise GatewayError("request_too_large", 413)
                content.extend(chunk)
        if len(content) != length:
            raise GatewayError("invalid_request_length", 422)
        try:
            body = StepRequest.model_validate_json(content)
            action = body.action
            if action.type not in binding.task.allowed_actions:
                raise GatewayError("policy_rejected", 403)
            if action.type == "tool.invoke":
                if action.tool_id not in binding.task.allowed_tools or action.tool_id not in TOOL_ARGUMENTS:
                    raise GatewayError("policy_rejected", 403)
                arguments = TOOL_ARGUMENTS[action.tool_id].model_validate(action.arguments)
                if hasattr(arguments, "asset_id") and arguments.asset_id not in binding.task.input_asset_refs:
                    raise GatewayError("policy_rejected", 403)
                if isinstance(arguments, BandMathArguments) and not {arguments.red_asset_id, arguments.nir_asset_id}.issubset(binding.task.input_asset_refs):
                    raise GatewayError("policy_rejected", 403)
                if isinstance(arguments, GridArguments) and not {arguments.source_asset_id, arguments.reference_asset_id}.issubset(binding.task.input_asset_refs):
                    raise GatewayError("policy_rejected", 403)
        except ValidationError:
            raise GatewayError("invalid_request", 422) from None
        await state()
        value = await upstream("POST", f"/v2/episodes/{binding.episode_id}/step", body.model_dump(mode="json"))
        if value["episode_id"] != binding.episode_id or value["task_manifest_hash"] != binding.task_manifest_hash:
            raise GatewayError("session_pin_mismatch", 409)
        return {"state": public_state(value["state"], binding),
                "observation": public_observation(value["observation"], binding),
                "terminated": value["terminated"], "truncated": value["truncated"]}

    async def artifact(artifact_id):
        try:
            TypeAdapter(ArtifactId).validate_python(artifact_id)
        except ValidationError:
            raise GatewayError("invalid_artifact_id", 422) from None
        await state()
        result = await upstream("GET", f"/v2/artifacts/{artifact_id}?episode_id={binding.episode_id}")
        value = TypeAdapter(Artifact).validate_python(result["artifact"])
        if value.artifact_id != artifact_id:
            raise GatewayError("upstream_scope_mismatch")
        if binding.task.artifact_identity == DERIVATION_SCHEME:
            try:
                validate_derivation_metadata(value.model_dump(mode="json"))
            except ValueError:
                raise GatewayError("upstream_scope_mismatch") from None
        elif value.sha256 != artifact_id[4:]:
            raise GatewayError("upstream_scope_mismatch")
        return value

    @app.get("/agent/artifacts/{artifact_id}")
    async def get_artifact(artifact_id: str):
        value = await artifact(artifact_id)
        result = value.model_dump(mode="json", exclude={"uri", "spatial"})
        if value.spatial:
            result["spatial"] = PublicSpatial.model_validate(value.spatial.model_dump()).model_dump(mode="json")
        return {"artifact": result}

    @app.get("/agent/artifacts/{artifact_id}/content")
    async def get_content(artifact_id: str):
        value = await artifact(artifact_id)
        approved_science = {(RASTER_TOOL,RASTER_VERSION),(RASTER_TOOL,RASTER_MASKED_VERSION),
                            (GRID_TOOL,GRID_VERSION)}
        task_inputs=set(binding.task.input_asset_refs)
        old_science=(len(value.lineage.input_refs)==2
                     and set(value.lineage.input_refs).issubset(task_inputs))
        masked_science=(value.lineage.tool_id==RASTER_TOOL
                        and value.lineage.tool_version==RASTER_MASKED_VERSION
                        and len(value.lineage.input_refs)==3
                        and set(value.lineage.input_refs[:2]).issubset(task_inputs))
        scientific = (value.media_type == "image/tiff" and value.kind == "raster" and value.spatial is not None
                      and value.size_bytes <= MAX_RASTER
                      and (value.lineage.tool_id,value.lineage.tool_version) in approved_science
                      and value.lineage.tool_id in binding.task.allowed_tools
                      and (old_science or masked_science))
        if (value.media_type != "image/png" and not scientific) or value.size_bytes > MAX_IMAGE:
            raise GatewayError("unsupported_public_artifact", 422)
        content = await upstream("GET", f"/v2/artifacts/{artifact_id}/content?episode_id={binding.episode_id}", image=True)
        if len(content) != value.size_bytes or hashlib.sha256(content).hexdigest() != value.sha256:
            raise GatewayError("artifact_checksum_mismatch")
        return Response(content, media_type=value.media_type, headers={"ETag": '"' + value.sha256 + '"'})

    return app

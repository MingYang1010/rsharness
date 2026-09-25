import base64
import hashlib
from typing import Dict, Optional

import httpx
from pydantic import JsonValue

from ..artifacts import ArtifactStore
from ..domain import V2DomainError
from ..events import canonical_json, sha256_json
from ..schemas import ArtifactLineage, MapState, SpatialExtent
from .base import RenderResult


class TerriaMapRenderer:
    def __init__(
        self,
        base_url: str,
        artifact_store: ArtifactStore,
        config: Dict[str, JsonValue],
        client: Optional[httpx.Client] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.artifact_store = artifact_store
        self.config = config
        timeout_ms = int(config.get("request_timeout_ms", 45000))
        self.client = client or httpx.Client(timeout=timeout_ms / 1000.0)

    def _request(
        self,
        method: str,
        path: str,
        json: Optional[dict] = None,
    ) -> dict:
        try:
            response = self.client.request(
                method,
                "%s%s" % (self.base_url, path),
                json=json,
            )
        except httpx.HTTPError as error:
            raise V2DomainError(
                "renderer_unavailable",
                "renderer service is unavailable",
                status_code=503,
                retryable=True,
                phase="renderer",
                details=[
                    {
                        "location": ["renderer"],
                        "message": type(error).__name__,
                        "type": "renderer_transport_error",
                    }
                ],
            )
        if response.status_code >= 400:
            try:
                error_body = response.json().get("error", {})
            except ValueError:
                error_body = {}
            code = str(error_body.get("code") or "renderer_failed")
            message = str(
                error_body.get("message") or "renderer could not complete the request"
            )
            raise V2DomainError(
                code,
                message,
                status_code=409 if response.status_code == 409 else 503,
                retryable=response.status_code >= 500,
                phase="renderer",
                details=[
                    {
                        "location": ["renderer"],
                        "message": canonical_json(error_body.get("details", {})),
                        "type": code,
                    }
                ],
            )
        try:
            return response.json()
        except ValueError:
            raise V2DomainError(
                "renderer_invalid_response",
                "renderer returned invalid JSON",
                status_code=503,
                retryable=True,
                phase="renderer",
            )

    def health(self) -> Dict[str, JsonValue]:
        return self._request("GET", "/healthz")

    def create_session(
        self,
        episode_id: str,
        renderer_config: Optional[Dict[str, JsonValue]] = None,
    ) -> None:
        self._request(
            "POST",
            "/renderer/sessions",
            json={"episode_id": episode_id},
        )

    def _ensure_session(self, episode_id: str) -> None:
        self.create_session(episode_id)

    def apply_state(
        self,
        episode_id: str,
        map_state: MapState,
    ) -> Dict[str, JsonValue]:
        self._ensure_session(episode_id)
        body = self._request(
            "PUT",
            "/renderer/sessions/%s/state" % episode_id,
            json={"map_state": map_state.model_dump(mode="json")},
        )
        data = body.get("data")
        if not isinstance(data, dict) or not data.get("consistent"):
            raise V2DomainError(
                "renderer_state_mismatch",
                "renderer read-back does not match desired map state",
                status_code=409,
                phase="renderer",
            )
        return data

    def capture(
        self,
        episode_id: str,
        map_state: MapState,
        semantic_state_hash: str,
    ) -> RenderResult:
        body = self._request(
            "POST",
            "/renderer/sessions/%s/capture" % episode_id,
        )
        data = body.get("data")
        if not isinstance(data, dict):
            raise V2DomainError(
                "renderer_invalid_response",
                "renderer capture response is missing data",
                status_code=503,
                retryable=True,
                phase="renderer",
            )
        try:
            content = base64.b64decode(data["content_base64"], validate=True)
        except (KeyError, ValueError):
            raise V2DomainError(
                "renderer_invalid_response",
                "renderer capture content is not valid base64",
                status_code=503,
                retryable=True,
                phase="renderer",
            )
        content_hash = hashlib.sha256(content).hexdigest()
        if content_hash != data.get("sha256") or len(content) != data.get("size_bytes"):
            raise V2DomainError(
                "renderer_checksum_mismatch",
                "renderer capture checksum or size does not match content",
                status_code=503,
                retryable=True,
                phase="renderer",
            )
        versions = data.get("versions")
        readback = data.get("readback")
        pixel_stats = data.get("pixel_stats")
        if not all(isinstance(value, dict) for value in (versions, readback, pixel_stats)):
            raise V2DomainError(
                "renderer_invalid_response",
                "renderer capture metadata is incomplete",
                status_code=503,
                retryable=True,
                phase="renderer",
            )
        lineage = ArtifactLineage(
            tool_id="renderer.terriamap.capture",
            tool_version=str(self.config.get("renderer_version", "1.0.0")),
            input_refs=[semantic_state_hash],
            parameters_hash=sha256_json(
                {
                    "profile": self.config.get("profile"),
                    "viewport": self.config.get("viewport"),
                    "device_pixel_ratio": self.config.get("device_pixel_ratio"),
                    "language": self.config.get("language"),
                    "basemap_id": self.config.get("basemap_id"),
                    "versions": versions,
                }
            ),
        )
        artifact = self.artifact_store.put_bytes(
            content=content,
            kind="image",
            media_type="image/png",
            lineage=lineage,
            spatial=SpatialExtent(crs="EPSG:4326", bbox=map_state.bbox),
            temporal=map_state.active_time_range,
        )
        return RenderResult(
            artifact=artifact,
            pixel_stats=pixel_stats,
            provenance={
                "profile": self.config.get("profile"),
                "versions": versions,
            },
            readback=readback,
        )

    def render(
        self,
        episode_id: str,
        map_state: MapState,
        semantic_state_hash: str,
    ) -> RenderResult:
        self.apply_state(episode_id, map_state)
        return self.capture(episode_id, map_state, semantic_state_hash)

    def inspect(self, episode_id: str) -> Dict[str, JsonValue]:
        body = self._request("GET", "/renderer/sessions/%s" % episode_id)
        data = body.get("data")
        if not isinstance(data, dict):
            raise V2DomainError(
                "renderer_invalid_response",
                "renderer session response is missing data",
                status_code=503,
                phase="renderer",
            )
        return data

    def close_session(self, episode_id: str) -> None:
        self._request("DELETE", "/renderer/sessions/%s" % episode_id)

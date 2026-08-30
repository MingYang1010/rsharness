from dataclasses import dataclass
from typing import Dict, Protocol

from pydantic import JsonValue

from ..schemas import ArtifactRef, MapState


@dataclass(frozen=True)
class RenderResult:
    artifact: ArtifactRef
    pixel_stats: Dict[str, JsonValue]
    provenance: Dict[str, JsonValue]
    readback: Dict[str, JsonValue]


class RendererAdapter(Protocol):
    def create_session(self, episode_id: str, renderer_config: Dict[str, JsonValue]) -> None:
        ...

    def apply_state(
        self,
        episode_id: str,
        map_state: MapState,
    ) -> Dict[str, JsonValue]:
        ...

    def capture(
        self,
        episode_id: str,
        map_state: MapState,
        semantic_state_hash: str,
    ) -> RenderResult:
        ...

    def render(
        self,
        episode_id: str,
        map_state: MapState,
        semantic_state_hash: str,
    ) -> RenderResult:
        ...

    def health(self) -> Dict[str, JsonValue]:
        ...

    def inspect(self, episode_id: str) -> Dict[str, JsonValue]:
        ...

    def close_session(self, episode_id: str) -> None:
        ...

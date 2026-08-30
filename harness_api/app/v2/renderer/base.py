from typing import Dict, Protocol

from pydantic import JsonValue

from ..schemas import ArtifactRef, MapState


class RendererAdapter(Protocol):
    def create_session(self, episode_id: str, renderer_config: Dict[str, JsonValue]) -> None:
        ...

    def apply_state(self, episode_id: str, map_state: MapState) -> None:
        ...

    def capture(
        self,
        episode_id: str,
        observation_spec: Dict[str, JsonValue],
    ) -> ArtifactRef:
        ...

    def inspect(self, episode_id: str) -> Dict[str, JsonValue]:
        ...

    def close_session(self, episode_id: str) -> None:
        ...

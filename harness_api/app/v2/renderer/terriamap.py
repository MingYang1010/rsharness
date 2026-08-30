from typing import Dict

from pydantic import JsonValue

from ..schemas import ArtifactRef, MapState


class TerriaMapRenderer:
    """Explicitly unavailable M1 scaffold; deterministic capture starts in M2."""

    def _unavailable(self) -> None:
        raise RuntimeError("TerriaMap renderer adapter is not implemented until M2")

    def create_session(self, episode_id: str, renderer_config: Dict[str, JsonValue]) -> None:
        self._unavailable()

    def apply_state(self, episode_id: str, map_state: MapState) -> None:
        self._unavailable()

    def capture(
        self,
        episode_id: str,
        observation_spec: Dict[str, JsonValue],
    ) -> ArtifactRef:
        self._unavailable()

    def inspect(self, episode_id: str) -> Dict[str, JsonValue]:
        self._unavailable()

    def close_session(self, episode_id: str) -> None:
        self._unavailable()

from typing import Any, Dict, List

from .events import semanticize, sha256_json
from .schemas import Observation, ObservationItem, V2EpisodeState


def state_hash(state: V2EpisodeState) -> str:
    return sha256_json(state.model_dump(mode="json"))


def semantic_state_hash(state: V2EpisodeState) -> str:
    return sha256_json(semanticize(state.model_dump(mode="json")))


def build_structural_observation(
    state: V2EpisodeState,
    observation_id: str,
    sequence: int,
    asset_refs: List[str],
    reason: str,
) -> Observation:
    map_payload: Dict[str, Any] = state.map.model_dump(mode="json")
    return Observation(
        observation_id=observation_id,
        sequence=sequence,
        primary_type="map_state",
        items=[
            ObservationItem(type="map_state", inline=map_payload),
            ObservationItem(type="asset_metadata", asset_refs=asset_refs),
        ],
        state_hash=state_hash(state),
        semantic_state_hash=semantic_state_hash(state),
        provenance={
            "builder": "structural-map-state",
            "builder_version": "1.0.0",
            "reason": reason,
            "task_manifest_hash": state.task_manifest_hash,
        },
        warnings=["rendered_view is unavailable until milestone M2"],
    )

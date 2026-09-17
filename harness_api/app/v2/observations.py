from typing import Any, Dict, List

from .events import semanticize, sha256_json
from .renderer.base import RenderResult
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
    if state.map is None or not state.map.layers:
        return Observation(
            observation_id=observation_id, sequence=sequence, primary_type="asset_metadata",
            items=[ObservationItem(type="asset_metadata", asset_refs=asset_refs)],
            state_hash=state_hash(state), semantic_state_hash=semantic_state_hash(state),
            provenance={"builder": "headless-assets", "builder_version": "1.0.0",
                        "reason": reason, "task_manifest_hash": state.task_manifest_hash}, warnings=[],
        )
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


def add_rendered_view(
    observation: Observation,
    render_result: RenderResult,
) -> Observation:
    rendered = observation.model_copy(deep=True)
    rendered.primary_type = "rendered_view"
    rendered.items.append(
        ObservationItem(
            type="rendered_view",
            artifact_ref=render_result.artifact.artifact_id,
        )
    )
    rendered.provenance = {
        **rendered.provenance,
        "renderer": render_result.provenance,
        "renderer_readback": render_result.readback,
        "renderer_pixel_stats": render_result.pixel_stats,
    }
    rendered.warnings = []
    return rendered

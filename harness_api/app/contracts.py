from typing import Any, Dict

from .domain import semantic_state_hash, state_hash
from .schemas import (
    EpisodeResultData,
    EpisodeState,
    MapObservation,
    StateData,
    TraceData,
    TraceTransition,
)


def public_state(raw: Dict[str, Any]) -> EpisodeState:
    return EpisodeState(
        episode_id=raw["episode_id"],
        state_version=raw["state_version"],
        status=raw["status"],
        step_count=raw["step_count"],
        max_steps=raw["max_steps"],
        task=raw["task"],
        view=raw["view"],
        layers={key: raw["layers"][key] for key in sorted(raw["layers"])},
        final_answer=raw.get("final_answer"),
        created_at=raw["created_at"],
        updated_at=raw["updated_at"],
    )


def public_observation(
    raw: Dict[str, Any], state: Dict[str, Any]
) -> MapObservation:
    return MapObservation(
        observation_type=raw["observation_type"],
        sequence=raw["sequence"],
        message=raw["message"],
        view=raw["view"],
        visible_layers=raw["visible_layers"],
        state_hash=state_hash(state),
        semantic_state_hash=semantic_state_hash(state),
    )


def public_episode_result(raw: Dict[str, Any]) -> EpisodeResultData:
    state = raw["state"]
    instance_hash = state_hash(state)
    semantic_hash = semantic_state_hash(state)
    return EpisodeResultData(
        episode_id=raw["episode_id"],
        state=public_state(state),
        observation=public_observation(raw["observation"], state),
        reward=raw.get("reward"),
        terminated=state["status"] == "terminated",
        truncated=state["status"] == "truncated",
        info={
            "client_action_id": raw.get("info", {}).get("client_action_id"),
            "remaining_steps": max(state["max_steps"] - state["step_count"], 0),
            "state_hash": instance_hash,
            "semantic_state_hash": semantic_hash,
        },
    )


def public_state_result(raw: Dict[str, Any]) -> StateData:
    state = raw["state"]
    return StateData(
        episode_id=raw["episode_id"],
        state=public_state(state),
        state_hash=state_hash(state),
        semantic_state_hash=semantic_state_hash(state),
    )


def public_trace(raw: Dict[str, Any]) -> TraceData:
    transitions = []
    for transition in raw["transitions"]:
        state = transition["state"]
        transitions.append(
            TraceTransition(
                sequence=transition["sequence"],
                client_action_id=transition["client_action_id"],
                created_at=transition["created_at"],
                action=transition["action"],
                observation=public_observation(transition["observation"], state),
                state_hash=state_hash(state),
                semantic_state_hash=semantic_state_hash(state),
                state=public_state(state),
            )
        )

    return TraceData(
        episode_id=raw["episode_id"],
        initial_state=public_state(raw["initial_state"]),
        transitions=transitions,
        final_state=public_state(raw["final_state"]),
        transition_count=len(transitions),
        hash_algorithm="sha256",
        trace_hash=raw["trace_hash"],
        semantic_trace_hash=raw["semantic_trace_hash"],
    )

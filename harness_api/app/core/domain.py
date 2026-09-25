import math
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .budgets import exhausted, initial_budget, update_budget
from .capabilities import IMPLEMENTED_ACTIONS
from .evidence import validate_evidence
from .observations import build_structural_observation
from .schemas import (
    AnswerRecord,
    AnswerSubmitAction,
    Artifact,
    TaskAsset,
    EvidenceRef,
    GeoPoint,
    MapLayerState,
    MapState,
    Observation,
    SpatialBoundingBox,
    TaskManifest,
    V2Action,
    V2EpisodeState,
)


DEFAULT_VIEW = SpatialBoundingBox(
    west=120.1,
    south=30.1,
    east=122.9,
    north=32.9,
)


class V2DomainError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 422,
        retryable: bool = False,
        phase: str = "state",
        details: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.phase = phase
        self.details = details or []


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def elapsed_ms(started_at: str, current_time: str) -> int:
    start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    end = datetime.fromisoformat(current_time.replace("Z", "+00:00"))
    return max(int((end - start).total_seconds() * 1000), 0)


def _center(bbox: SpatialBoundingBox) -> GeoPoint:
    return GeoPoint(
        longitude=(bbox.west + bbox.east) / 2.0,
        latitude=(bbox.south + bbox.north) / 2.0,
    )


def _bounded_range(
    center: float,
    span: float,
    lower: float,
    upper: float,
) -> Tuple[float, float]:
    span = min(max(span, 1e-6), upper - lower)
    start = center - span / 2.0
    end = center + span / 2.0
    if start < lower:
        end += lower - start
        start = lower
    if end > upper:
        start -= end - upper
        end = upper
    return max(start, lower), min(end, upper)


def _bbox_from_center(
    longitude: float,
    latitude: float,
    longitude_span: float,
    latitude_span: float,
) -> SpatialBoundingBox:
    west, east = _bounded_range(longitude, longitude_span, -180.0, 180.0)
    south, north = _bounded_range(latitude, latitude_span, -90.0, 90.0)
    return SpatialBoundingBox(west=west, south=south, east=east, north=north)


def _asset_map(manifest: TaskManifest) -> Dict[str, TaskAsset]:
    return {asset.asset_id: asset for asset in manifest.assets if asset.asset_id in manifest.task.inputs}


def _evidence_sources(
    manifest: TaskManifest,
    artifacts: Optional[Dict[str, Artifact]],
) -> Dict[str, Any]:
    sources: Dict[str, Any] = _asset_map(manifest)
    sources.update(artifacts or {})
    return sources


def create_initial_state(
    episode_id: str,
    manifest: TaskManifest,
    seed: int,
    timestamp: Optional[str] = None,
) -> Tuple[V2EpisodeState, Observation]:
    created_at = timestamp or utc_now()
    input_assets = _asset_map(manifest)
    primary_asset = input_assets[manifest.task.inputs[0]]
    layer_id = "layer-%s" % primary_asset.asset_id
    observation_id = "obs-%s" % uuid.uuid4().hex
    state = V2EpisodeState(
        episode_id=episode_id,
        task_ref={
            "task_id": manifest.task.task_id,
            "task_version": manifest.task.task_version,
        },
        task_manifest_hash=manifest.task_manifest_hash,
        seed=seed,
        state_version=0,
        status="active",
        step_count=0,
        map=MapState(
            bbox=DEFAULT_VIEW,
            center=_center(DEFAULT_VIEW),
            layers={
                layer_id: MapLayerState(
                    layer_id=layer_id,
                    asset_id=primary_asset.asset_id,
                    name="ESA WorldCover 2021 N30E120",
                    visible=True,
                    opacity=0.85,
                    style_id="worldcover-official-rgb-v1",
                    time_range=primary_asset.temporal,
                )
            },
            active_time_range=primary_asset.temporal,
        ),
        budget=initial_budget(manifest.task.budget),
        accessible_asset_refs=list(manifest.task.inputs),
        observation_refs=[observation_id],
        evidence_refs=[],
        final_answer=None,
        evaluation=None,
        created_at=created_at,
        updated_at=created_at,
    )
    if manifest.task.metadata.get("observation_profile") == "headless-tools-v1":
        if any(asset.spatial is None for asset in input_assets.values()):
            # No fabricated geographic state for pixel-only or mixed inputs.
            state.map = None
        else:
            state.map.layers = {}
            state.map.bbox = primary_asset.spatial.bbox
            state.map.center = _center(primary_asset.spatial.bbox)
    observation = build_structural_observation(
        state=state,
        observation_id=observation_id,
        sequence=0,
        asset_refs=state.accessible_asset_refs,
        reason="episode.reset",
    )
    return state, observation


def _action_allowed(action_type: str, patterns: List[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith(".*") and action_type.startswith(pattern[:-1]):
            return True
        if pattern == action_type:
            return True
    return False


def _set_view(state: V2EpisodeState, bbox: SpatialBoundingBox) -> None:
    state.map.bbox = bbox
    state.map.center = _center(bbox)


def _validate_answer(answer: Any, schema: Dict[str, Any]) -> None:
    expected_type = schema.get("type")
    if expected_type == "object" and not isinstance(answer, dict):
        raise V2DomainError("invalid_answer", "answer must be an object")
    if not isinstance(answer, dict):
        return
    missing = [key for key in schema.get("required", []) if key not in answer]
    if missing:
        raise V2DomainError(
            "invalid_answer",
            "answer is missing required fields: %s" % ", ".join(missing),
        )
    properties = schema.get("properties", {})
    type_checks = {
        "array": list,
        "number": (int, float),
        "object": dict,
        "string": str,
    }
    for key, definition in properties.items():
        if key not in answer or not isinstance(definition, dict):
            continue
        expected = definition.get("type")
        python_type = type_checks.get(expected)
        if python_type is not None and not isinstance(answer[key], python_type):
            raise V2DomainError(
                "invalid_answer",
                "answer.%s must have JSON type %s" % (key, expected),
            )
        if "enum" in definition and answer[key] not in definition["enum"]:
            raise V2DomainError(
                "invalid_answer",
                "answer.%s is not an allowed value" % key,
            )
        if isinstance(answer[key], bool) and expected == "number":
            raise V2DomainError(
                "invalid_answer",
                "answer.%s must have JSON type number" % key,
            )
        if expected == "number":
            value = float(answer[key])
            if not math.isfinite(value):
                raise V2DomainError("invalid_answer", "answer.%s must be finite" % key)
            if "minimum" in definition and value < definition["minimum"]:
                raise V2DomainError("invalid_answer", "answer.%s is below minimum" % key)
            if "maximum" in definition and value > definition["maximum"]:
                raise V2DomainError("invalid_answer", "answer.%s is above maximum" % key)


def _existing_evidence(state: V2EpisodeState) -> Dict[str, EvidenceRef]:
    return {item.evidence_id: item for item in state.evidence_refs}


def _validate_evidence_ids(
    state: V2EpisodeState,
    evidence_ids: List[str],
    required: bool,
) -> None:
    if required and not evidence_ids:
        raise V2DomainError(
            "evidence_required",
            "at least one frozen evidence reference is required",
            phase="policy",
        )
    known = _existing_evidence(state)
    missing = [item for item in evidence_ids if item not in known]
    if missing:
        raise V2DomainError(
            "unknown_evidence",
            "unknown evidence IDs: %s" % ", ".join(missing),
            phase="policy",
        )


def apply_action(
    current_state: V2EpisodeState,
    action: V2Action,
    manifest: TaskManifest,
    current_time: Optional[str] = None,
    artifacts: Optional[Dict[str, Artifact]] = None,
) -> Tuple[V2EpisodeState, Observation, Dict[str, Any]]:
    if current_state.status != "active":
        raise V2DomainError(
            "episode_closed",
            "episode is %s and cannot accept actions" % current_state.status,
            status_code=409,
            phase="state",
        )

    action_type = action.type
    if action_type.startswith("map.") and current_state.map is None:
        raise V2DomainError(
            "coordinate_system_mismatch",
            "geographic map actions are unavailable for pixel-only inputs",
            status_code=403,
            phase="policy",
        )
    if action_type not in IMPLEMENTED_ACTIONS or not _action_allowed(
        action_type,
        manifest.scenario.allowed_actions,
    ):
        raise V2DomainError(
            "policy_rejected",
            "action is declared but unavailable in this task profile: %s" % action_type,
            status_code=403,
            phase="policy",
        )

    state = current_state.model_copy(deep=True)
    event_payload: Dict[str, Any] = {"action_type": action_type}

    if action_type == "map.set_view":
        _set_view(state, action.bbox)

    elif action_type == "map.pan":
        bbox = state.map.bbox
        center = state.map.center
        shifted = _bbox_from_center(
            center.longitude + action.delta_longitude,
            center.latitude + action.delta_latitude,
            bbox.east - bbox.west,
            bbox.north - bbox.south,
        )
        _set_view(state, shifted)

    elif action_type == "map.zoom":
        bbox = state.map.bbox
        center = state.map.center
        scale = 1.0 / action.factor if action.direction == "in" else action.factor
        zoomed = _bbox_from_center(
            center.longitude,
            center.latitude,
            (bbox.east - bbox.west) * scale,
            (bbox.north - bbox.south) * scale,
        )
        _set_view(state, zoomed)

    elif action_type == "map.layer.set_visibility":
        layer = state.map.layers.get(action.layer_id)
        if layer is None:
            raise V2DomainError("unknown_layer", "unknown layer_id: %s" % action.layer_id)
        layer.visible = action.visible

    elif action_type == "map.layer.set_opacity":
        layer = state.map.layers.get(action.layer_id)
        if layer is None:
            raise V2DomainError("unknown_layer", "unknown layer_id: %s" % action.layer_id)
        layer.opacity = action.opacity

    elif action_type == "map.time.set_range":
        state.map.active_time_range = action.time_range

    elif action_type == "memory.save_evidence":
        evidence_by_id = _existing_evidence(state)
        if action.evidence.evidence_id in evidence_by_id:
            raise V2DomainError(
                "evidence_conflict",
                "evidence_id already exists: %s" % action.evidence.evidence_id,
                status_code=409,
            )
        try:
            validate_evidence(
                action.evidence,
                _evidence_sources(manifest, artifacts),
            )
        except ValueError as error:
            raise V2DomainError(
                "invalid_evidence",
                str(error),
                phase="policy",
            )
        state.evidence_refs.append(action.evidence)
        event_payload["evidence"] = action.evidence.model_dump(mode="json")

    elif action_type == "answer.submit":
        _validate_answer(action.answer, manifest.task.answer_schema)
        _validate_evidence_ids(
            state,
            action.evidence_ids,
            required=manifest.scenario.evidence_required,
        )
        state.final_answer = AnswerRecord(
            outcome="submitted",
            answer=action.answer,
            confidence=action.confidence,
            evidence_ids=action.evidence_ids,
        )
        state.status = "terminated"

    elif action_type == "answer.abstain":
        if not manifest.scenario.abstention_allowed:
            raise V2DomainError(
                "policy_rejected",
                "abstention is not allowed by this scenario",
                status_code=403,
                phase="policy",
            )
        _validate_evidence_ids(state, action.evidence_ids, required=False)
        state.final_answer = AnswerRecord(
            outcome="abstained",
            evidence_ids=action.evidence_ids,
            rationale=action.rationale,
        )
        state.status = "terminated"

    elif action_type == "answer.request_human_review":
        if manifest.scenario.human_review_policy == "never":
            raise V2DomainError(
                "policy_rejected",
                "human review is not allowed by this scenario",
                status_code=403,
                phase="policy",
            )
        _validate_evidence_ids(state, action.evidence_ids, required=False)
        state.final_answer = AnswerRecord(
            outcome="human_review_requested",
            evidence_ids=action.evidence_ids,
            rationale=action.rationale,
        )
        state.status = "terminated"

    timestamp = current_time or utc_now()
    state.step_count += 1
    state.state_version += 1
    state.updated_at = timestamp
    state.budget = update_budget(
        state.budget,
        elapsed_wall_time_ms=elapsed_ms(state.created_at, timestamp),
        step_increment=1,
    )
    if state.status == "active" and exhausted(state.budget):
        state.status = "truncated"

    observation_id = "obs-%s" % uuid.uuid4().hex
    state.observation_refs.append(observation_id)
    observation = build_structural_observation(
        state=state,
        observation_id=observation_id,
        sequence=state.step_count,
        asset_refs=state.accessible_asset_refs,
        reason=action_type,
    )
    return state, observation, event_payload

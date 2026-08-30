import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_VIEW = {
    "west": 120.1,
    "south": 30.1,
    "east": 122.9,
    "north": 32.9,
}

DEFAULT_LAYER = {
    "layer_id": "esa-worldcover-2021",
    "name": "ESA WorldCover 2021 (10 m, Local N30E120)",
    "kind": "cog",
    "source": "data/worldcover-2021/ESA_WorldCover_10m_2021_v200_N30E120_Map_RGB.tif",
    "visible": True,
    "opacity": 0.85,
}

ACTION_SPACE = [
    {
        "type": "set_view",
        "description": "Replace the active map bounding box.",
        "fields": {"bbox": "{west, south, east, north}"},
    },
    {
        "type": "pan",
        "description": "Shift the active bounding box in decimal degrees.",
        "fields": {"delta_longitude": "float", "delta_latitude": "float"},
    },
    {
        "type": "zoom",
        "description": "Zoom around the current center.",
        "fields": {"direction": "in | out", "factor": "1 < float <= 8"},
    },
    {
        "type": "set_layer_visibility",
        "description": "Show or hide a catalog layer.",
        "fields": {"layer_id": "string", "visible": "boolean"},
    },
    {
        "type": "set_layer_opacity",
        "description": "Set a catalog layer opacity.",
        "fields": {"layer_id": "string", "opacity": "0 <= float <= 1"},
    },
    {
        "type": "submit_answer",
        "description": "Finish the episode with an evidence-linked answer.",
        "fields": {"answer": "string", "evidence_refs": "string[]"},
    },
]

ACTION_TYPES = {item["type"] for item in ACTION_SPACE}


class DomainError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def state_hash(state: Dict[str, Any]) -> str:
    semantic_state = {
        key: value
        for key, value in state.items()
        if key not in {"created_at", "updated_at"}
    }
    return hashlib.sha256(canonical_json(semantic_state).encode("utf-8")).hexdigest()


def semantic_state(state: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(state)
    for key in ("episode_id", "created_at", "updated_at"):
        result.pop(key, None)
    return result


def semantic_state_hash(state: Dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json(semantic_state(state)).encode("utf-8")
    ).hexdigest()


def semantic_trace_hash(
    initial_state: Dict[str, Any],
    transitions: List[Dict[str, Any]],
    final_state: Dict[str, Any],
) -> str:
    replay = {
        "initial_state": semantic_state(initial_state),
        "transitions": [
            {
                "sequence": transition["sequence"],
                "action": transition["action"],
                "state": semantic_state(transition["state"]),
            }
            for transition in transitions
        ],
        "final_state": semantic_state(final_state),
    }
    return hashlib.sha256(canonical_json(replay).encode("utf-8")).hexdigest()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise DomainError("invalid_number", "%s must be a finite number" % field)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise DomainError("invalid_number", "%s must be a finite number" % field)
    if not math.isfinite(result):
        raise DomainError("invalid_number", "%s must be a finite number" % field)
    return result


def validate_bbox(bbox: Any) -> Dict[str, float]:
    if not isinstance(bbox, dict):
        raise DomainError("invalid_bbox", "bbox must be an object")

    required = ("west", "south", "east", "north")
    missing = [key for key in required if key not in bbox]
    if missing:
        raise DomainError("invalid_bbox", "bbox is missing: %s" % ", ".join(missing))

    result = {key: _number(bbox[key], "bbox.%s" % key) for key in required}
    if not -180.0 <= result["west"] < result["east"] <= 180.0:
        raise DomainError(
            "invalid_bbox",
            "bbox longitude must satisfy -180 <= west < east <= 180",
        )
    if not -90.0 <= result["south"] < result["north"] <= 90.0:
        raise DomainError(
            "invalid_bbox",
            "bbox latitude must satisfy -90 <= south < north <= 90",
        )
    return result


def _center(bbox: Dict[str, float]) -> Dict[str, float]:
    return {
        "longitude": (bbox["west"] + bbox["east"]) / 2.0,
        "latitude": (bbox["south"] + bbox["north"]) / 2.0,
    }


def _bounded_range(
    center: float, span: float, lower: float, upper: float
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
    longitude: float, latitude: float, longitude_span: float, latitude_span: float
) -> Dict[str, float]:
    west, east = _bounded_range(longitude, longitude_span, -180.0, 180.0)
    south, north = _bounded_range(latitude, latitude_span, -90.0, 90.0)
    return {"west": west, "south": south, "east": east, "north": north}


def _set_view(state: Dict[str, Any], bbox: Dict[str, float]) -> None:
    state["view"] = {"bbox": bbox, "center": _center(bbox)}


def create_initial_state(episode_id: str, specification: Dict[str, Any]) -> Dict[str, Any]:
    max_steps = int(specification.get("max_steps", 20))
    if not 1 <= max_steps <= 1000:
        raise DomainError("invalid_budget", "max_steps must be between 1 and 1000")

    bbox = validate_bbox(specification.get("initial_view") or DEFAULT_VIEW)
    timestamp = utc_now()
    return {
        "episode_id": episode_id,
        "state_version": 0,
        "status": "active",
        "step_count": 0,
        "max_steps": max_steps,
        "task": {
            "task_id": specification.get("task_id") or "interactive-map-task",
            "prompt": specification.get("prompt")
            or "Inspect the active Earth-observation layers.",
            "seed": int(specification.get("seed", 0)),
            "metadata": copy.deepcopy(specification.get("metadata") or {}),
        },
        "view": {"bbox": bbox, "center": _center(bbox)},
        "layers": {DEFAULT_LAYER["layer_id"]: copy.deepcopy(DEFAULT_LAYER)},
        "final_answer": None,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def build_observation(state: Dict[str, Any], message: str) -> Dict[str, Any]:
    visible_layers = [
        copy.deepcopy(layer)
        for layer in state["layers"].values()
        if layer["visible"]
    ]
    return {
        "observation_type": "map_state",
        "sequence": state["step_count"],
        "message": message,
        "view": copy.deepcopy(state["view"]),
        "visible_layers": visible_layers,
        "state_hash": state_hash(state),
        "semantic_state_hash": semantic_state_hash(state),
    }


def _layer(state: Dict[str, Any], layer_id: Any) -> Dict[str, Any]:
    if not isinstance(layer_id, str) or layer_id not in state["layers"]:
        raise DomainError("unknown_layer", "Unknown layer_id: %s" % layer_id)
    return state["layers"][layer_id]


def apply_action(
    current_state: Dict[str, Any], action: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if current_state["status"] != "active":
        raise DomainError(
            "episode_closed",
            "Episode is %s and cannot accept more actions" % current_state["status"],
            status_code=409,
        )

    action_type = action.get("type")
    if action_type not in ACTION_TYPES:
        raise DomainError("unsupported_action", "Unsupported action type: %s" % action_type)

    state = copy.deepcopy(current_state)
    message = "Action %s completed" % action_type

    if action_type == "set_view":
        _set_view(state, validate_bbox(action.get("bbox")))

    elif action_type == "pan":
        delta_longitude = _number(action.get("delta_longitude"), "delta_longitude")
        delta_latitude = _number(action.get("delta_latitude"), "delta_latitude")
        if delta_longitude == 0.0 and delta_latitude == 0.0:
            raise DomainError("invalid_pan", "pan must change longitude or latitude")
        bbox = state["view"]["bbox"]
        center = state["view"]["center"]
        shifted = _bbox_from_center(
            center["longitude"] + delta_longitude,
            center["latitude"] + delta_latitude,
            bbox["east"] - bbox["west"],
            bbox["north"] - bbox["south"],
        )
        _set_view(state, shifted)

    elif action_type == "zoom":
        direction = action.get("direction")
        if direction not in {"in", "out"}:
            raise DomainError("invalid_zoom", "zoom direction must be 'in' or 'out'")
        factor = _number(action.get("factor", 2.0), "factor")
        if not 1.0 < factor <= 8.0:
            raise DomainError("invalid_zoom", "zoom factor must satisfy 1 < factor <= 8")
        bbox = state["view"]["bbox"]
        center = state["view"]["center"]
        scale = 1.0 / factor if direction == "in" else factor
        zoomed = _bbox_from_center(
            center["longitude"],
            center["latitude"],
            (bbox["east"] - bbox["west"]) * scale,
            (bbox["north"] - bbox["south"]) * scale,
        )
        _set_view(state, zoomed)

    elif action_type == "set_layer_visibility":
        layer = _layer(state, action.get("layer_id"))
        visible = action.get("visible")
        if not isinstance(visible, bool):
            raise DomainError("invalid_visibility", "visible must be a boolean")
        layer["visible"] = visible

    elif action_type == "set_layer_opacity":
        layer = _layer(state, action.get("layer_id"))
        opacity = _number(action.get("opacity"), "opacity")
        if not 0.0 <= opacity <= 1.0:
            raise DomainError("invalid_opacity", "opacity must be between 0 and 1")
        layer["opacity"] = opacity

    elif action_type == "submit_answer":
        answer = action.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise DomainError("invalid_answer", "answer must be a non-empty string")
        evidence_refs = action.get("evidence_refs") or []
        if not isinstance(evidence_refs, list) or not all(
            isinstance(item, str) and item for item in evidence_refs
        ):
            raise DomainError("invalid_evidence", "evidence_refs must contain strings")
        state["final_answer"] = {
            "answer": answer.strip(),
            "evidence_refs": list(evidence_refs),
        }
        state["status"] = "terminated"
        message = "Answer submitted; episode terminated"

    state["step_count"] += 1
    state["state_version"] += 1
    if state["status"] == "active" and state["step_count"] >= state["max_steps"]:
        state["status"] = "truncated"
        message = "Step budget exhausted; episode truncated"
    state["updated_at"] = utc_now()

    return state, build_observation(state, message)

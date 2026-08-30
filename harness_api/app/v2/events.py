import hashlib
import json
import uuid
from typing import Any, Dict, Iterable, List

from .schemas import EventRecord


INSTANCE_KEYS = {
    "client_action_id",
    "created_at",
    "evidence_id",
    "evidence_ids",
    "episode_id",
    "event_id",
    "observation_id",
    "observation_refs",
    "request_id",
    "state_hash",
    "updated_at",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def semanticize(value: Any) -> Any:
    if isinstance(value, list):
        return [semanticize(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in sorted(value.items()):
        if key in INSTANCE_KEYS:
            continue
        if key == "wall_time_ms" and isinstance(item, dict):
            result[key] = {"limit": item.get("limit")}
            continue
        result[key] = semanticize(item)
    return result


def create_event(
    episode_id: str,
    sequence: int,
    event_type: str,
    state_version: int,
    created_at: str,
    payload: Dict[str, Any],
) -> EventRecord:
    return EventRecord(
        event_id="evt-%s" % uuid.uuid4().hex,
        episode_id=episode_id,
        sequence=sequence,
        event_type=event_type,
        state_version=state_version,
        created_at=created_at,
        payload=payload,
    )


def trace_hash(events: Iterable[EventRecord]) -> str:
    serialized = [event.model_dump(mode="json") for event in events]
    return sha256_json(serialized)


def semantic_trace_hash(
    task_manifest_hash: str,
    events: Iterable[EventRecord],
) -> str:
    serialized: List[Dict[str, Any]] = []
    for event in events:
        serialized.append(
            {
                "sequence": event.sequence,
                "event_type": event.event_type,
                "state_version": event.state_version,
                "payload": semanticize(event.payload),
            }
        )
    return sha256_json(
        {
            "task_manifest_hash": task_manifest_hash,
            "events": serialized,
        }
    )

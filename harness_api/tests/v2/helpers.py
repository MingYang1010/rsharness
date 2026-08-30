import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


HARNESS_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = HARNESS_ROOT.parent
TASKS_ROOT = PROJECT_ROOT / "tasks"
FIXTURES_ROOT = PROJECT_ROOT / "contracts" / "v2" / "fixtures"
sys.path.insert(0, str(HARNESS_ROOT))
os.environ.setdefault(
    "EO_HARNESS_DB",
    str(Path(tempfile.gettempdir()) / "eo-harness-v2-import.sqlite3"),
)

from app.main import create_app


RESET_REQUEST: Dict[str, Any] = {
    "task_ref": {
        "task_id": "worldcover-grounded-vqa",
        "task_version": "1.0.0",
    },
    "seed": 42,
}

ZOOM_REQUEST: Dict[str, Any] = {
    "client_action_id": "fixture-zoom-1",
    "expected_state_version": 0,
    "action": {
        "type": "map.zoom",
        "direction": "in",
        "factor": 2.0,
    },
}

EVIDENCE_REQUEST: Dict[str, Any] = {
    "client_action_id": "fixture-evidence-1",
    "expected_state_version": 1,
    "action": {
        "type": "memory.save_evidence",
        "evidence": {
            "evidence_id": "ev-fixture-aoi",
            "claim_id": "claim-1",
            "source_ref": "asset-worldcover-n30e120",
            "selector": {
                "bbox": {
                    "west": 120.5,
                    "south": 30.5,
                    "east": 121.0,
                    "north": 31.0,
                },
                "geometry": None,
                "time_range": None,
                "bands": ["visual"],
                "pixel_window": None,
            },
            "description": "Frozen WorldCover evidence for the selected AOI.",
            "frozen_sha256": (
                "9f376abaca38815c5c743126147aeffd1916bb1907ad98929d341d4e6c87381c"
            ),
        },
    },
}

ANSWER_REQUEST: Dict[str, Any] = {
    "client_action_id": "fixture-answer-1",
    "expected_state_version": 2,
    "action": {
        "type": "answer.submit",
        "answer": {
            "label": "built-up",
            "confidence": 0.84,
            "claims": [{"claim_id": "claim-1", "text": "Built-up is dominant."}],
        },
        "confidence": 0.84,
        "evidence_ids": ["ev-fixture-aoi"],
    },
}

DYNAMIC_ID_VALUES = {
    "episode_id": "ep2-00000000000000000000000000000000",
    "event_id": "evt-00000000000000000000000000000000",
    "observation_id": "obs-00000000000000000000000000000000",
}


def normalize_dynamic(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_dynamic(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized: Dict[str, Any] = {}
    for key, item in value.items():
        if key in DYNAMIC_ID_VALUES:
            normalized[key] = DYNAMIC_ID_VALUES[key]
        elif key in {"created_at", "updated_at"}:
            normalized[key] = "2026-01-01T00:00:00Z"
        elif key in {"state_hash", "trace_hash"}:
            normalized[key] = "0" * 64
        elif key == "wall_time_ms" and isinstance(item, dict):
            normalized[key] = {
                "limit": item["limit"],
                "used": 0,
                "remaining": item["limit"],
            }
        elif key == "observation_refs" and isinstance(item, list):
            normalized[key] = [DYNAMIC_ID_VALUES["observation_id"] for _ in item]
        else:
            normalized[key] = normalize_dynamic(item)
    return normalized


def read_fixture(name: str) -> Any:
    return json.loads((FIXTURES_ROOT / name).read_text(encoding="utf-8"))


def make_app(database_path: Path):
    return create_app(
        database_path=str(database_path),
        v2_tasks_path=str(TASKS_ROOT),
        v2_enabled=True,
    )

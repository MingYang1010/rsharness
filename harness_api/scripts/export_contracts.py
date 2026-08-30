import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


HARNESS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(HARNESS_ROOT))
os.environ.setdefault(
    "EO_HARNESS_DB",
    str(Path(tempfile.gettempdir()) / "eo-harness-openapi.sqlite3"),
)

from fastapi.testclient import TestClient

from app.main import create_app


FIXTURE_EPISODE_ID = "ep-00000000000000000000000000000000"
FIXTURE_TIMESTAMP = "2026-01-01T00:00:00Z"
FIXTURE_HASH = "0" * 64

RESET_REQUEST: Dict[str, Any] = {
    "task_id": "contract-fixture",
    "prompt": "Inspect the active WorldCover layer.",
    "seed": 42,
    "max_steps": 4,
    "metadata": {"split": "contract"},
}

STEP_REQUEST: Dict[str, Any] = {
    "client_action_id": "fixture-zoom-1",
    "action": {"type": "zoom", "direction": "in", "factor": 2.0},
}


def normalize_dynamic(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_dynamic(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized: Dict[str, Any] = {}
    for key, item in value.items():
        if key == "episode_id":
            normalized[key] = FIXTURE_EPISODE_ID
        elif key in {"created_at", "updated_at"}:
            normalized[key] = FIXTURE_TIMESTAMP
        elif key in {"state_hash", "trace_hash"}:
            normalized[key] = FIXTURE_HASH
        else:
            normalized[key] = normalize_dynamic(item)
    return normalized


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export() -> None:
    fixture_dir = PROJECT_ROOT / "contracts" / "fixtures"
    with tempfile.TemporaryDirectory() as tempdir:
        application = create_app(str(Path(tempdir) / "episodes.sqlite3"))
        with TestClient(application) as client:
            reset = client.post(
                "/v1/reset",
                headers={"X-Request-ID": "req-contract-reset"},
                json=RESET_REQUEST,
            )
            reset.raise_for_status()
            episode_id = reset.json()["data"]["episode_id"]

            step = client.post(
                "/v1/episodes/%s/step" % episode_id,
                headers={"X-Request-ID": "req-contract-step"},
                json=STEP_REQUEST,
            )
            step.raise_for_status()

            state = client.get(
                "/v1/episodes/%s/state" % episode_id,
                headers={"X-Request-ID": "req-contract-state"},
            )
            state.raise_for_status()

            trace = client.get(
                "/v1/episodes/%s/trace" % episode_id,
                headers={"X-Request-ID": "req-contract-trace"},
            )
            trace.raise_for_status()

            action_space = client.get(
                "/v1/action-space",
                headers={"X-Request-ID": "req-contract-actions"},
            )
            action_space.raise_for_status()

            validation_error = client.post(
                "/v1/episodes/%s/step" % episode_id,
                headers={"X-Request-ID": "req-contract-error"},
                json={
                    "action": {
                        "type": "zoom",
                        "direction": "in",
                        "factor": "2.0",
                    }
                },
            )
            if validation_error.status_code != 422:
                raise RuntimeError(
                    "Expected validation fixture to return 422, got %s"
                    % validation_error.status_code
                )

        write_json(fixture_dir / "reset-request.json", RESET_REQUEST)
        write_json(fixture_dir / "reset-response.json", normalize_dynamic(reset.json()))
        write_json(fixture_dir / "step-request.json", STEP_REQUEST)
        write_json(fixture_dir / "step-response.json", normalize_dynamic(step.json()))
        write_json(fixture_dir / "state-response.json", normalize_dynamic(state.json()))
        write_json(fixture_dir / "trace-response.json", normalize_dynamic(trace.json()))
        write_json(
            fixture_dir / "action-space-response.json",
            normalize_dynamic(action_space.json()),
        )
        write_json(
            fixture_dir / "validation-error-response.json",
            normalize_dynamic(validation_error.json()),
        )
        write_json(PROJECT_ROOT / "contracts" / "openapi-v1.json", application.openapi())


if __name__ == "__main__":
    export()

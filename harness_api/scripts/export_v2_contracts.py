import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


HARNESS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(HARNESS_ROOT))
sys.path.insert(0, str(HARNESS_ROOT / "tests"))
os.environ.setdefault(
    "EO_HARNESS_DB",
    str(Path(tempfile.gettempdir()) / "eo-harness-v2-export-import.sqlite3"),
)

from fastapi.testclient import TestClient

from app.v2.api import build_openapi_schema
from v2.helpers import (
    ANSWER_REQUEST,
    EVIDENCE_REQUEST,
    RESET_REQUEST,
    ZOOM_REQUEST,
    make_app,
    normalize_dynamic,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export() -> None:
    fixture_dir = PROJECT_ROOT / "contracts" / "v2" / "fixtures"
    with tempfile.TemporaryDirectory() as directory:
        application = make_app(Path(directory) / "episodes.sqlite3")
        with TestClient(application) as client:
            capabilities = client.get(
                "/v2/capabilities",
                headers={"X-Request-ID": "req-v2-capabilities"},
            )
            task = client.get(
                "/v2/tasks/worldcover-grounded-vqa/versions/1.0.0",
                headers={"X-Request-ID": "req-v2-task"},
            )
            reset = client.post(
                "/v2/reset",
                headers={"X-Request-ID": "req-v2-reset"},
                json=RESET_REQUEST,
            )
            reset.raise_for_status()
            episode_id = reset.json()["data"]["episode_id"]
            zoom = client.post(
                "/v2/episodes/%s/step" % episode_id,
                headers={"X-Request-ID": "req-v2-zoom"},
                json=ZOOM_REQUEST,
            )
            zoom.raise_for_status()
            evidence = client.post(
                "/v2/episodes/%s/step" % episode_id,
                headers={"X-Request-ID": "req-v2-evidence"},
                json=EVIDENCE_REQUEST,
            )
            evidence.raise_for_status()
            answer = client.post(
                "/v2/episodes/%s/step" % episode_id,
                headers={"X-Request-ID": "req-v2-answer"},
                json=ANSWER_REQUEST,
            )
            answer.raise_for_status()
            state = client.get(
                "/v2/episodes/%s/state" % episode_id,
                headers={"X-Request-ID": "req-v2-state"},
            )
            state.raise_for_status()
            trace = client.get(
                "/v2/episodes/%s/trace" % episode_id,
                headers={"X-Request-ID": "req-v2-trace"},
            )
            trace.raise_for_status()
            replay = client.post(
                "/v2/episodes/%s/replay" % episode_id,
                headers={"X-Request-ID": "req-v2-replay"},
            )
            replay.raise_for_status()
            validation_error = client.post(
                "/v2/episodes/%s/step" % episode_id,
                headers={"X-Request-ID": "req-v2-error"},
                json={
                    "client_action_id": "invalid-1",
                    "expected_state_version": 3,
                    "action": {
                        "type": "map.zoom",
                        "direction": "in",
                        "factor": "2.0",
                    },
                },
            )
            if validation_error.status_code != 422:
                raise RuntimeError(
                    "expected validation fixture status 422, got %s"
                    % validation_error.status_code
                )

        request_fixtures = {
            "reset-request.json": RESET_REQUEST,
            "zoom-step-request.json": ZOOM_REQUEST,
            "evidence-step-request.json": EVIDENCE_REQUEST,
            "answer-step-request.json": ANSWER_REQUEST,
        }
        response_fixtures = {
            "capabilities-response.json": capabilities.json(),
            "task-response.json": task.json(),
            "reset-response.json": reset.json(),
            "zoom-step-response.json": zoom.json(),
            "evidence-step-response.json": evidence.json(),
            "answer-step-response.json": answer.json(),
            "state-response.json": state.json(),
            "trace-response.json": trace.json(),
            "replay-response.json": replay.json(),
            "validation-error-response.json": validation_error.json(),
        }
        for name, value in request_fixtures.items():
            write_json(fixture_dir / name, value)
        for name, value in response_fixtures.items():
            write_json(fixture_dir / name, normalize_dynamic(value))
        write_json(
            PROJECT_ROOT / "contracts" / "v2" / "openapi-v2.json",
            build_openapi_schema(),
        )


if __name__ == "__main__":
    export()

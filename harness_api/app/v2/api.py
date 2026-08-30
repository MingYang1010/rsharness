from typing import Annotated, Optional, Type, TypeVar

from fastapi import APIRouter, FastAPI, Path, Query, Request
from fastapi.responses import JSONResponse

from . import API_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION
from .capabilities import build_capabilities
from .domain import V2DomainError
from .schemas import (
    EPISODE_ID_PATTERN,
    VERSION_PATTERN,
    CapabilitiesResponse,
    ReplayResponse,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepRequest,
    StepResponse,
    TaskData,
    TaskResponse,
    TraceResponse,
    V2ApiError,
    V2ErrorResponse,
    V2SuccessResponse,
    V2ValidationIssue,
    v2_meta,
)
from .store import V2EpisodeStore


EpisodePath = Annotated[str, Path(pattern=EPISODE_ID_PATTERN)]
TaskIdPath = Annotated[
    str,
    Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"),
]
TaskVersionPath = Annotated[str, Path(pattern=VERSION_PATTERN)]
ResponseT = TypeVar("ResponseT", bound=V2SuccessResponse)

V2_ERROR_RESPONSES = {
    403: {"model": V2ErrorResponse, "description": "Task policy rejected action"},
    404: {"model": V2ErrorResponse, "description": "V2 resource not found"},
    409: {"model": V2ErrorResponse, "description": "State or idempotency conflict"},
    422: {"model": V2ErrorResponse, "description": "V2 request validation failed"},
    500: {"model": V2ErrorResponse, "description": "Internal service error"},
    503: {"model": V2ErrorResponse, "description": "V2 service unavailable"},
}

router = APIRouter(prefix="/v2", tags=["V2 Environment"])


def _store(request: Request) -> V2EpisodeStore:
    store = getattr(request.app.state, "v2_store", None)
    if store is None:
        raise V2DomainError(
            "v2_disabled",
            "V2 routes are disabled by operator configuration",
            status_code=503,
            retryable=True,
            phase="service",
        )
    return store


def _request_id(request: Request) -> str:
    return request.state.request_id


def success(
    response_type: Type[ResponseT],
    request: Request,
    schema: str,
    data: object,
) -> ResponseT:
    return response_type(
        meta=v2_meta(schema, _request_id(request)),
        data=data,
    )


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    retryable: bool,
    phase: str,
    details: Optional[list[V2ValidationIssue]] = None,
) -> JSONResponse:
    body = V2ErrorResponse(
        meta=v2_meta("eo-harness.v2.error.response", _request_id(request)),
        error=V2ApiError(
            code=code,
            message=message,
            retryable=retryable,
            phase=phase,
            details=details or [],
        ),
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json", by_alias=True),
    )


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    responses=V2_ERROR_RESPONSES,
)
def capabilities(request: Request) -> CapabilitiesResponse:
    store = _store(request)
    data = build_capabilities(store.task_registry)
    return success(
        CapabilitiesResponse,
        request,
        "eo-harness.v2.capabilities.response",
        data,
    )


@router.get(
    "/tasks/{task_id}/versions/{task_version}",
    response_model=TaskResponse,
    responses=V2_ERROR_RESPONSES,
)
def get_task(
    request: Request,
    task_id: TaskIdPath,
    task_version: TaskVersionPath,
) -> TaskResponse:
    manifest = _store(request).get_task(task_id, task_version)
    return success(
        TaskResponse,
        request,
        "eo-harness.v2.task.response",
        TaskData(manifest=manifest),
    )


@router.post(
    "/reset",
    status_code=201,
    response_model=ResetResponse,
    responses=V2_ERROR_RESPONSES,
)
def reset(request: Request, body: ResetRequest) -> ResetResponse:
    data = _store(request).create_episode(
        body.task_ref.task_id,
        body.task_ref.task_version,
        body.seed,
    )
    return success(
        ResetResponse,
        request,
        "eo-harness.v2.reset.response",
        data,
    )


@router.get(
    "/episodes/{episode_id}/state",
    response_model=StateResponse,
    responses=V2_ERROR_RESPONSES,
)
def get_state(request: Request, episode_id: EpisodePath) -> StateResponse:
    data = _store(request).get_state(episode_id)
    return success(
        StateResponse,
        request,
        "eo-harness.v2.state.response",
        data,
    )


@router.post(
    "/episodes/{episode_id}/step",
    response_model=StepResponse,
    responses=V2_ERROR_RESPONSES,
)
def step(
    request: Request,
    episode_id: EpisodePath,
    body: StepRequest,
) -> StepResponse:
    data = _store(request).step(
        episode_id=episode_id,
        expected_state_version=body.expected_state_version,
        client_action_id=body.client_action_id,
        action=body.action,
    )
    return success(
        StepResponse,
        request,
        "eo-harness.v2.step.response",
        data,
    )


@router.get(
    "/episodes/{episode_id}/trace",
    response_model=TraceResponse,
    responses=V2_ERROR_RESPONSES,
)
def get_trace(
    request: Request,
    episode_id: EpisodePath,
    cursor: Optional[str] = Query(default=None, max_length=64),
    limit: int = Query(default=100, ge=1, le=1000),
) -> TraceResponse:
    data = _store(request).get_trace(episode_id, cursor, limit)
    return success(
        TraceResponse,
        request,
        "eo-harness.v2.trace.response",
        data,
    )


@router.post(
    "/episodes/{episode_id}/replay",
    response_model=ReplayResponse,
    responses=V2_ERROR_RESPONSES,
)
def replay(request: Request, episode_id: EpisodePath) -> ReplayResponse:
    data = _store(request).structural_replay(episode_id)
    return success(
        ReplayResponse,
        request,
        "eo-harness.v2.replay.response",
        data,
    )


def build_openapi_schema() -> dict:
    schema_app = FastAPI(
        title="EO Harness Environment API V2",
        version=IMPLEMENTATION_VERSION,
        description=(
            "Task-driven, evidence-native EO agent environment contract. "
            "Schema version %s." % SCHEMA_VERSION
        ),
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )
    schema_app.include_router(router)
    schema = schema_app.openapi()
    schema["info"]["x-api-version"] = API_VERSION
    schema["info"]["x-schema-version"] = SCHEMA_VERSION
    return schema

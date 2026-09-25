from typing import Annotated, Optional, Type, TypeVar

from fastapi import APIRouter, FastAPI, Header, Path, Query, Request
from fastapi.responses import JSONResponse, Response

from . import API_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION
from .artifacts import ArtifactStoreError
from .capabilities import build_capabilities
from .domain import V2DomainError
from .schemas import (
    ARTIFACT_ID_PATTERN,
    EPISODE_ID_PATTERN,
    OBSERVATION_ID_PATTERN,
    VERSION_PATTERN,
    ArtifactResponse,
    CapabilitiesResponse,
    EvaluationResponse,
    ObservationResponse,
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
EpisodeQuery = Annotated[str, Query(pattern=EPISODE_ID_PATTERN)]
ObservationPath = Annotated[str, Path(pattern=OBSERVATION_ID_PATTERN)]
ArtifactPath = Annotated[str, Path(pattern=ARTIFACT_ID_PATTERN)]
TaskIdPath = Annotated[
    str,
    Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"),
]
TaskVersionPath = Annotated[str, Path(pattern=VERSION_PATTERN)]
RangeHeader = Annotated[Optional[str], Header(alias="Range")]
ResponseT = TypeVar("ResponseT", bound=V2SuccessResponse)

V2_ERROR_RESPONSES = {
    403: {"model": V2ErrorResponse, "description": "Task policy rejected action"},
    404: {"model": V2ErrorResponse, "description": "V2 resource not found"},
    409: {"model": V2ErrorResponse, "description": "State or idempotency conflict"},
    416: {"model": V2ErrorResponse, "description": "Artifact range is not satisfiable"},
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
    data = build_capabilities(
        store.task_registry,
        store.schema_version(),
        store.renderer_capability(),
        store.evaluator_capability(),
    )
    if store.tool_executor is not None:
        data.actions = [*data.actions, "tool.invoke"]
        data.tools = (store.tool_executor.tool_ids if hasattr(store.tool_executor, "tool_ids")
                      else [store.tool_executor.tool_id])
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


@router.get(
    "/episodes/{episode_id}/observations/{observation_id}",
    response_model=ObservationResponse,
    responses=V2_ERROR_RESPONSES,
)
def get_observation(
    request: Request,
    episode_id: EpisodePath,
    observation_id: ObservationPath,
) -> ObservationResponse:
    data = _store(request).get_observation(episode_id, observation_id)
    return success(
        ObservationResponse,
        request,
        "eo-harness.v2.observation.response",
        data,
    )


@router.get(
    "/artifacts/{artifact_id}",
    response_model=ArtifactResponse,
    responses=V2_ERROR_RESPONSES,
)
def get_artifact(request: Request, artifact_id: ArtifactPath, episode_id: EpisodeQuery) -> ArtifactResponse:
    data = _store(request).get_artifact(artifact_id, episode_id)
    return success(
        ArtifactResponse,
        request,
        "eo-harness.v2.artifact.response",
        data,
    )


@router.get(
    "/artifacts/{artifact_id}/content",
    response_class=Response,
    responses=V2_ERROR_RESPONSES,
)
def get_artifact_content(
    request: Request,
    artifact_id: ArtifactPath,
    episode_id: EpisodeQuery,
    range_header: RangeHeader = None,
) -> Response:
    store = _store(request)
    artifact = store.get_artifact(artifact_id, episode_id).artifact
    if store.artifact_store is None:
        raise V2DomainError(
            "artifact_store_unavailable",
            "artifact content store is unavailable",
            status_code=503,
            retryable=True,
            phase="artifact",
        )
    try:
        content = store.artifact_store.read_content(artifact, range_header)
    except ArtifactStoreError as error:
        range_error = error.code in {"invalid_range", "range_not_satisfiable"}
        raise V2DomainError(
            error.code,
            error.message,
            status_code=416 if range_error else 503,
            retryable=not range_error,
            phase="artifact",
        )
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(len(content.content)),
        "ETag": '"%s"' % artifact.sha256,
    }
    if content.partial:
        headers["Content-Range"] = "bytes %s-%s/%s" % (
            content.start,
            content.end,
            content.total,
        )
    return Response(
        content=content.content,
        status_code=206 if content.partial else 200,
        media_type=artifact.media_type,
        headers=headers,
    )


@router.get(
    "/episodes/{episode_id}/evaluation",
    response_model=EvaluationResponse,
    responses=V2_ERROR_RESPONSES,
)
def get_evaluation(
    request: Request,
    episode_id: EpisodePath,
) -> EvaluationResponse:
    data = _store(request).get_evaluation(episode_id)
    return success(
        EvaluationResponse,
        request,
        "eo-harness.v2.evaluation.response",
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

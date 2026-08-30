import logging
import os
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, Optional, Type, TypeVar

from fastapi import Depends, FastAPI, Header, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from . import __version__
from .contracts import public_episode_result, public_state_result, public_trace
from .domain import ACTION_SPACE, DomainError
from .schemas import (
    API_VERSION,
    EPISODE_ID_PATTERN,
    REQUEST_ID_PATTERN,
    SCHEMA_VERSION,
    ActionSpaceData,
    ActionSpaceResponse,
    ApiError,
    ApiMeta,
    ErrorResponse,
    HealthData,
    HealthResponse,
    ResetRequest,
    ResetResponse,
    ServiceInfoData,
    ServiceInfoResponse,
    StateResponse,
    StepRequest,
    StepResponse,
    SuccessResponse,
    TraceResponse,
    ValidationIssue,
)
from .store import EpisodeStore


LOGGER = logging.getLogger(__name__)
DATABASE_PATH = os.environ.get("EO_HARNESS_DB", "/app/state/episodes.sqlite3")
REQUEST_ID_RE = re.compile(REQUEST_ID_PATTERN)
EpisodePath = Annotated[str, Path(pattern=EPISODE_ID_PATTERN)]
RequestIdHeader = Annotated[
    Optional[str],
    Header(
        alias="X-Request-ID",
        min_length=1,
        max_length=128,
        pattern=REQUEST_ID_PATTERN,
        description="Optional caller request ID echoed in the response body and header.",
    ),
]
ResponseT = TypeVar("ResponseT", bound=SuccessResponse)

ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Resource not found"},
    409: {"model": ErrorResponse, "description": "Episode or idempotency conflict"},
    422: {"model": ErrorResponse, "description": "Request validation failed"},
    500: {"model": ErrorResponse, "description": "Internal service error"},
}


def _validate_request_id(_x_request_id: RequestIdHeader = None) -> None:
    return None


def _request_id(request: Request) -> str:
    return request.state.request_id


def _meta(request: Request, schema: str) -> ApiMeta:
    return ApiMeta(
        api_version=API_VERSION,
        schema_version=SCHEMA_VERSION,
        schema=schema,
        request_id=_request_id(request),
    )


def _success(
    response_type: Type[ResponseT], request: Request, schema: str, data: object
) -> ResponseT:
    return response_type(meta=_meta(request, schema), data=data)


def _error(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Optional[list[ValidationIssue]] = None,
) -> JSONResponse:
    body = ErrorResponse(
        meta=_meta(request, "eo-harness.error.response"),
        error=ApiError(code=code, message=message, details=details or []),
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json", by_alias=True),
    )


def create_app(database_path: Optional[str] = None) -> FastAPI:
    episode_store = EpisodeStore(database_path or DATABASE_PATH)
    application = FastAPI(
        title="EO Harness Environment API",
        version=__version__,
        description=(
            "Stateful, deterministic map-operation environment for Earth-observation "
            "agent episodes. TerriaMap is a renderer; this API owns episode state."
        ),
        dependencies=[Depends(_validate_request_id)],
    )
    application.state.episode_store = episode_store

    @application.middleware("http")
    async def request_id_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        supplied = request.headers.get("X-Request-ID")
        request.state.request_id = (
            supplied
            if supplied is not None and REQUEST_ID_RE.fullmatch(supplied)
            else "req-%s" % uuid.uuid4().hex
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-EO-Harness-API-Version"] = API_VERSION
        response.headers["X-EO-Harness-Schema-Version"] = SCHEMA_VERSION
        return response

    @application.exception_handler(DomainError)
    async def domain_error_handler(
        request: Request, error: DomainError
    ) -> JSONResponse:
        return _error(
            request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        details = [
            ValidationIssue(
                location=list(item["loc"]),
                message=item["msg"],
                type=item["type"],
            )
            for item in error.errors()
        ]
        return _error(
            request,
            status_code=422,
            code="validation_error",
            message="Request validation failed",
            details=details,
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        code = {
            404: "not_found",
            405: "method_not_allowed",
        }.get(error.status_code, "http_error")
        message = error.detail if isinstance(error.detail, str) else "HTTP request failed"
        return _error(request, error.status_code, code, message)

    @application.exception_handler(Exception)
    async def internal_error_handler(request: Request, error: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled EO Harness API error")
        return _error(
            request,
            status_code=500,
            code="internal_error",
            message="The service could not complete the request",
        )

    @application.get(
        "/",
        response_model=ServiceInfoResponse,
        responses=ERROR_RESPONSES,
    )
    def service_info(request: Request) -> ServiceInfoResponse:
        data = ServiceInfoData(
            service="eo-harness-environment-api",
            version=__version__,
            docs="/docs",
            health="/healthz",
        )
        return _success(
            ServiceInfoResponse,
            request,
            "eo-harness.service-info.response",
            data,
        )

    @application.get(
        "/healthz",
        response_model=HealthResponse,
        responses=ERROR_RESPONSES,
    )
    def health(request: Request) -> HealthResponse:
        data = HealthData(status="ok", version=__version__, **episode_store.health())
        return _success(
            HealthResponse,
            request,
            "eo-harness.health.response",
            data,
        )

    @application.get(
        "/v1/action-space",
        response_model=ActionSpaceResponse,
        responses=ERROR_RESPONSES,
    )
    def action_space(request: Request) -> ActionSpaceResponse:
        data = ActionSpaceData(version="v1", actions=ACTION_SPACE)
        return _success(
            ActionSpaceResponse,
            request,
            "eo-harness.action-space.response",
            data,
        )

    @application.post(
        "/v1/reset",
        status_code=201,
        response_model=ResetResponse,
        responses=ERROR_RESPONSES,
    )
    def reset(request: Request, body: ResetRequest) -> ResetResponse:
        specification = body.model_dump(mode="json", exclude_none=True)
        data = public_episode_result(episode_store.create_episode(specification))
        return _success(
            ResetResponse,
            request,
            "eo-harness.reset.response",
            data,
        )

    @application.get(
        "/v1/episodes/{episode_id}/state",
        response_model=StateResponse,
        responses=ERROR_RESPONSES,
    )
    def get_state(request: Request, episode_id: EpisodePath) -> StateResponse:
        data = public_state_result(episode_store.get_state(episode_id))
        return _success(
            StateResponse,
            request,
            "eo-harness.state.response",
            data,
        )

    @application.post(
        "/v1/episodes/{episode_id}/step",
        response_model=StepResponse,
        responses=ERROR_RESPONSES,
    )
    def step(
        request: Request, episode_id: EpisodePath, body: StepRequest
    ) -> StepResponse:
        action = body.action.model_dump(mode="json")
        data = public_episode_result(
            episode_store.step(episode_id, action, body.client_action_id)
        )
        return _success(
            StepResponse,
            request,
            "eo-harness.step.response",
            data,
        )

    @application.get(
        "/v1/episodes/{episode_id}/trace",
        response_model=TraceResponse,
        responses=ERROR_RESPONSES,
    )
    def get_trace(request: Request, episode_id: EpisodePath) -> TraceResponse:
        data = public_trace(episode_store.get_trace(episode_id))
        return _success(
            TraceResponse,
            request,
            "eo-harness.trace.response",
            data,
        )

    return application


app = create_app()

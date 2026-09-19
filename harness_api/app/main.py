import json
import logging
import os
import re
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path as FilePath
from typing import Annotated, Optional, Type, TypeVar

from fastapi import Depends, FastAPI, Header, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from . import __v1_openapi_version__, __version__
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
from .v2.api import (
    build_openapi_schema as build_v2_openapi_schema,
    error_response as v2_error_response,
    router as v2_router,
)
from .v2.artifacts import ArtifactStore
from .v2.capabilities import TaskRegistry
from .v2.domain import V2DomainError
from .v2.evaluation import EvaluatorRegistry
from .v2.renderer.base import RendererAdapter
from .v2.renderer.terriamap import TerriaMapRenderer
from .v2.schemas import V2ValidationIssue
from .v2.store import V2EpisodeStore


LOGGER = logging.getLogger(__name__)
DATABASE_PATH = os.environ.get("EO_HARNESS_DB", "/app/state/episodes.sqlite3")
REQUEST_ID_RE = re.compile(REQUEST_ID_PATTERN)
AGENT_BACKEND_ROUTES = (
    ("GET", re.compile(r"^/healthz$")),
    ("GET", re.compile(r"^/v2/episodes/[^/]+/state$")),
    ("GET", re.compile(r"^/v2/episodes/[^/]+/observations/[^/]+$")),
    ("POST", re.compile(r"^/v2/episodes/[^/]+/step$")),
    ("GET", re.compile(r"^/v2/artifacts/[^/]+(?:/content)?$")),
)
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


def _agent_backend_route_allowed(method: str, path: str) -> bool:
    return any(
        method == allowed_method and pattern.fullmatch(path)
        for allowed_method, pattern in AGENT_BACKEND_ROUTES
    )


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


def create_app(
    database_path: Optional[str] = None,
    v2_tasks_path: Optional[str] = None,
    v2_enabled: Optional[bool] = None,
    v2_artifacts_path: Optional[str] = None,
    v2_datasets_path: Optional[str] = None,
    v2_renderer_config_path: Optional[str] = None,
    v2_renderer: Optional[RendererAdapter] = None,
    v2_evaluator_registry: Optional[EvaluatorRegistry] = None,
    v2_tool_executor=None,
    interface_role: Optional[str] = None,
) -> FastAPI:
    resolved_interface_role = (
        os.environ.get("EO_HARNESS_INTERFACE_ROLE", "operator")
        if interface_role is None
        else interface_role
    )
    if resolved_interface_role not in {"operator", "agent-backend"}:
        raise ValueError("Harness interface role must be operator or agent-backend")
    resolved_database_path = database_path or DATABASE_PATH
    episode_store = EpisodeStore(resolved_database_path)
    enabled = (
        os.environ.get("EO_HARNESS_V2_ENABLED", "1") not in {"0", "false", "False"}
        if v2_enabled is None
        else v2_enabled
    )
    source_root = FilePath(__file__).resolve().parents[2]
    source_tasks_path = source_root / "tasks"
    container_tasks_path = FilePath("/app/tasks")
    default_tasks_path = str(
        container_tasks_path if container_tasks_path.is_dir() else source_tasks_path
    )
    resolved_tasks_path = (
        v2_tasks_path
        or os.environ.get("EO_HARNESS_V2_TASKS")
        or default_tasks_path
    )
    v2_store = None
    if enabled:
        default_artifacts_path = str(
            FilePath(resolved_database_path).resolve().parent / "artifacts"
        )
        resolved_artifacts_path = (
            v2_artifacts_path
            or os.environ.get("EO_HARNESS_V2_ARTIFACTS")
            or default_artifacts_path
        )
        resolved_datasets_path = (
            v2_datasets_path
            or os.environ.get("EO_HARNESS_DATASETS")
            or str(source_root / "datasets")
        )
        resolved_renderer_config_path = FilePath(
            v2_renderer_config_path
            or os.environ.get("EO_HARNESS_V2_RENDERER_CONFIG")
            or source_root / "config" / "v2" / "renderer.json"
        )
        renderer_config = {}
        if resolved_renderer_config_path.is_file():
            with resolved_renderer_config_path.open(encoding="utf-8") as stream:
                renderer_config_value = json.load(stream)
            if not isinstance(renderer_config_value, dict):
                raise RuntimeError("V2 renderer config must contain a JSON object")
            renderer_config = renderer_config_value

        broker = None
        broker_url = os.environ.get("EO_HARNESS_ARTIFACT_BROKER_URL")
        if broker_url:
            from .v2.storage.client import BrokerClient
            token = FilePath(os.environ["EO_HARNESS_ARTIFACT_TOKEN_FILE"]).read_text().strip()
            broker = BrokerClient(broker_url, token)
        artifact_store = ArtifactStore(resolved_artifacts_path, broker=broker)
        tool_executor = v2_tool_executor
        provider_url = os.environ.get("EO_HARNESS_EO_GYM_URL")
        if tool_executor is None and provider_url:
            from .v2.tools.eo_gym import EOGymExecutor
            tool_executor = EOGymExecutor(provider_url, artifact_store)
        raster_url = os.environ.get("EO_HARNESS_RASTER_URL")
        if os.environ.get("EO_HARNESS_CATALOG_ENABLED") == "1" or raster_url:
            from .v2.tools.runtime import ToolRouter
            from .v2.tools.raster import RasterExecutor
            from .v2.tools.raster_grid import RasterGridExecutor
            from .v2.tools.temporal import TemporalExecutor
            tool_executor = ToolRouter(tool_executor,
                RasterExecutor(raster_url, artifact_store) if raster_url else None,
                RasterGridExecutor(raster_url, artifact_store) if raster_url else None,
                TemporalExecutor(raster_url, artifact_store) if raster_url else None)
        evaluator_registry = v2_evaluator_registry or EvaluatorRegistry(
            resolved_datasets_path,
            artifact_store,
        )
        renderer = v2_renderer
        renderer_url = os.environ.get("EO_HARNESS_V2_RENDERER_URL")
        if (
            renderer is None
            and renderer_url
            and renderer_config.get("enabled", False)
        ):
            renderer = TerriaMapRenderer(
                renderer_url,
                artifact_store,
                renderer_config,
            )
        v2_store = V2EpisodeStore(
            resolved_database_path,
            TaskRegistry(resolved_tasks_path),
            artifact_store=artifact_store,
            renderer=renderer,
            evaluator_registry=evaluator_registry,
            renderer_config=renderer_config,
            tool_executor=tool_executor,
        )
    application = FastAPI(
        title="EO Harness Environment API",
        version=__v1_openapi_version__,
        description=(
            "Stateful, deterministic map-operation environment for Earth-observation "
            "agent episodes. TerriaMap is a renderer; this API owns episode state."
        ),
        dependencies=[Depends(_validate_request_id)],
    )
    application.state.episode_store = episode_store
    application.state.v2_store = v2_store
    application.include_router(v2_router, include_in_schema=False)

    @application.get("/v2/openapi.json", include_in_schema=False)
    def v2_openapi() -> JSONResponse:
        return JSONResponse(content=build_v2_openapi_schema())

    @application.middleware("http")
    async def interface_boundary_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if (
            resolved_interface_role == "agent-backend"
            and not _agent_backend_route_allowed(request.method, request.url.path)
        ):
            return JSONResponse(
                {"error": {"code": "interface_route_denied"}},
                status_code=404,
            )
        return await call_next(request)

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
        is_v2 = request.url.path.startswith("/v2")
        response.headers["X-EO-Harness-API-Version"] = "v2" if is_v2 else API_VERSION
        response.headers["X-EO-Harness-Schema-Version"] = (
            "2.0.0" if is_v2 else SCHEMA_VERSION
        )
        return response

    @application.exception_handler(V2DomainError)
    async def v2_domain_error_handler(
        request: Request,
        error: V2DomainError,
    ) -> JSONResponse:
        details = [V2ValidationIssue.model_validate(item) for item in error.details]
        return v2_error_response(
            request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            retryable=error.retryable,
            phase=error.phase,
            details=details,
        )

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
        if request.url.path.startswith("/v2"):
            v2_details = [
                V2ValidationIssue(
                    location=list(item["loc"]),
                    message=item["msg"],
                    type=item["type"],
                    field=str(item["loc"][-1]) if item["loc"] else None,
                )
                for item in error.errors()
            ]
            return v2_error_response(
                request,
                status_code=422,
                code="validation_error",
                message="V2 request validation failed",
                retryable=False,
                phase="request",
                details=v2_details,
            )
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
        if request.url.path.startswith("/v2"):
            code = {
                404: "not_found",
                405: "method_not_allowed",
            }.get(error.status_code, "http_error")
            message = (
                error.detail
                if isinstance(error.detail, str)
                else "V2 HTTP request failed"
            )
            return v2_error_response(
                request,
                status_code=error.status_code,
                code=code,
                message=message,
                retryable=False,
                phase="request",
            )
        code = {
            404: "not_found",
            405: "method_not_allowed",
        }.get(error.status_code, "http_error")
        message = error.detail if isinstance(error.detail, str) else "HTTP request failed"
        return _error(request, error.status_code, code, message)

    @application.exception_handler(Exception)
    async def internal_error_handler(request: Request, error: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled EO Harness API error")
        if request.url.path.startswith("/v2"):
            return v2_error_response(
                request,
                status_code=500,
                code="internal_error",
                message="The V2 service could not complete the request",
                retryable=False,
                phase="service",
            )
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

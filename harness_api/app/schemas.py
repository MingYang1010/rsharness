from typing import Annotated, Any, Dict, Generic, List, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator


API_VERSION = "v1"
SCHEMA_VERSION = "1.0.0"
EPISODE_ID_PATTERN = r"^ep-[a-f0-9]{32}$"
REQUEST_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"
UTC_TIMESTAMP_PATTERN = (
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)

NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
TaskId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
LayerId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
EvidenceRef = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2048),
]
EpisodeId = Annotated[str, Field(pattern=EPISODE_ID_PATTERN)]
RequestId = Annotated[str, Field(pattern=REQUEST_ID_PATTERN)]
Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
UtcTimestamp = Annotated[str, Field(pattern=UTC_TIMESTAMP_PATTERN)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class RequestModel(ContractModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, strict=True)


class BoundingBox(RequestModel):
    west: float = Field(ge=-180.0, le=180.0, allow_inf_nan=False)
    south: float = Field(ge=-90.0, le=90.0, allow_inf_nan=False)
    east: float = Field(ge=-180.0, le=180.0, allow_inf_nan=False)
    north: float = Field(ge=-90.0, le=90.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_order(self) -> "BoundingBox":
        if self.west >= self.east:
            raise ValueError("west must be less than east")
        if self.south >= self.north:
            raise ValueError("south must be less than north")
        return self


class ResetRequest(RequestModel):
    task_id: TaskId = "interactive-map-task"
    prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=10000),
    ] = "Inspect the active Earth-observation layers."
    seed: int = Field(default=0, ge=0, le=2147483647)
    max_steps: int = Field(default=20, ge=1, le=1000)
    initial_view: Optional[BoundingBox] = None
    metadata: Dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_metadata_size(self) -> "ResetRequest":
        if len(self.model_dump_json().encode("utf-8")) > 65536:
            raise ValueError("reset request must not exceed 65536 JSON bytes")
        return self


class SetViewAction(RequestModel):
    type: Literal["set_view"]
    bbox: BoundingBox


class PanAction(RequestModel):
    type: Literal["pan"]
    delta_longitude: float = Field(ge=-360.0, le=360.0, allow_inf_nan=False)
    delta_latitude: float = Field(ge=-180.0, le=180.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_movement(self) -> "PanAction":
        if self.delta_longitude == 0.0 and self.delta_latitude == 0.0:
            raise ValueError("pan must change longitude or latitude")
        return self


class ZoomAction(RequestModel):
    type: Literal["zoom"]
    direction: Literal["in", "out"]
    factor: float = Field(default=2.0, gt=1.0, le=8.0, allow_inf_nan=False)


class SetLayerVisibilityAction(RequestModel):
    type: Literal["set_layer_visibility"]
    layer_id: LayerId
    visible: bool


class SetLayerOpacityAction(RequestModel):
    type: Literal["set_layer_opacity"]
    layer_id: LayerId
    opacity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class SubmitAnswerAction(RequestModel):
    type: Literal["submit_answer"]
    answer: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=50000),
    ]
    evidence_refs: List[EvidenceRef] = Field(default_factory=list, max_length=1000)


Action = Annotated[
    Union[
        SetViewAction,
        PanAction,
        ZoomAction,
        SetLayerVisibilityAction,
        SetLayerOpacityAction,
        SubmitAnswerAction,
    ],
    Field(discriminator="type"),
]


class StepRequest(RequestModel):
    action: Action
    client_action_id: Optional[
        Annotated[str, Field(min_length=1, max_length=128, pattern=REQUEST_ID_PATTERN)]
    ] = None


class ApiMeta(ContractModel):
    api_version: Literal["v1"]
    schema_version: Literal["1.0.0"]
    schema_id: Annotated[
        str,
        Field(alias="schema", serialization_alias="schema", min_length=1, max_length=128),
    ]
    request_id: RequestId


class ServiceInfoData(ContractModel):
    service: Literal["eo-harness-environment-api"]
    version: str
    docs: Literal["/docs"]
    health: Literal["/healthz"]


class HealthData(ContractModel):
    status: Literal["ok"]
    version: str
    database: Literal["ok"]


class ActionDescriptor(ContractModel):
    type: Literal[
        "set_view",
        "pan",
        "zoom",
        "set_layer_visibility",
        "set_layer_opacity",
        "submit_answer",
    ]
    description: str
    fields: Dict[str, str]


class ActionSpaceData(ContractModel):
    version: Literal["v1"]
    actions: List[ActionDescriptor]


class GeoPoint(ContractModel):
    longitude: float = Field(ge=-180.0, le=180.0, allow_inf_nan=False)
    latitude: float = Field(ge=-90.0, le=90.0, allow_inf_nan=False)


class MapView(ContractModel):
    bbox: BoundingBox
    center: GeoPoint


class LayerState(ContractModel):
    layer_id: LayerId
    name: str
    kind: str
    source: str
    visible: bool
    opacity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class TaskState(ContractModel):
    task_id: TaskId
    prompt: str
    seed: int = Field(ge=0, le=2147483647)
    metadata: Dict[str, JsonValue]


class FinalAnswer(ContractModel):
    answer: NonEmptyText
    evidence_refs: List[EvidenceRef]


class EpisodeState(ContractModel):
    episode_id: EpisodeId
    state_version: int = Field(ge=0)
    status: Literal["active", "terminated", "truncated"]
    step_count: int = Field(ge=0, le=1000)
    max_steps: int = Field(ge=1, le=1000)
    task: TaskState
    view: MapView
    layers: Dict[str, LayerState]
    final_answer: Optional[FinalAnswer]
    created_at: UtcTimestamp
    updated_at: UtcTimestamp


class MapObservation(ContractModel):
    observation_type: Literal["map_state"]
    sequence: int = Field(ge=0, le=1000)
    message: str
    view: MapView
    visible_layers: List[LayerState]
    state_hash: Sha256
    semantic_state_hash: Sha256


class StepInfo(ContractModel):
    client_action_id: Optional[str]
    remaining_steps: int = Field(ge=0, le=1000)
    state_hash: Sha256
    semantic_state_hash: Sha256


class EpisodeResultData(ContractModel):
    episode_id: EpisodeId
    state: EpisodeState
    observation: MapObservation
    reward: Optional[float]
    terminated: bool
    truncated: bool
    info: StepInfo


class StateData(ContractModel):
    episode_id: EpisodeId
    state: EpisodeState
    state_hash: Sha256
    semantic_state_hash: Sha256


class TraceTransition(ContractModel):
    sequence: int = Field(ge=1, le=1000)
    client_action_id: Optional[str]
    created_at: UtcTimestamp
    action: Action
    observation: MapObservation
    state_hash: Sha256
    semantic_state_hash: Sha256
    state: EpisodeState


class TraceData(ContractModel):
    episode_id: EpisodeId
    initial_state: EpisodeState
    transitions: List[TraceTransition] = Field(max_length=1000)
    final_state: EpisodeState
    transition_count: int = Field(ge=0, le=1000)
    hash_algorithm: Literal["sha256"]
    trace_hash: Sha256
    semantic_trace_hash: Sha256


DataT = TypeVar("DataT")


class SuccessResponse(ContractModel, Generic[DataT]):
    meta: ApiMeta
    data: DataT


class ServiceInfoResponse(SuccessResponse[ServiceInfoData]):
    pass


class HealthResponse(SuccessResponse[HealthData]):
    pass


class ActionSpaceResponse(SuccessResponse[ActionSpaceData]):
    pass


class ResetResponse(SuccessResponse[EpisodeResultData]):
    pass


class StepResponse(SuccessResponse[EpisodeResultData]):
    pass


class StateResponse(SuccessResponse[StateData]):
    pass


class TraceResponse(SuccessResponse[TraceData]):
    pass


class ValidationIssue(ContractModel):
    location: List[Union[str, int]]
    message: str
    type: str


class ApiError(ContractModel):
    code: str
    message: str
    details: List[ValidationIssue] = Field(default_factory=list)


class ErrorResponse(ContractModel):
    meta: ApiMeta
    error: ApiError

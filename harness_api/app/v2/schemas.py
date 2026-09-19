from typing import Annotated, Dict, Generic, List, Literal, Optional, TypeVar, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

from . import API_VERSION, SCHEMA_VERSION
from .artifact_identity import DERIVATION_SCHEME, LEGACY_SCHEME


EPISODE_ID_PATTERN = r"^ep2-[a-f0-9]{32}$"
OBSERVATION_ID_PATTERN = r"^obs-[a-f0-9]{32}$"
EVENT_ID_PATTERN = r"^evt-[a-f0-9]{32}$"
ARTIFACT_ID_PATTERN = r"^art-[a-f0-9]{64}$"
EVIDENCE_ID_PATTERN = r"^ev-[A-Za-z0-9][A-Za-z0-9._:-]{0,123}$"
REQUEST_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
ACTION_ID_PATTERN = REQUEST_ID_PATTERN
SHA256_PATTERN = r"^[a-f0-9]{64}$"
VERSION_PATTERN = r"^[0-9]+\.[0-9]+\.[0-9]+$"
UTC_TIMESTAMP_PATTERN = (
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)

NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
EpisodeId = Annotated[str, Field(pattern=EPISODE_ID_PATTERN)]
ObservationId = Annotated[str, Field(pattern=OBSERVATION_ID_PATTERN)]
EventId = Annotated[str, Field(pattern=EVENT_ID_PATTERN)]
ArtifactId = Annotated[str, Field(pattern=ARTIFACT_ID_PATTERN)]
EvidenceId = Annotated[str, Field(pattern=EVIDENCE_ID_PATTERN)]
RequestId = Annotated[str, Field(pattern=REQUEST_ID_PATTERN)]
Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
UtcTimestamp = Annotated[str, Field(pattern=UTC_TIMESTAMP_PATTERN)]
SemanticVersion = Annotated[str, Field(pattern=VERSION_PATTERN)]


class V2ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class V2RequestModel(V2ContractModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, strict=True)


class SpatialBoundingBox(V2ContractModel):
    west: float = Field(ge=-180.0, le=180.0, allow_inf_nan=False)
    south: float = Field(ge=-90.0, le=90.0, allow_inf_nan=False)
    east: float = Field(ge=-180.0, le=180.0, allow_inf_nan=False)
    north: float = Field(ge=-90.0, le=90.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_order(self) -> "SpatialBoundingBox":
        if self.west >= self.east:
            raise ValueError("west must be less than east")
        if self.south >= self.north:
            raise ValueError("south must be less than north")
        return self


class GeoPoint(V2ContractModel):
    longitude: float = Field(ge=-180.0, le=180.0, allow_inf_nan=False)
    latitude: float = Field(ge=-90.0, le=90.0, allow_inf_nan=False)


class SpatialExtent(V2ContractModel):
    crs: NonEmptyText
    bbox: SpatialBoundingBox
    geometry: Optional[Dict[str, JsonValue]] = None
    gsd_meters: Optional[float] = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
    )
    shape: Optional[List[int]] = Field(default=None, min_length=2, max_length=3)


class TemporalExtent(V2ContractModel):
    start: UtcTimestamp
    end: UtcTimestamp

    @model_validator(mode="after")
    def validate_order(self) -> "TemporalExtent":
        if self.start > self.end:
            raise ValueError("temporal start must not be after end")
        return self


class BudgetSpec(V2ContractModel):
    max_steps: int = Field(ge=1, le=10000)
    max_tool_calls: int = Field(ge=0, le=10000)
    max_wall_time_ms: int = Field(ge=1, le=86400000)
    max_input_bytes: int = Field(ge=0, le=1099511627776)
    max_artifact_bytes: int = Field(ge=0, le=1099511627776)


class TaskSpec(V2ContractModel):
    task_id: Identifier
    task_version: SemanticVersion
    family: Identifier
    prompt: Annotated[str, Field(min_length=1, max_length=50000)]
    inputs: List[Identifier] = Field(min_length=1, max_length=1000)
    scenario_profile: Identifier
    answer_schema: Dict[str, JsonValue]
    evaluator: Identifier
    budget: BudgetSpec
    seed: int = Field(ge=0, le=2147483647)
    metric_aggregation: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, JsonValue] = Field(default_factory=dict)


class ScenarioProfile(V2ContractModel):
    profile_id: Identifier
    domain: Identifier
    data_cutoff: UtcTimestamp
    freshness_max_age_seconds: Optional[int] = Field(default=None, ge=0)
    allowed_actions: List[str]
    allowed_tools: List[str]
    network_policy: Literal["none", "catalog_only", "allowlisted"]
    evidence_required: bool
    abstention_allowed: bool
    human_review_policy: Literal[
        "never",
        "allowed",
        "high_risk_or_low_confidence",
    ]


class AssetQuality(V2ContractModel):
    cloud_cover_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
    )
    nodata_fraction: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    coverage_fraction: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )


class AssetRef(V2ContractModel):
    asset_id: Identifier
    uri: NonEmptyText
    media_type: NonEmptyText
    roles: List[NonEmptyText] = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    spatial: SpatialExtent
    temporal: Optional[TemporalExtent] = None
    platform: Optional[str] = None
    instrument: Optional[str] = None
    bands: List[str] = Field(default_factory=list)
    polarizations: List[str] = Field(default_factory=list)
    quality: AssetQuality = Field(default_factory=AssetQuality)
    license: NonEmptyText
    source: NonEmptyText
    source_snapshot_hash: Optional[Sha256] = None


class PixelExtent(V2ContractModel):
    """Image coordinates: origin top-left, x right, y down, half-open windows."""

    coordinate_system: Literal["pixel"]
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    channels: int = Field(gt=0, strict=True)


class PixelAssetRef(AssetRef):
    """Separate variant keeps legacy georeferenced asset JSON byte-compatible."""

    spatial: None = None
    pixel: PixelExtent


TaskAsset = Union[AssetRef, PixelAssetRef]


class EvaluatorSpec(V2ContractModel):
    evaluator_id: Identifier
    evaluator_version: SemanticVersion
    metric_names: List[Identifier]
    aggregate_weights: Dict[str, float] = Field(default_factory=dict)
    config: Dict[str, JsonValue] = Field(default_factory=dict)


class TaskRef(V2ContractModel):
    task_id: Identifier
    task_version: SemanticVersion


class TaskManifest(V2ContractModel):
    task: TaskSpec
    scenario: ScenarioProfile
    assets: List[TaskAsset]
    evaluator: EvaluatorSpec
    task_manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_asset_coordinates(self) -> "TaskManifest":
        identity = self.task.metadata.get("artifact_identity", LEGACY_SCHEME)
        if identity not in (LEGACY_SCHEME, DERIVATION_SCHEME):
            raise ValueError("unsupported artifact identity policy")
        if identity == DERIVATION_SCHEME and self.task.metadata.get("observation_profile") != "headless-tools-v1":
            raise ValueError("derivation identity currently requires headless-tools-v1")
        asset_ids = [asset.asset_id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("duplicate asset IDs are ambiguous")
        if len(self.task.inputs) != len(set(self.task.inputs)):
            raise ValueError("duplicate task inputs are ambiguous")
        if not set(self.task.inputs).issubset(asset_ids):
            raise ValueError("task inputs must refer to manifest assets")
        inputs = [asset for asset in self.assets if asset.asset_id in self.task.inputs]
        if any(isinstance(asset, PixelAssetRef) for asset in inputs):
            if self.task.metadata.get("observation_profile") != "headless-tools-v1":
                raise ValueError("pixel-only inputs require headless-tools-v1")
            if any(action.startswith("map.") for action in self.scenario.allowed_actions):
                raise ValueError("pixel-only tasks cannot allow geographic map actions")
        return self


class ArtifactLineage(V2ContractModel):
    tool_id: Identifier
    tool_version: SemanticVersion
    input_refs: List[Identifier]
    parameters_hash: Sha256


class ArtifactRef(V2ContractModel):
    artifact_id: ArtifactId
    kind: Literal["raster", "vector", "table", "image", "text"]
    media_type: NonEmptyText
    uri: Annotated[str, Field(pattern=r"^artifact://sha256/[a-f0-9]{2}/[a-f0-9]{64}$")]
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    spatial: Optional[SpatialExtent] = None
    temporal: Optional[TemporalExtent] = None
    lineage: ArtifactLineage


class TemporalStackMember(V2ContractModel):
    item_id: Identifier
    acquired: UtcTimestamp
    platform: Identifier
    instrument: Identifier
    red_asset_id: Identifier
    scl_asset_id: Identifier
    bands: List[NonEmptyText] = Field(min_length=2, max_length=2)
    coverage_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    cloud_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def fixed_bands(self) -> "TemporalStackMember":
        if self.bands != ["red", "scl"]:
            raise ValueError("temporal stack member bands must be red then scl")
        if self.red_asset_id == self.scl_asset_id:
            raise ValueError("temporal stack member inputs must be distinct")
        return self


class TemporalStackDescriptor(V2ContractModel):
    before: TemporalStackMember
    after: TemporalStackMember
    grid_crs: NonEmptyText
    grid_transform: List[float] = Field(min_length=6, max_length=6)
    width: int = Field(gt=0, le=1024)
    height: int = Field(gt=0, le=1024)
    band_order: List[NonEmptyText] = Field(min_length=4, max_length=4)
    cloud_policy: Identifier
    alignment_method: Identifier

    @model_validator(mode="after")
    def ordered(self) -> "TemporalStackDescriptor":
        if self.band_order != ["before_red", "before_scl", "after_red", "after_scl"]:
            raise ValueError("temporal stack band order is fixed")
        if self.before.acquired >= self.after.acquired:
            raise ValueError("temporal stack members must be ordered")
        input_ids = [
            self.before.red_asset_id,
            self.before.scl_asset_id,
            self.after.red_asset_id,
            self.after.scl_asset_id,
        ]
        if len(set(input_ids)) != 4:
            raise ValueError("temporal stack input assets must be distinct")
        return self


class TemporalStackArtifactRef(ArtifactRef):
    kind: Literal["raster"]
    media_type: Literal["image/tiff"]
    spatial: SpatialExtent
    temporal: TemporalExtent
    temporal_stack: TemporalStackDescriptor

    @model_validator(mode="after")
    def validate_temporal_stack(self) -> "TemporalStackArtifactRef":
        stack = self.temporal_stack
        if (
            self.temporal.start != stack.before.acquired
            or self.temporal.end != stack.after.acquired
        ):
            raise ValueError("artifact temporal extent disagrees with stack members")
        if self.spatial.shape != [stack.height, stack.width, 4]:
            raise ValueError("artifact spatial shape disagrees with temporal stack")
        expected = [
            stack.before.red_asset_id,
            stack.before.scl_asset_id,
            stack.after.red_asset_id,
            stack.after.scl_asset_id,
        ]
        if self.lineage.input_refs != expected:
            raise ValueError("artifact lineage disagrees with temporal stack inputs")
        return self


class PixelArtifactRef(ArtifactRef):
    """Opt-in decoded image dimensions; legacy ArtifactRef JSON stays unchanged."""

    pixel: PixelExtent

    @model_validator(mode="after")
    def validate_pixel_kind(self) -> "PixelArtifactRef":
        if self.kind not in {"image", "raster"}:
            raise ValueError("pixel dimensions require an image or raster artifact")
        return self


Artifact = Union[TemporalStackArtifactRef, PixelArtifactRef, ArtifactRef]


class EvidenceSelector(V2ContractModel):
    geometry: Optional[Dict[str, JsonValue]] = None
    bbox: Optional[SpatialBoundingBox] = None
    time_range: Optional[TemporalExtent] = None
    bands: List[str] = Field(default_factory=list)
    pixel_window: Optional[List[Annotated[int, Field(strict=True)]]] = Field(default=None, min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_selector(self) -> "EvidenceSelector":
        if not any(
            (
                self.geometry is not None,
                self.bbox is not None,
                self.time_range is not None,
                bool(self.bands),
                self.pixel_window is not None,
            )
        ):
            raise ValueError("evidence selector must constrain the source")
        if self.pixel_window is not None:
            x, y, width, height = self.pixel_window
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError("pixel_window must be [x, y, positive width, positive height]")
        return self


class EvidenceRef(V2ContractModel):
    evidence_id: EvidenceId
    claim_id: Identifier
    source_ref: Identifier
    selector: EvidenceSelector
    description: Annotated[str, Field(min_length=1, max_length=10000)]
    frozen_sha256: Sha256


class Metric(V2ContractModel):
    name: Identifier
    value: float = Field(allow_inf_nan=False)
    weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )
    diagnostics: Dict[str, JsonValue] = Field(default_factory=dict)


class MetricResult(V2ContractModel):
    evaluation_id: Identifier
    status: Literal["pending", "completed", "failed"]
    metrics: List[Metric]
    aggregate_reward: Optional[float] = Field(default=None, allow_inf_nan=False)
    evaluator_id: Identifier
    evaluator_version: SemanticVersion
    diagnostics: Dict[str, JsonValue] = Field(default_factory=dict)


class MapLayerState(V2ContractModel):
    layer_id: Identifier
    asset_id: Identifier
    name: str
    visible: bool
    opacity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    style_id: Identifier
    time_range: Optional[TemporalExtent] = None


class MapState(V2ContractModel):
    bbox: SpatialBoundingBox
    center: GeoPoint
    layers: Dict[str, MapLayerState]
    active_time_range: Optional[TemporalExtent] = None


ObservationType = Literal[
    "map_state",
    "rendered_view",
    "raster_chip",
    "temporal_stack",
    "vector_features",
    "table",
    "tool_result",
    "asset_metadata",
]


class ObservationItem(V2ContractModel):
    type: ObservationType
    inline: Optional[Dict[str, JsonValue]] = None
    artifact_ref: Optional[ArtifactId] = None
    asset_refs: List[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_payload(self) -> "ObservationItem":
        present = sum(
            (
                self.inline is not None,
                self.artifact_ref is not None,
                bool(self.asset_refs),
            )
        )
        if present != 1:
            raise ValueError("observation item must have exactly one payload source")
        return self


class Observation(V2ContractModel):
    observation_id: ObservationId
    sequence: int = Field(ge=0)
    primary_type: ObservationType
    items: List[ObservationItem] = Field(min_length=1)
    state_hash: Sha256
    semantic_state_hash: Sha256
    provenance: Dict[str, JsonValue]
    warnings: List[str]


class BudgetCounter(V2ContractModel):
    limit: int = Field(ge=0)
    used: int = Field(ge=0)
    remaining: int = Field(ge=0)


class BudgetCounters(V2ContractModel):
    steps: BudgetCounter
    tool_calls: BudgetCounter
    wall_time_ms: BudgetCounter
    input_bytes: BudgetCounter
    artifact_bytes: BudgetCounter


class AnswerRecord(V2ContractModel):
    outcome: Literal["submitted", "abstained", "human_review_requested"]
    answer: Optional[JsonValue] = None
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    evidence_ids: List[EvidenceId]
    rationale: Optional[str] = None


class V2EpisodeState(V2ContractModel):
    episode_id: EpisodeId
    task_ref: TaskRef
    task_manifest_hash: Sha256
    seed: int = Field(ge=0, le=2147483647)
    state_version: int = Field(ge=0)
    status: Literal["active", "terminated", "truncated", "failed"]
    step_count: int = Field(ge=0)
    map: Optional[MapState]
    budget: BudgetCounters
    accessible_asset_refs: List[Identifier]
    observation_refs: List[ObservationId]
    evidence_refs: List[EvidenceRef]
    final_answer: Optional[AnswerRecord]
    evaluation: Optional[MetricResult]
    created_at: UtcTimestamp
    updated_at: UtcTimestamp


class MapSetViewAction(V2RequestModel):
    type: Literal["map.set_view"]
    bbox: SpatialBoundingBox


class MapPanAction(V2RequestModel):
    type: Literal["map.pan"]
    delta_longitude: float = Field(ge=-360.0, le=360.0, allow_inf_nan=False)
    delta_latitude: float = Field(ge=-180.0, le=180.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_movement(self) -> "MapPanAction":
        if self.delta_longitude == 0.0 and self.delta_latitude == 0.0:
            raise ValueError("pan must change longitude or latitude")
        return self


class MapZoomAction(V2RequestModel):
    type: Literal["map.zoom"]
    direction: Literal["in", "out"]
    factor: float = Field(default=2.0, gt=1.0, le=8.0, allow_inf_nan=False)


class MapLayerVisibilityAction(V2RequestModel):
    type: Literal["map.layer.set_visibility"]
    layer_id: Identifier
    visible: bool


class MapLayerOpacityAction(V2RequestModel):
    type: Literal["map.layer.set_opacity"]
    layer_id: Identifier
    opacity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class MapTimeRangeAction(V2RequestModel):
    type: Literal["map.time.set_range"]
    time_range: TemporalExtent


class MemorySaveEvidenceAction(V2RequestModel):
    type: Literal["memory.save_evidence"]
    evidence: EvidenceRef


class MemoryBookmarkAoiAction(V2RequestModel):
    type: Literal["memory.bookmark_aoi"]
    bookmark_id: Identifier
    bbox: SpatialBoundingBox
    label: NonEmptyText


class ToolInvokeAction(V2RequestModel):
    type: Literal["tool.invoke"]
    tool_id: Identifier
    arguments: Dict[str, JsonValue]


class AnswerSubmitAction(V2RequestModel):
    type: Literal["answer.submit"]
    answer: JsonValue
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    evidence_ids: List[EvidenceId]


class AnswerAbstainAction(V2RequestModel):
    type: Literal["answer.abstain"]
    rationale: Annotated[str, Field(min_length=1, max_length=10000)]
    evidence_ids: List[EvidenceId] = Field(default_factory=list)


class AnswerHumanReviewAction(V2RequestModel):
    type: Literal["answer.request_human_review"]
    rationale: Annotated[str, Field(min_length=1, max_length=10000)]
    evidence_ids: List[EvidenceId] = Field(default_factory=list)


V2Action = Annotated[
    Union[
        MapSetViewAction,
        MapPanAction,
        MapZoomAction,
        MapLayerVisibilityAction,
        MapLayerOpacityAction,
        MapTimeRangeAction,
        MemorySaveEvidenceAction,
        MemoryBookmarkAoiAction,
        ToolInvokeAction,
        AnswerSubmitAction,
        AnswerAbstainAction,
        AnswerHumanReviewAction,
    ],
    Field(discriminator="type"),
]


class ResetRequest(V2RequestModel):
    task_ref: TaskRef
    seed: Optional[int] = Field(default=None, ge=0, le=2147483647)


class StepRequest(V2RequestModel):
    client_action_id: Annotated[
        str,
        Field(min_length=1, max_length=128, pattern=ACTION_ID_PATTERN),
    ]
    expected_state_version: int = Field(ge=0)
    action: V2Action


class V2ApiMeta(V2ContractModel):
    api_version: Literal["v2"]
    schema_version: Literal["2.0.0"]
    schema_id: Annotated[
        str,
        Field(alias="schema", serialization_alias="schema", min_length=1, max_length=128),
    ]
    request_id: RequestId


class V2ValidationIssue(V2ContractModel):
    location: List[Union[str, int]]
    message: str
    type: str
    field: Optional[str] = None
    tool_id: Optional[Identifier] = None


class V2ApiError(V2ContractModel):
    code: Identifier
    message: str
    retryable: bool
    phase: Literal[
        "request",
        "policy",
        "state",
        "tool",
        "renderer",
        "artifact",
        "evaluation",
        "service",
    ]
    details: List[V2ValidationIssue] = Field(default_factory=list)


class V2ErrorResponse(V2ContractModel):
    meta: V2ApiMeta
    error: V2ApiError


DataT = TypeVar("DataT")


class V2SuccessResponse(V2ContractModel, Generic[DataT]):
    meta: V2ApiMeta
    data: DataT


class CapabilityStatus(V2ContractModel):
    status: Literal["available", "unavailable", "not_implemented"]
    version: Optional[str] = None
    details: Dict[str, JsonValue] = Field(default_factory=dict)


class CapabilitiesData(V2ContractModel):
    implementation_version: SemanticVersion
    store_schema_version: int = Field(ge=1)
    enabled: bool
    task_count: int = Field(ge=0)
    actions: List[str]
    declared_actions: List[str]
    observation_types: List[ObservationType]
    tools: List[Identifier]
    renderer: CapabilityStatus
    evaluator: CapabilityStatus
    structural_replay: CapabilityStatus


class TaskData(V2ContractModel):
    manifest: TaskManifest


class EpisodeResultData(V2ContractModel):
    episode_id: EpisodeId
    task_manifest_hash: Sha256
    state: V2EpisodeState
    observation: Observation
    terminated: bool
    truncated: bool


class StateData(V2ContractModel):
    episode_id: EpisodeId
    state: V2EpisodeState
    state_hash: Sha256
    semantic_state_hash: Sha256


class ObservationData(V2ContractModel):
    episode_id: EpisodeId
    observation: Observation


class ArtifactData(V2ContractModel):
    artifact: Artifact


class EvaluationData(V2ContractModel):
    episode_id: EpisodeId
    evaluation: MetricResult


class EventRecord(V2ContractModel):
    event_id: EventId
    episode_id: EpisodeId
    sequence: int = Field(ge=0)
    event_type: Literal[
        "episode.created",
        "action.accepted",
        "action.completed",
        "action.failed",
        "observation.emitted",
        "artifact.created",
        "artifact.failed",
        "evidence.saved",
        "evaluation.completed",
        "evaluation.failed",
        "episode.terminated",
        "episode.truncated",
        "episode.failed",
    ]
    state_version: int = Field(ge=0)
    created_at: UtcTimestamp
    payload: Dict[str, JsonValue]


class TraceData(V2ContractModel):
    episode_id: EpisodeId
    events: List[EventRecord]
    limit: int = Field(ge=1, le=1000)
    next_cursor: Optional[str]
    has_more: bool
    total_events: int = Field(ge=0)
    hash_algorithm: Literal["sha256"]
    trace_hash: Sha256
    semantic_trace_hash: Sha256


class ReplayCheck(V2ContractModel):
    name: Identifier
    passed: bool
    expected: Optional[str] = None
    actual: Optional[str] = None


class ReplayData(V2ContractModel):
    episode_id: EpisodeId
    mode: Literal["structural"]
    status: Literal["passed", "failed"]
    checked_event_count: int = Field(ge=0)
    checks: List[ReplayCheck]
    trace_hash: Sha256
    semantic_trace_hash: Sha256


class CapabilitiesResponse(V2SuccessResponse[CapabilitiesData]):
    pass


class TaskResponse(V2SuccessResponse[TaskData]):
    pass


class ResetResponse(V2SuccessResponse[EpisodeResultData]):
    pass


class StepResponse(V2SuccessResponse[EpisodeResultData]):
    pass


class StateResponse(V2SuccessResponse[StateData]):
    pass


class ObservationResponse(V2SuccessResponse[ObservationData]):
    pass


class ArtifactResponse(V2SuccessResponse[ArtifactData]):
    pass


class EvaluationResponse(V2SuccessResponse[EvaluationData]):
    pass


class TraceResponse(V2SuccessResponse[TraceData]):
    pass


class ReplayResponse(V2SuccessResponse[ReplayData]):
    pass


def v2_meta(schema: str, request_id: str) -> V2ApiMeta:
    return V2ApiMeta(
        api_version=API_VERSION,
        schema_version=SCHEMA_VERSION,
        schema=schema,
        request_id=request_id,
    )

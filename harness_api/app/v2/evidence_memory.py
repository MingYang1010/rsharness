"""Versioned, policy-bound geographic evidence memory.

The memory is separate from episode state.  Writes are trusted-operator actions;
Agents receive only a bounded public projection through a pinned snapshot.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from .events import canonical_json, sha256_json
from .schemas import (
    Artifact,
    EpisodeId,
    EvidenceId,
    EvidenceRef,
    EvidenceSelector,
    Identifier,
    MetricResult,
    NonEmptyText,
    SemanticVersion,
    Sha256,
    SpatialBoundingBox,
    TaskAsset,
    TaskManifest,
    TaskRef,
    TemporalExtent,
    UtcTimestamp,
    V2EpisodeState,
    V2RequestModel,
)
from .evidence import validate_evidence


MEMORY_ID_PATTERN = r"^mem-[a-f0-9]{64}$"
MemoryId = Annotated[str, Field(pattern=MEMORY_ID_PATTERN)]
MAX_POLICY_BYTES = 256 * 1024
MAX_EVENT_BYTES = 32 * 1024
MAX_STORE_BYTES = 256 * 1024 * 1024
MAX_PUBLIC_RECORD_BYTES = 8192
MAX_RESULT_BYTES = 64 * 1024
MAX_EVENTS = 20000
EMPTY_SNAPSHOT_SCHEMA = "eo-harness.evidence-memory.empty@1.0.0"


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _after(value: str, seconds: int) -> str:
    return (_timestamp(value) + timedelta(seconds=seconds)).isoformat().replace(
        "+00:00", "Z"
    )


class MemoryWriter(V2RequestModel):
    actor_id: Identifier
    certificate_sha256: Sha256


class MemorySourceGrant(V2RequestModel):
    task_id: Identifier
    task_versions: list[SemanticVersion] = Field(min_length=1, max_length=64)
    evaluator_id: Identifier
    min_aggregate_reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def unique_versions(self) -> "MemorySourceGrant":
        if len(self.task_versions) != len(set(self.task_versions)):
            raise ValueError("memory source grant versions must be unique")
        return self


class MemoryReaderGrant(V2RequestModel):
    task_id: Identifier
    task_versions: list[SemanticVersion] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_versions(self) -> "MemoryReaderGrant":
        if len(self.task_versions) != len(set(self.task_versions)):
            raise ValueError("memory reader grant versions must be unique")
        return self


class EvidenceMemoryPolicy(V2RequestModel):
    schema_version: Literal["1.0.0"]
    policy_id: Identifier
    scope_id: Identifier
    valid_from: UtcTimestamp
    expires_at: UtcTimestamp
    writers: list[MemoryWriter] = Field(min_length=1, max_length=64)
    source_grants: list[MemorySourceGrant] = Field(min_length=1, max_length=256)
    reader_grants: list[MemoryReaderGrant] = Field(min_length=1, max_length=256)
    max_record_ttl_seconds: int = Field(ge=60, le=365 * 24 * 60 * 60)
    max_active_records: int = Field(ge=1, le=10000)
    max_query_results: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def valid_policy(self) -> "EvidenceMemoryPolicy":
        if _timestamp(self.valid_from) >= _timestamp(self.expires_at):
            raise ValueError("memory policy validity window is empty")
        actor_ids = [item.actor_id for item in self.writers]
        certificate_hashes = [item.certificate_sha256 for item in self.writers]
        if len(actor_ids) != len(set(actor_ids)) or len(certificate_hashes) != len(
            set(certificate_hashes)
        ):
            raise ValueError("memory writer identities and certificates must be unique")
        source_keys = [(item.task_id, version) for item in self.source_grants for version in item.task_versions]
        reader_keys = [(item.task_id, version) for item in self.reader_grants for version in item.task_versions]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("memory source grants overlap")
        if len(reader_keys) != len(set(reader_keys)):
            raise ValueError("memory reader grants overlap")
        return self


class EvidenceMemoryBinding(V2RequestModel):
    schema_version: Literal["1.0.0"]
    policy_id: Identifier
    policy_sha256: Sha256
    scope_id: Identifier
    snapshot_sequence: int = Field(ge=0, le=MAX_EVENTS)
    snapshot_sha256: Sha256
    as_of: UtcTimestamp


class MemorySource(V2RequestModel):
    task_ref: TaskRef
    task_manifest_hash: Sha256
    episode_id: EpisodeId
    evidence_id: EvidenceId
    source_ref: Identifier
    source_sha256: Sha256
    selector: EvidenceSelector
    evaluation_id: Identifier
    evaluation_sha256: Sha256


class EvidenceMemoryRecord(V2RequestModel):
    schema_version: Literal["1.0.0"]
    memory_id: MemoryId
    policy_id: Identifier
    policy_sha256: Sha256
    scope_id: Identifier
    object_type: Identifier
    public_summary: Annotated[str, Field(min_length=1, max_length=4096)]
    bbox: SpatialBoundingBox
    time_range: TemporalExtent
    platform: Annotated[str, Field(min_length=1, max_length=128)]
    instrument: Annotated[str, Field(min_length=1, max_length=128)]
    available_at: UtcTimestamp
    expires_at: UtcTimestamp
    source: MemorySource
    provenance_sha256: Sha256

    @model_validator(mode="after")
    def valid_record(self) -> "EvidenceMemoryRecord":
        if _timestamp(self.available_at) >= _timestamp(self.expires_at):
            raise ValueError("memory record validity window is empty")
        if self.source.selector.bbox != self.bbox:
            raise ValueError("memory bbox must match the frozen evidence selector")
        if self.source.selector.time_range != self.time_range:
            raise ValueError("memory time range must match the frozen evidence selector")
        return self


class EvidenceMemoryInvalidation(V2RequestModel):
    memory_id: MemoryId
    reason: Annotated[str, Field(min_length=1, max_length=1024)]
    replacement_memory_id: MemoryId | None = None
    invalidated_at: UtcTimestamp

    @model_validator(mode="after")
    def not_self_replacement(self) -> "EvidenceMemoryInvalidation":
        if self.replacement_memory_id == self.memory_id:
            raise ValueError("memory invalidation cannot replace itself")
        return self


class EvidenceMemoryEvent(V2RequestModel):
    schema_version: Literal["1.0.0"]
    scope_id: Identifier
    sequence: int = Field(ge=1, le=MAX_EVENTS)
    event_type: Literal["memory.published", "memory.invalidated"]
    policy_id: Identifier
    policy_sha256: Sha256
    actor_id: Identifier
    actor_certificate_sha256: Sha256
    occurred_at: UtcTimestamp
    previous_event_sha256: Sha256 | None
    record: EvidenceMemoryRecord | None = None
    invalidation: EvidenceMemoryInvalidation | None = None
    event_sha256: Sha256

    @model_validator(mode="after")
    def payload_matches_type(self) -> "EvidenceMemoryEvent":
        if self.event_type == "memory.published":
            if self.record is None or self.invalidation is not None:
                raise ValueError("memory publish event requires exactly one record")
            if self.record.scope_id != self.scope_id:
                raise ValueError("memory event scope disagrees with record")
        elif self.invalidation is None or self.record is not None:
            raise ValueError("memory invalidation event requires exactly one invalidation")
        return self


class EvidenceMemorySnapshot(V2RequestModel):
    scope_id: Identifier
    sequence: int = Field(ge=0, le=MAX_EVENTS)
    snapshot_sha256: Sha256
    records: list[EvidenceMemoryRecord]
    invalidated_memory_ids: list[MemoryId]

    @property
    def active_records(self) -> list[EvidenceMemoryRecord]:
        invalidated = set(self.invalidated_memory_ids)
        return [item for item in self.records if item.memory_id not in invalidated]


class MemorySearchArguments(V2RequestModel):
    bbox: SpatialBoundingBox
    time_range: TemporalExtent
    platform: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    instrument: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    object_type: Identifier | None = None
    limit: int = Field(default=10, ge=1, le=20)
    offset: int = Field(default=0, ge=0, le=1000)


class PublicEvidenceMemoryRecord(V2RequestModel):
    memory_id: MemoryId
    object_type: Identifier
    public_summary: Annotated[str, Field(min_length=1, max_length=4096)]
    bbox: SpatialBoundingBox
    time_range: TemporalExtent
    platform: Annotated[str, Field(min_length=1, max_length=128)]
    instrument: Annotated[str, Field(min_length=1, max_length=128)]
    source_task: TaskRef
    source_sha256: Sha256
    provenance_sha256: Sha256
    available_at: UtcTimestamp
    expires_at: UtcTimestamp


class MemorySearchCost(V2RequestModel):
    model: Literal["logical-evidence-memory-v1"]
    records_scanned: int = Field(ge=0, le=10000)
    input_bytes: int = Field(ge=0, le=MAX_STORE_BYTES)


class MemorySearchResult(V2RequestModel):
    records: list[PublicEvidenceMemoryRecord] = Field(max_length=20)
    matched_count: int = Field(ge=0, le=10000)
    next_offset: int | None = Field(default=None, ge=0, le=1000)
    snapshot_sequence: int = Field(ge=0, le=MAX_EVENTS)
    snapshot_sha256: Sha256
    cost: MemorySearchCost


def _read_policy(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("evidence memory policy must be a regular file")
    details = path.stat()
    if details.st_size <= 0 or details.st_size > MAX_POLICY_BYTES:
        raise ValueError("evidence memory policy exceeds the allowed size")
    if details.st_mode & 0o022:
        raise ValueError("evidence memory policy must not be group/world writable")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        content = os.read(descriptor, MAX_POLICY_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(content) > MAX_POLICY_BYTES:
        raise ValueError("evidence memory policy exceeds the allowed size")
    return content


def load_evidence_memory_policy(
    path: Path, expected_sha256: str
) -> tuple[EvidenceMemoryPolicy, str]:
    content = _read_policy(path)
    digest = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(digest, expected_sha256):
        raise ValueError("evidence memory policy SHA-256 mismatch")
    try:
        return EvidenceMemoryPolicy.model_validate_json(content), digest
    except (ValidationError, ValueError):
        raise ValueError("evidence memory policy is invalid") from None


def authorize_memory_writer(
    policy: EvidenceMemoryPolicy,
    actor_id: str,
    actor_certificate_sha256: str,
    *,
    now: str | None = None,
) -> None:
    current = _timestamp(now or utc_now())
    if not (_timestamp(policy.valid_from) <= current < _timestamp(policy.expires_at)):
        raise ValueError("evidence memory policy is not currently valid")
    pins = {item.actor_id: item.certificate_sha256 for item in policy.writers}
    expected = pins.get(actor_id)
    if expected is None or not hmac.compare_digest(expected, actor_certificate_sha256):
        raise ValueError("evidence memory writer identity is not allowed")


def authorize_memory_source(
    policy: EvidenceMemoryPolicy,
    manifest: TaskManifest,
    evaluation: MetricResult,
) -> None:
    grants = [
        item
        for item in policy.source_grants
        if item.task_id == manifest.task.task_id
        and manifest.task.task_version in item.task_versions
    ]
    if len(grants) != 1:
        raise ValueError("source task is not granted by the evidence memory policy")
    grant = grants[0]
    if evaluation.status != "completed" or evaluation.evaluator_id != grant.evaluator_id:
        raise ValueError("source episode does not have the required completed evaluation")
    if (
        evaluation.aggregate_reward is None
        or evaluation.aggregate_reward < grant.min_aggregate_reward
    ):
        raise ValueError("source episode reward is below the memory publication threshold")


def memory_binding(manifest: TaskManifest) -> EvidenceMemoryBinding:
    value = manifest.task.metadata.get("evidence_memory")
    try:
        return EvidenceMemoryBinding.model_validate(value)
    except (ValidationError, ValueError):
        raise ValueError("task evidence memory binding is invalid") from None


def authorize_memory_reader(
    policy: EvidenceMemoryPolicy,
    policy_sha256: str,
    manifest: TaskManifest,
    binding: EvidenceMemoryBinding,
) -> None:
    if (
        binding.policy_id != policy.policy_id
        or binding.policy_sha256 != policy_sha256
        or binding.scope_id != policy.scope_id
    ):
        raise ValueError("task evidence memory policy pin does not match")
    if not (_timestamp(policy.valid_from) <= _timestamp(binding.as_of) < _timestamp(policy.expires_at)):
        raise ValueError("task evidence memory as_of is outside the policy window")
    if _timestamp(binding.as_of) > _timestamp(manifest.scenario.data_cutoff):
        raise ValueError("task evidence memory as_of exceeds the scenario data cutoff")
    granted = any(
        item.task_id == manifest.task.task_id
        and manifest.task.task_version in item.task_versions
        for item in policy.reader_grants
    )
    if not granted:
        raise ValueError("task is not granted evidence memory read access")


def _canonical_event(event: EvidenceMemoryEvent) -> bytes:
    return canonical_json(
        event.model_dump(mode="json", exclude={"event_sha256"})
    ).encode("utf-8")


def empty_snapshot_sha256(scope_id: str) -> str:
    return sha256_json({"schema": EMPTY_SNAPSHOT_SCHEMA, "scope_id": scope_id})


def _record_identity(record: EvidenceMemoryRecord) -> dict:
    return record.model_dump(mode="json", exclude={"memory_id"})


def validate_record_identity(record: EvidenceMemoryRecord) -> None:
    expected_provenance = sha256_json(record.source.model_dump(mode="json"))
    if not hmac.compare_digest(expected_provenance, record.provenance_sha256):
        raise ValueError("evidence memory provenance hash mismatch")
    expected_id = "mem-" + sha256_json(_record_identity(record))
    if not hmac.compare_digest(expected_id, record.memory_id):
        raise ValueError("evidence memory ID mismatch")


def _source_sensor(
    source: TaskAsset | Artifact, manifest: TaskManifest
) -> tuple[str, str]:
    if hasattr(source, "platform") and source.platform and source.instrument:
        return source.platform, source.instrument
    input_ids = set(source.lineage.input_refs) if hasattr(source, "lineage") else set()
    assets = [item for item in manifest.assets if item.asset_id in input_ids]
    platforms = {item.platform for item in assets if item.platform}
    instruments = {item.instrument for item in assets if item.instrument}
    if len(platforms) != 1 or len(instruments) != 1:
        raise ValueError("memory source must resolve to one reviewed platform and instrument")
    return next(iter(platforms)), next(iter(instruments))


def load_source_evidence(
    database: Path,
    tasks: Path,
    episode_id: str,
    evidence_id: str,
) -> tuple[TaskManifest, V2EpisodeState, MetricResult, EvidenceRef, TaskAsset | Artifact]:
    """Read one evaluated evidence source without migrating or writing its DB."""
    if database.is_symlink() or not database.is_file():
        raise ValueError("source episode database must be a regular file")
    if tasks.is_symlink() or not tasks.is_dir():
        raise ValueError("source task registry must be a regular directory")
    from .capabilities import TaskRegistry

    registry = TaskRegistry(str(tasks))
    uri = "file:" + quote(str(database.resolve()), safe="/") + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30.0)
    connection.row_factory = sqlite3.Row
    try:
        episode = connection.execute(
            "SELECT task_id,task_version,task_manifest_hash,state_json FROM v2_episodes WHERE episode_id=?",
            (episode_id,),
        ).fetchone()
        if episode is None:
            raise ValueError("source episode does not exist")
        state = V2EpisodeState.model_validate_json(episode["state_json"])
        manifest = registry.get(episode["task_id"], episode["task_version"])
        if (
            state.episode_id != episode_id
            or state.task_ref.task_id != episode["task_id"]
            or state.task_ref.task_version != episode["task_version"]
            or episode["task_manifest_hash"] != manifest.task_manifest_hash
            or state.task_manifest_hash != manifest.task_manifest_hash
        ):
            raise ValueError("source episode task manifest pin does not match")
        evaluation_row = connection.execute(
            "SELECT evaluation_json FROM v2_evaluations WHERE episode_id=?",
            (episode_id,),
        ).fetchone()
        if evaluation_row is None:
            raise ValueError("source episode has no evaluation")
        evaluation = MetricResult.model_validate_json(evaluation_row["evaluation_json"])
        evidence_items = [
            item for item in state.evidence_refs if item.evidence_id == evidence_id
        ]
        if len(evidence_items) != 1:
            raise ValueError("source evidence does not exist exactly once")
        evidence = evidence_items[0]
        source: TaskAsset | Artifact | None = next(
            (item for item in manifest.assets if item.asset_id == evidence.source_ref),
            None,
        )
        if source is None:
            association = connection.execute(
                "SELECT 1 FROM v2_episode_artifacts WHERE episode_id=? AND artifact_id=?",
                (episode_id, evidence.source_ref),
            ).fetchone()
            row = connection.execute(
                "SELECT artifact_json FROM v2_artifacts WHERE artifact_id=? AND status='created'",
                (evidence.source_ref,),
            ).fetchone()
            if association is None or row is None:
                raise ValueError(
                    "source evidence artifact is not accessible in the episode"
                )
            source = TypeAdapter(Artifact).validate_json(row["artifact_json"])
        return manifest, state, evaluation, evidence, source
    except (sqlite3.DatabaseError, KeyError, ValidationError):
        raise ValueError("source episode database is invalid") from None
    finally:
        connection.close()


def build_evidence_memory_record(
    *,
    policy: EvidenceMemoryPolicy,
    policy_sha256: str,
    manifest: TaskManifest,
    state: V2EpisodeState,
    evaluation: MetricResult,
    evidence: EvidenceRef,
    source: TaskAsset | Artifact,
    object_type: str,
    public_summary: str,
    ttl_seconds: int,
) -> EvidenceMemoryRecord:
    authorize_memory_source(policy, manifest, evaluation)
    if state.status != "terminated" or state.final_answer is None:
        raise ValueError("only a terminated answered episode can publish memory")
    if (
        state.task_ref != TaskRef(
            task_id=manifest.task.task_id, task_version=manifest.task.task_version
        )
        or state.task_manifest_hash != manifest.task_manifest_hash
    ):
        raise ValueError("source episode task pin does not match the manifest")
    if evidence.evidence_id not in state.final_answer.evidence_ids:
        raise ValueError("memory evidence was not cited by the final answer")
    if evidence.source_ref != getattr(source, "asset_id", getattr(source, "artifact_id", None)):
        raise ValueError("memory evidence source does not match resolved source")
    if evidence.frozen_sha256 != source.sha256:
        raise ValueError("memory evidence source hash is not frozen")
    validate_evidence(evidence, {evidence.source_ref: source})
    if evidence.selector.bbox is None or evidence.selector.time_range is None:
        raise ValueError("geographic memory requires bbox and time_range evidence selectors")
    if source.spatial is None or source.spatial.crs.upper() not in {
        "EPSG:4326",
        "OGC:CRS84",
    }:
        raise ValueError("geographic memory currently requires a WGS84 evidence source")
    if not 60 <= ttl_seconds <= policy.max_record_ttl_seconds:
        raise ValueError("evidence memory TTL is outside the policy limit")
    expires_at = _after(state.updated_at, ttl_seconds)
    if _timestamp(expires_at) > _timestamp(policy.expires_at):
        raise ValueError("evidence memory record expires after its policy")
    platform, instrument = _source_sensor(source, manifest)
    source_value = MemorySource(
        task_ref=state.task_ref,
        task_manifest_hash=state.task_manifest_hash,
        episode_id=state.episode_id,
        evidence_id=evidence.evidence_id,
        source_ref=evidence.source_ref,
        source_sha256=evidence.frozen_sha256,
        selector=evidence.selector,
        evaluation_id=evaluation.evaluation_id,
        evaluation_sha256=sha256_json(evaluation.model_dump(mode="json")),
    )
    body = {
        "schema_version": "1.0.0",
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256,
        "scope_id": policy.scope_id,
        "object_type": TypeAdapter(Identifier).validate_python(object_type),
        "public_summary": public_summary,
        "bbox": evidence.selector.bbox,
        "time_range": evidence.selector.time_range,
        "platform": platform,
        "instrument": instrument,
        "available_at": state.updated_at,
        "expires_at": expires_at,
        "source": source_value,
        "provenance_sha256": sha256_json(source_value.model_dump(mode="json")),
    }
    body["memory_id"] = "mem-" + sha256_json(
        EvidenceMemoryRecord.model_validate(
            {**body, "memory_id": "mem-" + "0" * 64}
        ).model_dump(mode="json", exclude={"memory_id"})
    )
    record = EvidenceMemoryRecord.model_validate(body)
    validate_record_identity(record)
    if len(canonical_json(record.model_dump(mode="json")).encode("utf-8")) > MAX_EVENT_BYTES // 2:
        raise ValueError("evidence memory record exceeds the allowed size")
    return record


def _bbox_intersects(left: SpatialBoundingBox, right: SpatialBoundingBox) -> bool:
    return not (
        left.east < right.west
        or left.west > right.east
        or left.north < right.south
        or left.south > right.north
    )


def _time_intersects(left: TemporalExtent, right: TemporalExtent) -> bool:
    return not (
        _timestamp(left.end) < _timestamp(right.start)
        or _timestamp(left.start) > _timestamp(right.end)
    )


def public_memory_record(record: EvidenceMemoryRecord) -> PublicEvidenceMemoryRecord:
    value = PublicEvidenceMemoryRecord(
        memory_id=record.memory_id,
        object_type=record.object_type,
        public_summary=record.public_summary,
        bbox=record.bbox,
        time_range=record.time_range,
        platform=record.platform,
        instrument=record.instrument,
        source_task=record.source.task_ref,
        source_sha256=record.source.source_sha256,
        provenance_sha256=record.provenance_sha256,
        available_at=record.available_at,
        expires_at=record.expires_at,
    )
    if len(canonical_json(value.model_dump(mode="json")).encode("utf-8")) > MAX_PUBLIC_RECORD_BYTES:
        raise ValueError("public evidence memory record exceeds the allowed size")
    return value


class EvidenceMemoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._initialize()

    def _check_file(self) -> None:
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("evidence memory store must be a regular file")
        details = self.path.stat()
        if not stat.S_ISREG(details.st_mode) or details.st_mode & 0o077:
            raise ValueError("evidence memory store must be owner-private")
        if details.st_size > MAX_STORE_BYTES:
            raise ValueError("evidence memory store exceeds the allowed size")

    def _connect(self) -> sqlite3.Connection:
        self._check_file()
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        return connection

    def _initialize(self) -> None:
        if self.path.is_symlink():
            raise ValueError("evidence memory store path must not be a symlink")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.path.exists():
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            os.close(descriptor)
        self._check_file()
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_memory_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_memory_events (
                    scope_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (scope_id, sequence)
                )
                """
            )
            current = connection.execute(
                "SELECT value FROM evidence_memory_meta WHERE key='schema_version'"
            ).fetchone()
            if current is None:
                connection.execute(
                    "INSERT INTO evidence_memory_meta(key,value) VALUES('schema_version','1.0.0')"
                )
            elif current["value"] != "1.0.0":
                raise ValueError("unsupported evidence memory store schema")
        os.chmod(self.path, 0o600)

    @staticmethod
    def _verify_events(scope_id: str, rows: list[sqlite3.Row]) -> list[EvidenceMemoryEvent]:
        events: list[EvidenceMemoryEvent] = []
        previous = None
        for expected_sequence, row in enumerate(rows, start=1):
            try:
                event = EvidenceMemoryEvent.model_validate_json(row["event_json"])
            except (ValidationError, ValueError):
                raise ValueError("evidence memory event is invalid") from None
            if (
                row["sequence"] != expected_sequence
                or event.sequence != expected_sequence
                or event.scope_id != scope_id
                or event.previous_event_sha256 != previous
                or hashlib.sha256(_canonical_event(event)).hexdigest()
                != event.event_sha256
            ):
                raise ValueError("evidence memory event chain is invalid")
            if event.record is not None:
                validate_record_identity(event.record)
            previous = event.event_sha256
            events.append(event)
        return events

    def _rows(
        self, connection: sqlite3.Connection, scope_id: str, sequence: int | None = None
    ) -> list[sqlite3.Row]:
        query = (
            "SELECT sequence,event_json FROM evidence_memory_events "
            "WHERE scope_id=?"
        )
        parameters: list[object] = [scope_id]
        if sequence is not None:
            query += " AND sequence<=?"
            parameters.append(sequence)
        query += " ORDER BY sequence"
        return connection.execute(query, parameters).fetchall()

    def snapshot(self, scope_id: str, sequence: int | None = None) -> EvidenceMemorySnapshot:
        with self._connect() as connection:
            all_rows = self._rows(connection, scope_id)
        all_events = self._verify_events(scope_id, all_rows)
        latest = len(all_events)
        selected = latest if sequence is None else sequence
        if selected < 0 or selected > latest:
            raise ValueError("evidence memory snapshot sequence is unavailable")
        events = all_events[:selected]
        records: dict[str, EvidenceMemoryRecord] = {}
        invalidated: set[str] = set()
        for event in events:
            if event.record is not None:
                if event.record.memory_id in records:
                    raise ValueError("evidence memory record was published more than once")
                records[event.record.memory_id] = event.record
            else:
                invalidation = event.invalidation
                if invalidation.memory_id not in records or invalidation.memory_id in invalidated:
                    raise ValueError("evidence memory invalidation order is invalid")
                if (
                    invalidation.replacement_memory_id is not None
                    and invalidation.replacement_memory_id not in records
                ):
                    raise ValueError("evidence memory replacement is unavailable")
                invalidated.add(invalidation.memory_id)
        snapshot_sha256 = events[-1].event_sha256 if events else empty_snapshot_sha256(scope_id)
        return EvidenceMemorySnapshot(
            scope_id=scope_id,
            sequence=selected,
            snapshot_sha256=snapshot_sha256,
            records=[records[key] for key in sorted(records)],
            invalidated_memory_ids=sorted(invalidated),
        )

    def _append(
        self,
        *,
        scope_id: str,
        event_type: Literal["memory.published", "memory.invalidated"],
        policy_id: str,
        policy_sha256: str,
        actor_id: str,
        actor_certificate_sha256: str,
        occurred_at: str,
        record: EvidenceMemoryRecord | None = None,
        invalidation: EvidenceMemoryInvalidation | None = None,
    ) -> EvidenceMemorySnapshot:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = self._rows(connection, scope_id)
            events = self._verify_events(scope_id, rows)
            if len(events) >= MAX_EVENTS:
                raise ValueError("evidence memory event limit reached")
            page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
            page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
            if page_size * page_count + MAX_EVENT_BYTES > MAX_STORE_BYTES:
                raise ValueError("evidence memory store capacity exceeded")
            unsigned = EvidenceMemoryEvent(
                schema_version="1.0.0",
                scope_id=scope_id,
                sequence=len(events) + 1,
                event_type=event_type,
                policy_id=policy_id,
                policy_sha256=policy_sha256,
                actor_id=actor_id,
                actor_certificate_sha256=actor_certificate_sha256,
                occurred_at=occurred_at,
                previous_event_sha256=events[-1].event_sha256 if events else None,
                record=record,
                invalidation=invalidation,
                event_sha256="0" * 64,
            )
            event = unsigned.model_copy(
                update={"event_sha256": hashlib.sha256(_canonical_event(unsigned)).hexdigest()}
            )
            content = event.model_dump_json()
            if len(content.encode("utf-8")) > MAX_EVENT_BYTES:
                raise ValueError("evidence memory event exceeds the allowed size")
            connection.execute(
                "INSERT INTO evidence_memory_events(scope_id,sequence,event_json) VALUES(?,?,?)",
                (scope_id, event.sequence, content),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.snapshot(scope_id)

    def publish(
        self,
        policy: EvidenceMemoryPolicy,
        policy_sha256: str,
        record: EvidenceMemoryRecord,
        *,
        actor_id: str,
        actor_certificate_sha256: str,
        now: str | None = None,
    ) -> EvidenceMemorySnapshot:
        timestamp = now or utc_now()
        authorize_memory_writer(
            policy, actor_id, actor_certificate_sha256, now=timestamp
        )
        validate_record_identity(record)
        if (
            record.scope_id != policy.scope_id
            or record.policy_id != policy.policy_id
            or record.policy_sha256 != policy_sha256
        ):
            raise ValueError("evidence memory record policy scope does not match")
        if not (
            _timestamp(record.available_at)
            <= _timestamp(timestamp)
            < _timestamp(record.expires_at)
        ):
            raise ValueError("evidence memory record is not valid at publication time")
        if _timestamp(record.expires_at) > _timestamp(policy.expires_at):
            raise ValueError("evidence memory record expires after its policy")
        current = self.snapshot(policy.scope_id)
        existing = {item.memory_id: item for item in current.records}.get(record.memory_id)
        if existing is not None:
            if existing != record:
                raise ValueError("evidence memory ID already has different content")
            return current
        if len(current.active_records) >= policy.max_active_records:
            raise ValueError("evidence memory active-record limit reached")
        return self._append(
            scope_id=policy.scope_id,
            event_type="memory.published",
            policy_id=policy.policy_id,
            policy_sha256=policy_sha256,
            actor_id=actor_id,
            actor_certificate_sha256=actor_certificate_sha256,
            occurred_at=timestamp,
            record=record,
        )

    def invalidate(
        self,
        policy: EvidenceMemoryPolicy,
        policy_sha256: str,
        memory_id: str,
        reason: str,
        *,
        actor_id: str,
        actor_certificate_sha256: str,
        replacement_memory_id: str | None = None,
        now: str | None = None,
    ) -> EvidenceMemorySnapshot:
        timestamp = now or utc_now()
        authorize_memory_writer(
            policy, actor_id, actor_certificate_sha256, now=timestamp
        )
        current = self.snapshot(policy.scope_id)
        records = {item.memory_id: item for item in current.records}
        if memory_id not in records:
            raise ValueError("evidence memory record does not exist")
        if memory_id in current.invalidated_memory_ids:
            with self._connect() as connection:
                events = self._verify_events(
                    policy.scope_id, self._rows(connection, policy.scope_id)
                )
            prior = next(
                event
                for event in events
                if event.invalidation is not None
                and event.invalidation.memory_id == memory_id
            )
            if (
                prior.invalidation.reason != reason
                or prior.invalidation.replacement_memory_id
                != replacement_memory_id
                or prior.actor_id != actor_id
                or prior.actor_certificate_sha256 != actor_certificate_sha256
            ):
                raise ValueError("evidence memory invalidation already has different content")
            return current
        if replacement_memory_id is not None and (
            replacement_memory_id not in records
            or replacement_memory_id in current.invalidated_memory_ids
        ):
            raise ValueError("evidence memory replacement is not active")
        invalidation = EvidenceMemoryInvalidation(
            memory_id=memory_id,
            reason=reason,
            replacement_memory_id=replacement_memory_id,
            invalidated_at=timestamp,
        )
        return self._append(
            scope_id=policy.scope_id,
            event_type="memory.invalidated",
            policy_id=policy.policy_id,
            policy_sha256=policy_sha256,
            actor_id=actor_id,
            actor_certificate_sha256=actor_certificate_sha256,
            occurred_at=timestamp,
            invalidation=invalidation,
        )

    def search(
        self,
        policy: EvidenceMemoryPolicy,
        policy_sha256: str,
        manifest: TaskManifest,
        query: MemorySearchArguments,
    ) -> MemorySearchResult:
        binding = memory_binding(manifest)
        authorize_memory_reader(policy, policy_sha256, manifest, binding)
        if query.limit > policy.max_query_results:
            raise ValueError("evidence memory query limit exceeds policy")
        snapshot = self.snapshot(binding.scope_id, binding.snapshot_sequence)
        if not hmac.compare_digest(snapshot.snapshot_sha256, binding.snapshot_sha256):
            raise ValueError("evidence memory snapshot pin does not match")
        as_of = _timestamp(binding.as_of)
        active = []
        for record in snapshot.active_records:
            if not (_timestamp(record.available_at) <= as_of < _timestamp(record.expires_at)):
                continue
            if not _bbox_intersects(record.bbox, query.bbox) or not _time_intersects(
                record.time_range, query.time_range
            ):
                continue
            if query.platform is not None and record.platform != query.platform:
                continue
            if query.instrument is not None and record.instrument != query.instrument:
                continue
            if query.object_type is not None and record.object_type != query.object_type:
                continue
            active.append(record)
        active.sort(key=lambda item: item.memory_id)
        end = query.offset + query.limit
        records = [public_memory_record(item) for item in active[query.offset:end]]
        scan_bytes = sum(
            len(canonical_json(item.model_dump(mode="json")).encode("utf-8"))
            for item in snapshot.records
        )
        result = MemorySearchResult(
            records=records,
            matched_count=len(active),
            next_offset=end if end < len(active) and end <= 1000 else None,
            snapshot_sequence=snapshot.sequence,
            snapshot_sha256=snapshot.snapshot_sha256,
            cost=MemorySearchCost(
                model="logical-evidence-memory-v1",
                records_scanned=len(snapshot.records),
                input_bytes=scan_bytes,
            ),
        )
        if len(canonical_json(result.model_dump(mode="json")).encode("utf-8")) > MAX_RESULT_BYTES:
            raise ValueError("evidence memory result exceeds the allowed size")
        return result


__all__ = [
    "EvidenceMemoryBinding",
    "EvidenceMemoryPolicy",
    "EvidenceMemoryRecord",
    "EvidenceMemorySnapshot",
    "EvidenceMemoryStore",
    "MemorySearchArguments",
    "MemorySearchResult",
    "authorize_memory_reader",
    "authorize_memory_source",
    "authorize_memory_writer",
    "build_evidence_memory_record",
    "empty_snapshot_sha256",
    "load_evidence_memory_policy",
    "load_source_evidence",
    "memory_binding",
    "public_memory_record",
    "utc_now",
    "validate_record_identity",
]

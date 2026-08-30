import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import STORE_SCHEMA_VERSION
from .capabilities import TaskRegistry
from .domain import V2DomainError, apply_action, create_initial_state, utc_now
from .events import (
    canonical_json,
    create_event,
    semantic_trace_hash,
    trace_hash,
)
from .observations import semantic_state_hash, state_hash
from .schemas import (
    EpisodeResultData,
    EventRecord,
    Observation,
    ReplayCheck,
    ReplayData,
    StateData,
    TaskManifest,
    TraceData,
    V2Action,
    V2EpisodeState,
)


MIGRATION_1 = [
    """
    CREATE TABLE IF NOT EXISTS v2_episodes (
        episode_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        task_version TEXT NOT NULL,
        task_manifest_hash TEXT NOT NULL,
        seed INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        status TEXT NOT NULL,
        state_version INTEGER NOT NULL,
        step_count INTEGER NOT NULL,
        initial_state_json TEXT NOT NULL,
        state_json TEXT NOT NULL,
        initial_observation_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS v2_events (
        episode_id TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        event_id TEXT NOT NULL UNIQUE,
        event_type TEXT NOT NULL,
        state_version INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        event_json TEXT NOT NULL,
        PRIMARY KEY (episode_id, sequence),
        FOREIGN KEY (episode_id) REFERENCES v2_episodes(episode_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS v2_action_results (
        episode_id TEXT NOT NULL,
        client_action_id TEXT NOT NULL,
        request_json TEXT NOT NULL,
        outcome TEXT NOT NULL,
        response_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (episode_id, client_action_id),
        FOREIGN KEY (episode_id) REFERENCES v2_episodes(episode_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS v2_observations (
        observation_id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        observation_json TEXT NOT NULL,
        FOREIGN KEY (episode_id) REFERENCES v2_episodes(episode_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS v2_artifacts (
        artifact_id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL,
        status TEXT NOT NULL,
        sha256 TEXT,
        size_bytes INTEGER,
        artifact_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (episode_id) REFERENCES v2_episodes(episode_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS v2_evidence (
        evidence_id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL,
        evidence_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (episode_id) REFERENCES v2_episodes(episode_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS v2_tool_runs (
        tool_run_id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL,
        tool_id TEXT NOT NULL,
        status TEXT NOT NULL,
        run_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (episode_id) REFERENCES v2_episodes(episode_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS v2_evaluations (
        evaluation_id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL,
        evaluation_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (episode_id) REFERENCES v2_episodes(episode_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_v2_events_episode_sequence
    ON v2_events(episode_id, sequence)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_v2_observations_episode_sequence
    ON v2_observations(episode_id, sequence)
    """,
]


class V2EpisodeStore:
    def __init__(self, database_path: str, task_registry: TaskRegistry):
        self.database_path = str(database_path)
        self.task_registry = task_registry
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _migration_statements(self) -> Sequence[str]:
        return MIGRATION_1

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS v2_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM v2_schema_migrations"
            ).fetchone()[0]
            if current < STORE_SCHEMA_VERSION:
                for statement in self._migration_statements():
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO v2_schema_migrations(version, applied_at) VALUES (?, ?)",
                    (STORE_SCHEMA_VERSION, utc_now()),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def schema_version(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM v2_schema_migrations"
                ).fetchone()[0]
            )

    def health(self) -> Dict[str, Any]:
        with self._connect() as connection:
            episode_count = connection.execute(
                "SELECT COUNT(*) FROM v2_episodes"
            ).fetchone()[0]
        return {
            "database": "ok",
            "schema_version": self.schema_version(),
            "episode_count": episode_count,
        }

    def get_task(self, task_id: str, task_version: str) -> TaskManifest:
        try:
            return self.task_registry.get(task_id, task_version)
        except KeyError:
            raise V2DomainError(
                "task_not_found",
                "unknown immutable task: %s@%s" % (task_id, task_version),
                status_code=404,
                phase="request",
            )

    @staticmethod
    def _episode_result(
        state: V2EpisodeState,
        observation: Observation,
    ) -> EpisodeResultData:
        return EpisodeResultData(
            episode_id=state.episode_id,
            task_manifest_hash=state.task_manifest_hash,
            state=state,
            observation=observation,
            terminated=state.status == "terminated",
            truncated=state.status == "truncated",
        )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        event: EventRecord,
    ) -> None:
        connection.execute(
            """
            INSERT INTO v2_events (
                episode_id, sequence, event_id, event_type,
                state_version, created_at, event_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.episode_id,
                event.sequence,
                event.event_id,
                event.event_type,
                event.state_version,
                event.created_at,
                canonical_json(event.model_dump(mode="json")),
            ),
        )

    @staticmethod
    def _insert_observation(
        connection: sqlite3.Connection,
        episode_id: str,
        observation: Observation,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO v2_observations (
                observation_id, episode_id, sequence,
                created_at, observation_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                observation.observation_id,
                episode_id,
                observation.sequence,
                created_at,
                canonical_json(observation.model_dump(mode="json")),
            ),
        )

    def create_episode(
        self,
        task_id: str,
        task_version: str,
        seed: Optional[int],
    ) -> EpisodeResultData:
        manifest = self.get_task(task_id, task_version)
        resolved_seed = manifest.task.seed if seed is None else seed
        episode_id = "ep2-%s" % uuid.uuid4().hex
        timestamp = utc_now()
        state, observation = create_initial_state(
            episode_id,
            manifest,
            resolved_seed,
            timestamp=timestamp,
        )
        created = create_event(
            episode_id=episode_id,
            sequence=0,
            event_type="episode.created",
            state_version=0,
            created_at=timestamp,
            payload={
                "task_manifest_hash": manifest.task_manifest_hash,
                "seed": resolved_seed,
                "state": state.model_dump(mode="json"),
                "observation": observation.model_dump(mode="json"),
            },
        )
        emitted = create_event(
            episode_id=episode_id,
            sequence=1,
            event_type="observation.emitted",
            state_version=0,
            created_at=timestamp,
            payload={"observation": observation.model_dump(mode="json")},
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO v2_episodes (
                    episode_id, task_id, task_version, task_manifest_hash,
                    seed, created_at, updated_at, status, state_version,
                    step_count, initial_state_json, state_json,
                    initial_observation_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    task_id,
                    task_version,
                    manifest.task_manifest_hash,
                    resolved_seed,
                    timestamp,
                    timestamp,
                    state.status,
                    state.state_version,
                    state.step_count,
                    canonical_json(state.model_dump(mode="json")),
                    canonical_json(state.model_dump(mode="json")),
                    canonical_json(observation.model_dump(mode="json")),
                ),
            )
            self._insert_event(connection, created)
            self._insert_event(connection, emitted)
            self._insert_observation(connection, episode_id, observation, timestamp)
        return self._episode_result(state, observation)

    @staticmethod
    def _load_episode(
        connection: sqlite3.Connection,
        episode_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM v2_episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        if row is None:
            raise V2DomainError(
                "episode_not_found",
                "unknown V2 episode_id: %s" % episode_id,
                status_code=404,
                phase="request",
            )
        return row

    def _manifest_for_row(self, row: sqlite3.Row) -> TaskManifest:
        manifest = self.get_task(row["task_id"], row["task_version"])
        if manifest.task_manifest_hash != row["task_manifest_hash"]:
            raise V2DomainError(
                "task_manifest_mismatch",
                "registered task manifest no longer matches the episode",
                status_code=409,
                phase="state",
            )
        return manifest

    def get_state(self, episode_id: str) -> StateData:
        with self._connect() as connection:
            row = self._load_episode(connection, episode_id)
        state = V2EpisodeState.model_validate_json(row["state_json"])
        return StateData(
            episode_id=episode_id,
            state=state,
            state_hash=state_hash(state),
            semantic_state_hash=semantic_state_hash(state),
        )

    @staticmethod
    def _next_sequence(connection: sqlite3.Connection, episode_id: str) -> int:
        return int(
            connection.execute(
                """
                SELECT COALESCE(MAX(sequence), -1) + 1
                FROM v2_events WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()[0]
        )

    def step(
        self,
        episode_id: str,
        expected_state_version: int,
        client_action_id: str,
        action: V2Action,
    ) -> EpisodeResultData:
        request_value = {
            "expected_state_version": expected_state_version,
            "action": action.model_dump(mode="json"),
        }
        request_json = canonical_json(request_value)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._load_episode(connection, episode_id)
            prior = connection.execute(
                """
                SELECT request_json, outcome, response_json
                FROM v2_action_results
                WHERE episode_id = ? AND client_action_id = ?
                """,
                (episode_id, client_action_id),
            ).fetchone()
            if prior is not None:
                if prior["request_json"] != request_json:
                    raise V2DomainError(
                        "idempotency_conflict",
                        "client_action_id was already used with a different request",
                        status_code=409,
                        phase="state",
                    )
                connection.commit()
                if prior["outcome"] == "error":
                    stored_error = json.loads(prior["response_json"])
                    raise V2DomainError(**stored_error)
                return EpisodeResultData.model_validate_json(prior["response_json"])

            if row["state_version"] != expected_state_version:
                raise V2DomainError(
                    "state_version_conflict",
                    "expected_state_version does not match current state_version",
                    status_code=409,
                    retryable=True,
                    phase="state",
                    details=[
                        {
                            "location": ["body", "expected_state_version"],
                            "message": "expected %s, current %s"
                            % (expected_state_version, row["state_version"]),
                            "type": "state_version_conflict",
                            "field": "expected_state_version",
                        }
                    ],
                )

            manifest = self._manifest_for_row(row)
            current_state = V2EpisodeState.model_validate_json(row["state_json"])
            timestamp = utc_now()
            next_sequence = self._next_sequence(connection, episode_id)
            accepted_event = create_event(
                episode_id,
                next_sequence,
                "action.accepted",
                current_state.state_version,
                timestamp,
                {
                    "client_action_id": client_action_id,
                    "expected_state_version": expected_state_version,
                    "action": action.model_dump(mode="json"),
                },
            )
            try:
                next_state, observation, action_payload = apply_action(
                    current_state,
                    action,
                    manifest,
                    current_time=timestamp,
                )
            except V2DomainError as error:
                failed_event = create_event(
                    episode_id,
                    next_sequence + 1,
                    "action.failed",
                    current_state.state_version,
                    timestamp,
                    {
                        "action_type": action.type,
                        "code": error.code,
                        "phase": error.phase,
                        "retryable": error.retryable,
                    },
                )
                self._insert_event(connection, accepted_event)
                self._insert_event(connection, failed_event)
                stored_error = {
                    "code": error.code,
                    "message": error.message,
                    "status_code": error.status_code,
                    "retryable": error.retryable,
                    "phase": error.phase,
                    "details": error.details,
                }
                connection.execute(
                    """
                    INSERT INTO v2_action_results (
                        episode_id, client_action_id, request_json,
                        outcome, response_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        episode_id,
                        client_action_id,
                        request_json,
                        "error",
                        canonical_json(stored_error),
                        timestamp,
                    ),
                )
                connection.commit()
                raise

            events = [
                accepted_event,
                create_event(
                    episode_id,
                    next_sequence + 1,
                    "action.completed",
                    next_state.state_version,
                    timestamp,
                    {
                        **action_payload,
                        "state_hash": state_hash(next_state),
                        "semantic_state_hash": semantic_state_hash(next_state),
                        "observation_id": observation.observation_id,
                    },
                ),
            ]
            offset = 2
            if action.type == "memory.save_evidence":
                events.append(
                    create_event(
                        episode_id,
                        next_sequence + offset,
                        "evidence.saved",
                        next_state.state_version,
                        timestamp,
                        {"evidence": action.evidence.model_dump(mode="json")},
                    )
                )
                offset += 1
            events.append(
                create_event(
                    episode_id,
                    next_sequence + offset,
                    "observation.emitted",
                    next_state.state_version,
                    timestamp,
                    {"observation": observation.model_dump(mode="json")},
                )
            )
            offset += 1
            if next_state.status == "terminated":
                events.append(
                    create_event(
                        episode_id,
                        next_sequence + offset,
                        "episode.terminated",
                        next_state.state_version,
                        timestamp,
                        {"final_answer": next_state.final_answer.model_dump(mode="json")},
                    )
                )
            elif next_state.status == "truncated":
                events.append(
                    create_event(
                        episode_id,
                        next_sequence + offset,
                        "episode.truncated",
                        next_state.state_version,
                        timestamp,
                        {"reason": "budget_exhausted"},
                    )
                )

            response = self._episode_result(next_state, observation)
            response_json = canonical_json(response.model_dump(mode="json"))
            for event in events:
                self._insert_event(connection, event)
            self._insert_observation(connection, episode_id, observation, timestamp)
            if action.type == "memory.save_evidence":
                connection.execute(
                    """
                    INSERT INTO v2_evidence (
                        evidence_id, episode_id, evidence_json, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        action.evidence.evidence_id,
                        episode_id,
                        canonical_json(action.evidence.model_dump(mode="json")),
                        timestamp,
                    ),
                )
            connection.execute(
                """
                UPDATE v2_episodes
                SET updated_at = ?, status = ?, state_version = ?,
                    step_count = ?, state_json = ?
                WHERE episode_id = ?
                """,
                (
                    next_state.updated_at,
                    next_state.status,
                    next_state.state_version,
                    next_state.step_count,
                    canonical_json(next_state.model_dump(mode="json")),
                    episode_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO v2_action_results (
                    episode_id, client_action_id, request_json,
                    outcome, response_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    client_action_id,
                    request_json,
                    "success",
                    response_json,
                    timestamp,
                ),
            )
            connection.commit()
            return EpisodeResultData.model_validate_json(response_json)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _parse_cursor(cursor: Optional[str]) -> int:
        if cursor is None:
            return -1
        if not cursor.startswith("seq:"):
            raise V2DomainError(
                "invalid_cursor",
                "trace cursor must use seq:<integer>",
                phase="request",
            )
        try:
            sequence = int(cursor[4:])
        except ValueError:
            raise V2DomainError(
                "invalid_cursor",
                "trace cursor must use seq:<integer>",
                phase="request",
            )
        if sequence < -1:
            raise V2DomainError(
                "invalid_cursor",
                "trace cursor sequence must be non-negative",
                phase="request",
            )
        return sequence

    def _all_events(
        self,
        connection: sqlite3.Connection,
        episode_id: str,
    ) -> List[EventRecord]:
        rows = connection.execute(
            """
            SELECT event_json FROM v2_events
            WHERE episode_id = ? ORDER BY sequence ASC
            """,
            (episode_id,),
        ).fetchall()
        return [EventRecord.model_validate_json(row["event_json"]) for row in rows]

    def get_trace(
        self,
        episode_id: str,
        cursor: Optional[str],
        limit: int,
    ) -> TraceData:
        after_sequence = self._parse_cursor(cursor)
        with self._connect() as connection:
            episode = self._load_episode(connection, episode_id)
            all_events = self._all_events(connection, episode_id)
        selected = [event for event in all_events if event.sequence > after_sequence]
        page = selected[:limit]
        has_more = len(selected) > limit
        next_cursor = "seq:%s" % page[-1].sequence if has_more and page else None
        return TraceData(
            episode_id=episode_id,
            events=page,
            limit=limit,
            next_cursor=next_cursor,
            has_more=has_more,
            total_events=len(all_events),
            hash_algorithm="sha256",
            trace_hash=trace_hash(all_events),
            semantic_trace_hash=semantic_trace_hash(
                episode["task_manifest_hash"],
                all_events,
            ),
        )

    def structural_replay(self, episode_id: str) -> ReplayData:
        with self._connect() as connection:
            episode = self._load_episode(connection, episode_id)
            events = self._all_events(connection, episode_id)
        sequences = [event.sequence for event in events]
        expected_sequences = list(range(len(events)))
        event_ids = [event.event_id for event in events]
        state_versions = [event.state_version for event in events]
        checks = [
            ReplayCheck(
                name="sequence.contiguous",
                passed=sequences == expected_sequences,
                expected=str(expected_sequences),
                actual=str(sequences),
            ),
            ReplayCheck(
                name="event_id.unique",
                passed=len(event_ids) == len(set(event_ids)),
            ),
            ReplayCheck(
                name="state_version.monotonic",
                passed=state_versions == sorted(state_versions),
            ),
            ReplayCheck(
                name="task_manifest.pinned",
                passed=events[0].payload.get("task_manifest_hash")
                == episode["task_manifest_hash"],
                expected=episode["task_manifest_hash"],
                actual=str(events[0].payload.get("task_manifest_hash")),
            ),
        ]
        concrete_hash = trace_hash(events)
        semantic_hash = semantic_trace_hash(episode["task_manifest_hash"], events)
        passed = all(check.passed for check in checks)
        return ReplayData(
            episode_id=episode_id,
            mode="structural",
            status="passed" if passed else "failed",
            checked_event_count=len(events),
            checks=checks,
            trace_hash=concrete_hash,
            semantic_trace_hash=semantic_hash,
        )

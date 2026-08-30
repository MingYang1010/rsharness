import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .domain import (
    DomainError,
    apply_action,
    build_observation,
    canonical_json,
    create_initial_state,
    semantic_state_hash,
    semantic_trace_hash,
    state_hash,
    utc_now,
)


class EpisodeStore:
    def __init__(self, database_path: str):
        self.database_path = str(database_path)
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    step_count INTEGER NOT NULL,
                    max_steps INTEGER NOT NULL,
                    initial_state_json TEXT NOT NULL,
                    state_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS transitions (
                    episode_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    client_action_id TEXT,
                    created_at TEXT NOT NULL,
                    action_json TEXT NOT NULL,
                    observation_json TEXT NOT NULL,
                    state_hash TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    PRIMARY KEY (episode_id, sequence),
                    UNIQUE (episode_id, client_action_id),
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                );

                CREATE INDEX IF NOT EXISTS idx_transitions_episode
                ON transitions(episode_id, sequence);
                """
            )

    def health(self) -> Dict[str, Any]:
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()
        return {"database": "ok"}

    def create_episode(self, specification: Dict[str, Any]) -> Dict[str, Any]:
        episode_id = "ep-%s" % uuid.uuid4().hex
        state = create_initial_state(episode_id, specification)
        observation = build_observation(state, "Episode reset")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO episodes (
                    episode_id, created_at, updated_at, status, step_count,
                    max_steps, initial_state_json, state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    state["created_at"],
                    state["updated_at"],
                    state["status"],
                    state["step_count"],
                    state["max_steps"],
                    canonical_json(state),
                    canonical_json(state),
                ),
            )
        return self._response(state, observation, client_action_id=None)

    def _load_episode(self, connection: sqlite3.Connection, episode_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            raise DomainError(
                "episode_not_found",
                "Unknown episode_id: %s" % episode_id,
                status_code=404,
            )
        return row

    def get_state(self, episode_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = self._load_episode(connection, episode_id)
        state = json.loads(row["state_json"])
        return {
            "episode_id": episode_id,
            "state": state,
            "state_hash": state_hash(state),
        }

    def _response(
        self,
        state: Dict[str, Any],
        observation: Dict[str, Any],
        client_action_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "episode_id": state["episode_id"],
            "state": state,
            "observation": observation,
            "reward": None,
            "terminated": state["status"] == "terminated",
            "truncated": state["status"] == "truncated",
            "info": {
                "client_action_id": client_action_id,
                "remaining_steps": max(state["max_steps"] - state["step_count"], 0),
                "state_hash": observation["state_hash"],
                "semantic_state_hash": semantic_state_hash(state),
            },
        }

    def step(
        self,
        episode_id: str,
        action: Dict[str, Any],
        client_action_id: Optional[str],
    ) -> Dict[str, Any]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._load_episode(connection, episode_id)

            if client_action_id is not None:
                prior = connection.execute(
                    """
                    SELECT action_json, response_json FROM transitions
                    WHERE episode_id = ? AND client_action_id = ?
                    """,
                    (episode_id, client_action_id),
                ).fetchone()
                if prior is not None:
                    if prior["action_json"] != canonical_json(action):
                        raise DomainError(
                            "idempotency_conflict",
                            "client_action_id was already used for a different action",
                            status_code=409,
                        )
                    connection.commit()
                    return json.loads(prior["response_json"])

            row = self._load_episode(connection, episode_id)
            current_state = json.loads(row["state_json"])
            next_state, observation = apply_action(current_state, action)
            response = json.loads(
                canonical_json(
                    self._response(next_state, observation, client_action_id)
                )
            )
            timestamp = utc_now()

            connection.execute(
                """
                INSERT INTO transitions (
                    episode_id, sequence, client_action_id, created_at,
                    action_json, observation_json, state_hash, state_json,
                    response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    next_state["step_count"],
                    client_action_id,
                    timestamp,
                    canonical_json(action),
                    canonical_json(observation),
                    observation["state_hash"],
                    canonical_json(next_state),
                    canonical_json(response),
                ),
            )
            connection.execute(
                """
                UPDATE episodes
                SET updated_at = ?, status = ?, step_count = ?, state_json = ?
                WHERE episode_id = ?
                """,
                (
                    next_state["updated_at"],
                    next_state["status"],
                    next_state["step_count"],
                    canonical_json(next_state),
                    episode_id,
                ),
            )
            connection.commit()
            return response
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_trace(self, episode_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            episode = self._load_episode(connection, episode_id)
            rows = connection.execute(
                """
                SELECT sequence, client_action_id, created_at, action_json,
                       observation_json, state_hash, state_json
                FROM transitions
                WHERE episode_id = ?
                ORDER BY sequence ASC
                """,
                (episode_id,),
            ).fetchall()

        transitions = [
            {
                "sequence": row["sequence"],
                "client_action_id": row["client_action_id"],
                "created_at": row["created_at"],
                "action": json.loads(row["action_json"]),
                "observation": json.loads(row["observation_json"]),
                "state_hash": row["state_hash"],
                "state": json.loads(row["state_json"]),
            }
            for row in rows
        ]
        final_state = json.loads(episode["state_json"])
        initial_state = json.loads(episode["initial_state_json"])
        return {
            "episode_id": episode_id,
            "initial_state": initial_state,
            "transitions": transitions,
            "final_state": final_state,
            "transition_count": len(transitions),
            "trace_hash": state_hash(
                {
                    "episode_id": episode_id,
                    "initial_state": json.loads(episode["initial_state_json"]),
                    "transitions": transitions,
                    "final_state": final_state,
                }
            ),
            "semantic_trace_hash": semantic_trace_hash(
                initial_state,
                transitions,
                final_state,
            ),
        }

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from app.core import STORE_SCHEMA_VERSION
from app.core.schemas import Artifact
from app.store import EpisodeStore
from app.core.capabilities import TaskRegistry
from app.core.store import V2EpisodeStore

from .helpers import EVIDENCE_REQUEST, TASKS_ROOT, make_app


class FailingMigrationStore(V2EpisodeStore):
    def _migration_statements(self, version):
        return [
            "CREATE TABLE migration_should_rollback(value TEXT)",
            "THIS IS NOT VALID SQLITE",
        ]


class V2StoreMigrationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Path(self.tempdir.name) / "episodes.sqlite3"
        self.registry = TaskRegistry(str(TASKS_ROOT))

    def tearDown(self):
        self.tempdir.cleanup()

    def table_names(self):
        with sqlite3.connect(self.database) as connection:
            return {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }

    def test_empty_database_migrates_idempotently(self):
        first = V2EpisodeStore(str(self.database), self.registry)
        self.assertEqual(first.schema_version(), 4)
        expected_tables = {
            "v2_episodes",
            "v2_events",
            "v2_action_results",
            "v2_observations",
            "v2_artifacts",
            "v2_evidence",
            "v2_tool_runs",
            "v2_evaluations",
            "v2_episode_artifacts",
            "v2_observation_artifacts",
        }
        self.assertTrue(expected_tables.issubset(self.table_names()))
        second = V2EpisodeStore(str(self.database), self.registry)
        self.assertEqual(second.schema_version(), 4)
        with sqlite3.connect(self.database) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM v2_schema_migrations"
            ).fetchone()[0]
        self.assertEqual(count, 4)

    def test_existing_v1_rows_are_not_read_or_modified(self):
        v1_store = EpisodeStore(str(self.database))
        reset = v1_store.create_episode(
            {"task_id": "v1-preserve", "prompt": "Do not modify me."}
        )
        episode_id = reset["episode_id"]
        with sqlite3.connect(self.database) as connection:
            before = connection.execute(
                "SELECT state_json FROM episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()[0]
        V2EpisodeStore(str(self.database), self.registry)
        with sqlite3.connect(self.database) as connection:
            after = connection.execute(
                "SELECT state_json FROM episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()[0]
            v2_count = connection.execute(
                "SELECT COUNT(*) FROM v2_episodes"
            ).fetchone()[0]
        self.assertEqual(json.loads(before), json.loads(after))
        self.assertEqual(before, after)
        self.assertEqual(v2_count, 0)

    def test_schema_one_v2_episode_is_preserved_when_migrating_to_two(self):
        store = V2EpisodeStore(str(self.database), self.registry)
        episode = store.create_episode("worldcover-grounded-vqa", "1.0.0", 42)
        with sqlite3.connect(self.database) as connection:
            before = connection.execute(
                "SELECT state_json FROM v2_episodes WHERE episode_id = ?",
                (episode.episode_id,),
            ).fetchone()[0]
            connection.execute("DROP TABLE v2_observation_artifacts")
            connection.execute("DROP TABLE v2_episode_artifacts")
            connection.execute("DROP TABLE v2_artifacts")
            connection.execute("""
                CREATE TABLE v2_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sha256 TEXT,
                    size_bytes INTEGER,
                    artifact_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES v2_episodes(episode_id)
                )
            """)
            connection.execute("DROP TABLE IF EXISTS old_v2_observation_artifacts")
            connection.execute("DELETE FROM v2_schema_migrations WHERE version = 2")
            connection.execute("DELETE FROM v2_schema_migrations WHERE version = 3")
            connection.execute("DELETE FROM v2_schema_migrations WHERE version = 4")

        migrated = V2EpisodeStore(str(self.database), self.registry)
        with sqlite3.connect(self.database) as connection:
            after = connection.execute(
                "SELECT state_json FROM v2_episodes WHERE episode_id = ?",
                (episode.episode_id,),
            ).fetchone()[0]
        self.assertEqual(migrated.schema_version(), STORE_SCHEMA_VERSION)
        self.assertEqual(before, after)
        self.assertIn("v2_episode_artifacts", self.table_names())
        self.assertIn("v2_observation_artifacts", self.table_names())

    def test_schema_two_evidence_rows_are_preserved_when_scoping_to_episode(self):
        store = V2EpisodeStore(str(self.database), self.registry)
        episode = store.create_episode("worldcover-grounded-vqa", "1.0.0", 42)
        evidence_json = json.dumps({"evidence_id": "ev-preserve-scope"})
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                INSERT INTO v2_evidence (
                    evidence_id, episode_id, evidence_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                ("ev-preserve-scope", episode.episode_id, evidence_json,
                 "2026-09-25T00:00:00Z"),
            )
            connection.commit()
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("BEGIN")
            connection.execute(
                """
                CREATE TABLE v2_evidence_schema_two (
                    evidence_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES v2_episodes(episode_id)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO v2_evidence_schema_two (
                    evidence_id, episode_id, evidence_json, created_at
                )
                SELECT evidence_id, episode_id, evidence_json, created_at
                FROM v2_evidence
                """
            )
            connection.execute("DROP TABLE v2_evidence")
            connection.execute("ALTER TABLE v2_evidence_schema_two RENAME TO v2_evidence")
            connection.execute("DELETE FROM v2_schema_migrations WHERE version = 3")
            connection.execute("DELETE FROM v2_schema_migrations WHERE version = 4")
            connection.commit()
            connection.execute("PRAGMA foreign_keys = ON")

        migrated = V2EpisodeStore(str(self.database), self.registry)
        with sqlite3.connect(self.database) as connection:
            primary_key = [
                row[1]
                for row in connection.execute("PRAGMA table_info(v2_evidence)")
                if row[5]
            ]
            preserved = connection.execute(
                """
                SELECT evidence_json FROM v2_evidence
                WHERE episode_id = ? AND evidence_id = ?
                """,
                (episode.episode_id, "ev-preserve-scope"),
            ).fetchone()

        self.assertEqual(migrated.schema_version(), STORE_SCHEMA_VERSION)
        self.assertEqual(primary_key, ["evidence_id", "episode_id"])
        self.assertEqual(json.loads(preserved[0])["evidence_id"], "ev-preserve-scope")

    def test_evidence_ids_are_local_to_each_episode(self):
        application = make_app(self.database)
        with TestClient(application, raise_server_exceptions=False) as client:
            episodes = []
            for index in range(2):
                reset = client.post("/v2/reset", json={
                    "task_ref": {
                        "task_id": "worldcover-grounded-vqa",
                        "task_version": "1.0.0",
                    },
                    "seed": 42,
                })
                self.assertEqual(reset.status_code, 201, reset.text)
                episode_id = reset.json()["data"]["episode_id"]
                episodes.append(episode_id)
                evidence = dict(EVIDENCE_REQUEST)
                evidence["client_action_id"] = "evidence-episode-%s" % index
                evidence["expected_state_version"] = 0
                saved = client.post("/v2/episodes/%s/step" % episode_id, json=evidence)
                self.assertEqual(saved.status_code, 200, saved.text)

            duplicate = dict(EVIDENCE_REQUEST)
            duplicate["client_action_id"] = "evidence-duplicate"
            duplicate["expected_state_version"] = 1
            conflict = client.post("/v2/episodes/%s/step" % episodes[0], json=duplicate)
            self.assertEqual(conflict.status_code, 409, conflict.text)
            self.assertEqual(conflict.json()["error"]["code"], "evidence_conflict")

        self.assertNotEqual(*episodes)
        with sqlite3.connect(self.database) as connection:
            rows = connection.execute(
                """
                SELECT episode_id, evidence_id FROM v2_evidence
                WHERE evidence_id = ?
                ORDER BY episode_id
                """,
                (EVIDENCE_REQUEST["action"]["evidence"]["evidence_id"],),
            ).fetchall()
        self.assertEqual(len(rows), 2)

    def test_schema_three_artifact_rows_are_preserved_when_scoping_to_episode(self):
        store = V2EpisodeStore(str(self.database), self.registry)
        episode = store.create_episode("worldcover-grounded-vqa", "1.0.0", 42)
        artifact_id = "art-" + "4" * 64
        artifact_json = json.dumps({"artifact_id": artifact_id})
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                INSERT INTO v2_artifacts (
                    artifact_id, episode_id, status, sha256,
                    size_bytes, artifact_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (artifact_id, episode.episode_id, "created", "4" * 64, 1,
                 artifact_json, "2026-09-25T00:00:00Z"),
            )
            connection.execute(
                """
                INSERT INTO v2_episode_artifacts (
                    episode_id, artifact_id, created_at
                ) VALUES (?, ?, ?)
                """,
                (episode.episode_id, artifact_id, "2026-09-25T00:00:00Z"),
            )
            connection.commit()
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("BEGIN")
            connection.execute(
                """
                CREATE TABLE v2_artifacts_schema_three (
                    artifact_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sha256 TEXT,
                    size_bytes INTEGER,
                    artifact_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES v2_episodes(episode_id)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO v2_artifacts_schema_three (
                    artifact_id, episode_id, status, sha256,
                    size_bytes, artifact_json, created_at
                )
                SELECT artifact_id, episode_id, status, sha256,
                       size_bytes, artifact_json, created_at
                FROM v2_artifacts
                """
            )
            connection.execute("DROP TABLE v2_artifacts")
            connection.execute(
                "ALTER TABLE v2_artifacts_schema_three RENAME TO v2_artifacts"
            )
            connection.execute("DELETE FROM v2_schema_migrations WHERE version = 4")
            connection.commit()
            connection.execute("PRAGMA foreign_keys = ON")

        migrated = V2EpisodeStore(str(self.database), self.registry)
        with sqlite3.connect(self.database) as connection:
            primary_key = [
                row[1] for row in connection.execute("PRAGMA table_info(v2_artifacts)")
                if row[5]
            ]
            preserved = connection.execute(
                """
                SELECT artifact_json FROM v2_artifacts
                WHERE episode_id = ? AND artifact_id = ?
                """,
                (episode.episode_id, artifact_id),
            ).fetchone()

        self.assertEqual(migrated.schema_version(), 4)
        self.assertEqual(primary_key, ["artifact_id", "episode_id"])
        self.assertEqual(json.loads(preserved[0])["artifact_id"], artifact_id)

    def test_deterministic_artifact_ids_are_local_to_each_episode(self):
        store = V2EpisodeStore(str(self.database), self.registry)
        artifact_json = json.dumps({
            "artifact_id": "art-" + "5" * 64,
            "kind": "image",
            "media_type": "image/png",
            "sha256": "5" * 64,
            "size_bytes": 1,
            "uri": "artifact://sha256/55/" + "5" * 64,
            "lineage": {
                "tool_id": "tool", "tool_version": "1.0.0",
                "input_refs": [], "parameters_hash": "0" * 64,
            },
        }, sort_keys=True, separators=(",", ":"))
        episodes = []
        for index in range(2):
            episode = store.create_episode(
                "worldcover-grounded-vqa", "1.0.0", 42
            )
            episodes.append(episode.episode_id)
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            for index in range(2):
                V2EpisodeStore._register_artifact(
                    connection, episodes[index], "obs-" + str(index),
                    TypeAdapter(Artifact).validate_python(json.loads(artifact_json)),
                    "2026-09-25T00:00:00Z",
                )
            connection.commit()
        with sqlite3.connect(self.database) as connection:
            rows = connection.execute(
                """
                SELECT episode_id, artifact_id FROM v2_artifacts
                WHERE artifact_id = ? ORDER BY episode_id
                """,
                ("art-" + "5" * 64,),
            ).fetchall()
        self.assertEqual(len(rows), 2)

    def test_schema_three_duplicate_association_materializes_episode_row(self):
        store = V2EpisodeStore(str(self.database), self.registry)
        first = store.create_episode("worldcover-grounded-vqa", "1.0.0", 42)
        second = store.create_episode("worldcover-grounded-vqa", "1.0.0", 42)
        artifact_id = "art-" + "6" * 64
        artifact_json = json.dumps({"artifact_id": artifact_id})
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                INSERT INTO v2_artifacts (
                    artifact_id, episode_id, status, sha256,
                    size_bytes, artifact_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (artifact_id, first.episode_id, "created", "6" * 64, 1,
                 artifact_json, "2026-09-25T00:00:00Z"),
            )
            for episode_id in (first.episode_id, second.episode_id):
                observation_id = "obs-" + episode_id[4:12]
                connection.execute(
                    """
                    INSERT INTO v2_observations (
                        observation_id, episode_id, sequence,
                        created_at, observation_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (observation_id, episode_id, 1,
                     "2026-09-25T00:00:00Z", "{}"),
                )
                connection.execute(
                    """
                    INSERT INTO v2_episode_artifacts (
                        episode_id, artifact_id, created_at
                    ) VALUES (?, ?, ?)
                    """,
                    (episode_id, artifact_id, "2026-09-25T00:00:00Z"),
                )
            connection.execute("DELETE FROM v2_schema_migrations WHERE version = 4")
            connection.execute(
                """
                CREATE TABLE v2_observation_artifacts_schema_three (
                    observation_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (observation_id, artifact_id),
                    FOREIGN KEY (observation_id)
                        REFERENCES v2_observations(observation_id)
                )
                """
            )
            for observation_id in (
                "obs-" + first.episode_id[4:12],
                "obs-" + second.episode_id[4:12],
            ):
                connection.execute(
                    """
                    INSERT INTO v2_observation_artifacts_schema_three (
                        observation_id, artifact_id, created_at
                    ) VALUES (?, ?, ?)
                    """,
                    (observation_id, artifact_id, "2026-09-25T00:00:00Z"),
                )
            connection.execute(
                """
                INSERT INTO v2_observation_artifacts_schema_three (
                    observation_id, artifact_id, created_at
                )
                SELECT observation_id, artifact_id, created_at
                FROM v2_observation_artifacts
                """
            )
            connection.execute("DROP TABLE v2_observation_artifacts")
            connection.execute(
                """
                ALTER TABLE v2_observation_artifacts_schema_three
                RENAME TO v2_observation_artifacts
                """
            )

        migrated = V2EpisodeStore(str(self.database), self.registry)
        with sqlite3.connect(self.database) as connection:
            rows = connection.execute(
                """
                SELECT episode_id FROM v2_artifacts
                WHERE artifact_id = ? ORDER BY episode_id
                """,
                (artifact_id,),
            ).fetchall()
        self.assertEqual(migrated.schema_version(), 4)
        self.assertEqual(rows, sorted([
            (first.episode_id,), (second.episode_id,),
        ]))

    def test_failed_migration_does_not_advance_version_or_leave_partial_table(self):
        with self.assertRaises(sqlite3.OperationalError):
            FailingMigrationStore(str(self.database), self.registry)
        self.assertNotIn("migration_should_rollback", self.table_names())
        with sqlite3.connect(self.database) as connection:
            version = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM v2_schema_migrations"
            ).fetchone()[0]
        self.assertEqual(version, 0)


if __name__ == "__main__":
    unittest.main()

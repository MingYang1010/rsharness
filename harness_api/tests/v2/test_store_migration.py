import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.store import EpisodeStore
from app.v2.capabilities import TaskRegistry
from app.v2.store import V2EpisodeStore

from .helpers import TASKS_ROOT


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
        self.assertEqual(first.schema_version(), 2)
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
        self.assertEqual(second.schema_version(), 2)
        with sqlite3.connect(self.database) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM v2_schema_migrations"
            ).fetchone()[0]
        self.assertEqual(count, 2)

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
            connection.execute(
                """
                INSERT INTO v2_artifacts (
                    artifact_id, episode_id, status, sha256,
                    size_bytes, artifact_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "art-" + "1" * 64,
                    episode.episode_id,
                    "created",
                    "1" * 64,
                    1,
                    "{}",
                    "2026-08-30T00:00:00Z",
                ),
            )
            connection.execute("DELETE FROM v2_schema_migrations WHERE version = 2")

        migrated = V2EpisodeStore(str(self.database), self.registry)
        with sqlite3.connect(self.database) as connection:
            after = connection.execute(
                "SELECT state_json FROM v2_episodes WHERE episode_id = ?",
                (episode.episode_id,),
            ).fetchone()[0]
        self.assertEqual(migrated.schema_version(), 2)
        self.assertEqual(before, after)
        self.assertIn("v2_episode_artifacts", self.table_names())
        self.assertIn("v2_observation_artifacts", self.table_names())
        with sqlite3.connect(self.database) as connection:
            association = connection.execute(
                """
                SELECT episode_id FROM v2_episode_artifacts
                WHERE artifact_id = ?
                """,
                ("art-" + "1" * 64,),
            ).fetchone()
        self.assertEqual(association[0], episode.episode_id)

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

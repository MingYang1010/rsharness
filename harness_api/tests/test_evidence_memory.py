import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from app.core.capabilities import TaskRegistry
from app.core.domain import create_initial_state
from app.core.evidence_memory import (
    EvidenceMemoryBinding,
    EvidenceMemoryPolicy,
    EvidenceMemoryStore,
    MemorySearchArguments,
    authorize_memory_writer,
    build_evidence_memory_record,
    load_evidence_memory_policy,
    load_source_evidence,
)
from app.core.events import sha256_json
from app.core.schemas import AnswerRecord, EvidenceRef, MetricResult, TaskManifest


ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "tasks"
ACTOR_CERTIFICATE = "1" * 64
POLICY_SHA = "2" * 64
NOW = "2026-08-02T00:00:00Z"


def policy_value():
    return {
        "schema_version": "1.0.0",
        "policy_id": "research-memory-v1",
        "scope_id": "nanjing-temporal-memory",
        "valid_from": "2026-01-01T00:00:00Z",
        "expires_at": "2027-01-01T00:00:00Z",
        "writers": [
            {
                "actor_id": "trusted-memory-curator",
                "certificate_sha256": ACTOR_CERTIFICATE,
            }
        ],
        "source_grants": [
            {
                "task_id": "worldcover-grounded-vqa",
                "task_versions": ["1.1.0"],
                "evaluator_id": "worldcover-grounded-v1",
                "min_aggregate_reward": 0.8,
            }
        ],
        "reader_grants": [
            {
                "task_id": "worldcover-grounded-vqa",
                "task_versions": ["1.1.0"],
            }
        ],
        "max_record_ttl_seconds": 30 * 24 * 60 * 60,
        "max_active_records": 8,
        "max_query_results": 5,
    }


class EvidenceMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.policy = EvidenceMemoryPolicy.model_validate(policy_value())
        self.store = EvidenceMemoryStore(self.root / "memory" / "events.sqlite3")
        self.manifest, self.state, self.evaluation, self.evidence, self.source = (
            self.source_episode()
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def source_episode():
        original = TaskRegistry(str(TASKS)).get("worldcover-grounded-vqa", "1.1.0")
        source = original.assets[0].model_copy(update={"instrument": "WorldCover-map"})
        body = {
            "task": original.task,
            "scenario": original.scenario,
            "assets": [source, *original.assets[1:]],
            "evaluator": original.evaluator,
        }
        manifest = TaskManifest(
            **body,
            task_manifest_hash=sha256_json(
                {
                    "task": body["task"].model_dump(mode="json"),
                    "scenario": body["scenario"].model_dump(mode="json"),
                    "assets": [item.model_dump(mode="json") for item in body["assets"]],
                    "evaluator": body["evaluator"].model_dump(mode="json"),
                }
            ),
        )
        state, _ = create_initial_state(
            "ep2-" + "a" * 32,
            manifest,
            42,
            "2026-08-01T00:00:00Z",
        )
        evidence = EvidenceRef(
            evidence_id="ev-memory-source",
            claim_id="dominant-class",
            source_ref=source.asset_id,
            selector={
                "bbox": {
                    "west": 120.5,
                    "south": 30.5,
                    "east": 121.0,
                    "north": 31.0,
                },
                "time_range": source.temporal,
                "bands": ["visual"],
            },
            description="Reviewed source evidence for cross-task memory.",
            frozen_sha256=source.sha256,
        )
        state = state.model_copy(
            update={
                "status": "terminated",
                "state_version": 2,
                "step_count": 2,
                "evidence_refs": [evidence],
                "final_answer": AnswerRecord(
                    outcome="submitted",
                    answer={"label": "built-up"},
                    confidence=1.0,
                    evidence_ids=[evidence.evidence_id],
                ),
                "updated_at": "2026-08-01T00:00:00Z",
            }
        )
        evaluation = MetricResult(
            evaluation_id="eval-memory-source",
            status="completed",
            metrics=[],
            aggregate_reward=1.0,
            evaluator_id="worldcover-grounded-v1",
            evaluator_version="1.1.0",
        )
        return manifest, state, evaluation, evidence, source

    def record(self, summary="Built-up was dominant in the reviewed area."):
        return build_evidence_memory_record(
            policy=self.policy,
            policy_sha256=POLICY_SHA,
            manifest=self.manifest,
            state=self.state,
            evaluation=self.evaluation,
            evidence=self.evidence,
            source=self.source,
            object_type="land-cover-assessment",
            public_summary=summary,
            ttl_seconds=14 * 24 * 60 * 60,
        )

    def publish(self, record=None):
        return self.store.publish(
            self.policy,
            POLICY_SHA,
            record or self.record(),
            actor_id="trusted-memory-curator",
            actor_certificate_sha256=ACTOR_CERTIFICATE,
            now=NOW,
        )

    def reader_manifest(self, sequence, digest, as_of=NOW):
        binding = EvidenceMemoryBinding(
            schema_version="1.0.0",
            policy_id=self.policy.policy_id,
            policy_sha256=POLICY_SHA,
            scope_id=self.policy.scope_id,
            snapshot_sequence=sequence,
            snapshot_sha256=digest,
            as_of=as_of,
        )
        task = self.manifest.task.model_copy(
            update={
                "metadata": {
                    **self.manifest.task.metadata,
                    "evidence_memory": binding.model_dump(mode="json"),
                }
            }
        )
        return self.manifest.model_copy(update={"task": task})

    def query(self):
        return MemorySearchArguments(
            bbox={"west": 120.6, "south": 30.6, "east": 120.9, "north": 30.9},
            time_range={
                "start": "2021-06-01T00:00:00Z",
                "end": "2021-06-30T00:00:00Z",
            },
            object_type="land-cover-assessment",
            limit=5,
        )

    def test_policy_pin_permissions_and_uniqueness(self):
        path = self.root / "policy.json"
        content = json.dumps(policy_value(), sort_keys=True).encode()
        path.write_bytes(content)
        os.chmod(path, 0o644)
        digest = hashlib.sha256(content).hexdigest()
        loaded, actual = load_evidence_memory_policy(path, digest)
        self.assertEqual(loaded, self.policy)
        self.assertEqual(actual, digest)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_evidence_memory_policy(path, "f" * 64)
        os.chmod(path, 0o666)
        with self.assertRaisesRegex(ValueError, "writable"):
            load_evidence_memory_policy(path, digest)
        invalid = policy_value()
        invalid["writers"].append(dict(invalid["writers"][0]))
        with self.assertRaises(ValidationError):
            EvidenceMemoryPolicy.model_validate(invalid)

    def test_writer_certificate_and_source_evaluation_fail_closed(self):
        authorize_memory_writer(
            self.policy,
            "trusted-memory-curator",
            ACTOR_CERTIFICATE,
            now=NOW,
        )
        with self.assertRaisesRegex(ValueError, "writer identity"):
            authorize_memory_writer(
                self.policy,
                "trusted-memory-curator",
                "3" * 64,
                now=NOW,
            )
        weak = self.evaluation.model_copy(update={"aggregate_reward": 0.79})
        with self.assertRaisesRegex(ValueError, "below"):
            build_evidence_memory_record(
                policy=self.policy,
                policy_sha256=POLICY_SHA,
                manifest=self.manifest,
                state=self.state,
                evaluation=weak,
                evidence=self.evidence,
                source=self.source,
                object_type="land-cover-assessment",
                public_summary="This must not be published.",
                ttl_seconds=3600,
            )

    def test_source_loader_is_read_only_and_manifest_pinned(self):
        original = TaskRegistry(str(TASKS)).get(
            "worldcover-grounded-vqa", "1.1.0"
        )
        state, _ = create_initial_state(
            "ep2-" + "b" * 32,
            original,
            42,
            "2026-08-01T00:00:00Z",
        )
        source = original.assets[0]
        evidence = EvidenceRef(
            evidence_id="ev-loader-source",
            claim_id="loader-claim",
            source_ref=source.asset_id,
            selector={
                "bbox": {
                    "west": 120.5,
                    "south": 30.5,
                    "east": 121.0,
                    "north": 31.0,
                },
                "time_range": source.temporal,
                "bands": ["visual"],
            },
            description="Source-loader fixture evidence.",
            frozen_sha256=source.sha256,
        )
        state = state.model_copy(update={"evidence_refs": [evidence]})
        database = self.root / "source episodes.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                "CREATE TABLE v2_episodes(episode_id TEXT,task_id TEXT,task_version TEXT,task_manifest_hash TEXT,state_json TEXT)"
            )
            connection.execute(
                "CREATE TABLE v2_evaluations(episode_id TEXT,evaluation_json TEXT)"
            )
            connection.execute(
                "CREATE TABLE v2_episode_artifacts(episode_id TEXT,artifact_id TEXT)"
            )
            connection.execute(
                "CREATE TABLE v2_artifacts(artifact_id TEXT,status TEXT,artifact_json TEXT)"
            )
            connection.execute(
                "INSERT INTO v2_episodes VALUES(?,?,?,?,?)",
                (
                    state.episode_id,
                    state.task_ref.task_id,
                    state.task_ref.task_version,
                    state.task_manifest_hash,
                    state.model_dump_json(),
                ),
            )
            connection.execute(
                "INSERT INTO v2_evaluations VALUES(?,?)",
                (state.episode_id, self.evaluation.model_dump_json()),
            )
            connection.commit()
        finally:
            connection.close()
        before = database.read_bytes()
        loaded = load_source_evidence(
            database, TASKS, state.episode_id, evidence.evidence_id
        )
        self.assertEqual(loaded[0].task_manifest_hash, original.task_manifest_hash)
        self.assertEqual(loaded[1], state)
        self.assertEqual(loaded[3], evidence)
        self.assertEqual(loaded[4], source)
        self.assertEqual(database.read_bytes(), before)

    def test_append_only_snapshots_invalidation_and_historical_replay(self):
        record = self.record()
        first = self.publish(record)
        self.assertEqual(first.sequence, 1)
        self.assertEqual(first.active_records, [record])
        self.assertEqual(self.publish(record), first)
        second = self.store.invalidate(
            self.policy,
            POLICY_SHA,
            record.memory_id,
            "superseded by operator review",
            actor_id="trusted-memory-curator",
            actor_certificate_sha256=ACTOR_CERTIFICATE,
            now="2026-08-03T00:00:00Z",
        )
        self.assertEqual(second.sequence, 2)
        self.assertEqual(second.active_records, [])
        retry = self.store.invalidate(
            self.policy,
            POLICY_SHA,
            record.memory_id,
            "superseded by operator review",
            actor_id="trusted-memory-curator",
            actor_certificate_sha256=ACTOR_CERTIFICATE,
            now="2026-08-04T00:00:00Z",
        )
        self.assertEqual(retry, second)
        with self.assertRaisesRegex(ValueError, "different content"):
            self.store.invalidate(
                self.policy,
                POLICY_SHA,
                record.memory_id,
                "a different retry reason",
                actor_id="trusted-memory-curator",
                actor_certificate_sha256=ACTOR_CERTIFICATE,
                now="2026-08-04T00:00:00Z",
            )
        historical = self.store.snapshot(self.policy.scope_id, 1)
        self.assertEqual(historical.active_records, [record])
        self.assertNotEqual(historical.snapshot_sha256, second.snapshot_sha256)

    def test_pinned_query_public_projection_cost_and_ttl(self):
        snapshot = self.publish()
        manifest = self.reader_manifest(snapshot.sequence, snapshot.snapshot_sha256)
        result = self.store.search(self.policy, POLICY_SHA, manifest, self.query())
        self.assertEqual(result.matched_count, 1)
        self.assertEqual(len(result.records), 1)
        public = result.records[0].model_dump(mode="json")
        self.assertNotIn("episode_id", public)
        self.assertNotIn("evidence_id", public)
        self.assertNotIn("source_ref", public)
        self.assertEqual(result.cost.records_scanned, 1)
        expired = self.reader_manifest(
            snapshot.sequence,
            snapshot.snapshot_sha256,
            "2026-08-20T00:00:00Z",
        )
        self.assertEqual(
            self.store.search(self.policy, POLICY_SHA, expired, self.query()).records,
            [],
        )
        wrong = self.reader_manifest(snapshot.sequence, "f" * 64)
        with self.assertRaisesRegex(ValueError, "snapshot pin"):
            self.store.search(self.policy, POLICY_SHA, wrong, self.query())

    def test_tampered_event_and_wide_store_permissions_fail_closed(self):
        self.publish()
        connection = sqlite3.connect(self.store.path)
        try:
            connection.execute(
                "UPDATE evidence_memory_events SET event_json=replace(event_json,'Built-up','Forest')"
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(ValueError, "chain is invalid"):
            self.store.snapshot(self.policy.scope_id)
        os.chmod(self.store.path, 0o644)
        with self.assertRaisesRegex(ValueError, "owner-private"):
            self.store.snapshot(self.policy.scope_id)


if __name__ == "__main__":
    unittest.main()

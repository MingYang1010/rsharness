import hashlib
import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.agent_credentials import (
    AgentCredentialRegistry,
    AgentSessionCredential,
    PublicTask,
)
from app.agent_gateway import AgentBinding
from app.control_plane import (
    AgentIssuancePolicy,
    ControlEventInput,
    append_control_event,
    authorize_issuance,
    authorize_management,
    load_issuance_policy,
    verify_control_audit,
)
from app.v2.schemas import BudgetSpec


PROJECT = Path(__file__).resolve().parents[2]
QUERY_SPEC = importlib.util.spec_from_file_location(
    "query_control_plane_audit",
    PROJECT / "scripts" / "query_control_plane_audit.py",
)
QUERY = importlib.util.module_from_spec(QUERY_SPEC)
assert QUERY_SPEC.loader is not None
QUERY_SPEC.loader.exec_module(QUERY)


NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
POLICY_SHA = "a" * 64
MANIFEST_SHA = "b" * 64
CERTIFICATE_SHA = "c" * 64


def policy() -> AgentIssuancePolicy:
    return AgentIssuancePolicy(
        policy_id="research-policy-v1",
        valid_from=NOW - timedelta(days=1),
        expires_at=NOW + timedelta(days=1),
        issuers=["trusted-operator"],
        subjects=["approved-runner"],
        grants=[
            {
                "task_id": "task-a",
                "task_version": "1.0.0",
                "max_ttl_seconds": 3600,
            }
        ],
        max_active_sessions_per_subject=2,
    )


def event(event_type: str, *, completed: bool = False) -> ControlEventInput:
    return ControlEventInput(
        event_type=event_type,
        operation_id="op-" + "d" * 32,
        actor_id="trusted-operator",
        subject_id="approved-runner",
        policy_id="research-policy-v1",
        policy_sha256=POLICY_SHA,
        task_id="task-a",
        task_version="1.0.0",
        task_manifest_hash=MANIFEST_SHA,
        episode_id="ep2-" + "1" * 32 if completed else None,
        generation=1 if completed else None,
    )


def binding() -> AgentBinding:
    return AgentBinding(
        episode_id="ep2-" + "1" * 32,
        task_manifest_hash=MANIFEST_SHA,
        token_sha256="c" * 64,
        task=PublicTask(
            task_id="task-a",
            task_version="1.0.0",
            prompt="reviewed prompt",
            answer_schema={"type": "object"},
            budget=BudgetSpec(
                max_steps=2,
                max_tool_calls=0,
                max_wall_time_ms=1000,
                max_input_bytes=0,
                max_artifact_bytes=0,
            ),
            input_asset_refs=["asset-a"],
            allowed_actions=["answer.abstain"],
            allowed_tools=[],
        ),
    )


class IssuancePolicyTests(unittest.TestCase):
    def test_policy_checksum_identity_task_ttl_and_session_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            content = policy().model_dump_json(indent=2).encode()
            path.write_bytes(content)
            path.chmod(0o644)
            loaded, digest = load_issuance_policy(path, hashlib.sha256(content).hexdigest())
            self.assertEqual(loaded, policy())
            self.assertEqual(digest, hashlib.sha256(content).hexdigest())
            authorize_issuance(
                loaded,
                actor_id="trusted-operator",
                subject_id="approved-runner",
                task_id="task-a",
                task_version="1.0.0",
                ttl_seconds=3600,
                active_subject_sessions=1,
                now=NOW,
            )
            denied = [
                {"actor_id": "other-operator"},
                {"subject_id": "other-runner"},
                {"task_id": "task-b"},
                {"ttl_seconds": 3601},
                {"active_subject_sessions": 2},
                {"now": loaded.expires_at},
            ]
            base = {
                "actor_id": "trusted-operator",
                "subject_id": "approved-runner",
                "task_id": "task-a",
                "task_version": "1.0.0",
                "ttl_seconds": 3600,
                "active_subject_sessions": 0,
                "now": NOW,
            }
            for change in denied:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    authorize_issuance(loaded, **{**base, **change})
            with self.assertRaisesRegex(ValueError, "checksum changed"):
                load_issuance_policy(path, "f" * 64)
            path.chmod(0o666)
            with self.assertRaisesRegex(ValueError, "permissions"):
                load_issuance_policy(path, hashlib.sha256(content).hexdigest())

    def test_certificate_policy_binds_each_subject_to_one_sha256(self):
        certificate_policy = AgentIssuancePolicy(
            schema_version="1.1.0",
            policy_id="certificate-policy-v1",
            valid_from=NOW - timedelta(days=1),
            expires_at=NOW + timedelta(days=1),
            issuers=["trusted-operator"],
            subjects=["approved-runner"],
            subject_certificates=[{
                "subject_id": "approved-runner",
                "certificate_sha256": CERTIFICATE_SHA,
            }],
            grants=[{
                "task_id": "task-a",
                "task_version": "1.0.0",
                "max_ttl_seconds": 3600,
            }],
            max_active_sessions_per_subject=1,
        )
        arguments = {
            "actor_id": "trusted-operator",
            "subject_id": "approved-runner",
            "task_id": "task-a",
            "task_version": "1.0.0",
            "ttl_seconds": 3600,
            "active_subject_sessions": 0,
            "subject_certificate_sha256": CERTIFICATE_SHA,
            "now": NOW,
        }
        authorize_issuance(certificate_policy, **arguments)
        with self.assertRaisesRegex(ValueError, "certificate identity"):
            authorize_issuance(
                certificate_policy,
                **{**arguments, "subject_certificate_sha256": "d" * 64},
            )
        with self.assertRaisesRegex(ValueError, "does not grant certificate"):
            authorize_issuance(
                policy(),
                **arguments,
            )
        with self.assertRaisesRegex(ValueError, "every subject"):
            AgentIssuancePolicy(
                **certificate_policy.model_dump(
                    exclude={"subject_certificates"}
                ),
                subject_certificates=[],
            )

    def test_expired_policy_still_allows_revocation_but_not_rotation(self):
        expired = policy().model_copy(
            update={
                "valid_from": NOW - timedelta(days=2),
                "expires_at": NOW - timedelta(days=1),
            }
        )
        authorize_management(
            expired,
            actor_id="trusted-operator",
            subject_id="approved-runner",
            task_id="task-a",
            task_version="1.0.0",
            ttl_seconds=None,
            rotate=False,
            now=NOW,
        )
        with self.assertRaisesRegex(ValueError, "not currently valid"):
            authorize_management(
                expired,
                actor_id="trusted-operator",
                subject_id="approved-runner",
                task_id="task-a",
                task_version="1.0.0",
                ttl_seconds=3600,
                rotate=True,
                now=NOW,
            )


class ControlAuditTests(unittest.TestCase):
    def test_hash_chain_is_private_append_only_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            first = append_control_event(
                path,
                event("issuance_started"),
                now=NOW,
            )
            second = append_control_event(
                path,
                event("issuance_completed", completed=True),
                now=NOW + timedelta(seconds=1),
            )
            self.assertEqual((first.sequence, second.sequence), (1, 2))
            self.assertEqual(second.previous_event_sha256, first.event_sha256)
            self.assertEqual(verify_control_audit(path), [first, second])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            summary = QUERY.query(
                path,
                operation_id="op-" + "d" * 32,
                subject_id="approved-runner",
                policy_sha256=POLICY_SHA,
                limit=1,
            )
            self.assertTrue(summary["chain_valid"])
            self.assertEqual(summary["event_count"], 2)
            self.assertEqual(summary["matched_count"], 2)
            self.assertEqual(summary["returned_count"], 1)
            self.assertTrue(summary["truncated"])
            self.assertEqual(summary["head_event_sha256"], second.event_sha256)
            raw = path.read_bytes()
            self.assertNotIn(b"agent-token", raw)
            path.write_bytes(raw.replace(b"trusted-operator", b"trusted-operatox", 1))
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "chain is invalid"):
                verify_control_audit(path)
            with self.assertRaisesRegex(ValueError, "chain is invalid"):
                append_control_event(path, event("binding_reused", completed=True))

    def test_partial_insecure_and_symlink_audits_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "audit.jsonl"
            path.write_text("{}")
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "partial final event"):
                verify_control_audit(path)
            path.write_text("{}\n")
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "owner-private"):
                verify_control_audit(path)
            target = root / "target.jsonl"
            target.write_text("{}\n")
            target.chmod(0o600)
            link = root / "link.jsonl"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                verify_control_audit(link)


class GovernedRegistryTests(unittest.TestCase):
    def test_schema_1_1_requires_complete_governance(self):
        issued = NOW - timedelta(minutes=1)
        legacy = AgentSessionCredential(
            binding=binding(),
            issued_at=issued,
            expires_at=issued + timedelta(hours=1),
        )
        governed = legacy.model_copy(
            update={
                "issuer_id": "trusted-operator",
                "subject_id": "approved-runner",
                "issuance_policy_id": "research-policy-v1",
                "issuance_policy_sha256": POLICY_SHA,
            }
        )
        AgentCredentialRegistry(schema_version="1.0.0", sessions=[legacy])
        AgentCredentialRegistry(schema_version="1.1.0", sessions=[governed])
        certificate_bound = governed.model_copy(
            update={"subject_certificate_sha256": CERTIFICATE_SHA}
        )
        AgentCredentialRegistry(
            schema_version="1.2.0", sessions=[certificate_bound]
        )
        with self.assertRaisesRegex(ValueError, "legacy registry"):
            AgentCredentialRegistry(schema_version="1.0.0", sessions=[governed])
        with self.assertRaisesRegex(ValueError, "governed registry"):
            AgentCredentialRegistry(schema_version="1.1.0", sessions=[legacy])
        with self.assertRaisesRegex(ValueError, "does not support certificate"):
            AgentCredentialRegistry(
                schema_version="1.1.0", sessions=[certificate_bound]
            )
        with self.assertRaisesRegex(ValueError, "requires a pin"):
            AgentCredentialRegistry(schema_version="1.2.0", sessions=[governed])


if __name__ == "__main__":
    unittest.main()

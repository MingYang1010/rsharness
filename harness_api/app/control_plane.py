"""Pinned issuance policy and hash-chained control-plane audit records."""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import ssl
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from .v2.schemas import EpisodeId, Identifier, SemanticVersion, Sha256, V2RequestModel

MAX_POLICY_BYTES = 1024 * 1024
MAX_AUDIT_BYTES = 64 * 1024 * 1024
MAX_AUDIT_EVENT_BYTES = 16 * 1024
IDENTITY_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
ZERO = timedelta(0)


class IssuanceGrant(V2RequestModel):
    task_id: Identifier
    task_version: SemanticVersion
    max_ttl_seconds: int = Field(ge=60, le=7 * 24 * 60 * 60)


class SubjectCertificate(V2RequestModel):
    subject_id: str
    certificate_sha256: Sha256

    @model_validator(mode="after")
    def valid_subject(self):
        if IDENTITY_PATTERN.fullmatch(self.subject_id) is None:
            raise ValueError("certificate subject identity is invalid")
        return self


class IssuerCertificate(V2RequestModel):
    issuer_id: str
    certificate_sha256: Sha256

    @model_validator(mode="after")
    def valid_issuer(self):
        if IDENTITY_PATTERN.fullmatch(self.issuer_id) is None:
            raise ValueError("certificate issuer identity is invalid")
        return self


class AgentIssuancePolicy(V2RequestModel):
    schema_version: Literal["1.0.0", "1.1.0", "1.2.0"] = "1.0.0"
    policy_id: Identifier
    valid_from: AwareDatetime
    expires_at: AwareDatetime
    issuers: list[str] = Field(min_length=1, max_length=64)
    subjects: list[str] = Field(min_length=1, max_length=256)
    grants: list[IssuanceGrant] = Field(min_length=1, max_length=256)
    max_active_sessions_per_subject: int = Field(ge=1, le=32)
    subject_certificates: list[SubjectCertificate] = Field(
        default_factory=list, max_length=256
    )
    issuer_certificates: list[IssuerCertificate] = Field(
        default_factory=list, max_length=64
    )

    @model_validator(mode="after")
    def valid_policy(self):
        if (
            self.valid_from.utcoffset() != ZERO
            or self.expires_at.utcoffset() != ZERO
            or self.expires_at <= self.valid_from
            or self.expires_at - self.valid_from > timedelta(days=366)
        ):
            raise ValueError("issuance policy requires a bounded UTC validity window")
        for values, name in ((self.issuers, "issuer"), (self.subjects, "subject")):
            if len(set(values)) != len(values) or any(
                IDENTITY_PATTERN.fullmatch(value) is None for value in values
            ):
                raise ValueError(f"issuance policy {name} identities are invalid")
        references = [(grant.task_id, grant.task_version) for grant in self.grants]
        if len(set(references)) != len(references):
            raise ValueError("issuance policy contains duplicate task grants")
        certificate_subjects = [item.subject_id for item in self.subject_certificates]
        subject_hashes = [item.certificate_sha256 for item in self.subject_certificates]
        certificate_issuers = [item.issuer_id for item in self.issuer_certificates]
        issuer_hashes = [item.certificate_sha256 for item in self.issuer_certificates]
        if self.schema_version == "1.0.0" and (
            self.subject_certificates or self.issuer_certificates
        ):
            raise ValueError("policy schema does not support certificate identities")
        if self.schema_version == "1.1.0" and (
            set(certificate_subjects) != set(self.subjects)
            or len(set(certificate_subjects)) != len(certificate_subjects)
            or len(set(subject_hashes)) != len(subject_hashes)
            or self.issuer_certificates
        ):
            raise ValueError("certificate policy must bind every subject exactly once")
        if self.schema_version == "1.2.0" and (
            set(certificate_subjects) != set(self.subjects)
            or len(set(certificate_subjects)) != len(certificate_subjects)
            or len(set(subject_hashes)) != len(subject_hashes)
            or set(certificate_issuers) != set(self.issuers)
            or len(set(certificate_issuers)) != len(certificate_issuers)
            or len(set(issuer_hashes)) != len(issuer_hashes)
            or not set(subject_hashes).isdisjoint(issuer_hashes)
        ):
            raise ValueError(
                "operator certificate policy must bind each role exactly once"
            )
        return self


class ControlEventInput(V2RequestModel):
    event_type: Literal[
        "issuance_started",
        "issuance_completed",
        "binding_reused",
        "revocation_started",
        "revocation_completed",
        "rotation_started",
        "rotation_completed",
    ]
    operation_id: str = Field(pattern=r"^op-[0-9a-f]{32}$")
    actor_id: str
    subject_id: str
    policy_id: Identifier
    policy_sha256: Sha256
    task_id: Identifier
    task_version: SemanticVersion
    task_manifest_hash: Sha256
    episode_id: EpisodeId | None = None
    generation: int | None = Field(default=None, ge=1, le=1_000_000)
    subject_certificate_sha256: Sha256 | None = None
    actor_certificate_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def identities(self):
        if (
            IDENTITY_PATTERN.fullmatch(self.actor_id) is None
            or IDENTITY_PATTERN.fullmatch(self.subject_id) is None
        ):
            raise ValueError("control-plane identities are invalid")
        completed = self.event_type.endswith("_completed") or self.event_type == "binding_reused"
        if completed != (self.episode_id is not None and self.generation is not None):
            raise ValueError("completed control events require episode and generation")
        return self


class ControlPlaneEvent(ControlEventInput):
    schema_version: Literal["1.0.0", "1.1.0"] = "1.0.0"
    sequence: int = Field(ge=1)
    occurred_at: AwareDatetime
    previous_event_sha256: Sha256 | None
    event_sha256: Sha256

    @model_validator(mode="after")
    def utc_timestamp(self):
        if self.occurred_at.utcoffset() != ZERO:
            raise ValueError("control-plane timestamp must use UTC")
        return self


def _read_bounded_regular(path: Path, maximum: int, *, private: bool) -> bytes:
    path = Path(path)
    try:
        if path.parent.resolve(strict=True) != path.parent.absolute():
            raise ValueError("parent path must not contain symlinks")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except (OSError, RuntimeError) as error:
        raise ValueError("bounded control file is unavailable") from error
    try:
        metadata = os.fstat(descriptor)
        forbidden_mode = 0o077 if private else 0o022
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & forbidden_mode:
            raise ValueError("control file permissions or type are invalid")
        if not 0 < metadata.st_size <= maximum:
            raise ValueError("control file size is invalid")
        content = bytearray()
        while chunk := os.read(descriptor, min(65536, maximum + 1 - len(content))):
            content.extend(chunk)
            if len(content) > maximum:
                raise ValueError("control file exceeds size limit")
        return bytes(content)
    finally:
        os.close(descriptor)


def load_issuance_policy(path: Path, expected_sha256: str) -> tuple[AgentIssuancePolicy, str]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("issuance policy checksum is invalid")
    content = _read_bounded_regular(Path(path), MAX_POLICY_BYTES, private=False)
    digest = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(digest, expected_sha256):
        raise ValueError("issuance policy checksum changed")
    try:
        return AgentIssuancePolicy.model_validate_json(content), digest
    except Exception as error:
        raise ValueError("issuance policy is invalid") from error


def certificate_file_sha256(path: Path) -> str:
    """Hash exactly one bounded PEM certificate as DER without following links."""
    content = _read_bounded_regular(Path(path), MAX_POLICY_BYTES, private=False)
    try:
        text = content.decode("ascii")
        begin = "-----BEGIN CERTIFICATE-----"
        end = "-----END CERTIFICATE-----"
        if text.count(begin) != 1 or text.count(end) != 1:
            raise ValueError
        start = text.index(begin)
        stop = text.index(end, start) + len(end)
        if text[:start].strip() or text[stop:].strip():
            raise ValueError
        der = ssl.PEM_cert_to_DER_cert(text[start:stop])
        if not der:
            raise ValueError
    except (UnicodeError, ValueError):
        raise ValueError("operator certificate file is invalid") from None
    return hashlib.sha256(der).hexdigest()


def _authorize_certificate_identities(
    policy: AgentIssuancePolicy,
    *,
    actor_id: str,
    subject_id: str,
    actor_certificate_sha256: str | None,
    subject_certificate_sha256: str | None,
) -> None:
    subject_pins = {
        item.subject_id: item.certificate_sha256
        for item in policy.subject_certificates
    }
    issuer_pins = {
        item.issuer_id: item.certificate_sha256
        for item in policy.issuer_certificates
    }
    if policy.schema_version in {"1.1.0", "1.2.0"}:
        expected = subject_pins.get(subject_id)
        if (
            subject_certificate_sha256 is None
            or expected is None
            or not hmac.compare_digest(expected, subject_certificate_sha256)
        ):
            raise ValueError("subject certificate identity is not allowed")
    elif subject_certificate_sha256 is not None:
        raise ValueError("issuance policy does not grant certificate identity")
    if policy.schema_version == "1.2.0":
        expected = issuer_pins.get(actor_id)
        if (
            actor_certificate_sha256 is None
            or expected is None
            or not hmac.compare_digest(expected, actor_certificate_sha256)
        ):
            raise ValueError("operator certificate identity is not allowed")
    elif actor_certificate_sha256 is not None:
        raise ValueError("issuance policy does not grant operator certificate identity")


def authorize_issuance(
    policy: AgentIssuancePolicy,
    *,
    actor_id: str,
    subject_id: str,
    task_id: str,
    task_version: str,
    ttl_seconds: int,
    active_subject_sessions: int,
    actor_certificate_sha256: str | None = None,
    subject_certificate_sha256: str | None = None,
    now: datetime | None = None,
) -> IssuanceGrant:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() != ZERO:
        raise ValueError("policy evaluation requires UTC")
    if not policy.valid_from <= current < policy.expires_at:
        raise ValueError("issuance policy is not currently valid")
    if actor_id not in policy.issuers or subject_id not in policy.subjects:
        raise ValueError("issuance identity is not allowed")
    _authorize_certificate_identities(
        policy,
        actor_id=actor_id,
        subject_id=subject_id,
        actor_certificate_sha256=actor_certificate_sha256,
        subject_certificate_sha256=subject_certificate_sha256,
    )
    grants = [
        grant
        for grant in policy.grants
        if grant.task_id == task_id and grant.task_version == task_version
    ]
    if len(grants) != 1:
        raise ValueError("task is not granted by issuance policy")
    if type(ttl_seconds) is not int or not 60 <= ttl_seconds <= grants[0].max_ttl_seconds:
        raise ValueError("credential TTL exceeds issuance grant")
    if (
        type(active_subject_sessions) is not int
        or active_subject_sessions < 0
        or active_subject_sessions >= policy.max_active_sessions_per_subject
    ):
        raise ValueError("subject active-session limit reached")
    return grants[0]


def authorize_management(
    policy: AgentIssuancePolicy,
    *,
    actor_id: str,
    subject_id: str,
    task_id: str,
    task_version: str,
    ttl_seconds: int | None,
    rotate: bool,
    actor_certificate_sha256: str | None = None,
    subject_certificate_sha256: str | None = None,
    now: datetime | None = None,
) -> None:
    if actor_id not in policy.issuers or subject_id not in policy.subjects:
        raise ValueError("management identity is not allowed")
    _authorize_certificate_identities(
        policy,
        actor_id=actor_id,
        subject_id=subject_id,
        actor_certificate_sha256=actor_certificate_sha256,
        subject_certificate_sha256=subject_certificate_sha256,
    )
    grants = [
        grant
        for grant in policy.grants
        if grant.task_id == task_id and grant.task_version == task_version
    ]
    if len(grants) != 1:
        raise ValueError("task is not granted by issuance policy")
    if rotate:
        authorize_issuance(
            policy,
            actor_id=actor_id,
            subject_id=subject_id,
            task_id=task_id,
            task_version=task_version,
            ttl_seconds=ttl_seconds,
            active_subject_sessions=0,
            actor_certificate_sha256=actor_certificate_sha256,
            subject_certificate_sha256=subject_certificate_sha256,
            now=now,
        )


def _canonical_event(event: ControlPlaneEvent) -> bytes:
    excluded = {"event_sha256"}
    if event.schema_version == "1.0.0":
        excluded.add("actor_certificate_sha256")
    return json.dumps(
        event.model_dump(mode="json", exclude=excluded),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _parse_audit(content: bytes) -> list[ControlPlaneEvent]:
    if not content:
        return []
    if not content.endswith(b"\n"):
        raise ValueError("control-plane audit has a partial final event")
    events = []
    previous = None
    for sequence, line in enumerate(content.splitlines(), start=1):
        if not line or len(line) > MAX_AUDIT_EVENT_BYTES:
            raise ValueError("control-plane audit event size is invalid")
        try:
            event = ControlPlaneEvent.model_validate_json(line)
        except Exception as error:
            raise ValueError("control-plane audit event is invalid") from error
        digest = hashlib.sha256(_canonical_event(event)).hexdigest()
        if (
            event.sequence != sequence
            or event.previous_event_sha256 != previous
            or not hmac.compare_digest(event.event_sha256, digest)
        ):
            raise ValueError("control-plane audit chain is invalid")
        events.append(event)
        previous = event.event_sha256
    return events


def _audit_descriptor(path: Path, *, create: bool, writable: bool = True) -> int:
    path = Path(path)
    if path.parent.resolve(strict=True) != path.parent.absolute():
        raise ValueError("audit parent must not contain symlinks")
    flags = (
        (os.O_RDWR | os.O_APPEND) if writable else os.O_RDONLY
    ) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if create:
        if not writable:
            raise ValueError("read-only audit open cannot create data")
        flags |= os.O_CREAT | os.O_EXCL
    try:
        return os.open(path, flags, 0o600)
    except OSError as error:
        raise ValueError("control-plane audit is unavailable") from error


def _checked_audit_content(descriptor: int) -> bytes:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
        raise ValueError("control-plane audit must be owner-private regular data")
    if metadata.st_size > MAX_AUDIT_BYTES:
        raise ValueError("control-plane audit exceeds size limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    content = bytearray()
    while chunk := os.read(descriptor, min(65536, MAX_AUDIT_BYTES + 1 - len(content))):
        content.extend(chunk)
        if len(content) > MAX_AUDIT_BYTES:
            raise ValueError("control-plane audit exceeds size limit")
    return bytes(content)


def append_control_event(
    path: Path,
    value: ControlEventInput,
    *,
    now: datetime | None = None,
) -> ControlPlaneEvent:
    path = Path(path)
    try:
        descriptor = _audit_descriptor(path, create=False)
    except ValueError:
        if path.exists() or path.is_symlink():
            raise
        descriptor = _audit_descriptor(path, create=True)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        events = _parse_audit(_checked_audit_content(descriptor))
        occurred_at = now or datetime.now(timezone.utc)
        unsigned = ControlPlaneEvent(
            **value.model_dump(),
            schema_version=(
                "1.1.0" if value.actor_certificate_sha256 is not None else "1.0.0"
            ),
            sequence=len(events) + 1,
            occurred_at=occurred_at,
            previous_event_sha256=events[-1].event_sha256 if events else None,
            event_sha256="0" * 64,
        )
        event = unsigned.model_copy(
            update={"event_sha256": hashlib.sha256(_canonical_event(unsigned)).hexdigest()}
        )
        line = event.model_dump_json(
            exclude=(
                {"actor_certificate_sha256"}
                if event.schema_version == "1.0.0"
                else None
            )
        ).encode("utf-8") + b"\n"
        if len(line) > MAX_AUDIT_EVENT_BYTES:
            raise ValueError("control-plane audit event exceeds size limit")
        if os.fstat(descriptor).st_size + len(line) > MAX_AUDIT_BYTES:
            raise ValueError("control-plane audit exceeds size limit")
        os.lseek(descriptor, 0, os.SEEK_END)
        view = memoryview(line)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("audit append made no progress")
            view = view[written:]
        os.fsync(descriptor)
        return event
    finally:
        os.close(descriptor)


def verify_control_audit(path: Path) -> list[ControlPlaneEvent]:
    descriptor = _audit_descriptor(Path(path), create=False, writable=False)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        return _parse_audit(_checked_audit_content(descriptor))
    finally:
        os.close(descriptor)


__all__ = [
    "AgentIssuancePolicy",
    "ControlEventInput",
    "ControlPlaneEvent",
    "IssuerCertificate",
    "IssuanceGrant",
    "SubjectCertificate",
    "MAX_AUDIT_BYTES",
    "append_control_event",
    "authorize_issuance",
    "authorize_management",
    "certificate_file_sha256",
    "load_issuance_policy",
    "verify_control_audit",
]

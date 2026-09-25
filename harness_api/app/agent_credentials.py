"""Bounded bearer credential registry for the scoped Agent gateway."""
from __future__ import annotations

import hmac
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import AwareDatetime, Field, model_validator

from .core.artifact_identity import LEGACY_SCHEME
from .core.schemas import (BudgetSpec, EpisodeId, Identifier, SemanticVersion,
                         Sha256, V2RequestModel)

MAX_REGISTRY_BYTES = 2 * 1024 * 1024
MAX_SESSIONS = 256
MAX_TTL_SECONDS = 7 * 24 * 60 * 60


class PublicTask(V2RequestModel):
    task_id: Identifier
    task_version: SemanticVersion
    prompt: str = Field(min_length=1, max_length=50000)
    answer_schema: dict[str, Any]
    budget: BudgetSpec
    input_asset_refs: list[Identifier] = Field(min_length=1, max_length=1000)
    allowed_actions: list[str]
    allowed_tools: list[str]
    artifact_identity: Literal["content-sha256-v1", "derivation-sha256-v1"] = LEGACY_SCHEME


class AgentBinding(V2RequestModel):
    episode_id: EpisodeId
    task_manifest_hash: Sha256
    token_sha256: Sha256
    task: PublicTask


class AgentSessionCredential(V2RequestModel):
    binding: AgentBinding
    issued_at: AwareDatetime
    expires_at: AwareDatetime
    status: Literal["active", "revoked"] = "active"
    revoked_at: AwareDatetime | None = None
    generation: int = Field(default=1, ge=1, le=1_000_000)
    issuer_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9._-]{0,63}$")
    subject_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9._-]{0,63}$")
    issuance_policy_id: Identifier | None = None
    issuance_policy_sha256: Sha256 | None = None
    subject_certificate_sha256: Sha256 | None = None
    issuer_certificate_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def valid_lifecycle(self):
        zero = timedelta(0)
        if self.issued_at.utcoffset() != zero or self.expires_at.utcoffset() != zero:
            raise ValueError("credential timestamps must use UTC")
        lifetime = self.expires_at - self.issued_at
        if lifetime <= zero or lifetime > timedelta(seconds=MAX_TTL_SECONDS):
            raise ValueError("credential lifetime is outside the supported bound")
        if self.status == "active" and self.revoked_at is not None:
            raise ValueError("active credential cannot have revoked_at")
        if self.status == "revoked" and self.revoked_at is None:
            raise ValueError("revoked credential requires revoked_at")
        if self.revoked_at is not None:
            if self.revoked_at.utcoffset() != zero or self.revoked_at < self.issued_at:
                raise ValueError("invalid revocation timestamp")
        governance = (
            self.issuer_id,
            self.subject_id,
            self.issuance_policy_id,
            self.issuance_policy_sha256,
        )
        if any(value is not None for value in governance) and not all(
            value is not None for value in governance
        ):
            raise ValueError("credential governance metadata must be complete")
        return self


class AgentCredentialRegistry(V2RequestModel):
    schema_version: Literal["1.0.0", "1.1.0", "1.2.0", "1.3.0"] = "1.0.0"
    sessions: list[AgentSessionCredential] = Field(default_factory=list, max_length=MAX_SESSIONS)

    @model_validator(mode="after")
    def unique_scopes(self):
        token_hashes = [item.binding.token_sha256 for item in self.sessions]
        episode_ids = [item.binding.episode_id for item in self.sessions]
        if len(set(token_hashes)) != len(token_hashes):
            raise ValueError("duplicate credential hash")
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("duplicate episode scope")
        governed = [item.issuer_id is not None for item in self.sessions]
        certificate_bound = [
            item.subject_certificate_sha256 is not None for item in self.sessions
        ]
        issuer_certificate_bound = [
            item.issuer_certificate_sha256 is not None for item in self.sessions
        ]
        if self.schema_version == "1.0.0" and any(governed):
            raise ValueError("legacy registry cannot contain governed sessions")
        if self.schema_version in {"1.1.0", "1.2.0", "1.3.0"} and not all(governed):
            raise ValueError("governed registry requires policy metadata for every session")
        if self.schema_version in {"1.0.0", "1.1.0"} and any(certificate_bound):
            raise ValueError("registry schema does not support certificate-bound sessions")
        if self.schema_version in {"1.2.0", "1.3.0"} and not all(certificate_bound):
            raise ValueError("certificate-bound registry requires a pin for every session")
        if self.schema_version in {"1.0.0", "1.1.0", "1.2.0"} and any(
            issuer_certificate_bound
        ):
            raise ValueError("registry schema does not support operator certificate pins")
        if self.schema_version == "1.3.0" and not all(issuer_certificate_bound):
            raise ValueError("operator-bound registry requires a pin for every session")
        return self


class CredentialError(Exception):
    def __init__(self, code: str, status: int):
        self.code = code
        self.status = status


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_agent_registry(path: Path) -> AgentCredentialRegistry:
    """Read one owner-private regular file without following any symlink."""
    path = Path(path)
    try:
        if path.parent.resolve(strict=True) != path.parent.absolute():
            raise ValueError("registry parent must not contain symlinks")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except (OSError, RuntimeError) as error:
        raise ValueError("credential registry is unavailable") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ValueError("credential registry must be an owner-private regular file")
        if metadata.st_size > MAX_REGISTRY_BYTES:
            raise ValueError("credential registry exceeds size limit")
        content = bytearray()
        while True:
            chunk = os.read(descriptor, min(65536, MAX_REGISTRY_BYTES + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > MAX_REGISTRY_BYTES:
                raise ValueError("credential registry exceeds size limit")
    finally:
        os.close(descriptor)
    try:
        return AgentCredentialRegistry.model_validate_json(content)
    except Exception as error:
        raise ValueError("credential registry is invalid") from error


class CredentialResolver:
    """Resolve a token hash to one binding; file-backed registries reload per request."""

    def __init__(self, *, binding: AgentBinding | None = None,
                 registry: AgentCredentialRegistry | None = None,
                 registry_path: Path | None = None,
                 validate_binding: Callable[[AgentBinding], None] | None = None,
                 now: Callable[[], datetime] = utc_now):
        if sum(value is not None for value in (binding, registry, registry_path)) != 1:
            raise ValueError("exactly one credential source is required")
        self.binding = binding
        self.registry = registry
        self.registry_path = Path(registry_path) if registry_path is not None else None
        self.validate_binding = validate_binding or (lambda value: None)
        self.now = now
        if binding is not None:
            self.validate_binding(binding)
        else:
            self._validated_registry()

    def _validated_registry(self) -> AgentCredentialRegistry:
        registry = load_agent_registry(self.registry_path) if self.registry_path is not None else self.registry
        if registry is None:
            raise ValueError("credential registry is unavailable")
        for item in registry.sessions:
            self.validate_binding(item.binding)
        return registry

    def ready(self) -> None:
        if self.binding is None:
            self._validated_registry()

    def _resolve(self, token_sha256: str) -> tuple[AgentBinding, str | None]:
        if self.binding is not None:
            if hmac.compare_digest(token_sha256, self.binding.token_sha256):
                return self.binding, None
            raise CredentialError("unauthorized", 401)
        try:
            registry = self._validated_registry()
        except ValueError:
            raise CredentialError("credential_registry_unavailable", 503) from None
        selected = None
        for item in registry.sessions:
            if hmac.compare_digest(token_sha256, item.binding.token_sha256):
                selected = item
        if selected is None:
            raise CredentialError("unauthorized", 401)
        if selected.status == "revoked":
            raise CredentialError("session_revoked", 401)
        current = self.now()
        if current.tzinfo is None or current.utcoffset() != timedelta(0):
            raise CredentialError("credential_registry_unavailable", 503)
        if current < selected.issued_at:
            raise CredentialError("session_not_yet_valid", 401)
        if current >= selected.expires_at:
            raise CredentialError("session_expired", 401)
        return selected.binding, selected.subject_certificate_sha256

    def resolve(self, token_sha256: str) -> AgentBinding:
        return self._resolve(token_sha256)[0]

    def resolve_with_certificate(self, token_sha256: str) -> tuple[AgentBinding, str | None]:
        return self._resolve(token_sha256)

"""Pinned, budget-accounted search over governed evidence memory."""
from __future__ import annotations

from pydantic import ValidationError

from ..domain import V2DomainError
from ..evidence_memory import (
    EvidenceMemoryPolicy,
    EvidenceMemoryStore,
    MemorySearchArguments,
)
from ..schemas import Artifact, TaskManifest, ToolInvokeAction
from .runtime import PreparedTool, ToolOutput


TOOL_ID = "memory.search"
TOOL_VERSION = "1.0.0"


class MemorySearchExecutor:
    tool_id = TOOL_ID
    tool_version = TOOL_VERSION

    def __init__(
        self,
        store: EvidenceMemoryStore,
        policy: EvidenceMemoryPolicy,
        policy_sha256: str,
    ) -> None:
        self.store = store
        self.policy = policy
        self.policy_sha256 = policy_sha256

    def plan(
        self,
        action: ToolInvokeAction,
        manifest: TaskManifest,
        accessible_asset_refs: list[str],
        episode_artifacts: dict[str, Artifact] | None = None,
    ) -> PreparedTool:
        del accessible_asset_refs, episode_artifacts
        if (
            action.tool_id != self.tool_id
            or action.tool_id not in manifest.scenario.allowed_tools
        ):
            raise V2DomainError(
                "policy_rejected",
                "tool is not allowed by this task",
                403,
                phase="policy",
            )
        try:
            query = MemorySearchArguments.model_validate(action.arguments)
        except (ValidationError, ValueError):
            raise V2DomainError(
                "invalid_tool_arguments",
                "invalid evidence memory query; use the documented fields and bounds",
                phase="request",
            ) from None
        try:
            result = self.store.search(
                self.policy,
                self.policy_sha256,
                manifest,
                query,
            )
        except ValueError:
            # Policy pins, store integrity and private provenance stay behind the
            # same public denial. The operator can inspect the server-side cause.
            raise V2DomainError(
                "policy_rejected",
                "evidence memory is unavailable for this immutable task",
                403,
                phase="policy",
            ) from None
        output = ToolOutput(
            artifact=None,
            metadata=result.model_dump(mode="json"),
            input_bytes=result.cost.input_bytes,
        )
        return PreparedTool(
            tool_version=self.tool_version,
            input_bytes=result.cost.input_bytes,
            max_output_bytes=0,
            invoke=lambda: output,
            metadata_only=True,
        )


__all__ = ["MemorySearchExecutor", "TOOL_ID", "TOOL_VERSION"]

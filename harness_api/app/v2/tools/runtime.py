"""Small dispatch adapter; legacy single-image executors remain supported."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..domain import V2DomainError
from ..schemas import Artifact, TaskManifest, ToolInvokeAction


@dataclass(frozen=True)
class ToolOutput:
    artifact: Artifact | None
    metadata: dict[str, Any]
    input_bytes: int


@dataclass(frozen=True)
class PreparedTool:
    tool_version: str
    input_bytes: int
    max_output_bytes: int
    invoke: Callable[[], ToolOutput]
    metadata_only: bool = False


def prepare_tool(executor, action: ToolInvokeAction, manifest: TaskManifest,
                 accessible_asset_refs: list[str],
                 episode_artifacts: Mapping[str, Artifact] | None = None) -> PreparedTool:
    if isinstance(executor, ToolRouter):
        return executor.plan(action, manifest, accessible_asset_refs, episode_artifacts)
    _, asset = executor.prepare(action, manifest)
    if asset.asset_id not in accessible_asset_refs:
        raise V2DomainError("policy_rejected", "asset is not accessible in this episode", 403, phase="policy")
    return PreparedTool(executor.tool_version, asset.size_bytes, executor.max_output_bytes,
                        lambda: executor.invoke(action, manifest))


class ToolRouter:
    """Opt-in local catalog, optionally alongside one legacy provider."""

    def __init__(self, provider=None, raster=None, grid=None):
        from .catalog import CatalogExecutor
        self.catalog = CatalogExecutor()
        self.provider = provider
        self.raster = raster
        self.grid = grid
        self.tool_ids = [*self.catalog.tool_ids, *([provider.tool_id] if provider else []),
                         *([raster.tool_id] if raster else []), *([grid.tool_id] if grid else [])]

    def plan(self, action: ToolInvokeAction, manifest: TaskManifest,
             accessible_asset_refs: list[str],
             episode_artifacts: Mapping[str, Artifact] | None = None) -> PreparedTool:
        if action.tool_id in self.catalog.tool_ids:
            return self.catalog.plan(action, manifest, accessible_asset_refs)
        if self.raster is not None and action.tool_id == self.raster.tool_id:
            return self.raster.plan(action, manifest, accessible_asset_refs,
                                    episode_artifacts or {})
        if self.grid is not None and action.tool_id == self.grid.tool_id:
            return self.grid.plan(action, manifest, accessible_asset_refs)
        if self.provider is not None and action.tool_id == self.provider.tool_id:
            return prepare_tool(self.provider, action, manifest, accessible_asset_refs)
        raise V2DomainError("policy_rejected", "tool is not available", 403, phase="policy")

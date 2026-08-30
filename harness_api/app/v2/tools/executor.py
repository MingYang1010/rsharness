from typing import Dict

from pydantic import JsonValue

from .registry import ToolRegistry


class ToolExecutor:
    """M3 execution boundary; M1 deliberately registers no executable tools."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def invoke(self, tool_id: str, arguments: Dict[str, JsonValue]) -> None:
        self.registry.get(tool_id)
        raise NotImplementedError("tool execution begins in milestone M3")

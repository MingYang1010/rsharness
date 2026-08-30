from typing import Dict, List

from .specs import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, specification: ToolSpec) -> None:
        if specification.tool_id in self._tools:
            raise ValueError("duplicate tool_id: %s" % specification.tool_id)
        self._tools[specification.tool_id] = specification

    def get(self, tool_id: str) -> ToolSpec:
        try:
            return self._tools[tool_id]
        except KeyError:
            raise KeyError("tool is not allowlisted: %s" % tool_id)

    def list(self) -> List[ToolSpec]:
        return [self._tools[key] for key in sorted(self._tools)]

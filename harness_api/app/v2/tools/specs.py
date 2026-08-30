from typing import Dict, List, Literal

from pydantic import JsonValue

from ..schemas import Identifier, SemanticVersion, V2ContractModel


class ToolSpec(V2ContractModel):
    tool_id: Identifier
    tool_version: SemanticVersion
    input_schema: Dict[str, JsonValue]
    output_schema: Dict[str, JsonValue]
    determinism: Literal["deterministic", "seeded", "nondeterministic"]
    timeout_ms: int
    resource_class: Literal["cpu-small", "cpu-large", "gpu"]
    network_policy: Literal["none", "catalog_only", "allowlisted"]
    allowed_media_types: List[str]
    cost_estimator: Identifier

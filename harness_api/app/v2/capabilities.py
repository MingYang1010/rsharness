import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from . import IMPLEMENTATION_VERSION
from .events import sha256_json
from .schemas import (
    AssetRef,
    CapabilitiesData,
    CapabilityStatus,
    EvaluatorSpec,
    ScenarioProfile,
    TaskManifest,
    TaskSpec,
)


IMPLEMENTED_ACTIONS = [
    "answer.abstain",
    "answer.request_human_review",
    "answer.submit",
    "map.layer.set_opacity",
    "map.layer.set_visibility",
    "map.pan",
    "map.set_view",
    "map.time.set_range",
    "map.zoom",
    "memory.save_evidence",
]

DECLARED_ACTIONS = IMPLEMENTED_ACTIONS + [
    "memory.bookmark_aoi",
    "tool.invoke",
]

OBSERVATION_TYPES = [
    "map_state",
    "asset_metadata",
    "rendered_view",
    "raster_chip",
    "temporal_stack",
    "vector_features",
    "table",
    "tool_result",
]


class TaskRegistry:
    def __init__(self, root: str):
        self.root = Path(root)
        self._manifests: Dict[Tuple[str, str], TaskManifest] = {}
        self._load()

    @staticmethod
    def _read_json(path: Path) -> object:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)

    def _load(self) -> None:
        if not self.root.is_dir():
            raise RuntimeError("V2 task root does not exist: %s" % self.root)
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir():
                continue
            task = TaskSpec.model_validate(self._read_json(directory / "task.json"))
            scenario = ScenarioProfile.model_validate(
                self._read_json(directory / "scenario.json")
            )
            evaluator = EvaluatorSpec.model_validate(
                self._read_json(directory / "evaluator.json")
            )
            assets_value = self._read_json(directory / "assets.json")
            if not isinstance(assets_value, list):
                raise RuntimeError("assets.json must contain an array")
            assets = [AssetRef.model_validate(item) for item in assets_value]
            self._validate_links(task, scenario, evaluator, assets)
            manifest_body = {
                "task": task.model_dump(mode="json"),
                "scenario": scenario.model_dump(mode="json"),
                "assets": [asset.model_dump(mode="json") for asset in assets],
                "evaluator": evaluator.model_dump(mode="json"),
            }
            manifest = TaskManifest(
                **manifest_body,
                task_manifest_hash=sha256_json(manifest_body),
            )
            key = (task.task_id, task.task_version)
            if key in self._manifests:
                raise RuntimeError("duplicate immutable V2 task: %s@%s" % key)
            self._manifests[key] = manifest
        if not self._manifests:
            raise RuntimeError("V2 task registry is empty")

    @staticmethod
    def _validate_links(
        task: TaskSpec,
        scenario: ScenarioProfile,
        evaluator: EvaluatorSpec,
        assets: Iterable[AssetRef],
    ) -> None:
        if task.scenario_profile != scenario.profile_id:
            raise RuntimeError("TaskSpec scenario_profile does not match scenario.json")
        if task.evaluator != evaluator.evaluator_id:
            raise RuntimeError("TaskSpec evaluator does not match evaluator.json")
        asset_ids = {asset.asset_id for asset in assets}
        missing = sorted(set(task.inputs) - asset_ids)
        if missing:
            raise RuntimeError("TaskSpec inputs are missing assets: %s" % missing)

    def get(self, task_id: str, task_version: str) -> TaskManifest:
        try:
            return self._manifests[(task_id, task_version)]
        except KeyError:
            raise KeyError("unknown immutable task: %s@%s" % (task_id, task_version))

    def count(self) -> int:
        return len(self._manifests)


def build_capabilities(
    registry: TaskRegistry,
    store_schema_version: int,
    renderer_status: Tuple[str, Optional[str], Dict[str, Any]],
    evaluator_status: Tuple[str, Optional[str], Dict[str, Any]],
) -> CapabilitiesData:
    renderer_state, renderer_version, renderer_details = renderer_status
    evaluator_state, evaluator_version, evaluator_details = evaluator_status
    return CapabilitiesData(
        implementation_version=IMPLEMENTATION_VERSION,
        store_schema_version=store_schema_version,
        enabled=True,
        task_count=registry.count(),
        actions=IMPLEMENTED_ACTIONS,
        declared_actions=DECLARED_ACTIONS,
        observation_types=OBSERVATION_TYPES,
        tools=[],
        renderer=CapabilityStatus(
            status=renderer_state,
            version=renderer_version,
            details=renderer_details,
        ),
        evaluator=CapabilityStatus(
            status=evaluator_state,
            version=evaluator_version,
            details=evaluator_details,
        ),
        structural_replay=CapabilityStatus(
            status="available",
            version="1.0.0",
        ),
    )

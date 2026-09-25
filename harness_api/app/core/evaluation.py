import hashlib
import math
import uuid
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Optional, Tuple

from pydantic import ValidationError

from .artifacts import ArtifactStore
from .schemas import (
    ArtifactRef,
    AssetRef,
    Metric,
    MetricResult,
    PixelArtifactRef,
    PixelAssetRef,
    SpatialBoundingBox,
    TaskManifest,
    TemporalStackArtifactRef,
    V2EpisodeState,
)
from .temporal import (
    TOOL_ID as TEMPORAL_TOOL_ID,
    VERSION as TEMPORAL_TOOL_VERSION,
    TemporalToolResult,
)
from .evidence_memory import MemorySearchResult
from .tools.memory import TOOL_ID as MEMORY_TOOL_ID, TOOL_VERSION as MEMORY_TOOL_VERSION


class EvaluatorError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def aggregate_metrics(metrics: Iterable[Metric]) -> Optional[float]:
    weighted_sum = 0.0
    total_weight = 0.0
    for metric in metrics:
        if metric.weight is None:
            continue
        weighted_sum += metric.value * metric.weight
        total_weight += metric.weight
    if total_weight == 0.0:
        return None
    return weighted_sum / total_weight


def _contains(outer: SpatialBoundingBox, inner: SpatialBoundingBox) -> bool:
    return (
        outer.west <= inner.west
        and outer.south <= inner.south
        and outer.east >= inner.east
        and outer.north >= inner.north
    )


def _same_bbox(left: SpatialBoundingBox, right: SpatialBoundingBox) -> bool:
    return all(
        abs(first - second) <= 1e-9
        for first, second in (
            (left.west, right.west),
            (left.south, right.south),
            (left.east, right.east),
            (left.north, right.north),
        )
    )


class EvaluatorRegistry:
    def __init__(self, dataset_root: str, artifact_store: ArtifactStore):
        self.dataset_root = Path(dataset_root).resolve()
        self.artifact_store = artifact_store

    def capability(self) -> Tuple[str, Dict[str, object]]:
        if not self.dataset_root.is_dir():
            return "unavailable", {"reason": "dataset_root_missing"}
        return "available", {
            "evaluator_id": "worldcover-grounded-v1",
            "version": "1.1.0",
        }

    def evaluate_safely(
        self,
        manifest: TaskManifest,
        state: V2EpisodeState,
        artifacts: Dict[str, ArtifactRef],
        renderer_calls: int,
        failed_actions: int,
        wall_time_ms: int,
        tool_results: Optional[list[dict]] = None,
    ) -> MetricResult:
        evaluation_id = "eval-%s" % uuid.uuid4().hex
        try:
            evaluator_id = manifest.evaluator.evaluator_id
            arguments = {
                "evaluation_id": evaluation_id,
                "manifest": manifest,
                "state": state,
                "artifacts": artifacts,
                "renderer_calls": renderer_calls,
                "failed_actions": failed_actions,
                "wall_time_ms": wall_time_ms,
            }
            if evaluator_id == "worldcover-grounded-v1":
                return self._evaluate_worldcover(**arguments)
            if evaluator_id == "whu-building-change-v1":
                return self._evaluate_whu_change(**arguments)
            if evaluator_id == "temporal-selection-v1":
                return self._evaluate_temporal_selection(
                    **arguments,
                    tool_results=tool_results or [],
                )
            if evaluator_id == "evidence-memory-v1":
                return self._evaluate_evidence_memory(
                    **arguments,
                    tool_results=tool_results or [],
                )
            raise EvaluatorError(
                "evaluator_not_supported",
                "registered evaluator is not implemented",
            )
        except EvaluatorError as error:
            return MetricResult(
                evaluation_id=evaluation_id,
                status="failed",
                metrics=[],
                aggregate_reward=None,
                evaluator_id=manifest.evaluator.evaluator_id,
                evaluator_version=manifest.evaluator.evaluator_version,
                diagnostics={"code": error.code, "message": error.message},
            )
        except Exception as error:
            return MetricResult(
                evaluation_id=evaluation_id,
                status="failed",
                metrics=[],
                aggregate_reward=None,
                evaluator_id=manifest.evaluator.evaluator_id,
                evaluator_version=manifest.evaluator.evaluator_version,
                diagnostics={
                    "code": "evaluator_internal_error",
                    "error_type": type(error).__name__,
                },
            )

    def _resolve_asset_path(self, asset: AssetRef) -> Path:
        prefix = "local://dataset/"
        if not asset.uri.startswith(prefix):
            raise EvaluatorError(
                "unsupported_gold_uri",
                "canonical evaluator asset must use local://dataset/",
            )
        relative = PurePosixPath(asset.uri[len(prefix) :])
        if relative.is_absolute() or ".." in relative.parts:
            raise EvaluatorError(
                "invalid_gold_uri",
                "canonical evaluator asset URI is outside the dataset root",
            )
        path = (self.dataset_root / Path(*relative.parts)).resolve()
        try:
            path.relative_to(self.dataset_root)
        except ValueError:
            raise EvaluatorError(
                "invalid_gold_uri",
                "canonical evaluator asset URI is outside the dataset root",
            )
        if not path.is_file():
            raise EvaluatorError(
                "gold_asset_missing",
                "canonical evaluator asset is unavailable",
            )
        if path.stat().st_size != asset.size_bytes:
            raise EvaluatorError(
                "gold_asset_size_mismatch",
                "canonical evaluator asset size does not match its manifest",
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != asset.sha256:
            raise EvaluatorError(
                "gold_asset_checksum_mismatch",
                "canonical evaluator asset checksum does not match its manifest",
            )
        return path

    @staticmethod
    def _canonical_asset(manifest: TaskManifest) -> AssetRef:
        asset_id = manifest.evaluator.config.get("canonical_asset_id")
        for asset in manifest.assets:
            if asset.asset_id == asset_id and "evaluator" in asset.roles:
                return asset
        raise EvaluatorError(
            "canonical_asset_not_registered",
            "task evaluator canonical asset is not registered as evaluator-only",
        )

    def _class_distribution(
        self,
        manifest: TaskManifest,
        aoi: SpatialBoundingBox,
    ) -> Tuple[AssetRef, Dict[int, int]]:
        asset = self._canonical_asset(manifest)
        path = self._resolve_asset_path(asset)
        try:
            import numpy
            import rasterio
        except ImportError:
            raise EvaluatorError(
                "evaluator_dependency_missing",
                "rasterio and numpy are required by the WorldCover evaluator",
            )
        with rasterio.open(path) as dataset:
            if dataset.crs is None or dataset.crs.to_string() != "EPSG:4326":
                raise EvaluatorError(
                    "gold_asset_crs_mismatch",
                    "canonical WorldCover asset must use EPSG:4326",
                )
            bounds = SpatialBoundingBox(
                west=dataset.bounds.left,
                south=dataset.bounds.bottom,
                east=dataset.bounds.right,
                north=dataset.bounds.top,
            )
            if not _contains(bounds, aoi):
                raise EvaluatorError(
                    "evaluation_aoi_outside_gold",
                    "evaluation AOI is outside the canonical WorldCover asset",
                )
            window = dataset.window(aoi.west, aoi.south, aoi.east, aoi.north)
            window = window.round_offsets().round_lengths()
            values = dataset.read(1, window=window, masked=True)
        compressed = values.compressed()
        classes, counts = numpy.unique(compressed, return_counts=True)
        distribution = {
            int(class_value): int(count)
            for class_value, count in zip(classes.tolist(), counts.tolist())
        }
        expected_count = int(manifest.evaluator.config.get("expected_pixel_count", 0))
        if expected_count and int(compressed.size) != expected_count:
            raise EvaluatorError(
                "gold_pixel_count_mismatch",
                "canonical AOI pixel count does not match evaluator config",
            )
        expected_raw = manifest.evaluator.config.get("expected_distribution", {})
        expected = {int(key): int(value) for key, value in expected_raw.items()}
        if expected and distribution != expected:
            raise EvaluatorError(
                "gold_distribution_mismatch",
                "canonical AOI class distribution does not match evaluator config",
            )
        return asset, distribution

    def _faithfulness(
        self,
        manifest: TaskManifest,
        state: V2EpisodeState,
        artifacts: Dict[str, ArtifactRef],
        aoi: SpatialBoundingBox,
        submitted_label: Optional[str],
        dominant_label: str,
        dominant_fraction: float,
    ) -> Tuple[float, Dict[str, object]]:
        answer = state.final_answer
        evidence_ids = set(answer.evidence_ids if answer is not None else [])
        evidence = [item for item in state.evidence_refs if item.evidence_id in evidence_ids]
        accessible_assets = {
            asset.asset_id: asset
            for asset in manifest.assets
            if asset.asset_id in state.accessible_asset_refs
        }
        source_valid = False
        rendered_valid = False
        for item in evidence:
            if item.selector.bbox is None or not _same_bbox(item.selector.bbox, aoi):
                continue
            asset = accessible_assets.get(item.source_ref)
            if asset is not None and item.frozen_sha256 == asset.sha256:
                source_valid = True
                continue
            artifact = artifacts.get(item.source_ref)
            if (
                artifact is not None
                and artifact.kind == "image"
                and artifact.media_type == "image/png"
                and artifact.spatial is not None
                and _contains(artifact.spatial.bbox, aoi)
                and item.frozen_sha256 == artifact.sha256
                and self.artifact_store.audit_exists(artifact)
            ):
                rendered_valid = True
        minimum_fraction = float(
            manifest.evaluator.config.get("minimum_dominant_fraction", 0.5)
        )
        claim_supported = (
            submitted_label == dominant_label
            and dominant_fraction >= minimum_fraction
        )
        passed = source_valid and rendered_valid and claim_supported
        return float(passed), {
            "claim_supported": claim_supported,
            "evidence_count": len(evidence),
            "rendered_evidence_valid": rendered_valid,
            "source_evidence_valid": source_valid,
        }

    @staticmethod
    def _efficiency(
        manifest: TaskManifest,
        state: V2EpisodeState,
        renderer_calls: int,
        failed_actions: int,
        wall_time_ms: int,
    ) -> Tuple[float, Dict[str, object]]:
        config = manifest.evaluator.config.get("efficiency", {})
        ideal_steps = max(int(config.get("ideal_steps", 1)), 1)
        ideal_renderer_calls = max(int(config.get("ideal_renderer_calls", 1)), 1)
        wall_limit = max(int(config.get("wall_time_soft_limit_ms", 1)), 1)
        step_score = min(1.0, ideal_steps / max(state.step_count, 1))
        renderer_score = (
            min(1.0, ideal_renderer_calls / renderer_calls)
            if renderer_calls > 0
            else 0.0
        )
        wall_score = min(1.0, wall_limit / max(wall_time_ms, 1))
        failure_score = 1.0 / (1.0 + max(failed_actions, 0))
        score = round(
            step_score * renderer_score * wall_score * failure_score,
            6,
        )
        return score, {
            "failed_actions": failed_actions,
            "ideal_renderer_calls": ideal_renderer_calls,
            "ideal_steps": ideal_steps,
            "renderer_calls": renderer_calls,
            "steps": state.step_count,
            "wall_time_ms": wall_time_ms,
            "wall_time_soft_limit_ms": wall_limit,
        }

    @staticmethod
    def _temporal_expected(manifest: TaskManifest) -> Dict[str, object]:
        config = manifest.evaluator.config
        status = config.get("expected_selection_status")
        reason = config.get("expected_selection_reason")
        coverage = config.get("expected_coverage_decision")
        reasons = {
            "selected",
            "wrong_date",
            "insufficient_coverage",
            "cloudy",
            "sensor_mismatch",
        }
        if (
            status not in {"selected", "rejected"}
            or reason not in reasons
            or coverage not in {"pass", "fail", "not_evaluated"}
            or (status == "selected") != (reason == "selected")
        ):
            raise EvaluatorError(
                "evaluator_config_invalid",
                "temporal truth status, reason and coverage decision are invalid",
            )
        expected: Dict[str, object] = {
            "status": status,
            "reason": reason,
            "coverage_decision": coverage,
        }
        input_ids = config.get("expected_input_asset_ids", [])
        if status == "selected":
            before_item = config.get("expected_before_item_id")
            after_item = config.get("expected_after_item_id")
            if (
                not isinstance(before_item, str)
                or not before_item
                or not isinstance(after_item, str)
                or not after_item
                or not isinstance(input_ids, list)
                or len(input_ids) != 4
                or len(set(input_ids)) != 4
                or not all(isinstance(value, str) and value for value in input_ids)
            ):
                raise EvaluatorError(
                    "evaluator_config_invalid",
                    "selected temporal truth requires two items and four inputs",
                )
            expected.update(
                before_item_id=before_item,
                after_item_id=after_item,
                input_asset_ids=input_ids,
            )
        elif input_ids:
            raise EvaluatorError(
                "evaluator_config_invalid",
                "rejected temporal truth must not declare selected inputs",
            )
        return expected

    @staticmethod
    def _temporal_tool_result(
        tool_results: list[dict],
    ) -> Tuple[Optional[TemporalToolResult], int, list[str]]:
        matching = [
            value
            for value in tool_results
            if isinstance(value, dict)
            and value.get("tool_id") == TEMPORAL_TOOL_ID
        ]
        completed = [value for value in matching if value.get("status") == "completed"]
        statuses = [str(value.get("status")) for value in matching]
        if not completed:
            return None, len(matching), statuses
        value = completed[-1]
        if value.get("tool_version") != TEMPORAL_TOOL_VERSION:
            raise EvaluatorError(
                "temporal_tool_version_mismatch",
                "temporal tool result version does not match the evaluator",
            )
        try:
            result = TemporalToolResult.model_validate(
                {
                    "selection": value["selection"],
                    "stack": value.get("stack"),
                }
            )
        except (KeyError, TypeError, ValidationError):
            raise EvaluatorError(
                "temporal_tool_result_invalid",
                "temporal tool result failed contract validation",
            ) from None
        return result, len(matching), statuses

    @staticmethod
    def _temporal_coverage_decision(
        result: Optional[TemporalToolResult],
    ) -> str:
        if result is None or result.selection.reason == "wrong_date":
            return "not_evaluated"
        if result.selection.reason == "insufficient_coverage":
            return "fail"
        return "pass"

    @staticmethod
    def _temporal_efficiency(
        manifest: TaskManifest,
        state: V2EpisodeState,
        renderer_calls: int,
        failed_actions: int,
        wall_time_ms: int,
        temporal_call_count: int,
    ) -> Tuple[float, Dict[str, object]]:
        config = manifest.evaluator.config.get("efficiency", {})
        ideal_steps = max(int(config.get("ideal_steps", 1)), 1)
        ideal_tool_calls = max(int(config.get("ideal_tool_calls", 1)), 1)
        wall_limit = max(int(config.get("wall_time_soft_limit_ms", 1)), 1)
        expected_renderer_calls = max(
            int(config.get("expected_renderer_calls", 0)), 0
        )
        step_score = min(1.0, ideal_steps / max(state.step_count, 1))
        tool_score = (
            min(1.0, ideal_tool_calls / temporal_call_count)
            if temporal_call_count > 0
            else 0.0
        )
        renderer_score = 1.0 / (
            1.0 + abs(renderer_calls - expected_renderer_calls)
        )
        wall_score = min(1.0, wall_limit / max(wall_time_ms, 1))
        failure_score = 1.0 / (1.0 + max(failed_actions, 0))
        score = round(
            step_score
            * tool_score
            * renderer_score
            * wall_score
            * failure_score,
            6,
        )
        return score, {
            "expected_renderer_calls": expected_renderer_calls,
            "failed_actions": failed_actions,
            "ideal_steps": ideal_steps,
            "ideal_tool_calls": ideal_tool_calls,
            "renderer_calls": renderer_calls,
            "steps": state.step_count,
            "temporal_tool_calls": temporal_call_count,
            "wall_time_ms": wall_time_ms,
            "wall_time_soft_limit_ms": wall_limit,
        }

    def _evaluate_temporal_selection(
        self,
        evaluation_id: str,
        manifest: TaskManifest,
        state: V2EpisodeState,
        artifacts: Dict[str, ArtifactRef],
        renderer_calls: int,
        failed_actions: int,
        wall_time_ms: int,
        tool_results: list[dict],
    ) -> MetricResult:
        if manifest.evaluator.evaluator_version != "1.0.0":
            raise EvaluatorError(
                "evaluator_version_not_supported",
                "temporal selection evaluator version is not implemented",
            )
        expected = self._temporal_expected(manifest)
        result, temporal_call_count, tool_statuses = self._temporal_tool_result(
            tool_results
        )
        temporal_outputs = [
            artifact
            for artifact in artifacts.values()
            if artifact.lineage.tool_id == TEMPORAL_TOOL_ID
        ]
        temporal_artifacts = [
            artifact
            for artifact in temporal_outputs
            if isinstance(artifact, TemporalStackArtifactRef)
        ]
        status_matches = (
            result is not None
            and result.selection.status == expected["status"]
            and result.selection.reason == expected["reason"]
        )
        artifact_matches = False
        pair_matches = expected["status"] == "rejected"
        if expected["status"] == "selected" and result is not None:
            selection = result.selection
            stack = result.stack
            input_ids = expected["input_asset_ids"]
            pair_matches = bool(
                selection.before is not None
                and selection.after is not None
                and stack is not None
                and selection.before.item_id == expected["before_item_id"]
                and selection.after.item_id == expected["after_item_id"]
                and stack.before_item_id == expected["before_item_id"]
                and stack.after_item_id == expected["after_item_id"]
                and stack.input_asset_ids == input_ids
            )
            if len(temporal_artifacts) == 1 and stack is not None:
                artifact = temporal_artifacts[0]
                descriptor = artifact.temporal_stack
                artifact_matches = bool(
                    self.artifact_store.audit_exists(artifact)
                    and artifact.lineage.tool_version == TEMPORAL_TOOL_VERSION
                    and artifact.lineage.input_refs == input_ids
                    and descriptor.before.item_id == expected["before_item_id"]
                    and descriptor.after.item_id == expected["after_item_id"]
                    and descriptor.grid_crs == stack.crs
                    and descriptor.grid_transform == stack.transform
                    and descriptor.width == stack.width
                    and descriptor.height == stack.height
                    and descriptor.band_order == stack.band_order
                    and descriptor.cloud_policy == stack.cloud_policy
                    and descriptor.before.coverage_fraction
                    == selection.before.coverage_fraction
                    and descriptor.after.coverage_fraction
                    == selection.after.coverage_fraction
                    and descriptor.before.cloud_fraction
                    == stack.before_cloud_fraction
                    and descriptor.after.cloud_fraction
                    == stack.after_cloud_fraction
                )
        elif expected["status"] == "rejected":
            artifact_matches = not temporal_outputs

        temporal_validity = float(
            status_matches and pair_matches and artifact_matches
        )
        actual_coverage = self._temporal_coverage_decision(result)
        coverage_valid = actual_coverage == expected["coverage_decision"]
        if result is not None and result.selection.status == "selected":
            before = result.selection.before
            after = result.selection.after
            stack = result.stack
            coverage_valid = bool(
                coverage_valid
                and before is not None
                and after is not None
                and stack is not None
                and before.coverage_fraction
                >= result.selection.minimum_coverage_fraction
                and after.coverage_fraction
                >= result.selection.minimum_coverage_fraction
                and stack.aligned_coverage_fraction
                >= float(
                    manifest.evaluator.config.get(
                        "minimum_aligned_coverage_fraction", 0.0
                    )
                )
            )

        answer = state.final_answer
        actual_outcome = answer.outcome if answer is not None else None
        expected_outcome = (
            "submitted" if expected["status"] == "selected" else "abstained"
        )
        false_confidence = bool(
            expected["status"] == "rejected" and actual_outcome == "submitted"
        )
        unnecessary_abstention = bool(
            expected["status"] == "selected" and actual_outcome == "abstained"
        )
        abstention_correctness = float(actual_outcome == expected_outcome)

        faithfulness = 0.0
        faithfulness_diagnostics: Dict[str, object]
        if expected["status"] == "selected" and len(temporal_artifacts) == 1:
            artifact = temporal_artifacts[0]
            selected_evidence = set(answer.evidence_ids if answer is not None else [])
            evidence = [
                item
                for item in state.evidence_refs
                if item.evidence_id in selected_evidence
                and item.source_ref == artifact.artifact_id
            ]
            valid = [
                item
                for item in evidence
                if item.frozen_sha256 == artifact.sha256
                and item.selector.bbox is not None
                and _same_bbox(item.selector.bbox, artifact.spatial.bbox)
                and item.selector.time_range == artifact.temporal
            ]
            faithfulness = float(
                actual_outcome == "submitted"
                and artifact_matches
                and len(valid) == 1
            )
            faithfulness_diagnostics = {
                "artifact_id": artifact.artifact_id,
                "selected_evidence_count": len(evidence),
                "valid_full_extent_evidence_count": len(valid),
            }
        else:
            no_claim_evidence = bool(
                answer is not None
                and not answer.evidence_ids
                and not temporal_outputs
            )
            faithfulness = float(
                actual_outcome == "abstained"
                and status_matches
                and no_claim_evidence
            )
            faithfulness_diagnostics = {
                "basis": "metadata-only-rejection",
                "no_claim_evidence": no_claim_evidence,
                "temporal_artifact_count": len(temporal_outputs),
            }

        efficiency, efficiency_diagnostics = self._temporal_efficiency(
            manifest,
            state,
            renderer_calls,
            failed_actions,
            wall_time_ms,
            temporal_call_count,
        )
        values = {
            "temporal.validity": (
                temporal_validity,
                {
                    "artifact_matches": artifact_matches,
                    "expected_reason": expected["reason"],
                    "expected_status": expected["status"],
                    "observed_reason": (
                        result.selection.reason if result is not None else None
                    ),
                    "observed_status": (
                        result.selection.status if result is not None else None
                    ),
                    "pair_matches": pair_matches,
                    "tool_result_statuses": tool_statuses,
                },
            ),
            "spatial.coverage": (
                float(coverage_valid),
                {
                    "actual_decision": actual_coverage,
                    "expected_decision": expected["coverage_decision"],
                },
            ),
            "answer.abstention_correctness": (
                abstention_correctness,
                {
                    "actual_outcome": actual_outcome,
                    "expected_outcome": expected_outcome,
                    "false_confidence": false_confidence,
                    "unnecessary_abstention": unnecessary_abstention,
                    "insufficient_input_asset_ids": insufficient_inputs,
                    "minimum_input_coverage_fraction": minimum_coverage,
                },
            ),
            "evidence.faithfulness": (
                faithfulness,
                faithfulness_diagnostics,
            ),
            "process.efficiency": (
                efficiency,
                efficiency_diagnostics,
            ),
        }
        if any(name not in values for name in manifest.evaluator.metric_names):
            raise EvaluatorError(
                "evaluator_metric_not_supported",
                "temporal evaluator metric list contains an unknown metric",
            )
        metrics = [
            Metric(
                name=name,
                value=values[name][0],
                weight=manifest.evaluator.aggregate_weights.get(name),
                diagnostics=values[name][1],
            )
            for name in manifest.evaluator.metric_names
        ]
        return MetricResult(
            evaluation_id=evaluation_id,
            status="completed",
            metrics=metrics,
            aggregate_reward=aggregate_metrics(metrics),
            evaluator_id=manifest.evaluator.evaluator_id,
            evaluator_version=manifest.evaluator.evaluator_version,
            diagnostics={
                "expected_selection_reason": expected["reason"],
                "expected_selection_status": expected["status"],
                "false_confidence": false_confidence,
                "temporal_artifact_ids": [
                    artifact.artifact_id for artifact in temporal_outputs
                ],
                "temporal_tool_calls": temporal_call_count,
                "unnecessary_abstention": unnecessary_abstention,
            },
        )

    @staticmethod
    def _memory_tool_results(
        tool_results: list[dict],
    ) -> Tuple[list[MemorySearchResult], int, list[str]]:
        matching = [
            value
            for value in tool_results
            if isinstance(value, dict) and value.get("tool_id") == MEMORY_TOOL_ID
        ]
        completed = [value for value in matching if value.get("status") == "completed"]
        results: list[MemorySearchResult] = []
        for value in completed:
            if value.get("tool_version") != MEMORY_TOOL_VERSION:
                raise EvaluatorError(
                    "memory_tool_version_mismatch",
                    "memory tool result version does not match the evaluator",
                )
            try:
                results.append(
                    MemorySearchResult.model_validate(
                        {key: value[key] for key in MemorySearchResult.model_fields}
                    )
                )
            except (KeyError, TypeError, ValidationError):
                raise EvaluatorError(
                    "memory_tool_result_invalid",
                    "memory tool result failed contract validation",
                ) from None
        return results, len(matching), [str(value.get("status")) for value in matching]

    def _evaluate_evidence_memory(
        self,
        evaluation_id: str,
        manifest: TaskManifest,
        state: V2EpisodeState,
        artifacts: Dict[str, ArtifactRef],
        renderer_calls: int,
        failed_actions: int,
        wall_time_ms: int,
        tool_results: list[dict],
    ) -> MetricResult:
        del artifacts
        if manifest.evaluator.evaluator_version != "1.0.0":
            raise EvaluatorError(
                "evaluator_version_not_supported",
                "evidence memory evaluator version is not implemented",
            )
        config = manifest.evaluator.config
        treatment = config.get("treatment")
        expected_label = config.get("expected_label")
        expected_memory_id = config.get("expected_memory_id")
        expected_snapshot = config.get("expected_snapshot_sha256")
        if (
            treatment not in {"with_memory", "without_memory"}
            or not isinstance(expected_label, str)
            or not expected_label
            or not isinstance(expected_memory_id, str)
            or not expected_memory_id.startswith("mem-")
            or len(expected_memory_id) != 68
            or not isinstance(expected_snapshot, str)
            or len(expected_snapshot) != 64
        ):
            raise EvaluatorError(
                "evaluator_config_invalid",
                "evidence memory benchmark truth or treatment is invalid",
            )
        results, call_count, statuses = self._memory_tool_results(tool_results)
        result = results[0] if len(results) == 1 else None
        returned_ids = (
            [record.memory_id for record in result.records] if result is not None else []
        )
        retrieval_valid = bool(
            treatment == "with_memory"
            and call_count == 1
            and len(results) == 1
            and result.snapshot_sha256 == expected_snapshot
            and expected_memory_id in returned_ids
        )
        protocol_valid = (
            retrieval_valid if treatment == "with_memory" else call_count == 0
        )

        answer = state.final_answer
        submitted = bool(answer is not None and answer.outcome == "submitted")
        answer_value = answer.answer if answer is not None else None
        label = answer_value.get("label") if isinstance(answer_value, dict) else None
        cited_memory_ids = (
            answer_value.get("memory_ids", []) if isinstance(answer_value, dict) else []
        )
        if not isinstance(cited_memory_ids, list) or not all(
            isinstance(value, str) for value in cited_memory_ids
        ):
            cited_memory_ids = []
        accuracy = float(submitted and label == expected_label)
        faithfulness = float(
            accuracy == 1.0
            and retrieval_valid
            and cited_memory_ids == [expected_memory_id]
        )

        efficiency_config = config.get("efficiency", {})
        ideal_steps = int(
            efficiency_config.get(
                "ideal_steps", 2 if treatment == "with_memory" else 1
            )
        )
        wall_limit = int(efficiency_config.get("wall_time_soft_limit_ms", 30000))
        if ideal_steps <= 0 or wall_limit <= 0:
            raise EvaluatorError(
                "evaluator_config_invalid",
                "evidence memory efficiency bounds must be positive",
            )
        step_score = min(1.0, ideal_steps / max(state.step_count, 1))
        wall_score = min(1.0, wall_limit / max(wall_time_ms, 1))
        failure_score = 1.0 / (1.0 + max(failed_actions, 0))
        renderer_score = 1.0 if renderer_calls == 0 else 0.0
        efficiency = round(
            float(protocol_valid)
            * step_score
            * wall_score
            * failure_score
            * renderer_score,
            6,
        )
        values = {
            "task.accuracy": (
                accuracy,
                {
                    "expected_label": expected_label,
                    "submitted_label": label,
                    "submitted": submitted,
                },
            ),
            "evidence.memory_faithfulness": (
                faithfulness,
                {
                    "cited_memory_ids": cited_memory_ids,
                    "expected_memory_id": expected_memory_id,
                    "retrieval_valid": retrieval_valid,
                    "returned_memory_ids": returned_ids,
                },
            ),
            "process.efficiency": (
                efficiency,
                {
                    "failed_actions": failed_actions,
                    "ideal_steps": ideal_steps,
                    "memory_tool_calls": call_count,
                    "protocol_valid": protocol_valid,
                    "renderer_calls": renderer_calls,
                    "steps": state.step_count,
                    "wall_time_ms": wall_time_ms,
                    "wall_time_soft_limit_ms": wall_limit,
                },
            ),
        }
        if any(name not in values for name in manifest.evaluator.metric_names):
            raise EvaluatorError(
                "evaluator_metric_not_supported",
                "evidence memory evaluator metric list contains an unknown metric",
            )
        metrics = [
            Metric(
                name=name,
                value=values[name][0],
                weight=manifest.evaluator.aggregate_weights.get(name),
                diagnostics=values[name][1],
            )
            for name in manifest.evaluator.metric_names
        ]
        return MetricResult(
            evaluation_id=evaluation_id,
            status="completed",
            metrics=metrics,
            aggregate_reward=aggregate_metrics(metrics),
            evaluator_id=manifest.evaluator.evaluator_id,
            evaluator_version=manifest.evaluator.evaluator_version,
            diagnostics={
                "memory_tool_result_statuses": statuses,
                "retrieval_valid": retrieval_valid,
                "treatment": treatment,
            },
        )

    @staticmethod
    def _change_asset(
        manifest: TaskManifest,
        config_key: str,
        required_roles: set[str],
        *,
        input_required: bool,
    ) -> PixelAssetRef:
        asset_id = manifest.evaluator.config.get(config_key)
        asset = next(
            (
                item
                for item in manifest.assets
                if item.asset_id == asset_id
            ),
            None,
        )
        if not isinstance(asset, PixelAssetRef):
            raise EvaluatorError(
                "change_asset_not_registered",
                "temporal change asset is not a registered pixel asset",
            )
        if not required_roles.issubset(set(asset.roles)):
            raise EvaluatorError(
                "change_asset_role_mismatch",
                "temporal change asset roles do not match evaluator config",
            )
        is_input = asset.asset_id in manifest.task.inputs
        if is_input != input_required:
            raise EvaluatorError(
                "change_asset_visibility_mismatch",
                "labels must be evaluator-only and images must be task inputs",
            )
        return asset

    def _change_assets(
        self,
        manifest: TaskManifest,
    ) -> Tuple[PixelAssetRef, PixelAssetRef, PixelAssetRef, PixelAssetRef]:
        before_input = self._change_asset(
            manifest,
            "before_input_asset_id",
            {"input_image", "temporal_before"},
            input_required=True,
        )
        after_input = self._change_asset(
            manifest,
            "after_input_asset_id",
            {"input_image", "temporal_after"},
            input_required=True,
        )
        before_label = self._change_asset(
            manifest,
            "before_label_asset_id",
            {"evaluator", "building_label", "temporal_before"},
            input_required=False,
        )
        after_label = self._change_asset(
            manifest,
            "after_label_asset_id",
            {"evaluator", "building_label", "temporal_after"},
            input_required=False,
        )
        assets = (before_input, after_input, before_label, after_label)
        if any(asset.temporal is None for asset in assets):
            raise EvaluatorError(
                "temporal_extent_missing",
                "all temporal change assets require immutable timestamps",
            )
        if before_input.temporal.end >= after_input.temporal.start:
            raise EvaluatorError(
                "temporal_order_mismatch",
                "before input must end before the after input starts",
            )
        if (
            before_label.temporal != before_input.temporal
            or after_label.temporal != after_input.temporal
        ):
            raise EvaluatorError(
                "gold_temporal_mismatch",
                "hidden label timestamps must match their public inputs",
            )
        return before_input, after_input, before_label, after_label

    def _read_binary_label(
        self,
        asset: PixelAssetRef,
        expected_width: int,
        expected_height: int,
        expected_pixel_count: int,
    ):
        path = self._resolve_asset_path(asset)
        try:
            import numpy
            import rasterio
        except ImportError:
            raise EvaluatorError(
                "evaluator_dependency_missing",
                "rasterio and numpy are required by the change evaluator",
            )
        try:
            with rasterio.open(path) as dataset:
                if dataset.count != 1:
                    raise EvaluatorError(
                        "gold_label_band_mismatch",
                        "building labels must contain exactly one band",
                    )
                if dataset.crs is not None:
                    raise EvaluatorError(
                        "gold_label_crs_mismatch",
                        "split WHU labels must remain pixel-only",
                    )
                if dataset.dtypes != ("uint8",):
                    raise EvaluatorError(
                        "gold_label_dtype_mismatch",
                        "building labels must use uint8",
                    )
                if (
                    dataset.width != expected_width
                    or dataset.height != expected_height
                    or asset.pixel.width != expected_width
                    or asset.pixel.height != expected_height
                    or asset.pixel.channels != 1
                ):
                    raise EvaluatorError(
                        "gold_label_shape_mismatch",
                        "building label dimensions do not match evaluator config",
                    )
                values = dataset.read(1)
        except EvaluatorError:
            raise
        except Exception:
            raise EvaluatorError(
                "gold_label_unreadable",
                "building label could not be decoded",
            ) from None
        unique = {int(value) for value in numpy.unique(values).tolist()}
        if not unique.issubset({0, 255}):
            raise EvaluatorError(
                "gold_label_value_domain_mismatch",
                "building labels must contain only 0 and 255",
            )
        if int(values.size) != expected_pixel_count:
            raise EvaluatorError(
                "gold_pixel_count_mismatch",
                "building label pixel count does not match evaluator config",
            )
        return values == 255

    @staticmethod
    def _change_truth(
        manifest: TaskManifest,
        before_mask,
        after_mask,
    ) -> Dict[str, object]:
        import numpy

        new_pixels = int(numpy.count_nonzero(~before_mask & after_mask))
        demolished_pixels = int(
            numpy.count_nonzero(before_mask & ~after_mask)
        )
        changed_pixels = new_pixels + demolished_pixels
        total = int(before_mask.size)
        changed_fraction = changed_pixels / total
        minor_limit = float(
            manifest.evaluator.config.get("minor_change_max_fraction", 0.1)
        )
        dominance = float(
            manifest.evaluator.config.get("direction_dominance_ratio", 1.5)
        )
        if not 0.0 < minor_limit < 1.0 or dominance < 1.0:
            raise EvaluatorError(
                "evaluator_config_invalid",
                "change thresholds are outside their valid ranges",
            )
        if changed_pixels == 0:
            change_class = "no_change"
            direction = "no_change"
        else:
            change_class = (
                "minor_change"
                if changed_fraction <= minor_limit
                else "major_change"
            )
            if new_pixels > demolished_pixels * dominance:
                direction = "expansion"
            elif demolished_pixels > new_pixels * dominance:
                direction = "reduction"
            else:
                direction = "mixed"
        expected = {
            "expected_changed_pixels": changed_pixels,
            "expected_new_pixels": new_pixels,
            "expected_demolished_pixels": demolished_pixels,
        }
        if any(
            int(manifest.evaluator.config.get(key, -1)) != value
            for key, value in expected.items()
        ):
            raise EvaluatorError(
                "gold_change_truth_mismatch",
                "computed change counts do not match immutable evaluator config",
            )
        return {
            "change_class": change_class,
            "change_direction": direction,
            "changed_fraction": changed_fraction,
            "changed_pixels": changed_pixels,
            "new_pixels": new_pixels,
            "demolished_pixels": demolished_pixels,
            "pixel_count": total,
        }

    def _change_faithfulness(
        self,
        manifest: TaskManifest,
        state: V2EpisodeState,
        artifacts: Dict[str, ArtifactRef],
        inputs: Tuple[PixelAssetRef, PixelAssetRef],
    ) -> Tuple[float, Dict[str, object]]:
        answer = state.final_answer
        selected = set(answer.evidence_ids if answer is not None else [])
        evidence = [
            item
            for item in state.evidence_refs
            if item.evidence_id in selected
        ]
        required_tool = str(
            manifest.evaluator.config.get(
                "required_evidence_tool_id", "eo_gym.crop"
            )
        )
        covered: set[str] = set()
        for item in evidence:
            artifact = artifacts.get(item.source_ref)
            if (
                not isinstance(artifact, PixelArtifactRef)
                or artifact.kind != "image"
                or artifact.media_type != "image/png"
                or item.frozen_sha256 != artifact.sha256
                or not self.artifact_store.audit_exists(artifact)
                or artifact.lineage.tool_id != required_tool
            ):
                continue
            for asset in inputs:
                if artifact.lineage.input_refs != [asset.asset_id]:
                    continue
                if (
                    artifact.pixel.width != asset.pixel.width
                    or artifact.pixel.height != asset.pixel.height
                    or item.selector.pixel_window
                    != [0, 0, artifact.pixel.width, artifact.pixel.height]
                ):
                    continue
                covered.add(asset.asset_id)
        expected = [asset.asset_id for asset in inputs]
        missing = [asset_id for asset_id in expected if asset_id not in covered]
        return float(not missing), {
            "covered_input_asset_ids": [
                asset_id for asset_id in expected if asset_id in covered
            ],
            "evidence_count": len(evidence),
            "missing_input_asset_ids": missing,
            "required_evidence_tool_id": required_tool,
        }

    def _evaluate_whu_change(
        self,
        evaluation_id: str,
        manifest: TaskManifest,
        state: V2EpisodeState,
        artifacts: Dict[str, ArtifactRef],
        renderer_calls: int,
        failed_actions: int,
        wall_time_ms: int,
    ) -> MetricResult:
        if manifest.evaluator.evaluator_version != "1.0.0":
            raise EvaluatorError(
                "evaluator_version_not_supported",
                "WHU change evaluator version is not implemented",
            )
        before_input, after_input, before_label, after_label = (
            self._change_assets(manifest)
        )
        expected_width = int(
            manifest.evaluator.config.get("expected_width", 0)
        )
        expected_height = int(
            manifest.evaluator.config.get("expected_height", 0)
        )
        expected_count = int(
            manifest.evaluator.config.get("expected_pixel_count", 0)
        )
        if (
            expected_width <= 0
            or expected_height <= 0
            or expected_count != expected_width * expected_height
            or before_input.pixel.width != expected_width
            or before_input.pixel.height != expected_height
            or after_input.pixel != before_input.pixel
        ):
            raise EvaluatorError(
                "evaluator_shape_config_mismatch",
                "public inputs and evaluator shape config disagree",
            )
        before_mask = self._read_binary_label(
            before_label,
            expected_width,
            expected_height,
            expected_count,
        )
        after_mask = self._read_binary_label(
            after_label,
            expected_width,
            expected_height,
            expected_count,
        )
        truth = self._change_truth(manifest, before_mask, after_mask)
        answer_value = (
            state.final_answer.answer
            if state.final_answer is not None
            else None
        )
        submitted = (
            state.final_answer is not None
            and state.final_answer.outcome == "submitted"
            and isinstance(answer_value, dict)
        )
        actual_outcome = (
            state.final_answer.outcome if state.final_answer is not None else None
        )
        expected_outcome = str(
            manifest.evaluator.config.get("expected_outcome", "submitted")
        )
        if expected_outcome not in {"submitted", "abstained"}:
            raise EvaluatorError(
                "evaluator_config_invalid",
                "expected outcome must be submitted or abstained",
            )
        required_inputs = (before_input, after_input)
        minimum_coverage = float(
            manifest.evaluator.config.get(
                "minimum_input_coverage_fraction", 0.0
            )
        )
        if not 0.0 <= minimum_coverage <= 1.0:
            raise EvaluatorError(
                "evaluator_config_invalid",
                "minimum input coverage must be in [0, 1]",
            )
        availability = {
            asset.asset_id: (
                asset.quality.coverage_fraction
                if asset.quality.coverage_fraction is not None
                else 1.0
            )
            for asset in required_inputs
        }
        insufficient_inputs = sorted(
            asset_id
            for asset_id, coverage in availability.items()
            if coverage < minimum_coverage
        )
        metadata_requires_abstention = bool(insufficient_inputs)
        if expected_outcome == "abstained" and not metadata_requires_abstention:
            raise EvaluatorError(
                "expected_abstention_unsupported",
                "task metadata does not justify required abstention",
            )
        if expected_outcome == "submitted" and metadata_requires_abstention:
            raise EvaluatorError(
                "expected_submission_unsupported",
                "insufficient input coverage contradicts expected submission",
            )
        false_confidence = bool(
            expected_outcome == "abstained" and actual_outcome == "submitted"
        )
        unnecessary_abstention = bool(
            expected_outcome == "submitted" and actual_outcome == "abstained"
        )
        abstention_correctness = float(actual_outcome == expected_outcome)
        submitted_class = (
            answer_value.get("change_class") if submitted else None
        )
        submitted_direction = (
            answer_value.get("change_direction") if submitted else None
        )
        submitted_fraction = (
            answer_value.get("changed_fraction") if submitted else None
        )
        valid_fraction = (
            isinstance(submitted_fraction, (int, float))
            and not isinstance(submitted_fraction, bool)
            and math.isfinite(float(submitted_fraction))
            and 0.0 <= float(submitted_fraction) <= 1.0
        )
        class_accuracy = float(
            submitted_class == truth["change_class"]
        )
        direction_accuracy = float(
            submitted_direction == truth["change_direction"]
        )
        tolerance = float(
            manifest.evaluator.config.get(
                "changed_fraction_tolerance", 0.05
            )
        )
        if tolerance <= 0.0 or tolerance > 1.0:
            raise EvaluatorError(
                "evaluator_config_invalid",
                "changed fraction tolerance must be in (0, 1]",
            )
        fraction_error = (
            abs(float(submitted_fraction) - truth["changed_fraction"])
            if valid_fraction
            else None
        )
        fraction_score = (
            max(0.0, 1.0 - fraction_error / tolerance)
            if fraction_error is not None
            else 0.0
        )
        faithfulness, faithfulness_diagnostics = (
            self._change_faithfulness(
                manifest,
                state,
                artifacts,
                (before_input, after_input),
            )
        )
        efficiency, efficiency_diagnostics = self._efficiency(
            manifest,
            state,
            renderer_calls,
            failed_actions,
            wall_time_ms,
        )
        values = {
            "task.change_class_accuracy": (
                class_accuracy,
                {
                    "submitted": submitted_class,
                    "truth": truth["change_class"],
                },
            ),
            "task.direction_accuracy": (
                direction_accuracy,
                {
                    "submitted": submitted_direction,
                    "truth": truth["change_direction"],
                },
            ),
            "task.changed_fraction_score": (
                round(fraction_score, 6),
                {
                    "absolute_error": fraction_error,
                    "submitted": submitted_fraction,
                    "tolerance": tolerance,
                    "truth": truth["changed_fraction"],
                },
            ),
            "evidence.faithfulness": (
                faithfulness,
                faithfulness_diagnostics,
            ),
            "answer.abstention_correctness": (
                abstention_correctness,
                {
                    "actual_outcome": actual_outcome,
                    "expected_outcome": expected_outcome,
                    "false_confidence": false_confidence,
                    "insufficient_input_asset_ids": insufficient_inputs,
                    "minimum_input_coverage_fraction": minimum_coverage,
                    "unnecessary_abstention": unnecessary_abstention,
                },
            ),
            "process.efficiency": (
                efficiency,
                efficiency_diagnostics,
            ),
        }
        if any(name not in values for name in manifest.evaluator.metric_names):
            raise EvaluatorError(
                "evaluator_metric_not_supported",
                "WHU evaluator metric list contains an unknown metric",
            )
        metrics = [
            Metric(
                name=name,
                value=values[name][0],
                weight=manifest.evaluator.aggregate_weights.get(name),
                diagnostics=values[name][1],
            )
            for name in manifest.evaluator.metric_names
        ]
        return MetricResult(
            evaluation_id=evaluation_id,
            status="completed",
            metrics=metrics,
            aggregate_reward=aggregate_metrics(metrics),
            evaluator_id=manifest.evaluator.evaluator_id,
            evaluator_version=manifest.evaluator.evaluator_version,
            diagnostics={
                "before_label_asset_id": before_label.asset_id,
                "after_label_asset_id": after_label.asset_id,
                "false_confidence": false_confidence,
                "insufficient_input_asset_ids": insufficient_inputs,
                "truth": truth,
                "unnecessary_abstention": unnecessary_abstention,
            },
        )

    def _evaluate_worldcover(
        self,
        evaluation_id: str,
        manifest: TaskManifest,
        state: V2EpisodeState,
        artifacts: Dict[str, ArtifactRef],
        renderer_calls: int,
        failed_actions: int,
        wall_time_ms: int,
    ) -> MetricResult:
        if manifest.evaluator.evaluator_id != "worldcover-grounded-v1":
            raise EvaluatorError(
                "evaluator_not_supported",
                "registered evaluator is not implemented",
            )
        aoi = SpatialBoundingBox.model_validate(
            manifest.evaluator.config.get("evaluation_aoi")
        )
        canonical_asset, distribution = self._class_distribution(manifest, aoi)
        total = sum(distribution.values())
        if total <= 0:
            raise EvaluatorError(
                "gold_aoi_empty",
                "canonical evaluation AOI contains no valid pixels",
            )
        dominant_class, dominant_count = max(
            distribution.items(),
            key=lambda item: (item[1], -item[0]),
        )
        labels = {
            int(key): str(value)
            for key, value in manifest.evaluator.config.get("class_labels", {}).items()
        }
        dominant_label = labels.get(dominant_class)
        if dominant_label is None:
            raise EvaluatorError(
                "gold_class_unknown",
                "canonical dominant class is missing from evaluator labels",
            )
        answer_value = state.final_answer.answer if state.final_answer is not None else None
        submitted_label = (
            str(answer_value.get("label"))
            if isinstance(answer_value, dict) and answer_value.get("label") is not None
            else None
        )
        accuracy = float(
            state.final_answer is not None
            and state.final_answer.outcome == "submitted"
            and submitted_label == dominant_label
        )
        dominant_fraction = dominant_count / total
        faithfulness, faithfulness_diagnostics = self._faithfulness(
            manifest=manifest,
            state=state,
            artifacts=artifacts,
            aoi=aoi,
            submitted_label=submitted_label,
            dominant_label=dominant_label,
            dominant_fraction=dominant_fraction,
        )
        efficiency, efficiency_diagnostics = self._efficiency(
            manifest=manifest,
            state=state,
            renderer_calls=renderer_calls,
            failed_actions=failed_actions,
            wall_time_ms=wall_time_ms,
        )
        values = {
            "task.accuracy": (
                accuracy,
                {
                    "dominant_class": dominant_class,
                    "dominant_fraction": dominant_fraction,
                    "dominant_label": dominant_label,
                    "submitted_label": submitted_label,
                },
            ),
            "evidence.faithfulness": (faithfulness, faithfulness_diagnostics),
            "process.efficiency": (efficiency, efficiency_diagnostics),
        }
        metrics = [
            Metric(
                name=name,
                value=values[name][0],
                weight=manifest.evaluator.aggregate_weights.get(name),
                diagnostics=values[name][1],
            )
            for name in manifest.evaluator.metric_names
        ]
        return MetricResult(
            evaluation_id=evaluation_id,
            status="completed",
            metrics=metrics,
            aggregate_reward=aggregate_metrics(metrics),
            evaluator_id=manifest.evaluator.evaluator_id,
            evaluator_version=manifest.evaluator.evaluator_version,
            diagnostics={
                "canonical_asset_id": canonical_asset.asset_id,
                "canonical_sha256": canonical_asset.sha256,
                "class_distribution": {
                    str(key): value for key, value in sorted(distribution.items())
                },
                "evaluation_aoi": aoi.model_dump(mode="json"),
                "observation_profile": manifest.evaluator.config.get(
                    "observation_profile"
                ),
            },
        )

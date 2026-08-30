import hashlib
import uuid
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Optional, Tuple

from .artifacts import ArtifactStore
from .schemas import (
    ArtifactRef,
    AssetRef,
    Metric,
    MetricResult,
    SpatialBoundingBox,
    TaskManifest,
    V2EpisodeState,
)


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
    ) -> MetricResult:
        evaluation_id = "eval-%s" % uuid.uuid4().hex
        try:
            return self._evaluate_worldcover(
                evaluation_id=evaluation_id,
                manifest=manifest,
                state=state,
                artifacts=artifacts,
                renderer_calls=renderer_calls,
                failed_actions=failed_actions,
                wall_time_ms=wall_time_ms,
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

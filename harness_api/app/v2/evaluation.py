import hashlib
import math
import uuid
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Optional, Tuple

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
                "truth": truth,
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

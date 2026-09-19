"""Independent validation for the pinned Sentinel-2 SCL exclusion policy.

CloudSEN12 high-quality manual labels use 0=clear, 1=thick cloud,
2=thin cloud and 3=cloud shadow.  Positive means unusable for the reviewed
temporal policy, so manual classes 1, 2 and 3 are compared with the current
SCL exclusion set.  This module performs no IO and does not silently invent
scores when a metric denominator is empty.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from .raster_math import CLOUD_EXCLUDED_CLASSES, CLOUD_POLICY

VERSION = "1.0.0"
BENCHMARK_ID = "cloud-policy-validation-v1"
MANUAL_CLASSES = (0, 1, 2, 3)
MANUAL_INVALID_CLASSES = (1, 2, 3)
SCL_CLASSES = tuple(range(1, 12))
DEFAULT_THRESHOLDS = (0.001, 0.2)


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _metrics(confusion: dict[str, int]) -> dict[str, float | None]:
    true_negative = confusion["true_negative"]
    false_positive = confusion["false_positive"]
    false_negative = confusion["false_negative"]
    true_positive = confusion["true_positive"]
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    specificity = _ratio(true_negative, true_negative + false_positive)
    balanced_accuracy = (
        float((recall + specificity) / 2)
        if recall is not None and specificity is not None
        else None
    )
    return {
        "precision": precision,
        "recall": recall,
        "intersection_over_union": _ratio(
            true_positive,
            true_positive + false_positive + false_negative,
        ),
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }


def _class_array(values, valid, *, allowed: Iterable[int], name: str):
    import numpy

    array = numpy.asarray(values)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if valid is None:
        valid_array = numpy.ones(array.shape, dtype=bool)
    else:
        valid_array = numpy.asarray(valid)
        if valid_array.shape != array.shape or valid_array.dtype != numpy.bool_:
            raise ValueError(f"{name} validity mask must be aligned boolean data")
    selected = array[valid_array]
    if selected.size:
        if not numpy.isfinite(selected).all() or not numpy.equal(
            selected, numpy.floor(selected)
        ).all():
            raise ValueError(f"{name} valid labels must be finite integers")
        if not numpy.isin(selected, tuple(allowed)).all():
            raise ValueError(f"{name} valid labels are outside the reviewed class domain")
    return array, valid_array


def evaluate_cloud_policy(
    manual,
    scl,
    *,
    manual_valid=None,
    scl_valid=None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict:
    """Compare one aligned manual/SCL window and return JSON-safe metrics."""
    import numpy

    manual_array, manual_mask = _class_array(
        manual,
        manual_valid,
        allowed=MANUAL_CLASSES,
        name="manual",
    )
    scl_array, scl_mask = _class_array(
        scl,
        scl_valid,
        allowed=SCL_CLASSES,
        name="scl",
    )
    if scl_array.shape != manual_array.shape:
        raise ValueError("manual and SCL labels must have the same grid")
    reviewed_thresholds = []
    for threshold in thresholds:
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not numpy.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("thresholds must be finite fractions in [0, 1]")
        value = float(threshold)
        if value in reviewed_thresholds:
            raise ValueError("thresholds must be unique")
        reviewed_thresholds.append(value)
    if not reviewed_thresholds:
        raise ValueError("at least one threshold is required")

    scored = manual_mask & scl_mask
    scored_pixels = int(scored.sum())
    if scored_pixels <= 0:
        raise ValueError("manual and SCL labels have no jointly valid pixels")
    manual_invalid = scored & numpy.isin(manual_array, MANUAL_INVALID_CLASSES)
    policy_invalid = scored & numpy.isin(scl_array, CLOUD_EXCLUDED_CLASSES)
    confusion = {
        "true_negative": int(numpy.count_nonzero(scored & ~manual_invalid & ~policy_invalid)),
        "false_positive": int(numpy.count_nonzero(scored & ~manual_invalid & policy_invalid)),
        "false_negative": int(numpy.count_nonzero(scored & manual_invalid & ~policy_invalid)),
        "true_positive": int(numpy.count_nonzero(scored & manual_invalid & policy_invalid)),
    }
    manual_fraction = float(numpy.count_nonzero(manual_invalid) / scored_pixels)
    policy_fraction = float(numpy.count_nonzero(policy_invalid) / scored_pixels)
    decisions = []
    for threshold in reviewed_thresholds:
        manual_decision = "accept" if manual_fraction <= threshold else "reject"
        policy_decision = "accept" if policy_fraction <= threshold else "reject"
        if manual_decision == policy_decision:
            outcome = "agree_" + manual_decision
        elif policy_decision == "accept":
            outcome = "false_accept"
        else:
            outcome = "false_reject"
        decisions.append(
            {
                "threshold": threshold,
                "manual_decision": manual_decision,
                "policy_decision": policy_decision,
                "outcome": outcome,
            }
        )
    return {
        "scored_pixels": scored_pixels,
        "unscored_pixels": int(scored.size - scored_pixels),
        "confusion": confusion,
        "metrics": _metrics(confusion),
        "fractions": {
            "manual_invalid": manual_fraction,
            "policy_invalid": policy_fraction,
        },
        "threshold_decisions": decisions,
    }


def aggregate_cloud_policy(
    windows: Sequence[dict],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict:
    """Aggregate evaluated windows without averaging per-window pixel metrics."""
    if not windows:
        raise ValueError("at least one evaluated window is required")
    reviewed_thresholds = [float(value) for value in thresholds]
    confusion = {
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 0,
    }
    scored_pixels = unscored_pixels = 0
    manual_invalid_pixels = policy_invalid_pixels = 0
    threshold_counts = {
        value: {"agree_accept": 0, "agree_reject": 0, "false_accept": 0, "false_reject": 0}
        for value in reviewed_thresholds
    }
    threshold_samples = {
        value: {"false_accept": [], "false_reject": []}
        for value in reviewed_thresholds
    }
    for window in windows:
        sample_id = window.get("sample_id")
        result = window.get("evaluation")
        if not isinstance(sample_id, str) or not sample_id or not isinstance(result, dict):
            raise ValueError("each window requires a sample_id and evaluation")
        for name in confusion:
            value = result["confusion"][name]
            if type(value) is not int or value < 0:
                raise ValueError("invalid confusion count")
            confusion[name] += value
        scored = result["scored_pixels"]
        unscored = result["unscored_pixels"]
        if type(scored) is not int or scored <= 0 or type(unscored) is not int or unscored < 0:
            raise ValueError("invalid pixel count")
        scored_pixels += scored
        unscored_pixels += unscored
        manual_invalid_pixels += result["confusion"]["true_positive"] + result["confusion"]["false_negative"]
        policy_invalid_pixels += result["confusion"]["true_positive"] + result["confusion"]["false_positive"]
        decisions = result.get("threshold_decisions", [])
        if [item.get("threshold") for item in decisions] != reviewed_thresholds:
            raise ValueError("window thresholds disagree with aggregate thresholds")
        for item in decisions:
            outcome = item["outcome"]
            if outcome not in threshold_counts[item["threshold"]]:
                raise ValueError("invalid threshold outcome")
            threshold_counts[item["threshold"]][outcome] += 1
            if outcome in threshold_samples[item["threshold"]]:
                threshold_samples[item["threshold"]][outcome].append(sample_id)

    decision_summary = []
    for threshold in reviewed_thresholds:
        counts = threshold_counts[threshold]
        agreements = counts["agree_accept"] + counts["agree_reject"]
        decision_summary.append(
            {
                "threshold": threshold,
                "window_count": len(windows),
                "correct_window_decisions": agreements,
                "window_decision_accuracy": float(agreements / len(windows)),
                "outcomes": counts,
                "false_accept_samples": threshold_samples[threshold]["false_accept"],
                "false_reject_samples": threshold_samples[threshold]["false_reject"],
            }
        )
    return {
        "window_count": len(windows),
        "scored_pixels": scored_pixels,
        "unscored_pixels": unscored_pixels,
        "confusion": confusion,
        "metrics": _metrics(confusion),
        "fractions": {
            "manual_invalid": float(manual_invalid_pixels / scored_pixels),
            "policy_invalid": float(policy_invalid_pixels / scored_pixels),
        },
        "threshold_decisions": decision_summary,
    }


__all__ = [
    "BENCHMARK_ID",
    "CLOUD_POLICY",
    "DEFAULT_THRESHOLDS",
    "MANUAL_CLASSES",
    "MANUAL_INVALID_CLASSES",
    "VERSION",
    "aggregate_cloud_policy",
    "evaluate_cloud_policy",
]

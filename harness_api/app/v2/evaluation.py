from typing import Iterable, Optional

from .schemas import Metric


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

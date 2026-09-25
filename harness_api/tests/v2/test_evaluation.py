import unittest

from app.core.evaluation import aggregate_metrics
from app.core.schemas import Metric


class V2EvaluationTests(unittest.TestCase):
    def test_metric_vector_aggregates_only_explicit_weights(self):
        metrics = [
            Metric(name="task.accuracy", value=1.0, weight=0.6),
            Metric(name="evidence.faithfulness", value=0.8, weight=0.3),
            Metric(name="process.efficiency", value=0.5, weight=0.1),
            Metric(name="calibration.confidence", value=0.7, weight=None),
        ]
        self.assertAlmostEqual(aggregate_metrics(metrics), 0.89)

    def test_missing_weights_keep_aggregate_nullable(self):
        metrics = [Metric(name="task.accuracy", value=1.0)]
        self.assertIsNone(aggregate_metrics(metrics))


if __name__ == "__main__":
    unittest.main()

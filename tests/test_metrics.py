import unittest

import numpy as np
import pandas as pd

from src.metrics import continuity, observed_arc_set_metrics


class MetricsTests(unittest.TestCase):
    def test_continuity_does_not_bridge_missing_predictions(self):
        arcs = np.array([[0, 1], [1, 2], [5, 6]], dtype=np.int64)
        # The only valid adjacent pair is (2, 2). Arc 0 and arc 2 are separated by a missing
        # prediction and must not be compared with each other.
        pred = np.array([0, -1, 2, 2], dtype=np.int64)
        self.assertEqual(continuity(pred, arcs), 1.0)

    def test_observed_arc_set_metric_is_unordered_fix_level_overlap(self):
        obs = pd.DataFrame({"label_arc": [1, 2, 2], "pred_arc": [1, 3, 3]})
        precision, recall, f1 = observed_arc_set_metrics(obs)
        self.assertEqual(precision, 0.5)
        self.assertEqual(recall, 0.5)
        self.assertEqual(f1, 0.5)


if __name__ == "__main__":
    unittest.main()

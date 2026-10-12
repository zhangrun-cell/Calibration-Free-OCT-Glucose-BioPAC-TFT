from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biopac_tft_oct.reporting import common_evaluation_mask, subject_cluster_intervals


class PredictionReportingTests(unittest.TestCase):
    def points(self):
        rows = []
        for model in ["TFT", "TiDE"]:
            for subject, value in [("one", 6.0), ("two", 8.0)]:
                for index in [50, 51]:
                    rows.append({"model": model, "subject": subject, "session": subject + "/a.csv",
                        "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=index),
                        "row_index": index, "actual_glucose": value,
                        "predicted": value + 1, "reference_observed": True})
        return pd.DataFrame(rows)

    def test_common_mask_excludes_nonfinite_from_every_model(self):
        points = self.points()
        points.loc[(points.model == "TFT") & (points.subject == "one") & (points.row_index == 51),
                   "predicted"] = np.nan
        audit = common_evaluation_mask(points, ["TFT", "TiDE"])
        self.assertEqual(audit.loc[audit.included].groupby("model").size().to_dict(),
                         {"TFT": 3, "TiDE": 3})

    def test_missing_model_point_is_not_silently_compared(self):
        points = self.points().iloc[:-1]
        audit = common_evaluation_mask(points, ["TFT", "TiDE"])
        self.assertEqual(audit.loc[audit.included].groupby("model").size().to_dict(),
                         {"TFT": 3, "TiDE": 3})

    def test_cluster_bootstrap_is_deterministic_and_preserves_error(self):
        first = subject_cluster_intervals(self.points(), 40, 7)
        second = subject_cluster_intervals(self.points(), 40, 7)
        pd.testing.assert_frame_equal(first, second)
        rmse = first.loc[first.metric == "RMSE"]
        np.testing.assert_allclose(rmse[["estimate", "ci_lower", "ci_upper"]], 1)
        self.assertTrue(first.n_subjects.eq(2).all())


if __name__ == "__main__":
    unittest.main()

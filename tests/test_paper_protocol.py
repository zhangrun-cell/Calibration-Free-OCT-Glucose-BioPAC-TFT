"""Lightweight tests for the manuscript-aligned public protocol."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biopac_tft_oct import (  # noqa: E402
    PredictionProtocol,
    fit_dej_guided_dynamic_signal_extraction_rule,
    paired_physiological_trajectory_stress_test,
    shift_time_covariate,
    subjectwise_fivefold_splits,
)


class PaperProtocolTests(unittest.TestCase):
    def test_l0_and_model_history_are_distinct(self) -> None:
        protocol = PredictionProtocol()
        self.assertEqual(protocol.initial_reference_length, 10)
        self.assertEqual(protocol.first_forecastable_index, 50)
        self.assertEqual(int(protocol.reference_history_mask(80).sum()), 10)
        self.assertFalse(protocol.evaluable_target_mask(80)[49])
        self.assertTrue(protocol.evaluable_target_mask(80)[50])

    def test_subjects_are_tested_once_and_never_split_within_a_fold(self) -> None:
        subjects = [f"S{i:02d}" for i in range(1, 42)]
        folds = list(subjectwise_fivefold_splits(subjects, seed=7))
        self.assertEqual(len(folds), 5)
        all_tested = [subject for fold in folds for subject in fold.test_subjects]
        self.assertCountEqual(all_tested, subjects)
        for fold in folds:
            self.assertFalse(set(fold.train_subjects) & set(fold.validation_subjects))
            self.assertFalse(set(fold.train_subjects) & set(fold.test_subjects))
            self.assertFalse(set(fold.validation_subjects) & set(fold.test_subjects))

    def test_discovery_mapping_can_be_frozen_and_applied(self) -> None:
        table = pd.DataFrame(
            {
                "site": ["arm", "arm", "arm", "finger", "finger", "finger"],
                "dej_depth": [30, 35, 40, 60, 65, 70],
                "window_center_1": [35, 40, 45, 70, 75, 80],
                "window_center_2": [45, 50, 55, 80, 85, 90],
                "window_center_3": [55, 60, 65, 90, 95, 100],
                "window_center_4": [65, 70, 75, 100, 105, 110],
                "window_center_5": [75, 80, 85, 110, 115, 120],
            }
        )
        rule = fit_dej_guided_dynamic_signal_extraction_rule(table)
        np.testing.assert_array_equal(rule.centres(site="arm", dej_depth=42), np.array([47, 57, 67, 77, 87]))
        self.assertEqual(rule.window_ranges(site="finger", dej_depth=62, n_depth=150)[0], (67, 78))

    def test_stress_test_preserves_oct_reference_pairs_and_time_is_external(self) -> None:
        oct_features = np.array([[1.0], [2.0], [3.0], [4.0]])
        reference = np.array([10.0, 20.0, 30.0, 40.0])
        stressed_oct, stressed_reference = paired_physiological_trajectory_stress_test(
            oct_features, reference, mode="circular", shift=1
        )
        np.testing.assert_array_equal(stressed_reference, np.array([40.0, 10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(stressed_oct[:, 0] * 10.0, stressed_reference)
        np.testing.assert_array_equal(shift_time_covariate(np.arange(4), shift=1), np.array([3, 0, 1, 2]))


if __name__ == "__main__":
    unittest.main()

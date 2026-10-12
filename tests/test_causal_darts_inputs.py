import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_paper_darts.py"
spec = importlib.util.spec_from_file_location("run_paper_darts", SCRIPT)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class CausalInputTests(unittest.TestCase):
    def test_excel_serial_dates_are_not_nanoseconds(self):
        values = pd.Series(["45818.74305555555", "45818.74375", "45818.74444444444"])
        parsed = runner.parse_observation_times(values)
        self.assertEqual(parsed.iloc[0], pd.Timestamp("2025-06-10 17:50:00"))
        self.assertTrue((parsed.diff().dropna() == pd.Timedelta(minutes=1)).all())

    def test_mixed_text_dates_keep_all_observations(self):
        values = pd.Series(["2025-06-10 17:50:00", "2025/6/10 17:51", None])
        parsed = runner.parse_observation_times(values)
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2025-06-10 17:51:00"))
        self.assertTrue(pd.isna(parsed.iloc[2]))

    def test_identical_oct_is_grouped_without_merging_participants(self):
        sessions = []
        for index in range(11):
            frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=60, freq="min")})
            for column in runner.OCT_COLUMNS:
                frame[column] = np.arange(60, dtype=float) + index
            frame["glucose"] = index + 5.0
            sessions.append(SimpleNamespace(subject=f"subject-{index + 1}",
                                             file=f"subject-{index + 1}/session.csv", frame=frame))
        # Labels may differ; participant identities must still be retained.
        sessions[1].frame[runner.OCT_COLUMNS] = sessions[0].frame[runner.OCT_COLUMNS]
        folds, linked = runner.build_subject_folds(sessions)
        self.assertEqual(len(folds), 5)
        self.assertEqual(len(linked), 1)
        tested = []
        for fold in folds:
            roles = [set(fold.train_subjects), set(fold.validation_subjects), set(fold.test_subjects)]
            self.assertEqual(len(set.union(*roles)), 11)
            self.assertTrue(all(not roles[a] & roles[b] for a, b in ((0, 1), (0, 2), (1, 2))))
            self.assertEqual(next(i for i, role in enumerate(roles) if "subject-1" in role),
                             next(i for i, role in enumerate(roles) if "subject-2" in role))
            tested.extend(fold.test_subjects)
        self.assertEqual(len(tested), 11)
        self.assertEqual(len(set(tested)), 11)

    def test_causal_optical_preprocessing_cannot_see_future(self):
        from biopac_tft_oct.biopac import causal_biopac_process
        rng = np.random.default_rng(42)
        a = rng.normal(size=(16, 40))
        b = a.copy()
        b[12:] += 1000
        first = causal_biopac_process(a, n_segments=4, max_shift=2)
        second = causal_biopac_process(b, n_segments=4, max_shift=2)
        np.testing.assert_allclose(first.corrected[:12], second.corrected[:12], rtol=0, atol=1e-12)

    def test_hidden_reference_cannot_change_initial_context(self):
        protocol = runner.PredictionProtocol()
        a = np.arange(80, dtype=float).reshape(-1, 1)
        b = a.copy()
        b[10:] = 9999
        np.testing.assert_array_equal(runner.initial_reference_input(a, protocol),
                                      runner.initial_reference_input(b, protocol))
        self.assertTrue((runner.initial_reference_input(a, protocol)[10:] == 0).all())

    def test_future_oct_cannot_change_available_inputs(self):
        a = np.arange(400, dtype=float).reshape(80, 5)
        b = a.copy()
        b[50:] = -9999
        np.testing.assert_array_equal(runner.available_oct(a, 50), runner.available_oct(b, 50))

    def test_diagnostic_padding_uses_only_initial_reference(self):
        protocol = runner.PredictionProtocol()
        a = np.arange(80, dtype=float).reshape(-1, 1)
        b = a.copy()
        b[10:] = 9999
        for policy, expected in (("scaled-zero", 0), ("last-reference", 9),
                                 ("mean-reference", 4.5)):
            first = runner.initial_reference_input(a, protocol, policy)
            second = runner.initial_reference_input(b, protocol, policy)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first[:10], a[:10])
            self.assertTrue((first[10:] == expected).all())
        with self.assertRaises(ValueError):
            runner.initial_reference_input(a, protocol, "unknown")

    def test_density_aggregation_uses_candidates_not_reference(self):
        self.assertAlmostEqual(runner.density_mean([1, 1.05, 1.2, 99], 0.5), 1.025)

    def test_tuning_arguments_keep_protocol_defaults(self):
        args = runner.parse_args(["--data-dir", "data", "--output-dir", "new-output",
            "--learning-rate", "0.0003", "--tft-hidden-size", "32",
            "--epochs", "40", "--reference-padding", "last-reference"])
        self.assertEqual((args.initial_reference, args.input_window, args.horizon), (10, 50, 10))
        self.assertEqual(args.reference_padding, "last-reference")
        defaults = runner.parse_args(["--data-dir", "data", "--output-dir", "new-output"])
        self.assertEqual(defaults.reference_padding, "scaled-zero")
        with self.assertRaises(ValueError):
            runner.parse_args(["--data-dir", "data", "--output-dir", "new-output",
                "--tft-hidden-size", "17"])


if __name__ == "__main__":
    unittest.main()

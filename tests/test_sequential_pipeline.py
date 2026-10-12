import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from biopac_tft_oct.pipeline import load_rule, save_rule, sequential_signal_features, validate_cohorts
from biopac_tft_oct.rule_discovery import DEJGuidedDynamicSignalExtractionRule

spec = importlib.util.spec_from_file_location("runner_sequential_tests", ROOT / "scripts/run_paper_darts.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class SequentialPipelineTests(unittest.TestCase):
    def make_rule(self):
        return DEJGuidedDynamicSignalExtractionRule({"arm": (1., 1., 1., 1., 1.)},
                                                   {"arm": (15., 27., 39., 51., 63.)})

    def test_dynamic_centres_and_prefix_invariance(self):
        z = np.arange(140)
        profiles = np.array([1 + 5*np.exp(-((z-8)/3)**2) + 4*np.exp(-((z-dej)/4)**2)
                             for dej in (35, 37, 39, 41, 43)])
        kwargs = dict(rule=self.make_rule(), site="arm", epidermal_range=(3, 15),
                      dej_range=(25, 50), n_segments=4, max_shift=0, min_history=2)
        features, peaks, centres = sequential_signal_features(profiles, **kwargs)
        self.assertTrue(np.all(centres[1:] > centres[:-1]))
        np.testing.assert_array_equal(peaks[:, 2], peaks[:, 1] - peaks[:, 0])
        changed = profiles.copy()
        changed[3:] *= 20
        shorter = sequential_signal_features(profiles[:3], **kwargs)
        disturbed = sequential_signal_features(changed, **kwargs)
        np.testing.assert_allclose(features[:3], shorter[0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(features[:3], disturbed[0][:3], rtol=0, atol=1e-12)

    def test_rule_roundtrip_and_full_window_width(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rule.json"
            save_rule(self.make_rule(), path, discovery_subjects=["D1"], discovery_sessions=["d1"])
            rule, payload = load_rule(path)
            self.assertEqual(payload["discovery_subjects"], ["D1"])
            self.assertEqual([b-a for a, b in rule.window_ranges(site="arm", dej_depth=30, n_depth=140)], [11]*5)
            with self.assertRaises(FileExistsError):
                save_rule(rule, path, discovery_subjects=[], discovery_sessions=[])
        with self.assertRaises(ValueError):
            self.make_rule().window_ranges(site="arm", dej_depth=120, n_depth=140)

    def test_discovery_participant_cannot_enter_prediction(self):
        manifest = pd.DataFrame({"cohort": ["discovery", "prediction"],
                                 "subject": ["D1", "P1"], "session": ["d.csv", "p.csv"]})
        self.assertEqual(validate_cohorts(manifest)["prediction_sessions"], 1)
        manifest.loc[1, "subject"] = "D1"
        with self.assertRaises(ValueError):
            validate_cohorts(manifest)

    def make_session(self):
        frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=60, freq="min"),
                              "glucose": np.arange(60.) * 10, "reference_observed": True})
        for column in runner.OCT_COLUMNS:
            frame[column] = np.arange(60.)
        for column in runner.STATIC_COLUMNS:
            frame[column] = 1.
        return runner.Session("subject-1/s.csv", "subject-1", frame, "synthetic", 0, 0, 0)

    def test_paired_stress_preserves_time_and_pairing(self):
        source = self.make_session()
        for mode in ("paired-shuffle", "paired-circular"):
            args = SimpleNamespace(test_perturbation=mode, seed=42, perturbation_shift=3)
            stressed = runner.perturb_test_session(source, args)
            np.testing.assert_array_equal(stressed.frame.glucose, stressed.frame.slopmean1 * 10)
            np.testing.assert_array_equal(stressed.frame.elapsed_time, np.arange(60, dtype=np.float32)/240.)
            pd.testing.assert_frame_equal(source.frame[self.static_columns()], stressed.frame[self.static_columns()])
        self.assertNotIn("elapsed_time", source.frame)

    @staticmethod
    def static_columns():
        return ["date", *runner.STATIC_COLUMNS]

    def test_time_shift_preserves_original_physiology(self):
        source = self.make_session()
        args = SimpleNamespace(test_perturbation="time-shift", seed=42, perturbation_shift=3)
        changed = runner.perturb_test_session(source, args)
        pd.testing.assert_frame_equal(changed.frame[source.frame.columns], source.frame)
        np.testing.assert_array_equal(changed.frame.elapsed_time,
                                      np.roll(np.arange(60, dtype=np.float32)/240., 3))

    def test_main_lengths_cannot_change(self):
        for setting in ("--initial-reference", "--input-window", "--horizon"):
            with self.assertRaises(ValueError):
                runner.parse_args(["--data-dir", "data", "--output-dir", "out", setting, "11"])

    @unittest.skipUnless(importlib.util.find_spec("darts"), "Requires training dependencies")
    def test_supplied_time_channel_survives_forecast_preparation(self):
        source = self.make_session()
        args = SimpleNamespace(test_perturbation="time-shift", seed=42, perturbation_shift=3)
        _, _, time = runner.to_series(runner.perturb_test_session(source, args))
        extended = runner.known_time_prefix(time, 70)
        np.testing.assert_array_equal(extended.values()[:60], time.values())
        _, _, no_time = runner.to_series(source, "oct_only")
        self.assertTrue(np.all(runner.known_time_prefix(no_time, 70).values() == 0))


if __name__ == "__main__":
    unittest.main()

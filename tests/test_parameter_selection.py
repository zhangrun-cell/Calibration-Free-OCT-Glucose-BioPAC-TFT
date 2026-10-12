import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd


@unittest.skipUnless(importlib.util.find_spec("torch"), "Training dependencies are optional for protocol-only checks.")
class ParameterSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/tune_subjectwise_fivefold.py"
        spec = importlib.util.spec_from_file_location("parameter_selection_test_module", path)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def test_test_rmse_cannot_rank_candidates(self):
        scores = [
            {"candidate": "validation-winner", "validation_RMSE": 1.8, "test_RMSE": 9.0},
            {"candidate": "test-winner", "validation_RMSE": 2.1, "test_RMSE": 0.1},
        ]
        chosen = self.module.select_candidate(scores)
        self.assertEqual(chosen["candidate"], "validation-winner")

    def test_nonfinite_validation_is_not_selected(self):
        scores = [{"candidate": "invalid", "validation_RMSE": np.nan},
                  {"candidate": "finite", "validation_RMSE": 2.0}]
        self.assertEqual(self.module.select_candidate(scores)["candidate"], "finite")
        with self.assertRaises(ValueError):
            self.module.select_candidate(scores[:1])
        with self.assertRaises(ValueError):
            self.module.select_candidate([])

    def test_cached_candidates_require_identical_protocol_and_sources(self):
        config = {"initial_reference": 10, "input_window": 50, "horizon": 10,
                  "folds": 5, "seed": 42, "torch_threads": 2, "accelerator": "gpu",
                  "run_kind": "full_tuning"}
        manifest = pd.DataFrame([{"fold": 1, "subject": "subject-1", "session": "a.csv",
                                  "role": "train", "sha256": "original"}])
        self.module.verify_reuse_protocol(config, config, manifest, manifest)
        with self.assertRaises(ValueError):
            self.module.verify_reuse_protocol(config, {**config, "initial_reference": 50},
                                              manifest, manifest)
        changed = manifest.copy()
        changed.loc[0, "sha256"] = "changed"
        with self.assertRaises(ValueError):
            self.module.verify_reuse_protocol(config, config, manifest, changed)
        changed = manifest.copy()
        changed.loc[0, "role"] = "test"
        with self.assertRaises(ValueError):
            self.module.verify_reuse_protocol(config, config, manifest, changed)

    def test_cached_parameters_include_base_defaults(self):
        config = {key: None for key in self.module.ALLOWED - {"name"}}
        config.update(batch_size=32, learning_rate=0.001)
        args = SimpleNamespace(**config)
        old = {"name": "baseline", "epochs": 20}
        new = {"name": "baseline", "epochs": 20, "batch_size": 32}
        self.assertTrue(self.module.same_effective_parameters(config, old, args, new))
        self.assertFalse(self.module.same_effective_parameters(config, old, args,
                                                              {**new, "learning_rate": 0.0003}))


if __name__ == "__main__":
    unittest.main()

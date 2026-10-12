"""Verify the trained TFT cannot use hidden glucose or unobserved OCT."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from darts.models import TFTModel

RUNNER = Path(__file__).resolve().with_name("run_paper_darts.py")
spec = importlib.util.spec_from_file_location("paper_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    config = json.loads((args.run_dir / "run_config.json").read_text())
    if config.get("input_source", "full") != "full" or config.get("zero_reference", False) or config.get("test_perturbation", "none") != "none":
        raise ValueError("This causality check requires the unperturbed full-input TFT run.")
    protocol = runner.PredictionProtocol(config["initial_reference"], config["input_window"], config["horizon"])
    torch.set_num_threads(2)
    subject_map = Path(config["subject_map"]) if config.get("subject_map") else None
    sessions = runner.load_sessions(args.data_dir, subject_map)
    folds, _ = runner.build_subject_folds(sessions, seed=config["seed"])
    session = next(s for s in sessions if s.subject in folds[args.fold - 1].test_subjects
                   and len(s.frame) >= protocol.input_chunk_length + args.steps)
    if args.steps < 1:
        raise ValueError("At least one forecast step must be checked.")
    target, oct_cov, time_cov = runner.to_series(session)
    selection_file = args.run_dir / "selected_parameters_by_fold.csv"
    parameters = {}
    if selection_file.exists():
        selected = pd.read_csv(selection_file)
        row = selected.loc[(selected.fold == args.fold) & (selected.model == "TFT")]
        if len(row) != 1:
            raise ValueError("The requested TFT fold has not completed parameter selection.")
        row = row.iloc[0]
        model_path, scaler_path = Path(row.model_path), Path(row.scaler_path)
        parameters = json.loads(row.candidate_parameters)
    else:
        model_path = args.run_dir / f"models/fold{args.fold}_TFT.pkl"
        scaler_path = args.run_dir / f"scalers/fold{args.fold}.pkl"
    scalers = joblib.load(scaler_path)
    target = scalers["target"].transform(target)
    oct_cov = scalers["oct"].transform(oct_cov)
    model = TFTModel.load(str(model_path), map_location="cpu")
    model.trainer_params["accelerator"] = "cpu"
    model.trainer_params["devices"] = 1
    values = target.values(copy=True)
    values[protocol.initial_reference_length:] = 9999
    changed_target = target.with_values(values)
    values = oct_cov.values(copy=True)
    values[protocol.input_chunk_length + args.steps - 1:] = -9999
    changed_oct = oct_cov.with_values(values)
    padding = parameters.get("reference_padding", config.get("reference_padding", "scaled-zero"))
    def forecast(y, x):
        return runner.causal_forecast(model, "TFT", y, x, time_cov, protocol,
                                      max_steps=args.steps, reference_padding=padding).values()
    baseline = forecast(target, oct_cov)
    hidden_target = forecast(changed_target, oct_cov)
    future_oct = forecast(target, changed_oct)
    without_reuse = runner.causal_forecast(model, "TFT", target, oct_cov, time_cov,
                                         protocol, max_steps=args.steps, reference_padding=padding,
                                         reuse_trainer=False).values()
    np.testing.assert_allclose(baseline, hidden_target, atol=1e-6, rtol=0)
    np.testing.assert_allclose(baseline, future_oct, atol=1e-6, rtol=0)
    np.testing.assert_allclose(baseline, without_reuse, atol=1e-6, rtol=0)
    report = {"fold": args.fold, "session": session.file, "model_path": str(model_path),
              "scope": f"first {args.steps} prospective forecasts from the saved TFT checkpoint",
              "hidden_glucose_after_L0_delta": float(np.max(np.abs(baseline-hidden_target))),
              "unobserved_oct_delta": float(np.max(np.abs(baseline-future_oct))),
              "trainer_reuse_delta": float(np.max(np.abs(baseline-without_reuse))), "passed": True}
    (args.run_dir / f"causality_check_fold{args.fold}_steps{args.steps}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

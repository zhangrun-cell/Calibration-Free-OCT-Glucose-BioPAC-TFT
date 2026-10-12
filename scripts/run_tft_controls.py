"""Evaluate held-out stress controls with the same saved full-input TFT models."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import torch
from darts.models import TFTModel

spec = importlib.util.spec_from_file_location("paper_runner_controls", Path(__file__).with_name("run_paper_darts.py"))
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

SCENARIOS = {"baseline": (False, "none"), "zero-reference": (True, "none"),
             "time-shift": (False, "time-shift"), "paired-shuffle": (False, "paired-shuffle"),
             "paired-circular": (False, "paired-circular")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--shift", type=int, default=10)
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--max-prediction-steps", type=int, help="Smoke checks only, not full clinical evaluation.")
    own = parser.parse_args()
    config = json.loads((own.run_dir / "run_config.json").read_text(encoding="utf-8"))
    if config.get("input_source", "full") != "full" or config.get("zero_reference", False) or config.get("test_perturbation", "none") != "none":
        raise ValueError("Controls require models trained on unperturbed full-input trajectories.")
    args = runner.parse_args(["--data-dir", str(own.data_dir), "--output-dir", str(own.output_dir)])
    for key, value in config.items():
        if hasattr(args, key) and key not in {"data_dir", "output_dir"}:
            setattr(args, key, Path(value) if key in {"subject_map", "cohort_manifest", "evaluation_mask"} and value else value)
    args.accelerator = own.accelerator
    args.perturbation_shift = own.shift
    args.max_prediction_steps = own.max_prediction_steps
    runner.validate_training_args(args)
    sessions = runner.load_sessions(args.data_dir, args.subject_map)
    runner.check_cohort_manifest(args, sessions)
    folds, _ = runner.build_subject_folds(sessions, seed=args.seed)
    manifest = pd.read_csv(own.run_dir / "fold_manifest.csv", dtype={"subject": str})
    for session in sessions:
        rows = manifest.loc[manifest.session.eq(session.file)]
        if rows.empty or not rows.sha256.eq(session.source_sha256).all() or not rows.subject.eq(session.subject).all():
            raise ValueError("Source data or participant identities changed since model training.")
    if set(manifest.session) != {s.file for s in sessions}:
        raise ValueError("The source run and controls must use the same sessions.")
    for fold in folds:
        held_out = set(manifest.loc[(manifest.fold == fold.fold_index + 1) & manifest.role.eq("test"), "subject"])
        if held_out != set(fold.test_subjects):
            raise ValueError("The saved run and control folds differ.")
    if own.output_dir.exists():
        raise FileExistsError(own.output_dir)
    own.output_dir.mkdir(parents=True)
    protocol = runner.PredictionProtocol()
    selection_file = own.run_dir / "selected_parameters_by_fold.csv"
    selections = pd.read_csv(selection_file) if selection_file.exists() else None
    torch.set_num_threads(args.torch_threads)
    for scenario in own.scenarios:
        output = own.output_dir / scenario
        output.mkdir()
        arguments = copy.deepcopy(args)
        arguments.zero_reference, arguments.test_perturbation = SCENARIOS[scenario]
        all_points, all_candidates = [], []
        for fold in folds:
            number = fold.fold_index + 1
            selected_args = copy.deepcopy(arguments)
            if selections is not None:
                row = selections.loc[(selections.fold == number) & selections.model.eq("TFT")]
                if len(row) != 1:
                    raise ValueError("All five selected full-input TFT models are required.")
                row = row.iloc[0]
                for key, value in json.loads(row.candidate_parameters).items():
                    if key != "name":
                        setattr(selected_args, key, value)
                model_path, scaler_path = Path(row.model_path), Path(row.scaler_path)
            else:
                model_path = own.run_dir / f"models/fold{number}_TFT.pkl"
                scaler_path = own.run_dir / f"scalers/fold{number}.pkl"
            scalers = joblib.load(scaler_path)
            training = [s for s in sessions if s.subject in fold.train_subjects and len(s.frame) >= 60]
            y = [scalers["target"].transform(runner.to_series(s)[0]).values(copy=False) for s in training]
            import numpy as np
            fold_data = {"target_scaler": scalers["target"], "oct_scaler": scalers["oct"],
                         "population_reference_scaled": float(np.mean(np.concatenate(y)))}
            model = TFTModel.load(str(model_path), map_location="cpu")
            model.trainer_params["accelerator"] = own.accelerator
            model.trainer_params["devices"] = 1
            test = [s for s in sessions if s.subject in fold.test_subjects]
            rows, candidates, _ = runner.predict_sessions(model, "TFT", test, fold_data,
                                                          protocol, selected_args, number)
            all_points.extend(rows)
            all_candidates.extend(candidates)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report_sessions = [runner.perturb_test_session(s, arguments) for s in sessions]
        runner.write_prediction_reports(pd.DataFrame(all_points), report_sessions, output,
            expected_models=["TFT"], evaluation_mask=args.evaluation_mask,
            bootstrap_repetitions=args.bootstrap_repetitions, seed=args.seed)
        pd.DataFrame(all_candidates).to_csv(output / "horizon_candidates_scaled.csv", index=False)
        details = {"scenario": scenario, "trained_run": str(own.run_dir), "test_only_change": True,
                   "time_unchanged_in_paired_stress": True, "oct_reference_pairing_preserved": True,
                   "shift": own.shift, "max_prediction_steps": own.max_prediction_steps,
                   "source_manifest_sha256": hashlib.sha256((own.run_dir / "fold_manifest.csv").read_bytes()).hexdigest()}
        (output / "control_contract.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        print(f"Completed held-out control: {scenario}", flush=True)


if __name__ == "__main__":
    main()

"""Select parameters inside each outer subject fold; test its winner only once.

The outer test participants are never used to rank a candidate or choose a
reference-padding policy. These are prospective new results, not a mechanism
for forcing the archived manuscript metrics.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import torch

RUNNER = Path(__file__).resolve().with_name("run_paper_darts.py")
spec = importlib.util.spec_from_file_location("paper_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

ALLOWED = {"name", "epochs", "batch_size", "learning_rate", "tft_hidden_size",
           "tft_lstm_layers", "tft_attention_heads", "dropout",
           "gradient_clip", "scheduler_patience", "tft_quantiles", "reference_padding",
           "target_scaling"}


def select_candidate(validation_scores):
    if not validation_scores:
        raise ValueError("No validation scores.")
    finite = [row for row in validation_scores if np.isfinite(row["validation_RMSE"])]
    if not finite:
        raise ValueError("Every candidate produced a nonfinite validation score.")
    return min(finite, key=lambda row: row["validation_RMSE"])


def verify_reuse_protocol(previous, current, previous_manifest, current_manifest):
    for key in ("initial_reference", "input_window", "horizon", "folds", "seed",
                "torch_threads", "accelerator", "max_prediction_steps",
                "test_sessions_per_fold", "input_source", "zero_reference", "test_perturbation"):
        if previous.get(key) != current.get(key):
            raise ValueError(f"Cached-run protocol differs: {key}")
    if previous.get("run_kind") != "full_tuning":
        raise ValueError("Only a completed full tuning run can supply cached candidates.")
    keys = ["fold", "subject", "session", "role", "sha256"]
    def normalized(frame):
        return frame[keys].sort_values(["fold", "session"]).reset_index(drop=True)
    if not normalized(previous_manifest).equals(normalized(current_manifest)):
        raise ValueError("Cached-run source hashes or subject folds differ.")


def same_effective_parameters(previous_config, previous_candidate, current_args, candidate):
    return all(previous_candidate.get(key, previous_config.get(key)) ==
               candidate.get(key, getattr(current_args, key, None))
               for key in ALLOWED - {"name"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-space", type=Path, required=True)
    parser.add_argument("--reuse-candidates-from", type=Path,
                        help="Reuse exact candidate checkpoints and validation scores from an audited run.")
    own, remaining = parser.parse_known_args()
    args = runner.parse_args(remaining)
    if args.only_fold is not None or args.dry_run:
        raise ValueError("This tuning entry point always executes all five outer folds.")
    if args.test_perturbation != "none" or args.zero_reference:
        raise ValueError("Tune the unperturbed model before testing perturbations.")
    space = json.loads(own.search_space.read_text(encoding="utf-8"))
    for name in args.models:
        if not space.get(name):
            raise ValueError(f"No candidates for {name}.")
        for candidate in space[name]:
            if set(candidate) - ALLOWED or not candidate.get("name"):
                raise ValueError(f"Invalid candidate: {candidate}")
            if any(c in candidate["name"] for c in '/\\:'):
                raise ValueError("Candidate names must be safe filename components.")
        if len({c["name"] for c in space[name]}) != len(space[name]):
            raise ValueError("Candidate names must be unique within each model.")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    sessions = runner.load_sessions(args.data_dir, args.subject_map)
    cohort_counts = runner.check_cohort_manifest(args, sessions)
    folds, linked = runner.build_subject_folds(sessions, seed=args.seed)
    protocol = runner.PredictionProtocol(args.initial_reference, args.input_window, args.horizon, args.folds)
    args.output_dir.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(search_space=space, outer_folds=5, selection_metric="validation-only pooled RMSE",
                  hyperparameters_selected="separately inside each outer fold",
                  observed_subject_groups=len({s.subject for s in sessions}),
                  observed_sessions=len(sessions), cohort_counts=cohort_counts,
                  run_kind="functional_tuning_smoke" if args.test_sessions_per_fold or args.max_prediction_steps or any(c.get("epochs", args.epochs) < 20 for name in args.models for c in space[name]) else "full_tuning",
                  repeated_oct_sessions_kept_together=linked,
                  test_metrics_used_for_selection=False,
                  reuse_candidates_from=str(own.reuse_candidates_from) if own.reuse_candidates_from else None)
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = []
    for fold in folds:
        for session in sessions:
            role = "test" if session.subject in fold.test_subjects else "validation" if session.subject in fold.validation_subjects else "train"
            manifest.append({"fold": fold.fold_index + 1, "subject": session.subject,
                             "session": session.file, "role": role, "sha256": session.source_sha256})
    pd.DataFrame(manifest).to_csv(args.output_dir / "fold_manifest.csv", index=False, encoding="utf-8-sig")
    previous_config, cached_scores = None, []
    if own.reuse_candidates_from:
        previous_dir = own.reuse_candidates_from
        audit = json.loads((previous_dir / "completed_run_audit.json").read_text(encoding="utf-8"))
        if not audit.get("passed") or not audit.get("parameters_selected_on_validation_only"):
            raise ValueError("Cached candidates require a successful completed-run audit.")
        previous_config = json.loads((previous_dir / "run_config.json").read_text(encoding="utf-8"))
        verify_reuse_protocol(previous_config, config,
            pd.read_csv(previous_dir / "fold_manifest.csv", encoding="utf-8-sig"), pd.DataFrame(manifest))
        cached_scores = pd.read_csv(previous_dir / "validation_candidate_scores.csv").to_dict("records")
        print("Verified cached run: identical source hashes, subject folds and inference protocol.", flush=True)
    all_points, all_candidates, excluded, scores, selections = [], [], [], [], []
    for fold in folds:
        number = fold.fold_index + 1
        fold_dir = args.output_dir / f"fold{number}"
        fold_dir.mkdir()
        data_by_scaling = {}
        def get_fold_data(parameters):
            key = parameters.target_scaling
            if key not in data_by_scaling:
                data_by_scaling[key] = runner.prepare_fold(sessions, fold, parameters)
            return data_by_scaling[key]
        test = [s for s in sessions if s.subject in fold.test_subjects]
        if args.test_sessions_per_fold:
            test = test[:args.test_sessions_per_fold]
        for name in args.models:
            candidate_scores = []
            for candidate in space[name]:
                candidate_args = copy.deepcopy(args)
                for key, value in candidate.items():
                    if key != "name":
                        setattr(candidate_args, key, value)
                runner.validate_training_args(candidate_args)
                fold_data = get_fold_data(candidate_args)
                cached = next((row for row in cached_scores if row["fold"] == number and
                    row["model"] == name and row["candidate"] == candidate["name"] and
                    same_effective_parameters(previous_config, json.loads(row["candidate_parameters"]),
                                              args, candidate)), None)
                if cached is not None:
                    path = fold_dir / f"{name}_{candidate['name']}.pkl"
                    source_path = Path(cached["model_path"])
                    scaler_path = fold_dir / f"{name}_{candidate['name']}_scalers.pkl"
                    shutil.copy2(source_path, path)
                    if name != "XGBoost":
                        shutil.copy2(Path(str(source_path) + ".ckpt"), Path(str(path) + ".ckpt"))
                    shutil.copy2(Path(cached["scaler_path"]), scaler_path)
                    record = {**cached, "model_path": str(path), "scaler_path": str(scaler_path),
                              "reused_from": str(own.reuse_candidates_from)}
                    candidate_scores.append(record)
                    scores.append(record)
                    pd.DataFrame(scores).to_csv(args.output_dir / "validation_candidate_scores.csv", index=False)
                    model_class = type(runner.create_model(name, candidate_args, args.seed))
                    print(f"Fold {number}/5 {name}: preserved candidate {candidate['name']}; "
                          f"validation RMSE={cached['validation_RMSE']:.6f}", flush=True)
                    continue
                print(f"Fold {number}/5 {name}: training candidate {candidate['name']}", flush=True)
                model = runner.fit_fold_model(name, candidate_args, fold_data)
                validation_args = copy.deepcopy(candidate_args)
                validation_args.export_candidates = False
                val_points, _, _ = runner.predict_sessions(model, name, fold_data["validation"],
                    fold_data, protocol, validation_args, number)
                val_frame = pd.DataFrame(val_points)
                val_frame = val_frame.loc[val_frame.reference_observed]
                if val_frame.empty:
                    raise ValueError("No prospective validation points.")
                if not np.isfinite(val_frame.predicted).all():
                    raise ValueError("A candidate has nonfinite validation predictions; do not score a smaller subset.")
                if candidate_scores and len(val_frame) != candidate_scores[0]["n_validation_points"]:
                    raise ValueError("Candidates must use the same physical validation points.")
                score = runner.regression_metrics(val_frame.actual_glucose.to_numpy(), val_frame.predicted.to_numpy())["RMSE"]
                path = fold_dir / f"{name}_{candidate['name']}.pkl"
                model.save(str(path))
                scaler_path = fold_dir / f"{name}_{candidate['name']}_scalers.pkl"
                joblib.dump({"target": fold_data["target_scaler"], "oct": fold_data["oct_scaler"]}, scaler_path)
                record = {"fold": number, "model": name, "candidate": candidate["name"],
                          "validation_RMSE": score, "n_validation_points": len(val_frame),
                          "candidate_parameters": json.dumps(candidate), "model_path": str(path),
                          "scaler_path": str(scaler_path)}
                candidate_scores.append(record)
                scores.append(record)
                pd.DataFrame(scores).to_csv(args.output_dir / "validation_candidate_scores.csv", index=False)
                print(f"  Validation-only RMSE={score:.6f}, points={len(val_frame)}", flush=True)
                model_class = type(model)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            winner = select_candidate(candidate_scores)
            selected = next(c for c in space[name] if c["name"] == winner["candidate"])
            selected_args = copy.deepcopy(args)
            for key, value in selected.items():
                if key != "name":
                    setattr(selected_args, key, value)
            model = model_class.load(winner["model_path"])
            fold_data = get_fold_data(selected_args).copy()
            scalers = joblib.load(winner["scaler_path"])
            fold_data.update(target_scaler=scalers["target"], oct_scaler=scalers["oct"])
            selections.append(winner)
            pd.DataFrame(selections).to_csv(args.output_dir / "selected_parameters_by_fold.csv", index=False)
            print(f"Fold {number}/5 {name}: locked winner {winner['candidate']}; evaluating outer test", flush=True)
            rows, candidates, skipped = runner.predict_sessions(model, name, test, fold_data,
                                                               protocol, selected_args, number)
            all_points.extend(rows)
            all_candidates.extend(candidates)
            excluded.extend(skipped)
            pd.DataFrame(all_points).to_csv(args.output_dir / "predictions_long.csv", index=False, encoding="utf-8-sig")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    metrics = runner.write_prediction_reports(pd.DataFrame(all_points), sessions, args.output_dir,
        expected_models=args.models, evaluation_mask=args.evaluation_mask,
        bootstrap_repetitions=args.bootstrap_repetitions, seed=args.seed)
    pd.DataFrame(excluded, columns=["fold", "model", "session", "reason"]).to_csv(
        args.output_dir / "excluded_sessions.csv", index=False)
    if all_candidates:
        pd.DataFrame(all_candidates).to_csv(args.output_dir / "horizon_candidates_scaled.csv", index=False)
    print(metrics[["model", "n_points", "RMSE", "A_plus_B_pct"]].to_string(index=False), flush=True)
    print("All five outer folds completed; test metrics were not used for candidate selection.", flush=True)


if __name__ == "__main__":
    main()

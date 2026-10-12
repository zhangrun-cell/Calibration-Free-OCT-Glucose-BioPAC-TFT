"""Audit completed prediction runs without selecting parameters on test errors."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(run_dir, data_dir):
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    contract = json.loads((run_dir / "evaluation_contract.json").read_text(encoding="utf-8"))
    folds = pd.read_csv(run_dir / "fold_manifest.csv", encoding="utf-8-sig")
    predictions = pd.read_csv(run_dir / "predictions_long.csv", encoding="utf-8-sig")
    metrics = pd.read_csv(run_dir / "pooled_metrics.csv", encoding="utf-8-sig")
    selected = pd.read_csv(run_dir / "selected_parameters_by_fold.csv")
    candidates = pd.read_csv(run_dir / "validation_candidate_scores.csv")
    require((config["initial_reference"], config["input_window"], config["horizon"]) == (10, 50, 10),
            "Unexpected L0/W/H settings.")
    require(contract["main_offset_minutes"] == 0 and not contract["individual_offset_optimization"],
            "The primary results must not use individual delay fitting.")
    require(set(folds.fold) == set(range(1, 6)), "Five outer folds are required.")
    require(not folds.duplicated(["fold", "session"]).any(), "Repeated session roles.")
    require(folds.groupby(["fold", "subject"]).role.nunique().eq(1).all(),
            "A participant crosses roles within an outer fold.")
    test = folds.loc[folds.role.eq("test")]
    require(test.session.nunique() == folds.session.nunique(), "Some sessions were never held out.")
    require(test.groupby("session").size().eq(1).all(), "A session was tested more than once.")
    require(test.groupby("subject").fold.nunique().eq(1).all(), "A participant crosses outer test folds.")
    for pair in config.get("repeated_oct_sessions_kept_together", []):
        role_counts = folds.loc[folds.subject.isin(pair["subjects"])].groupby("fold").role.nunique()
        require(role_counts.eq(1).all(), "Repeated OCT inputs cross training/validation/test roles.")
    for row in folds.drop_duplicates("session").itertuples():
        path = data_dir / row.session
        require(path.is_file(), f"Missing source: {path}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256,
                f"Source changed after training began: {path}")
    models = config["models"]
    require(set(metrics.model) == set(models), "Missing model summary.")
    require(len(selected) == 5 * len(models), "Parameter selection is incomplete.")
    for choice in selected.itertuples():
        scores = candidates.loc[(candidates.fold == choice.fold) & (candidates.model == choice.model)]
        require(np.isclose(choice.validation_RMSE, scores.validation_RMSE.min()),
                "Selected model does not have the lowest validation-only RMSE.")
    require(not predictions.duplicated(["model", "session", "row_index"]).any(),
            "Repeated model predictions.")
    require(predictions.row_index.ge(50).all(), "A forecast precedes the input-window boundary.")
    identity = test.set_index("session").fold
    require(predictions.session.map(identity).eq(predictions.fold).all(),
            "Predictions do not come from the assigned held-out fold.")
    flags = predictions.included.astype(str).str.lower().isin(["true", "1"])
    included = predictions.loc[flags]
    require(np.isfinite(included[["actual_glucose", "predicted"]]).all().all(),
            "Included predictions contain nonfinite values.")
    require(included.groupby(["session", "row_index"]).model.nunique().eq(len(models)).all(),
            "Models do not use a common physical-point mask.")
    for row in metrics.itertuples():
        points = included.loc[included.model.eq(row.model)]
        rmse = float(np.sqrt(np.mean((points.predicted - points.actual_glucose) ** 2)))
        require(np.isclose(row.RMSE, rmse, atol=1e-10), "Summary RMSE is not independently reproduced.")
        require(row.n_points == len(points), "Summary denominator disagrees with predictions.")
    horizon_path = run_dir / "horizon_candidates_scaled.csv"
    if horizon_path.exists():
        horizons = pd.read_csv(horizon_path)
        require(horizons.latest_observed_oct_row_index.eq(horizons.origin_row_index).all(),
                "Candidate OCT boundary differs from forecast origin.")
        require(horizons.target_row_index.gt(horizons.origin_row_index).all(),
                "A candidate target is not in the forecast horizon.")
        require((horizons.target_row_index - horizons.origin_row_index).eq(horizons.horizon).all(),
                "Candidate horizon index is inconsistent.")
    environment = {"python": platform.python_version(), "platform": platform.platform()}
    for distribution in ["u8darts", "torch", "pytorch-lightning", "numpy", "pandas", "scipy", "scikit-learn", "xgboost"]:
        try:
            environment[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            environment[distribution] = None
    distribution = importlib.metadata.distribution("u8darts")
    library_files = [p for p in distribution.files if p.parts[0] == "darts"
                     and p.suffix == ".py" and p.hash and p.hash.mode == "sha256"]
    modified = [str(p) for p in library_files if base64.urlsafe_b64encode(
        hashlib.sha256(distribution.locate_file(p).read_bytes()).digest()).decode().rstrip("=") != p.hash.value]
    environment["darts_python_files_checked_against_installation_record"] = len(library_files)
    environment["darts_modified_python_files"] = modified
    report = {
        "passed": True, "scope": "completed five-fold prediction run on precomputed OCT features",
        "audit_signoff": "protocol and output integrity only; not manuscript accuracy reproduction",
        "subjects_retained": int(folds.subject.nunique()), "sessions_retained": int(folds.session.nunique()),
        "tested_once_per_session": True, "participant_roles_disjoint": True,
        "source_csv_hashes_unchanged": True, "parameters_selected_on_validation_only": True,
        "first_forecast_after_W": 50, "observed_reference_history_L0": 10,
        "evaluation_points": int(contract["n_evaluable_physical_points"]),
        "identity_note": "Folder identities supplied by the author; identical OCT-input pairs kept in the same fold without merging participants.",
        "optical_preprocessing_reproduction_verified": False,
        "discovery_cohort_reproduction_verified": False, "environment": environment,
    }
    (run_dir / "completed_run_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.run_dir, arguments.data_dir)

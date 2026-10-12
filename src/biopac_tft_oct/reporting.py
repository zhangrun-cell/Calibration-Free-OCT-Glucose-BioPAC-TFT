"""Locked-mask, session-level and participant-cluster prediction reporting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import clarke_percentages, regression_metrics

KEYS = ["session", "row_index"]


def metric_record(group):
    y, p = group["actual_glucose"].to_numpy(), group["predicted"].to_numpy()
    return {"n_points": len(group), "n_sessions": group.session.nunique(),
            "n_subjects": group.subject.nunique(), **regression_metrics(y, p),
            **clarke_percentages(y, p)}


def common_evaluation_mask(points, expected_models, evaluation_mask=None):
    """Require finite predictions from every requested model at a physical point.

    An optional locked external mask only restricts these points. Its membership
    is never inferred from the prediction error or the reference glucose level.
    """
    frame = points.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if frame.duplicated(["model", *KEYS]).any():
        raise ValueError("Duplicate model predictions for the same physical point.")
    if set(frame.model.unique()) != set(expected_models):
        raise ValueError("Prediction reports require all requested models.")
    counts = frame.groupby(KEYS)["model"].nunique()
    finite = np.isfinite(frame.actual_glucose) & np.isfinite(frame.predicted)
    valid = pd.Series(finite & frame.reference_observed.astype(bool), index=frame.index)
    group_valid = valid.groupby([frame[k] for k in KEYS]).all()
    common = counts.eq(len(expected_models)) & group_valid
    keys = pd.MultiIndex.from_frame(frame[KEYS])
    frame["included"] = common.reindex(keys, fill_value=False).to_numpy()
    if frame.groupby(KEYS)["actual_glucose"].nunique(dropna=False).gt(1).any():
        raise ValueError("Models disagree on the reference for a physical point.")
    if evaluation_mask is not None:
        locked = pd.read_csv(evaluation_mask, encoding="utf-8-sig")
        if not {"session", "timestamp"}.issubset(locked.columns):
            raise ValueError("Locked mask requires session and timestamp columns.")
        locked["timestamp"] = pd.to_datetime(locked["timestamp"])
        locked["session"] = locked["session"].str.replace("\\", "/", regex=False)
        if locked.duplicated(["session", "timestamp"]).any():
            raise ValueError("Duplicate physical points in the locked mask.")
        if "included" in locked:
            flags = locked["included"].astype(str).str.lower()
            if not flags.isin(["true", "false", "1", "0"]).all():
                raise ValueError("Locked mask included values must be true/false or 1/0.")
            locked = locked.loc[flags.isin(["true", "1"])]
        locked_keys = pd.MultiIndex.from_frame(locked[["session", "timestamp"]])
        frame["included"] &= pd.MultiIndex.from_frame(frame[["session", "timestamp"]]).isin(locked_keys)
    return frame


def subject_cluster_intervals(points, repetitions=1000, seed=42):
    """Resample complete participant trajectories, retaining repeat multiplicity."""
    columns = ["model", "metric", "estimate", "ci_lower", "ci_upper", "confidence_level",
               "n_subjects", "bootstrap_repetitions", "finite_replicates"]
    rows = []
    if repetitions == 0:
        return pd.DataFrame(columns=columns)
    for name, group in points.groupby("model", sort=True):
        subjects = sorted(group.subject.unique())
        if len(subjects) < 2:
            continue
        arrays = {s: group.loc[group.subject.eq(s), ["actual_glucose", "predicted"]].to_numpy()
                  for s in subjects}
        estimates = metric_record(group)
        samples = {key: [] for key in ["RMSE", "MAE", "MARD_pct", "R2", "ME", "A_plus_B_pct"]}
        rng = np.random.default_rng(seed)
        for _ in range(repetitions):
            draw = rng.choice(subjects, size=len(subjects), replace=True)
            values = np.concatenate([arrays[s] for s in draw], axis=0)
            metrics = {**regression_metrics(values[:, 0], values[:, 1]),
                       **clarke_percentages(values[:, 0], values[:, 1])}
            for key in samples:
                samples[key].append(metrics[key])
        for key, values in samples.items():
            finite = np.asarray(values)[np.isfinite(values)]
            lower, upper = np.quantile(finite, [0.025, 0.975]) if len(finite) else [np.nan, np.nan]
            rows.append({"model": name, "metric": key, "estimate": estimates[key],
                         "ci_lower": lower, "ci_upper": upper, "confidence_level": 0.95,
                         "n_subjects": len(subjects), "bootstrap_repetitions": repetitions,
                         "finite_replicates": len(finite)})
    return pd.DataFrame(rows, columns=columns)


def fixed_offset_report(points, sessions, offsets=range(-30, 31)):
    """Compare target y[index + offset]; no per-session offset selection."""
    lookup = {s.file: s.frame for s in sessions}
    rows = []
    for name, group in points.groupby("model", sort=True):
        for offset in offsets:
            references, predictions = [], []
            for session_name, trajectory in group.groupby("session"):
                source = lookup[session_name]
                index = trajectory.row_index.to_numpy(dtype=int) + offset
                within = (index >= 0) & (index < len(source))
                index, predicted = index[within], trajectory.predicted.to_numpy()[within]
                observed = source.iloc[index].reference_observed.to_numpy(dtype=bool)
                references.extend(source.iloc[index].glucose.to_numpy()[observed])
                predictions.extend(predicted[observed])
            rows.append({"model": name, "global_offset_minutes": offset,
                         "n_points": len(references),
                         **regression_metrics(np.asarray(references), np.asarray(predictions))})
    return pd.DataFrame(rows)


def write_prediction_reports(points, sessions, output_dir, *, expected_models,
                             evaluation_mask=None, bootstrap_repetitions=1000, seed=42):
    output_dir = Path(output_dir)
    audit = common_evaluation_mask(points, expected_models, evaluation_mask)
    audit.to_csv(output_dir / "predictions_long.csv", index=False, encoding="utf-8-sig")
    audit[["fold", "subject", "session", "timestamp", "row_index", "included"]].drop_duplicates(
        KEYS).to_csv(output_dir / "evaluation_mask.csv", index=False, encoding="utf-8-sig")
    included = audit.loc[audit.included].copy()
    if included.empty:
        raise ValueError("No points remain under the common evaluation mask.")
    summary = pd.DataFrame([{"model": name, **metric_record(group)}
                            for name, group in included.groupby("model", sort=True)])
    summary.to_csv(output_dir / "pooled_metrics.csv", index=False, encoding="utf-8-sig")
    for level in ["fold", "session", "subject"]:
        records = []
        for (name, key), group in included.groupby(["model", level], sort=True):
            records.append({"model": name, level: key, **metric_record(group)})
        pd.DataFrame(records).to_csv(output_dir / f"{level}_metrics.csv", index=False, encoding="utf-8-sig")
    subject_cluster_intervals(included, bootstrap_repetitions, seed).to_csv(
        output_dir / "subject_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")
    fixed_offset_report(included, sessions).to_csv(output_dir / "fixed_global_offset_metrics.csv",
                                                  index=False, encoding="utf-8-sig")
    tested_sessions = set(audit.session.unique())
    for session in sessions:
        if session.file not in tested_sessions:
            continue
        frame = session.frame[["date", "glucose", "reference_observed"]].copy()
        frame["row_index"] = np.arange(len(frame))
        current = audit.loc[audit.session.eq(session.file)]
        for name in expected_models:
            values = current.loc[current.model.eq(name)].set_index("row_index").predicted
            frame[f"{name}_pred"] = frame.row_index.map(values)
        mask = current.drop_duplicates("row_index").set_index("row_index").included
        frame["included_in_metrics"] = frame.row_index.map(mask).eq(True)
        destination = output_dir / "per_session_predictions" / session.file
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
    status = {"main_offset_minutes": 0, "individual_offset_optimization": False,
              "prediction_time_loss": "first W points have no forecast",
              "mask_policy": "finite common-model predictions and observed references; optional locked mask",
              "locked_mask": str(evaluation_mask) if evaluation_mask else None,
              "expected_models": expected_models, "n_evaluable_physical_points": int(len(included) // len(expected_models)),
              "bootstrap_unit": "participant; all sessions and minutes retained together"}
    (output_dir / "evaluation_contract.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return summary

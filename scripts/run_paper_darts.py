"""Subject-grouped Darts training with ten reference points and causal OCT inputs.

CSV inputs are precomputed Bio-PAC/DEJ features. This runner does not claim that
static CSV features reproduce online Bio-PAC or reconstruct missing discovery
coefficients. Every run writes to a new directory and reports actual counts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biopac_tft_oct.metrics import clarke_percentages, regression_metrics
from biopac_tft_oct.protocol import PredictionProtocol, subjectwise_fivefold_splits
from biopac_tft_oct.aggregation import dense_candidate_average
from biopac_tft_oct.reporting import write_prediction_reports
from biopac_tft_oct.controls import INPUT_SOURCE_SETTINGS, paired_physiological_trajectory_stress_test
from biopac_tft_oct.pipeline import validate_cohorts

OCT_COLUMNS = [f"slopmean{i}" for i in range(1, 6)]
STATIC_COLUMNS = ["age", "gender", "diabetic_healthy", "diabetic_t1", "diabetic_t2", "finger", "arm"]


@dataclass
class Session:
    file: str
    subject: str
    frame: pd.DataFrame
    source_sha256: str
    dropped_unpaired_rows: int
    forward_filled_oct_values: int
    inserted_minutes: int


def parse_observation_times(values: pd.Series) -> pd.Series:
    """Read text dates and Excel day serials without interpreting them as nanoseconds."""
    numeric = pd.to_numeric(values, errors="coerce")
    excel_serial = numeric.between(20000, 80000)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    parsed.loc[~excel_serial] = pd.to_datetime(
        values.loc[~excel_serial], format="mixed", errors="coerce")
    # Excel floating-point day serials can leave sub-second conversion noise.
    parsed.loc[excel_serial] = pd.to_datetime(
        numeric.loc[excel_serial], unit="D", origin="1899-12-30", errors="coerce").dt.round("s")
    return parsed


def load_sessions(root: Path, subject_map: Path | None = None) -> list[Session]:
    sessions = []
    for path in sorted(root.glob("subject-*/*.csv")):
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(path, encoding="gb18030")
        required = ["date", "glucose", *OCT_COLUMNS, *STATIC_COLUMNS]
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {missing}")
        frame = frame[required].copy()
        frame["date"] = parse_observation_times(frame["date"])
        frame["glucose"] = pd.to_numeric(frame["glucose"], errors="coerce")
        n_raw = len(frame)
        frame = frame.loc[frame["date"].notna() & np.isfinite(frame["glucose"])].reset_index(drop=True)
        dropped = n_raw - len(frame)
        if frame.empty:
            raise ValueError(f"No valid time/reference pairs in {path}; do not silently discard the session.")
        if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
            raise ValueError(f"Duplicate or unordered timestamps in {path}")
        frame["reference_observed"] = True
        minute_index = pd.date_range(frame["date"].min(), frame["date"].max(), freq="1min")
        inserted = len(minute_index) - len(frame)
        frame = frame.set_index("date").reindex(minute_index)
        frame.index.name = "date"
        frame["reference_observed"] = frame["reference_observed"].eq(True)
        frame["glucose"] = frame["glucose"].interpolate(method="time")
        frame[STATIC_COLUMNS] = frame[STATIC_COLUMNS].ffill()
        frame = frame.reset_index()
        for col in required[1:]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        missing_oct = int(frame[OCT_COLUMNS].isna().sum().sum())
        frame[OCT_COLUMNS] = frame[OCT_COLUMNS].ffill()
        if not np.isfinite(frame[OCT_COLUMNS].to_numpy()).all():
            raise ValueError(f"Missing initial OCT features in {path}; no future-value fill is allowed.")
        if not np.isfinite(frame["glucose"].to_numpy()).all():
            raise ValueError(f"Missing reference values in {path}; supply the locked analysis table.")
        delta = frame["date"].diff().dropna()
        if len(delta) and not (delta == pd.Timedelta(minutes=1)).all():
            raise ValueError(f"Non-minute timestamps in {path}; no silent interpolation is performed.")
        for col in STATIC_COLUMNS:
            if frame[col].nunique(dropna=False) != 1 or frame[col].isna().any():
                raise ValueError(f"Static covariate {col} varies or is missing in {path}.")
        sessions.append(Session(path.relative_to(root).as_posix(), path.parent.name,
                                frame, hashlib.sha256(path.read_bytes()).hexdigest(), dropped, missing_oct, inserted))
    if not sessions:
        raise ValueError(f"No session CSV files in {root}")
    if subject_map is not None:
        mapping = pd.read_csv(subject_map, encoding="utf-8-sig", dtype=str)
        if not {"session", "subject"}.issubset(mapping.columns):
            raise ValueError("Subject map requires session and subject columns.")
        if mapping[["session", "subject"]].isna().any().any():
            raise ValueError("Subject map contains missing identifiers.")
        mapping["session"] = mapping["session"].str.replace("\\", "/", regex=False)
        if mapping["session"].duplicated().any():
            raise ValueError("Each session must occur exactly once in the subject map.")
        lookup = mapping.set_index("session")["subject"].to_dict()
        if set(lookup) != {s.file for s in sessions}:
            raise ValueError("Subject map must match the supplied prediction sessions exactly.")
        for session in sessions:
            session.subject = lookup[session.file]
    return sessions


def build_subject_folds(sessions, seed=42):
    """Keep participants and exactly repeated optical trajectories in one fold."""
    subjects = sorted({s.subject for s in sessions})
    parents = {s: s for s in subjects}

    def find(subject):
        while parents[subject] != subject:
            parents[subject] = parents[parents[subject]]
            subject = parents[subject]
        return subject

    repeated = {}
    for session in sessions:
        data = session.frame[["date", *OCT_COLUMNS]]
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(data, index=False).values.tobytes()).hexdigest()
        repeated.setdefault(fingerprint, []).append(session)
    linked = []
    for group in repeated.values():
        members = sorted({s.subject for s in group})
        if len(members) > 1:
            for subject in members[1:]:
                parents[find(subject)] = find(members[0])
            linked.append({"subjects": members, "sessions": [s.file for s in group],
                           "reason": "identical timestamp and five-window OCT sequence"})
    folded_groups = list(subjectwise_fivefold_splits([find(s) for s in subjects], seed=seed))
    from biopac_tft_oct.protocol import SubjectFold
    folds = []
    for fold in folded_groups:
        roles = [tuple(s for s in subjects if find(s) in members) for members in
                 (fold.train_subjects, fold.validation_subjects, fold.test_subjects)]
        folds.append(SubjectFold(fold.fold_index, *roles))
    return folds, linked


def initial_reference_input(values: np.ndarray, protocol: PredictionProtocol,
                            padding: str = "scaled-zero") -> np.ndarray:
    """Fill the W-length target context with only L0 reference samples.

    The default retains the recorded trial's scaled-zero padding. The other
    policies are diagnostic only and derive padding from the first L0 points.
    No policy uses hidden reference glucose after L0.
    """
    if len(values) < protocol.input_chunk_length:
        raise ValueError("Insufficient context for the model input window.")
    if padding == "scaled-zero":
        fill = 0.0
    elif padding == "last-reference":
        fill = float(np.asarray(values[protocol.initial_reference_length - 1]).item())
    elif padding == "mean-reference":
        fill = float(np.mean(values[:protocol.initial_reference_length]))
    else:
        raise ValueError(f"Unknown initial reference padding: {padding}")
    result = np.full((protocol.input_chunk_length, 1), fill, dtype=np.float32)
    result[:protocol.initial_reference_length] = values[:protocol.initial_reference_length]
    return result


def available_oct(values: np.ndarray, stop: int) -> np.ndarray:
    """Only observed OCT up to the forecast origin can reach the predictor."""
    return np.asarray(values[:stop], dtype=np.float32).copy()


def density_mean(values, keep_ratio: float):
    return dense_candidate_average(values, keep_ratio)


def create_model(name, args, seed):
    import torch
    from darts.models import TFTModel
    from darts.utils.likelihood_models import QuantileRegression
    trainer = {"accelerator": args.accelerator, "enable_progress_bar": False,
               "logger": False, "enable_checkpointing": False, "enable_model_summary": False}
    if args.accelerator == "gpu":
        trainer["devices"] = 1
    clip = getattr(args, "gradient_clip", 0.0)
    if clip > 0:
        trainer["gradient_clip_val"] = clip
    common = dict(input_chunk_length=args.input_window, output_chunk_length=args.horizon,
                  n_epochs=args.epochs, batch_size=args.batch_size, random_state=seed,
                  optimizer_kwargs={"lr": args.learning_rate}, pl_trainer_kwargs=trainer,
                  force_reset=True, save_checkpoints=False)
    scheduler_patience = getattr(args, "scheduler_patience", 5)
    if scheduler_patience > 0:
        common.update(lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
                      lr_scheduler_kwargs={"mode": "min", "factor": 0.5,
                          "patience": scheduler_patience, "threshold": 1e-4,
                          "min_lr": 1e-6, "monitor": "val_loss"})
    dropout = getattr(args, "dropout", 0.1)
    if name != "TFT":
        raise ValueError("This simplified prediction entry point supports TFT only.")
    return TFTModel(**common, hidden_size=args.tft_hidden_size,
                    lstm_layers=args.tft_lstm_layers,
                    num_attention_heads=args.tft_attention_heads,
                    use_static_covariates=INPUT_SOURCE_SETTINGS[args.input_source].use_static,
                    dropout=dropout, likelihood=QuantileRegression(
                        [0.1, 0.5, 0.9] if args.tft_quantiles == "compact" else None))


def to_series(session, input_source="full"):
    from darts import TimeSeries
    frame = session.frame
    static = frame[STATIC_COLUMNS].iloc[0].astype(float).copy()
    static["age"] /= 100.0
    times = pd.DatetimeIndex(frame["date"])
    setting = INPUT_SOURCE_SETTINGS[input_source]
    target = TimeSeries.from_times_and_values(times, frame[["glucose"]].to_numpy(dtype=np.float32),
                                              columns=["glucose"], freq="1min")
    if setting.use_static:
        target = target.with_static_covariates(static)
    optical_values = frame[OCT_COLUMNS].to_numpy(dtype=np.float32)
    if not setting.use_oct:
        optical_values = np.zeros_like(optical_values)
    oct_cov = TimeSeries.from_times_and_values(times, optical_values,
                                               columns=OCT_COLUMNS, freq="1min")
    elapsed = (frame["elapsed_time"].to_numpy(dtype=np.float32).reshape(-1, 1)
               if "elapsed_time" in frame else np.arange(len(frame), dtype=np.float32).reshape(-1, 1) / 240.0)
    if not setting.use_time:
        elapsed = np.zeros_like(elapsed)
    time_cov = TimeSeries.from_times_and_values(times, elapsed,
                                                columns=["elapsed_time"], freq="1min")
    return target, oct_cov, time_cov


def known_time_prefix(time_cov, length):
    """Retain explicit Time values; extend only the deterministic clock."""
    from darts import TimeSeries
    values = time_cov.values(copy=True)
    if length > len(values):
        step = 0.0 if np.all(values == 0) else 1.0 / 240.0
        tail = values[-1, 0] + step * np.arange(1, length - len(values) + 1)
        values = np.concatenate([values, tail.reshape(-1, 1)], axis=0)
    times = pd.date_range(time_cov.start_time(), periods=length, freq="1min")
    return TimeSeries.from_times_and_values(times, values[:length],
                                            columns=time_cov.components, freq="1min")


def causal_forecast(model, name, target, oct_cov, time_cov, protocol, max_steps=None,
                    reference_padding="scaled-zero", candidate_rows=None, reuse_trainer=True):
    from darts import TimeSeries, concatenate
    # The target supplied here is used only for L0 values, metadata and timing.
    times = target.time_index
    context = TimeSeries.from_times_and_values(times[:protocol.input_chunk_length],
                initial_reference_input(target.values(copy=False), protocol, reference_padding),
                columns=target.components, freq="1min").with_static_covariates(target.static_covariates)
    steps = len(target) - protocol.input_chunk_length
    if max_steps is not None:
        steps = min(steps, max_steps)
    candidates = [[] for _ in range(steps)]
    values = []
    if name != "TFT":
        raise ValueError("Only TFT is supported.")
    trainer = getattr(model, "trainer", None) if reuse_trainer else None
    for offset in range(steps):
        stop = protocol.input_chunk_length + offset
        past = oct_cov[:stop]
        future_time = known_time_prefix(time_cov, stop + protocol.output_chunk_length)
        forecast = model.predict(n=protocol.output_chunk_length, series=context,
                                 past_covariates=past, future_covariates=future_time,
                                 predict_likelihood_parameters=True, verbose=False,
                                 **({"trainer": trainer} if trainer is not None else {}))
        if reuse_trainer:
            trainer = getattr(model, "trainer", None)
        # A single probabilistic sample is not a reproducible point estimate.
        fc_values = forecast.values(copy=False)[:, model.likelihood.quantiles.index(0.5)]
        for horizon, value in enumerate(fc_values[:steps - offset]):
            candidates[offset + horizon].append(float(value))
            if candidate_rows is not None:
                candidate_rows.append({"origin_row_index": stop - 1,
                    "latest_observed_oct_row_index": stop - 1,
                    "target_row_index": stop + horizon, "horizon": horizon + 1,
                    "candidate_scaled": float(value)})
        value = density_mean(candidates[offset], protocol.aggregation_keep_ratio)
        if not np.isfinite(value):
            raise ValueError("Aggregation returned a non-finite value.")
        values.append(value)
        point = TimeSeries.from_times_and_values(pd.DatetimeIndex([times[stop]]),
                   np.array([[value]], dtype=np.float32), columns=target.components,
                   freq="1min").with_static_covariates(target.static_covariates)
        context = concatenate([context, point], axis=0)
    return TimeSeries.from_times_and_values(times[protocol.input_chunk_length:protocol.input_chunk_length + steps],
                np.asarray(values, dtype=np.float32).reshape(-1, 1), columns=target.components,
                freq="1min").with_static_covariates(target.static_covariates)


def prepare_fold(sessions, fold, args):
    """Fit both scalers on this outer fold's training participants only."""
    from darts.dataprocessing.transformers import Scaler
    train = [s for s in sessions if s.subject in fold.train_subjects
             and len(s.frame) >= args.input_window + args.horizon]
    validation = [s for s in sessions if s.subject in fold.validation_subjects
                  and len(s.frame) >= args.input_window + args.horizon]
    if not train or not validation:
        raise ValueError(f"Fold {fold.fold_index + 1} requires usable training and validation sessions.")
    train_series, val_series = [to_series(s, args.input_source) for s in train], [to_series(s, args.input_source) for s in validation]
    from sklearn.preprocessing import StandardScaler
    target_scaler = (Scaler(StandardScaler(), global_fit=True)
                     if getattr(args, "target_scaling", "minmax") == "standard"
                     else Scaler(global_fit=True))
    oct_scaler = Scaler(global_fit=True)
    train_y = target_scaler.fit_transform([item[0] for item in train_series])
    train_oct = oct_scaler.fit_transform([item[1] for item in train_series])
    val_y = target_scaler.transform([item[0] for item in val_series])
    val_oct = oct_scaler.transform([item[1] for item in val_series])
    return {"train": train, "validation": validation,
            "target_scaler": target_scaler, "oct_scaler": oct_scaler,
            "train_y": train_y, "train_oct": train_oct,
            "population_reference_scaled": float(np.mean(np.concatenate([y.values(copy=False) for y in train_y]))),
            "train_time": [item[2] for item in train_series],
            "val_y": val_y, "val_oct": val_oct, "val_time": [item[2] for item in val_series]}


def fit_fold_model(name, args, fold_data):
    import torch
    torch.set_num_threads(getattr(args, "torch_threads", 2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = create_model(name, args, args.seed)
    kwargs = {"series": fold_data["train_y"]}
    kwargs.update(past_covariates=fold_data["train_oct"], future_covariates=fold_data["train_time"],
                      val_series=fold_data["val_y"], val_past_covariates=fold_data["val_oct"],
                      val_future_covariates=fold_data["val_time"])
    model.fit(**kwargs)
    return model


def perturb_test_session(session, args):
    """Change held-out inputs only; paired stress leaves explicit Time intact."""
    mode = args.test_perturbation
    if mode == "none":
        return session
    frame = session.frame.copy()
    frame["elapsed_time"] = np.arange(len(frame), dtype=np.float32) / 240.0
    if mode == "time-shift":
        frame["elapsed_time"] = np.roll(frame["elapsed_time"].to_numpy(), args.perturbation_shift)
    else:
        optical, reference = paired_physiological_trajectory_stress_test(
            frame[OCT_COLUMNS].to_numpy(),
            frame[["glucose", "reference_observed"]].to_numpy(),
            mode="shuffle" if mode == "paired-shuffle" else "circular",
            seed=args.seed, shift=args.perturbation_shift)
        frame[OCT_COLUMNS] = optical
        frame["glucose"] = reference[:, 0].astype(float)
        frame["reference_observed"] = reference[:, 1].astype(bool)
    return replace(session, frame=frame)


def predict_sessions(model, name, sessions, fold_data, protocol, args, fold_number):
    points, candidates, excluded = [], [], []
    for session in sessions:
        session = perturb_test_session(session, args)
        if len(session.frame) <= args.input_window:
            excluded.append({"fold": fold_number, "model": name, "session": session.file,
                             "reason": "too short for W-length history"})
            continue
        if not session.frame["reference_observed"].iloc[:protocol.initial_reference_length].all():
            raise ValueError(f"Initial reference contains newly interpolated missing minutes: {session.file}")
        target, oct_, time_ = to_series(session, args.input_source)
        scaled_y = fold_data["target_scaler"].transform(target)
        scaled_oct = fold_data["oct_scaler"].transform(oct_)
        if args.zero_reference:
            masked = scaled_y.values(copy=True)
            masked[:protocol.initial_reference_length] = fold_data["population_reference_scaled"]
            scaled_y = scaled_y.with_values(masked)
        session_candidates = [] if args.export_candidates else None
        prediction = causal_forecast(model, name, scaled_y, scaled_oct, time_, protocol,
            args.max_prediction_steps, reference_padding=args.reference_padding,
            candidate_rows=session_candidates)
        values = fold_data["target_scaler"].inverse_transform(prediction).values(copy=False).reshape(-1)
        for index, value in enumerate(values, start=args.input_window):
            points.append({"fold": fold_number, "model": name, "subject": session.subject,
                "session": session.file, "timestamp": session.frame.iloc[index]["date"],
                "row_index": index, "reference_observed": bool(session.frame.iloc[index]["reference_observed"]),
                "actual_glucose": float(session.frame.iloc[index]["glucose"]), "predicted": float(value)})
        for row in session_candidates or []:
            candidates.append({"fold": fold_number, "model": name, "subject": session.subject,
                               "session": session.file, **row})
        print(f"    Predicted {session.file}: {len(values)} points", flush=True)
    return points, candidates, excluded


def validate_training_args(args):
    for key in ("epochs", "batch_size", "tft_hidden_size", "tft_lstm_layers",
                "tft_attention_heads"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive.")
    if args.learning_rate <= 0 or not 0 <= args.dropout < 1:
        raise ValueError("Learning rate must be positive and dropout must be in [0, 1).")
    if args.gradient_clip < 0 or args.scheduler_patience < 0:
        raise ValueError("Gradient clipping and scheduler patience cannot be negative.")
    if getattr(args, "bootstrap_repetitions", 0) < 0:
        raise ValueError("Bootstrap repetitions cannot be negative.")
    if args.tft_hidden_size % args.tft_attention_heads:
        raise ValueError("TFT hidden size must be divisible by the attention-head count.")
    if args.folds != 5:
        raise ValueError("The manuscript prediction entry point requires five subject folds.")
    if (args.initial_reference, args.input_window, args.horizon) != (10, 50, 10):
        raise ValueError("The main manuscript entry point fixes L0=10, W=50 and H=10.")
    if args.models != ["TFT"]:
        raise ValueError("The simplified entry point runs TFT only.")
    if args.max_prediction_steps is not None and args.max_prediction_steps < 1:
        raise ValueError("At least one prediction step is required.")
    if args.test_perturbation != "none" and args.input_source != "full":
        raise ValueError("Trajectory perturbations require an unperturbed full-input training model.")
    if args.reference_padding not in {"scaled-zero", "last-reference", "mean-reference"}:
        raise ValueError("Unsupported initial-reference padding policy.")
    if getattr(args, "tft_quantiles", "darts-default") not in {"darts-default", "compact"}:
        raise ValueError("Unsupported TFT quantile grid.")
    if getattr(args, "target_scaling", "minmax") not in {"minmax", "standard"}:
        raise ValueError("Unsupported target scaling.")
    if getattr(args, "torch_threads", 2) < 1:
        raise ValueError("Torch thread count must be positive.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--initial-reference", type=int, default=10)
    parser.add_argument("--input-window", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--folds", type=int, choices=[5], default=5)
    parser.add_argument("--subject-map", type=Path)
    parser.add_argument("--cohort-manifest", type=Path)
    parser.add_argument("--require-paper-counts", action="store_true")
    parser.add_argument("--evaluation-mask", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--tft-hidden-size", type=int, default=16)
    parser.add_argument("--tft-lstm-layers", type=int, default=1)
    parser.add_argument("--tft-attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--target-scaling", choices=["minmax", "standard"], default="minmax")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--tft-quantiles", choices=["darts-default", "compact"], default="darts-default")
    parser.add_argument("--reference-padding", choices=["scaled-zero", "last-reference", "mean-reference"],
                        default="scaled-zero")
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--models", nargs="+", choices=["TFT"], default=["TFT"])
    parser.add_argument("--input-source", choices=list(INPUT_SOURCE_SETTINGS), default="full")
    parser.add_argument("--zero-reference", action="store_true")
    parser.add_argument("--test-perturbation", choices=["none", "time-shift", "paired-shuffle", "paired-circular"], default="none")
    parser.add_argument("--perturbation-shift", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-sessions-per-fold", type=int)
    parser.add_argument("--max-prediction-steps", type=int)
    parser.add_argument("--only-fold", type=int, choices=list(range(1, 6)))
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--export-candidates", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    validate_training_args(args)
    return args


def run(args):
    validate_training_args(args)
    protocol = PredictionProtocol(args.initial_reference, args.input_window, args.horizon, args.folds)
    sessions = load_sessions(args.data_dir, args.subject_map)
    cohort_counts = check_cohort_manifest(args, sessions)
    folds, linked = build_subject_folds(sessions, seed=args.seed)
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(observed_sessions=len(sessions), observed_subject_folders=len({s.subject for s in sessions}),
                  dropped_unpaired_rows=sum(s.dropped_unpaired_rows for s in sessions),
                  oct_values_causally_forward_filled=sum(s.forward_filled_oct_values for s in sessions),
                  inserted_minutes_excluded_from_metrics=sum(s.inserted_minutes for s in sessions),
                  cohort_counts=cohort_counts, actual_identity_grouping=("supplied participant manifest"
                      if args.subject_map else "author-supplied subject folder names"),
                  reference_padding=args.reference_padding,
                  tft_point_estimate="predicted 0.5 quantile; no stochastic sampling",
                  future_oct="unavailable", preprocessing_input="precomputed Bio-PAC/DEJ CSV features",
                  training_target_history="teacher-forced training windows; test uses only first L0 reference values",
                  repeated_oct_sessions_kept_together=linked,
                  point_time_semantics="one-step-ahead target; past OCT ends at forecast origin",
                  legacy_settings_reused="default TFT quantiles, scheduler, model sizes, dense 30% aggregation",
                  run_kind="smoke" if args.epochs < 20 or args.test_sessions_per_fold or args.max_prediction_steps or args.only_fold else "full")
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = []
    for fold in folds:
        for session in sessions:
            role = "test" if session.subject in fold.test_subjects else "validation" if session.subject in fold.validation_subjects else "train"
            manifest.append({"fold": fold.fold_index + 1, "role": role, "subject": session.subject,
                             "session": session.file, "n_points": len(session.frame), "sha256": session.source_sha256,
                             "dropped_unpaired_rows": session.dropped_unpaired_rows,
                             "oct_values_causally_forward_filled": session.forward_filled_oct_values,
                             "inserted_minutes": session.inserted_minutes})
    pd.DataFrame(manifest).to_csv(args.output_dir / "fold_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Sessions={len(sessions)}, subject folders={len({s.subject for s in sessions})}, L0={args.initial_reference}, W={args.input_window}, H={args.horizon}", flush=True)
    if args.dry_run:
        print("Dry run completed: session schemas and grouped folds validated.", flush=True)
        return
    import torch
    import darts
    print(f"Darts={darts.__version__}, Torch={torch.__version__}, CUDA={torch.cuda.is_available()}", flush=True)
    points = []
    excluded = []
    candidates = []
    for fold in folds:
        number = fold.fold_index + 1
        if args.only_fold is not None and args.only_fold != number:
            continue
        test = [s for s in sessions if s.subject in fold.test_subjects]
        if args.test_sessions_per_fold:
            test = test[:args.test_sessions_per_fold]
        fold_data = prepare_fold(sessions, fold, args)
        import joblib
        scaler_dir = args.output_dir / "scalers"
        scaler_dir.mkdir(exist_ok=True)
        joblib.dump({"target": fold_data["target_scaler"], "oct": fold_data["oct_scaler"]},
                    scaler_dir / f"fold{number}.pkl")
        print(f"Fold {number}: train sessions={len(fold_data['train'])}, validation={len(fold_data['validation'])}, test={len(test)}", flush=True)
        for name in args.models:
            print(f"  Training {name}", flush=True)
            model = fit_fold_model(name, args, fold_data)
            model_dir = args.output_dir / "models"
            model_dir.mkdir(exist_ok=True)
            model.save(str(model_dir / f"fold{number}_{name}.pkl"))
            model_points, model_candidates, model_excluded = predict_sessions(
                model, name, test, fold_data, protocol, args, number)
            points.extend(model_points)
            candidates.extend(model_candidates)
            excluded.extend(model_excluded)
            pd.DataFrame(points).to_csv(args.output_dir / "predictions_long.csv", index=False, encoding="utf-8-sig")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not points:
        raise ValueError("No evaluable predictions were produced.")
    report_sessions = [perturb_test_session(session, args) for session in sessions]
    metrics = write_prediction_reports(pd.DataFrame(points), report_sessions, args.output_dir,
        expected_models=args.models, evaluation_mask=args.evaluation_mask,
        bootstrap_repetitions=args.bootstrap_repetitions, seed=args.seed)
    if candidates:
        pd.DataFrame(candidates).to_csv(args.output_dir / "horizon_candidates_scaled.csv", index=False,
                                       encoding="utf-8-sig")
    pd.DataFrame(excluded, columns=["fold", "model", "session", "reason"]).to_csv(args.output_dir / "excluded_sessions.csv", index=False)
    print(metrics[["model", "n_points", "RMSE", "A_plus_B_pct"]].to_string(index=False), flush=True)
    print("New-run metrics were measured without matching or forcing archived paper values.", flush=True)


def check_cohort_manifest(args, sessions):
    if args.cohort_manifest is None:
        if args.require_paper_counts:
            raise ValueError("Paper-cohort verification requires --cohort-manifest.")
        return {"verified": False, "reason": "Separate discovery cohort not supplied to this prediction runner."}
    manifest = pd.read_csv(args.cohort_manifest, dtype=str)
    counts = validate_cohorts(manifest, {s.subject for s in sessions},
                             require_paper_counts=args.require_paper_counts)
    prediction = manifest.loc[manifest.cohort.eq("prediction")]
    observed = {(s.subject, s.file) for s in sessions}
    expected = set(zip(prediction.subject, prediction.session))
    if observed != expected:
        raise ValueError("Prediction-session membership does not match the cohort manifest.")
    return {"verified": True, **counts}


if __name__ == "__main__":
    run(parse_args())

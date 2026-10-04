"""Subject-wise five-fold OCT-conditioned prediction pipeline skeleton.

This organized, de-identified script exposes the study protocol without
shipping clinical records, fitted model weights, or figure outputs. It makes
four boundaries explicit:

1. the 28-session rule-discovery cohort is handled separately from prediction;
2. the 72-session prediction cohort is split by subject in five folds;
3. L0 = 10 reference blood glucose samples initialize state but do not define
   the first forecastable point;
4. W = 50 and H = 10 define the model history and forecast horizon.

Install optional Darts dependencies before adapting this skeleton to ethically
approved local data:

    pip install -r requirements-optional.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from biopac_tft_oct.protocol import PredictionProtocol, subjectwise_fivefold_splits  # noqa: E402


DEFAULT_DYNAMIC_COVARIATES = ["slopmean1", "slopmean2", "slopmean3", "slopmean4", "slopmean5"]
DEFAULT_STATIC_COVARIATES = [
    "age",
    "gender",
    "diabetic_healthy",
    "diabetic_t1",
    "diabetic_t2",
    "finger",
    "arm",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Subject-wise five-fold OCT blood glucose pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("research_code/data"))
    parser.add_argument("--subject-pattern", type=str, default="subject-")
    parser.add_argument("--time-col", type=str, default="date")
    parser.add_argument("--target-col", type=str, default="glucose")
    parser.add_argument("--dynamic-covariates", nargs="+", default=DEFAULT_DYNAMIC_COVARIATES)
    parser.add_argument("--static-covariates", nargs="+", default=DEFAULT_STATIC_COVARIATES)
    parser.add_argument("--fold", type=int, default=None, help="Optional zero-based fold to display.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def discover_subject_files(data_dir: Path, subject_pattern: str) -> dict[str, list[Path]]:
    subjects: dict[str, list[Path]] = {}
    for subject_dir in sorted(data_dir.glob(f"{subject_pattern}*")):
        if not subject_dir.is_dir():
            continue
        csv_files = sorted(subject_dir.glob("*.csv"))
        if csv_files:
            subjects[subject_dir.name] = csv_files
    return subjects


def validate_session_csv(
    csv_path: Path,
    time_col: str,
    target_col: str,
    dynamic_covariates: Iterable[str],
    static_covariates: Iterable[str],
) -> pd.DataFrame:
    """Validate the local, de-identified session format before Darts conversion."""

    df = pd.read_csv(csv_path)
    required = [time_col, target_col, *dynamic_covariates, *static_covariates]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    return df.sort_values(time_col)


def main() -> None:
    args = build_arg_parser().parse_args()
    protocol = PredictionProtocol()
    subjects = discover_subject_files(args.data_dir, args.subject_pattern)
    if not subjects:
        print(f"No subject CSV files found under {args.data_dir}.")
        print("This is expected in the public repository because clinical data are not released.")
        print("Place de-identified local files under research_code/data/subject-*/ to inspect the folds.")
        return

    folds = list(subjectwise_fivefold_splits(subjects.keys(), n_splits=protocol.n_subject_folds, seed=args.seed))
    selected_folds = folds if args.fold is None else [folds[args.fold]]
    print("Paper protocol")
    print(f"  subject-wise folds:      {protocol.n_subject_folds}")
    print(f"  initial reference L0:    {protocol.initial_reference_length}")
    print(f"  input history W:         {protocol.input_chunk_length}")
    print(f"  forecast horizon H:      {protocol.output_chunk_length}")
    print(f"  first forecastable index: {protocol.first_forecastable_index}")
    print()
    for fold in selected_folds:
        print(f"Fold {fold.fold_index + 1}/{protocol.n_subject_folds}")
        print(f"  train subjects ({len(fold.train_subjects)}): {list(fold.train_subjects)}")
        print(f"  validation subjects ({len(fold.validation_subjects)}): {list(fold.validation_subjects)}")
        print(f"  test subjects ({len(fold.test_subjects)}): {list(fold.test_subjects)}")
    print()
    print("Each subject remains in one partition within a fold; sessions are never concatenated.")
    print("Reference blood glucose beyond L0 is masked at inference, and future OCT is unavailable.")


if __name__ == "__main__":
    main()

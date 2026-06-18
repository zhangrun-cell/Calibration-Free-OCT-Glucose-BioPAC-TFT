"""Subject-wise Darts training pipeline skeleton.

This file is an organized, de-identified version of the internal training
workflow. It intentionally does not include private subject CSV files, trained
weights, or generated figures. It documents the code structure used for the
paper:

- discover subject folders;
- split by subject, never by time point;
- load glucose, five Bio-PAC OCT features, and static covariates;
- fit shared Darts models;
- run overlapping-window prediction and dense candidate averaging.

Install optional dependencies before using real data:

    pip install -r requirements-optional.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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
    parser = argparse.ArgumentParser(description="Subject-wise OCT glucose model pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("research_code/data"))
    parser.add_argument("--subject-pattern", type=str, default="subject-")
    parser.add_argument("--time-col", type=str, default="date")
    parser.add_argument("--target-col", type=str, default="glucose")
    parser.add_argument("--dynamic-covariates", nargs="+", default=DEFAULT_DYNAMIC_COVARIATES)
    parser.add_argument("--static-covariates", nargs="+", default=DEFAULT_STATIC_COVARIATES)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--input-chunk-length", type=int, default=50)
    parser.add_argument("--output-chunk-length", type=int, default=10)
    parser.add_argument("--global-keep-ratio", type=float, default=0.3)
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


def split_subjects(
    subjects: dict[str, list[Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    names = sorted(subjects)
    random.Random(seed).shuffle(names)
    n_total = len(names)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = min(n_train, n_total)
    n_val = min(n_val, n_total - n_train)
    n_test = min(int(round(n_total * test_ratio)), n_total - n_train - n_val)
    return names[:n_train], names[n_train : n_train + n_val], names[n_train + n_val : n_train + n_val + n_test]


def validate_session_csv(
    csv_path: Path,
    time_col: str,
    target_col: str,
    dynamic_covariates: Iterable[str],
    static_covariates: Iterable[str],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [time_col, target_col, *dynamic_covariates, *static_covariates]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    return df.sort_values(time_col)


def dense_candidate_average(values: np.ndarray, keep_ratio: float = 0.3) -> float:
    """Average the most concentrated subset of overlapping predictions."""

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if x.size == 1:
        return float(x[0])
    x = np.sort(x)
    keep = max(1, int(np.ceil(x.size * keep_ratio)))
    if keep >= x.size:
        return float(np.mean(x))
    spans = x[keep - 1 :] - x[: x.size - keep + 1]
    start = int(np.argmin(spans))
    return float(np.mean(x[start : start + keep]))


def main() -> None:
    args = build_arg_parser().parse_args()
    subjects = discover_subject_files(args.data_dir, args.subject_pattern)
    if not subjects:
        print(f"No subject CSV files found under {args.data_dir}.")
        print("This is expected in the public repository because private clinical data are not released.")
        print("Place de-identified local files under research_code/data/subject-*/ to run the full pipeline.")
        return
    train_subjects, val_subjects, test_subjects = split_subjects(
        subjects,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    print("Subject-wise split")
    print(f"  train: {train_subjects}")
    print(f"  val:   {val_subjects}")
    print(f"  test:  {test_subjects}")
    print()
    print("Expected Darts model configuration")
    print(f"  input_chunk_length:  {args.input_chunk_length}")
    print(f"  output_chunk_length: {args.output_chunk_length}")
    print(f"  dynamic covariates:  {args.dynamic_covariates}")
    print(f"  static covariates:   {args.static_covariates}")
    print()
    print("This skeleton omits private training execution because clinical data are not released.")


if __name__ == "__main__":
    main()

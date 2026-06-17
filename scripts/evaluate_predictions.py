"""Evaluate glucose prediction CSV files.

Expected columns:
    actual_glucose,tft_global_pred,transformer_global_pred,xgboost_global_pred,tide_global_pred

Values are assumed to be in mmol/L.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biopac_tft_oct import clarke_percentages, regression_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="Prediction CSV to evaluate.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output summary CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    models = {
        "TFT": "tft_global_pred",
        "Transformer": "transformer_global_pred",
        "XGBoost": "xgboost_global_pred",
        "TiDE": "tide_global_pred",
    }
    rows = []
    for model, column in models.items():
        if column not in df.columns:
            continue
        sub = df[["actual_glucose", column]].dropna()
        metrics = regression_metrics(sub["actual_glucose"].to_numpy(), sub[column].to_numpy())
        clarke = clarke_percentages(sub["actual_glucose"].to_numpy(), sub[column].to_numpy())
        rows.append({"model": model, **metrics, **clarke})

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

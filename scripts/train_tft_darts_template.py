"""Template for TFT training with Darts.

This script documents the model configuration used by the paper workflow. It is
not executed by the demo because the clinical training data are not publicly
released. Users with their own data in the documented CSV format can adapt this
template.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biopac_tft_oct.protocol import PredictionProtocol


def expected_columns() -> list[str]:
    return [
        "date",
        "glucose",
        "slopmean1",
        "slopmean2",
        "slopmean3",
        "slopmean4",
        "slopmean5",
        "gender",
        "age",
        "diabetic_healthy",
        "diabetic_t1",
        "diabetic_t2",
        "finger",
        "arm",
    ]


def validate_csv(path: Path) -> None:
    df = pd.read_csv(path, nrows=5)
    missing = [c for c in expected_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def main() -> None:
    protocol = PredictionProtocol()
    print(
        "This is a training template. Install u8darts[torch] and adapt the "
        "subject-wise data loading routine for your local ethically approved data."
    )
    print("Expected columns:")
    for column in expected_columns():
        print(f"  - {column}")
    print("\nPaper configuration summary:")
    print(f"  initial_reference_length (L0): {protocol.initial_reference_length}")
    print(f"  input_chunk_length (W): {protocol.input_chunk_length}")
    print(f"  output_chunk_length (H): {protocol.output_chunk_length}")
    print(f"  first forecastable target index: {protocol.first_forecastable_index}")
    print("  static covariates: age, sex, diabetic status, measurement site")
    print("  dynamic OCT covariates: slopmean1--slopmean5")
    print("  split: five-fold subject-wise cross-validation, never by sampling point")
    print("  inference: future reference blood glucose masked; future OCT unavailable")


if __name__ == "__main__":
    main()

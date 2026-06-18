"""Run a small anonymized Bio-PAC and evaluation demo.

This demo uses synthetic/de-identified data with the same column conventions as
the manuscript workflow. It is intended to verify the code path, not to
reproduce the clinical results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biopac_tft_oct import (  # noqa: E402
    biopac_process,
    clarke_percentages,
    dej_anchored_features,
    dej_anchored_window_ranges,
    dense_candidate_average,
    regression_metrics,
)


def main() -> None:
    demo_csv = ROOT / "examples" / "demo_anonymized_sample.csv"
    df = pd.read_csv(demo_csv)

    depth_cols = [c for c in df.columns if c.startswith("depth_")]
    if depth_cols:
        oct_signal = df[depth_cols].to_numpy(dtype=float)
    else:
        oct_signal = synthesize_demo_oct(df["glucose_mmol_l"].to_numpy(dtype=float), n_depth=80)

    result = biopac_process(oct_signal, n_segments=6, max_shift=8, epidermis_depth=8)
    demo_offsets = (8, 18, 28, 38, 48)
    features, windows = dej_anchored_features(
        result.corrected,
        dej_index=18,
        offsets=demo_offsets,
        half_width=5,
    )
    shifted_windows = dej_anchored_window_ranges(
        dej_index=26,
        n_depth=result.corrected.shape[1],
        offsets=demo_offsets,
        half_width=5,
    )

    # The demo prediction is a simple noisy proxy so that metrics can be tested
    # without shipping clinical model weights.
    glucose = df["glucose_mmol_l"].to_numpy(dtype=float)
    proxy = glucose + 0.25 * np.sin(np.linspace(0, 2 * np.pi, glucose.size))
    smoothed_proxy = np.array(
        [
            dense_candidate_average(proxy[max(0, i - 2) : min(proxy.size, i + 3)], keep_ratio=0.5)
            for i in range(proxy.size)
        ]
    )

    metrics = regression_metrics(glucose, smoothed_proxy)
    clarke = clarke_percentages(glucose, smoothed_proxy)

    print("Bio-PAC demo completed")
    print(f"Corrected OCT shape: {result.corrected.shape}")
    print(f"DEJ windows: {windows}")
    print(f"DEJ windows after moving the DEJ anchor from 18 to 26 pixels: {shifted_windows}")
    print(f"Feature matrix shape: {features.shape}")
    print("Regression metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("Clarke percentages:")
    for key, value in clarke.items():
        print(f"  {key}: {value:.4f}")


def synthesize_demo_oct(glucose: np.ndarray, n_depth: int = 80) -> np.ndarray:
    """Generate a toy OCT depth-time map from anonymized glucose values.

    The generated values are synthetic and are not clinical measurements. They
    only provide a reproducible input for exercising the Bio-PAC code path.
    """

    rng = np.random.default_rng(42)
    n_time = glucose.size
    t = np.linspace(0, 1, n_time)
    z = np.arange(n_depth)
    base = 48.0 * np.exp(-z / 42.0) + 6.0 * np.exp(-((z - 22.0) ** 2) / 90.0)
    epidermal_drift = 2.2 * np.sin(2 * np.pi * t) + 0.8 * (t - 0.5)
    glucose_component = (glucose - np.mean(glucose))[:, None] * np.exp(-((z - 35.0) ** 2) / 180.0)
    motion = 0.6 * np.sin(2 * np.pi * (t[:, None] + z[None, :] / 55.0))
    noise = rng.normal(0.0, 0.25, size=(n_time, n_depth))
    return base[None, :] + epidermal_drift[:, None] + glucose_component + motion + noise


if __name__ == "__main__":
    main()

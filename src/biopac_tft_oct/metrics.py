"""Regression and Clarke error grid metrics for glucose prediction."""

from __future__ import annotations

import numpy as np

MMOL_TO_MG_DL = 18.0182


def clarke_zone_mgdl(reference_mgdl: np.ndarray, predicted_mgdl: np.ndarray) -> np.ndarray:
    """Return Clarke error grid zones A-E in mg/dL space."""

    r = np.asarray(reference_mgdl, dtype=float)
    p = np.asarray(predicted_mgdl, dtype=float)
    if r.shape != p.shape:
        raise ValueError("reference and predicted arrays must have the same shape.")

    zone_a = ((r <= 70) & (p <= 70)) | ((p >= 0.8 * r) & (p <= 1.2 * r))
    zone_e = ((r >= 180) & (p <= 70)) | ((r <= 70) & (p >= 180))
    zone_c = ((r >= 70) & (r <= 290) & (p >= r + 110)) | (
        (r >= 130) & (r <= 180) & (p <= (7.0 / 5.0) * r - 182)
    )
    zone_d = ((r >= 240) & (p >= 70) & (p <= 180)) | (
        (r <= 175.0 / 3.0) & (p >= 70) & (p <= 180)
    ) | ((r >= 175.0 / 3.0) & (r <= 70) & (p >= (6.0 / 5.0) * r))

    return np.select([zone_a, zone_e, zone_c, zone_d], ["A", "E", "C", "D"], default="B")


def clarke_percentages(
    reference_mmol: np.ndarray,
    predicted_mmol: np.ndarray,
) -> dict[str, float]:
    """Compute Clarke zone percentages from mmol/L glucose values."""

    ref = np.asarray(reference_mmol, dtype=float)
    pred = np.asarray(predicted_mmol, dtype=float)
    mask = np.isfinite(ref) & np.isfinite(pred)
    zones = clarke_zone_mgdl(ref[mask] * MMOL_TO_MG_DL, pred[mask] * MMOL_TO_MG_DL)
    n = len(zones)
    out: dict[str, float] = {"n": float(n)}
    for letter in "ABCDE":
        out[f"{letter}_pct"] = 100.0 * float(np.sum(zones == letter)) / n if n else float("nan")
    out["A_plus_B_pct"] = out["A_pct"] + out["B_pct"]
    return out


def regression_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return R2, MSE, MAE, RMSE, ME, and MARD."""

    y = np.asarray(reference, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if y.size == 0:
        return {k: float("nan") for k in ["R2", "MSE", "MAE", "RMSE", "ME", "MARD_pct"]}

    err = p - y
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    me = float(np.mean(err))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    positive = y > 0
    mard = float(100.0 * np.mean(np.abs(p[positive] - y[positive]) / y[positive])) if np.any(positive) else float("nan")
    return {"R2": r2, "MSE": mse, "MAE": mae, "RMSE": rmse, "ME": me, "MARD_pct": mard}

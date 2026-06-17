"""Distribution-aware autoregressive aggregation utilities."""

from __future__ import annotations

import numpy as np


def dense_candidate_average(values: list[float] | np.ndarray, keep_ratio: float = 0.3) -> float:
    """Average the most concentrated subset of candidate predictions.

    Multiple overlapping autoregressive windows may produce candidate
    predictions for the same physical time point. This function sorts the
    candidates, finds the contiguous subset with the smallest value range, and
    averages that subset. It is a deterministic approximation of the
    high-density candidate averaging used in the manuscript.
    """

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return float(arr[0])

    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    k = max(1, int(np.floor(arr.size * keep_ratio)))
    sorted_arr = np.sort(arr)
    if k >= sorted_arr.size:
        return float(np.mean(sorted_arr))

    widths = sorted_arr[k - 1 :] - sorted_arr[: sorted_arr.size - k + 1]
    start = int(np.argmin(widths))
    return float(np.mean(sorted_arr[start : start + k]))

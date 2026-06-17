"""DEJ-anchored dynamic OCT feature construction."""

from __future__ import annotations

import numpy as np


def dej_anchored_features(
    corrected_oct: np.ndarray,
    dej_index: int,
    offsets: tuple[int, ...] = (20, 40, 60, 80, 100),
    half_width: int = 10,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Average five depth windows anchored below the DEJ.

    Parameters
    ----------
    corrected_oct:
        Bio-PAC corrected OCT array with shape ``(time, depth)``.
    dej_index:
        DEJ anchor pixel index.
    offsets:
        Window-center offsets relative to the DEJ pixel.
    half_width:
        Half-width of each depth window in pixels.

    Returns
    -------
    features:
        Array with shape ``(time, n_windows)``.
    windows:
        Inclusive-exclusive depth ranges ``[(start, stop), ...]``.
    """

    x = np.asarray(corrected_oct, dtype=float)
    if x.ndim != 2:
        raise ValueError("corrected_oct must have shape (time, depth).")

    n_depth = x.shape[1]
    windows: list[tuple[int, int]] = []
    cols = []
    for offset in offsets:
        center = int(dej_index + offset)
        start = max(0, center - half_width)
        stop = min(n_depth, center + half_width + 1)
        if start >= stop:
            raise ValueError("Invalid DEJ window; check dej_index and offsets.")
        windows.append((start, stop))
        cols.append(np.nanmean(x[:, start:stop], axis=1))

    return np.column_stack(cols), windows

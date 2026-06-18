"""DEJ-anchored dynamic OCT feature construction."""

from __future__ import annotations

import numpy as np


def dej_anchored_window_ranges(
    dej_index: int,
    n_depth: int,
    offsets: tuple[int, ...] = (20, 40, 60, 80, 100),
    half_width: int = 5,
) -> list[tuple[int, int]]:
    """Return DEJ-anchored depth-window ranges.

    The DEJ pixel is supplied by anatomical annotation, segmentation, or the
    study-specific preprocessing protocol. Once ``dej_index`` is provided, all
    window centers are automatically shifted as

        center_j = dej_index + offset_j

    so different DEJ depths produce different five-window locations. Returned
    ranges follow Python indexing: ``start`` is inclusive and ``stop`` is
    exclusive. With the default ``half_width=5``, each full window contains 11
    pixels, matching the manuscript definition ``[c_j - 5, c_j + 5]``.
    """

    n_depth = int(n_depth)
    if n_depth <= 0:
        raise ValueError("n_depth must be positive.")

    dej_index = int(dej_index)
    half_width = int(max(0, half_width))
    windows: list[tuple[int, int]] = []
    for offset in offsets:
        center = int(dej_index + offset)
        start = max(0, center - half_width)
        stop = min(n_depth, center + half_width + 1)
        if start >= stop:
            raise ValueError("Invalid DEJ window; check dej_index, offsets, and depth size.")
        windows.append((start, stop))
    return windows


def dej_anchored_features(
    corrected_oct: np.ndarray,
    dej_index: int,
    offsets: tuple[int, ...] = (20, 40, 60, 80, 100),
    half_width: int = 5,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Average five depth windows anchored below the DEJ.

    Parameters
    ----------
    corrected_oct:
        Bio-PAC corrected OCT array with shape ``(time, depth)``.
    dej_index:
        DEJ anchor pixel index. The five window centers are shifted according
        to ``dej_index + offset``.
    offsets:
        Window-center offsets relative to the DEJ pixel.
    half_width:
        Half-width of each depth window in pixels. The default value of 5
        yields 11-pixel windows.

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
    windows = dej_anchored_window_ranges(
        dej_index=dej_index,
        n_depth=n_depth,
        offsets=offsets,
        half_width=half_width,
    )
    cols = []
    for start, stop in windows:
        cols.append(np.nanmean(x[:, start:stop], axis=1))

    return np.column_stack(cols), windows

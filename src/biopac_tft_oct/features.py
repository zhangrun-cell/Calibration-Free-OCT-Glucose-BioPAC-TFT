"""DEJ-anchored dynamic OCT feature construction."""

from __future__ import annotations

import numpy as np


SITE_DEJ_SEARCH_RANGES: dict[str, tuple[int, int]] = {
    # Approximate ranges inferred from the manuscript Fig. 4 and a depth
    # spacing of 5.8574 um/pixel: arm/wrist DEJ ~160-250 um, finger DEJ
    # ~320-420 um. The pixel ranges are deliberately slightly widened.
    "arm": (27, 45),
    "wrist": (27, 45),
    "forearm": (27, 45),
    "finger": (55, 75),
}


def _smooth_profile(values: np.ndarray, window: int) -> np.ndarray:
    window = int(max(1, window))
    if window == 1:
        return values.astype(float, copy=True)
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: values.size]


def detect_first_peak_anchor(
    oct_signal: np.ndarray,
    smooth_window: int = 5,
    search_start: int = 10,
    search_stop: int | None = None,
    site: str | None = None,
    search_range: tuple[int, int] | None = None,
) -> int:
    """Detect the first depth-axis intensity peak used as the anatomical anchor.

    The input is a Bio-PAC-corrected depth-time OCT signal with shape
    ``(time, depth)``. The function averages OCT intensity over time to obtain
    one representative depth-intensity profile, smooths that profile, skips the
    first 10 depth pixels by default, restricts the search to the expected
    site-specific DEJ range when ``site`` or ``search_range`` is provided, and
    returns the first interior local maximum in that range. This post-cutoff
    first peak is used as the DEJ/first-peak anchor for downstream window
    placement.
    """

    x = np.asarray(oct_signal, dtype=float)
    if x.ndim != 2:
        raise ValueError("oct_signal must have shape (time, depth).")

    n_depth = x.shape[1]
    if n_depth < 3:
        return int(np.nanargmax(np.nanmean(x, axis=0)))

    if search_range is not None:
        range_start, range_stop = search_range
    elif site is not None:
        site_key = site.lower()
        if site_key not in SITE_DEJ_SEARCH_RANGES:
            valid_sites = ", ".join(sorted(SITE_DEJ_SEARCH_RANGES))
            raise ValueError(f"Unknown site '{site}'. Expected one of: {valid_sites}.")
        range_start, range_stop = SITE_DEJ_SEARCH_RANGES[site_key]
    else:
        range_start, range_stop = search_start, search_stop

    start = int(max(1, 10, range_start))
    stop_value = n_depth - 1 if range_stop is None else range_stop
    stop = int(min(stop_value, n_depth - 1))
    if start >= stop:
        raise ValueError("Invalid peak search range.")

    depth_profile = np.nanmean(x, axis=0)
    profile = _smooth_profile(depth_profile, smooth_window)
    for z in range(start, stop):
        if profile[z - 1] < profile[z] and profile[z] >= profile[z + 1]:
            return z

    return int(start + np.nanargmax(profile[start:stop]))


def dej_anchored_window_ranges(
    dej_index: int | None,
    n_depth: int,
    offsets: tuple[int, ...] = (20, 40, 60, 80, 100),
    half_width: int = 5,
    anchor_signal: np.ndarray | None = None,
    site: str | None = None,
    search_range: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Return DEJ-anchored depth-window ranges.

    If ``dej_index`` is omitted, the anchor is automatically detected as the
    first local peak of the averaged depth-intensity OCT profile after skipping
    the first 10 pixels and applying the site-specific or user-provided search
    range. Once the anchor is available, all window centers are shifted as

        center_j = anchor + offset_j

    so different DEJ depths produce different five-window locations. Returned
    ranges follow Python indexing: ``start`` is inclusive and ``stop`` is
    exclusive. With the default ``half_width=5``, each full window contains 11
    pixels, matching the manuscript definition ``[c_j - 5, c_j + 5]``.
    """

    n_depth = int(n_depth)
    if n_depth <= 0:
        raise ValueError("n_depth must be positive.")

    if dej_index is None:
        if anchor_signal is None:
            raise ValueError("anchor_signal is required when dej_index is None.")
        dej_index = detect_first_peak_anchor(
            anchor_signal,
            site=site,
            search_range=search_range,
        )
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
    dej_index: int | None = None,
    offsets: tuple[int, ...] = (20, 40, 60, 80, 100),
    half_width: int = 5,
    site: str | None = None,
    search_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Average five depth windows anchored below the DEJ.

    Parameters
    ----------
    corrected_oct:
        Bio-PAC corrected OCT array with shape ``(time, depth)``.
    dej_index:
        Optional DEJ/first-peak anchor pixel index. When omitted, the anchor is
        detected automatically as the first local maximum of the averaged
        depth-intensity profile within the site-specific or user-provided
        search range. Manual ``dej_index`` takes precedence over automatic
        detection.
    offsets:
        Window-center offsets relative to the DEJ pixel.
    half_width:
        Half-width of each depth window in pixels. The default value of 5
        yields 11-pixel windows.
    site:
        Optional measurement site used for automatic anchor detection. Supported
        values include ``"arm"``, ``"wrist"``, ``"forearm"``, and ``"finger"``.
    search_range:
        Optional ``(start, stop)`` pixel range that overrides the site preset
        during automatic anchor detection.

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
        anchor_signal=x,
        site=site,
        search_range=search_range,
    )
    cols = []
    for start, stop in windows:
        cols.append(np.nanmean(x[:, start:stop], axis=1))

    return np.column_stack(cols), windows

"""Bio-PAC preprocessing for depth-resolved OCT time series.

Input convention
----------------
The main input is a two-dimensional OCT depth-time representation:

    oct_signal[t, z]

where ``t`` is the acquisition index/time point and ``z`` is the OCT depth
pixel. The value is OCT backscattered intensity after the B-scan/volume has
been surface-flattened, cropped, and laterally averaged.

Bio-PAC has two computational stages:
1. Morphology-aware refractive-distortion alignment.
2. Epidermis-referenced optical decoupling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BioPACResult:
    """Container for intermediate and final Bio-PAC outputs."""

    aligned: np.ndarray
    corrected: np.ndarray
    epidermal_fingerprint: np.ndarray
    depth_coefficients: np.ndarray
    shifts: np.ndarray


def _as_time_depth(oct_signal: np.ndarray) -> np.ndarray:
    x = np.asarray(oct_signal, dtype=float)
    if x.ndim != 2:
        raise ValueError("oct_signal must be a 2-D array with shape (time, depth).")
    return x


def _analytic_envelope(scan: np.ndarray) -> np.ndarray:
    """Return ``abs(hilbert(scan))`` using only NumPy FFT operations."""

    x = np.asarray(scan, dtype=float)
    n = x.size
    spectrum = np.fft.fft(x)
    multiplier = np.zeros(n, dtype=float)
    if n % 2 == 0:
        multiplier[0] = 1.0
        multiplier[n // 2] = 1.0
        multiplier[1 : n // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1 : (n + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * multiplier)
    return np.abs(analytic)


def _gaussian_smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    sigma = float(max(0.0, sigma))
    if sigma == 0.0:
        return values.astype(float, copy=True)
    radius = max(1, int(round(3.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(grid**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    pad = radius
    padded = np.pad(values, pad_width=pad, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: values.size]


def _structural_envelope(scan: np.ndarray, smooth_sigma: float) -> np.ndarray:
    """Hilbert structural envelope used for segment matching.

    This mirrors the original MATLAB preprocessing step:

        smoothdata(abs(hilbert(scan)), 'gaussian', sigma)
    """

    return _gaussian_smooth_1d(_analytic_envelope(scan), smooth_sigma)


def morphology_align(
    oct_signal: np.ndarray,
    n_segments: int = 10,
    max_shift: int = 30,
    envelope_sigma_ref: float = 10.0,
    envelope_sigma_current: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Align apparent OCT depth morphology over time.

    Each A-scan is compared with the first A-scan reference. For each depth
    segment, one shift is estimated by cross-correlating the whole segment
    envelope, not isolated local extrema. Segment shifts are interpolated into
    a continuous depth-warping field and the original intensity curve is
    resampled. The resampling changes depth coordinates but does not directly
    alter intensity values.

    Parameters
    ----------
    oct_signal:
        Array with shape ``(time, depth)``.
    n_segments:
        Number of depth segments used for segment-wise shift estimation.
    max_shift:
        Maximum tested integer shift in pixels.

    Returns
    -------
    aligned:
        Morphology-aligned OCT signal with shape ``(time, depth)``.
    shifts:
        Estimated segment shifts with shape ``(time, n_segments)``.
    """

    x = _as_time_depth(oct_signal)
    n_time, n_depth = x.shape
    n_segments = int(max(1, min(n_segments, n_depth)))
    max_shift = int(max(0, max_shift))

    reference = x[0]
    ref_envelope = _structural_envelope(reference, envelope_sigma_ref)
    edges = np.round(np.linspace(0, n_depth, n_segments + 1)).astype(int)
    centers = 0.5 * (edges[:-1] + edges[1:] - 1)

    aligned = np.zeros_like(x)
    shifts = np.zeros((n_time, n_segments), dtype=float)
    depth_axis = np.arange(n_depth, dtype=float)

    for t in range(n_time):
        current = x[t]
        cur_envelope = _structural_envelope(current, envelope_sigma_current)
        local_shifts = np.zeros(n_segments, dtype=float)

        for s in range(n_segments):
            start, stop = edges[s], edges[s + 1]
            ref_seg = ref_envelope[start:stop]
            cur_seg = cur_envelope[start:stop]
            best_shift = 0
            best_score = -np.inf

            for shift in range(-max_shift, max_shift + 1):
                shifted_idx = np.arange(start, stop) + shift
                valid = (shifted_idx >= 0) & (shifted_idx < n_depth)
                if np.count_nonzero(valid) < 3:
                    continue
                a = ref_seg[valid]
                b = cur_envelope[shifted_idx[valid]]
                if np.std(a) == 0 or np.std(b) == 0:
                    score = -np.inf
                else:
                    score = float(np.corrcoef(a, b)[0, 1])
                if score > best_score:
                    best_score = score
                    best_shift = shift

            local_shifts[s] = best_shift

        shifts[t] = local_shifts
        warp_field = np.interp(depth_axis, centers, local_shifts, left=local_shifts[0], right=local_shifts[-1])
        query_points = depth_axis + warp_field
        aligned[t] = np.interp(query_points, depth_axis, current, left=0.0, right=0.0)

    return aligned, shifts


def epidermis_referenced_decoupling(
    aligned_oct: np.ndarray,
    epidermis_depth: int = 10,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove common epidermis-referenced optical drift.

    The epidermal reference curve is averaged from superficial depth pixels and
    z-score standardized into a dimensionless fingerprint ``P(t)``. For each
    depth pixel ``z``, a one-dimensional regression

        I_align(t, z) = alpha_z P(t) + beta_z + residual_z(t)

    estimates the depth-specific contribution of the fingerprint. Bio-PAC
    subtracts ``alpha_z P(t)`` and preserves the residual depth dynamics.
    """

    x = _as_time_depth(aligned_oct)
    n_time, n_depth = x.shape
    epidermis_depth = int(max(1, min(epidermis_depth, n_depth)))

    raw_fingerprint = np.nanmean(x[:, :epidermis_depth], axis=1)
    std = float(np.nanstd(raw_fingerprint))
    if std < eps:
        fingerprint = np.zeros(n_time, dtype=float)
    else:
        fingerprint = (raw_fingerprint - np.nanmean(raw_fingerprint)) / (std + eps)

    denom = float(np.sum((fingerprint - np.mean(fingerprint)) ** 2))
    corrected = np.zeros_like(x)
    alpha = np.zeros(n_depth, dtype=float)

    if denom < eps:
        return x.copy(), fingerprint, alpha

    p_centered = fingerprint - np.mean(fingerprint)
    for z in range(n_depth):
        y = x[:, z]
        y_centered = y - np.nanmean(y)
        alpha[z] = float(np.nansum(p_centered * y_centered) / denom)
        corrected[:, z] = y - alpha[z] * fingerprint

    return corrected, fingerprint, alpha


def biopac_process(
    oct_signal: np.ndarray,
    n_segments: int = 10,
    max_shift: int = 30,
    epidermis_depth: int = 10,
) -> BioPACResult:
    """Run the complete Bio-PAC workflow."""

    aligned, shifts = morphology_align(
        oct_signal=oct_signal,
        n_segments=n_segments,
        max_shift=max_shift,
    )
    corrected, fingerprint, alpha = epidermis_referenced_decoupling(
        aligned,
        epidermis_depth=epidermis_depth,
    )
    return BioPACResult(
        aligned=aligned,
        corrected=corrected,
        epidermal_fingerprint=fingerprint,
        depth_coefficients=alpha,
        shifts=shifts,
    )

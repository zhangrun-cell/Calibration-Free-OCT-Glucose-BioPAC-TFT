"""Mechanistic control utilities used for the Figure 7 analyses.

These helpers make the information boundaries explicit. They do not retrain a
model; callers apply them to held-out sessions before inference with a model
trained on the original trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InputSourceSetting:
    """Which covariates are available to a Figure 7 input-source control."""

    use_time: bool
    use_static: bool
    use_oct: bool


INPUT_SOURCE_SETTINGS: dict[str, InputSourceSetting] = {
    "time_only": InputSourceSetting(use_time=True, use_static=False, use_oct=False),
    "time_plus_oct": InputSourceSetting(use_time=True, use_static=False, use_oct=True),
    "time_plus_static": InputSourceSetting(use_time=True, use_static=True, use_oct=False),
    "full": InputSourceSetting(use_time=True, use_static=True, use_oct=True),
}


def replace_initial_reference_with_population_value(
    reference_glucose: np.ndarray,
    *,
    initial_reference_length: int,
    population_reference_value: float,
) -> np.ndarray:
    """Create the zero-reference input while retaining only a population anchor."""

    result = np.asarray(reference_glucose, dtype=float).copy()
    result[: int(initial_reference_length)] = float(population_reference_value)
    return result


def shift_time_covariate(time_covariate: np.ndarray, shift: int) -> np.ndarray:
    """Shift explicit Time only; OCT and reference glucose remain unchanged."""

    return np.roll(np.asarray(time_covariate), int(shift), axis=0)


def paired_physiological_trajectory_stress_test(
    oct_features: np.ndarray,
    reference_glucose: np.ndarray,
    *,
    mode: str,
    seed: int = 42,
    shift: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder OCT-reference-glucose pairs while leaving Time unchanged.

    ``mode='shuffle'`` randomly reorders matched OCT-reference glucose pairs
    within a session. ``mode='circular'`` circularly shifts the matched pairs.
    Both stress tests preserve the pair itself but disrupt its relation to the
    original OGTT time template. Static covariates and explicit Time are not
    modified by this function.
    """

    oct_array = np.asarray(oct_features)
    glucose_array = np.asarray(reference_glucose)
    if oct_array.shape[0] != glucose_array.shape[0]:
        raise ValueError("oct_features and reference_glucose must share the time axis.")

    if mode == "shuffle":
        order = np.random.default_rng(seed).permutation(oct_array.shape[0])
    elif mode == "circular":
        order = np.roll(np.arange(oct_array.shape[0]), int(shift))
    else:
        raise ValueError("mode must be 'shuffle' or 'circular'.")
    return oct_array[order].copy(), glucose_array[order].copy()

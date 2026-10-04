"""DEJ-guided dynamic signal-extraction rule discovery and application.

The rule is fitted once in the independent rule-discovery cohort. It learns, for
each measurement site, the fixed linear relationship between OCT-derived DEJ
depth and the centres of five informative OCT windows. The frozen mapping is
then applied to prediction-cohort sessions using OCT morphology alone; reference
blood glucose values in the prediction cohort are never used to reselect or
adapt window centres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DEJGuidedDynamicSignalExtractionRule:
    """Frozen site-specific mapping from DEJ depth to five window centres."""

    slopes_by_site: Mapping[str, tuple[float, ...]]
    intercepts_by_site: Mapping[str, tuple[float, ...]]
    half_width: int = 5

    def centres(self, *, site: str, dej_depth: float) -> np.ndarray:
        """Calculate rounded five-window centres from OCT-derived DEJ depth."""

        key = str(site).lower()
        if key not in self.slopes_by_site or key not in self.intercepts_by_site:
            known = ", ".join(sorted(self.slopes_by_site))
            raise ValueError(f"No frozen DEJ rule for site '{site}'. Known sites: {known}.")
        slopes = np.asarray(self.slopes_by_site[key], dtype=float)
        intercepts = np.asarray(self.intercepts_by_site[key], dtype=float)
        if slopes.shape != intercepts.shape:
            raise ValueError(f"Inconsistent slope/intercept arrays for site '{site}'.")
        return np.rint(slopes * float(dej_depth) + intercepts).astype(int)

    def window_ranges(self, *, site: str, dej_depth: float, n_depth: int) -> list[tuple[int, int]]:
        """Return inclusive-exclusive ranges for the five 11-pixel windows."""

        n_depth = int(n_depth)
        if n_depth <= 0:
            raise ValueError("n_depth must be positive.")
        windows: list[tuple[int, int]] = []
        for center in self.centres(site=site, dej_depth=dej_depth):
            start = max(0, int(center) - self.half_width)
            stop = min(n_depth, int(center) + self.half_width + 1)
            if start >= stop:
                raise ValueError("A derived window lies outside the OCT depth range.")
            windows.append((start, stop))
        return windows


def fit_dej_guided_dynamic_signal_extraction_rule(
    discovery_table: pd.DataFrame,
    *,
    site_col: str = "site",
    dej_depth_col: str = "dej_depth",
    window_center_cols: Sequence[str] = (
        "window_center_1",
        "window_center_2",
        "window_center_3",
        "window_center_4",
        "window_center_5",
    ),
    half_width: int = 5,
) -> DEJGuidedDynamicSignalExtractionRule:
    """Fit the discovery-only site-specific linear DEJ-to-window mappings.

    ``discovery_table`` belongs exclusively to the rule-discovery cohort. Its
    five centre columns can be derived during the discovery analysis using
    reference blood glucose trajectories. The resulting rule contains only
    slopes and intercepts; it can therefore be applied to a prediction cohort
    without accessing its reference blood glucose values.
    """

    required = [site_col, dej_depth_col, *window_center_cols]
    missing = [column for column in required if column not in discovery_table.columns]
    if missing:
        raise ValueError(f"Discovery table is missing required columns: {missing}")
    if len(window_center_cols) != 5:
        raise ValueError("Exactly five window-centre columns are required.")

    slopes_by_site: dict[str, tuple[float, ...]] = {}
    intercepts_by_site: dict[str, tuple[float, ...]] = {}
    for site, group in discovery_table.groupby(discovery_table[site_col].astype(str).str.lower(), sort=True):
        numeric = group[[dej_depth_col, *window_center_cols]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(numeric) < 2:
            raise ValueError(f"Site '{site}' has fewer than two complete discovery sessions.")
        x = numeric[dej_depth_col].to_numpy(dtype=float)
        if np.allclose(x, x[0]):
            raise ValueError(f"Site '{site}' has no DEJ-depth variation for linear rule fitting.")
        slopes: list[float] = []
        intercepts: list[float] = []
        for column in window_center_cols:
            slope, intercept = np.polyfit(x, numeric[column].to_numpy(dtype=float), deg=1)
            slopes.append(float(slope))
            intercepts.append(float(intercept))
        slopes_by_site[str(site)] = tuple(slopes)
        intercepts_by_site[str(site)] = tuple(intercepts)

    return DEJGuidedDynamicSignalExtractionRule(
        slopes_by_site=slopes_by_site,
        intercepts_by_site=intercepts_by_site,
        half_width=int(half_width),
    )

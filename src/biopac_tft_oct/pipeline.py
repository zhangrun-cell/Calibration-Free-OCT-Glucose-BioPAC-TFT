"""Sequential Bio-PAC and frozen, morphology-guided OCT feature extraction."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .biopac import causal_biopac_process
from .features import _smooth_profile
from .rule_discovery import DEJGuidedDynamicSignalExtractionRule


def validate_cohorts(manifest: pd.DataFrame, prediction_subjects=None, *, require_paper_counts=False):
    """Keep discovery labels and their participants outside predictor evaluation."""
    required = {"cohort", "subject", "session"}
    if not required.issubset(manifest.columns) or manifest[list(required)].isna().any().any():
        raise ValueError("Cohort manifest requires nonempty cohort, subject and session columns.")
    if not manifest.cohort.isin(["discovery", "prediction"]).all():
        raise ValueError("Cohort must be discovery or prediction.")
    if manifest.session.duplicated().any():
        raise ValueError("Session IDs must be globally unique across the two cohorts.")
    discovery = manifest.loc[manifest.cohort.eq("discovery")]
    prediction = manifest.loc[manifest.cohort.eq("prediction")]
    if discovery.empty or prediction.empty:
        raise ValueError("Both rule-discovery and prediction cohorts must be documented.")
    if set(discovery.subject) & set(prediction.subject):
        raise ValueError("Rule-discovery and prediction participants must not overlap.")
    if prediction_subjects is not None and set(prediction.subject) != set(prediction_subjects):
        raise ValueError("Prediction participants do not match the cohort manifest.")
    if require_paper_counts:
        counts = (len(discovery), discovery.subject.nunique(), len(prediction), prediction.subject.nunique())
        if counts != (28, 21, 72, 41):
            raise ValueError(f"Expected discovery 28/21 and prediction 72/41; observed {counts}.")
    return {"discovery_sessions": len(discovery), "discovery_participants": discovery.subject.nunique(),
            "prediction_sessions": len(prediction), "prediction_participants": prediction.subject.nunique()}


def save_rule(rule, path, *, discovery_subjects, discovery_sessions):
    payload = asdict(rule)
    payload.update(schema_version=1, anatomical_scale="epidermal_peak_to_dej_pixels",
                   discovery_subjects=sorted(set(discovery_subjects)),
                   discovery_sessions=sorted(set(discovery_sessions)))
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("Do not overwrite a locked DEJ rule; use a new file.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_rule(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("anatomical_scale") != "epidermal_peak_to_dej_pixels":
        raise ValueError("The rule must declare the same morphology scale used in discovery.")
    if payload.get("half_width") != 5:
        raise ValueError("The locked rule must use 11-pixel windows.")
    rule = DEJGuidedDynamicSignalExtractionRule(payload["slopes_by_site"], payload["intercepts_by_site"], 5)
    return rule, payload


def morphology_thickness(profile, epidermal_range, dej_range, smooth_window=5):
    """Locate two morphology peaks in a current A-scan, never a session mean."""
    scan = np.asarray(profile, dtype=float)
    if scan.ndim != 1 or not np.isfinite(scan).all():
        raise ValueError("A morphology A-scan must be one-dimensional and finite.")
    smoothed = _smooth_profile(scan, smooth_window)
    peaks, _ = find_peaks(smoothed)
    selected = []
    for start, stop in (epidermal_range, dej_range):
        if not 0 <= start < stop <= len(scan):
            raise ValueError("Peak search ranges must be within the current A-scan.")
        candidates = peaks[(peaks >= start) & (peaks < stop)]
        if not len(candidates):
            raise ValueError("No morphology peak in the declared range; do not invent a DEJ anchor.")
        selected.append(int(candidates[0]))
    epidermal, dej = selected
    if dej <= epidermal:
        raise ValueError("The DEJ-associated peak must be deeper than the epidermal peak.")
    return epidermal, dej, dej - epidermal


def sequential_signal_features(oct_profiles, *, rule, site, epidermal_range, dej_range,
                               n_segments=10, max_shift=30, epidermis_depth=10, min_history=10):
    """Map acquired depth profiles to five features without any glucose argument.

    Surface flattening and lateral volume averaging precede this function.
    Search ranges are acquisition-specific inputs, not fabricated cohort coefficients.
    The existing rule API names its scalar ``dej_depth``; here it receives the
    peak-to-peak thickness scale T used in the manuscript's c_j = round(a_j T+b_j).
    """
    profiles = np.asarray(oct_profiles, dtype=float)
    result = causal_biopac_process(profiles, n_segments=n_segments, max_shift=max_shift,
                                  epidermis_depth=epidermis_depth, min_history=min_history)
    features, peaks, centres = [], [], []
    for t in range(len(profiles)):
        epidermal, dej, thickness = morphology_thickness(result.aligned[t], epidermal_range, dej_range)
        windows = rule.window_ranges(site=site, dej_depth=thickness, n_depth=profiles.shape[1])
        features.append([float(np.mean(result.corrected[t, start:stop])) for start, stop in windows])
        peaks.append((epidermal, dej, thickness))
        centres.append(rule.centres(site=site, dej_depth=thickness))
    return np.asarray(features), np.asarray(peaks), np.asarray(centres)

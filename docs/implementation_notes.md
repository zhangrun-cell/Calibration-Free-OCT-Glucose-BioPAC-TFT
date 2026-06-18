# Implementation Notes and Differences From the Original Research Code

This repository is a public, de-identified reproducibility package. It is meant
to explain the computational logic of the manuscript and to provide runnable
examples without releasing private clinical OCT-glucose data.

## What This Repository Reproduces

- Bio-PAC morphology-aware depth alignment.
- Epidermis-referenced optical decoupling using a standardized epidermal
  fingerprint and depth-wise linear regression.
- DEJ-anchored five-window dynamic OCT feature construction.
- Distribution-aware candidate averaging for overlapping predictions.
- Regression metrics and Clarke error-grid zone percentages.
- Organized MATLAB/Python research scripts that preserve the internal algorithm
  structure without releasing private data.
- A small synthetic/anonymized demo that verifies the public code path.

## What This Repository Does Not Reproduce

- It does not include the raw clinical OCT volumes.
- It does not include the subject-level clinical CSV files used for manuscript
  training and evaluation.
- It does not include trained model weights.
- It does not rerun the full four-model training experiment that generated the
  reported manuscript tables and figures.

These omissions are intentional because the original dataset contains human
measurements and is restricted by institutional ethics and privacy constraints.

## Differences From the Original Bio-PAC MATLAB Code

The original internal Bio-PAC preprocessing was implemented in MATLAB. The
public Python implementation follows the same algorithmic structure but is a
clean-room, lightweight version for reproducibility and review.

Key differences:

1. **Input format**
   - Original code starts from internal OCT files and MATLAB variables such as
     `OCTSignal`.
   - Public code starts from a two-dimensional `oct_signal[t, z]` array after
     volume preprocessing.

2. **Structural envelope**
   - Original MATLAB code uses `abs(hilbert(scan))` followed by Gaussian
     smoothing.
   - Public Python code now follows the same logic using a NumPy FFT
     implementation of the Hilbert analytic envelope and Gaussian smoothing, so
     SciPy is not required for the demo.

3. **Warping-field interpolation**
   - Original MATLAB code uses segment shifts and `pchip` interpolation.
   - Public Python code uses NumPy linear interpolation to keep dependencies
     minimal.

4. **Optical decoupling**
   - Both versions average the epidermis-dominated superficial depths to obtain
     `N_epi(t)`, standardize it into `P(t)`, fit a depth-wise first-order
     regression, and subtract `alpha_z P(t)`.
   - This is the closest one-to-one part between the internal and public
     implementations.

5. **Five-window feature construction**
   - Original MATLAB code uses fixed center rules and half widths tied to the
     internal 200-pixel depth representation.
   - Public Python code exposes a generic DEJ-anchored function with configurable
     `dej_index`, `offsets`, `half_width`, `site`, and `search_range`. By
     default, it detects the first depth-axis intensity peak after skipping the
     first 10 pixels and constrains the search to the site-specific DEJ range
     (`27-45` pixels for arm/wrist and `55-75` pixels for finger). A manually
     supplied `dej_index` overrides automatic detection. Each window center is
     shifted as `anchor + offset`, with the default `half_width=5` matching the
     manuscript's 11-pixel window definition.

## Differences From the Original Darts Training Pipeline

The original internal training code is a full Darts workflow with private
subject folders, subject-wise splitting, four models, long-horizon overlapping
prediction, prediction export, and plotting scripts.

The public repository provides:

- the expected clinical CSV column format,
- a Darts training template,
- an organized subject-wise Darts pipeline skeleton in `research_code/python/`,
- optional dependencies for Darts/XGBoost,
- evaluation utilities for prediction CSV files.

This is sufficient to explain the modeling design, but it is not a full
clinical-result reproduction package.

## Organized Original-Style Code

The `research_code/` folder contains cleaned code that is intentionally closer
to the internal research scripts than the compact public demo:

- `research_code/matlab/biopac_preprocess_original_style.m` keeps the original
  MATLAB Bio-PAC sequence: 200-pixel crop, Hilbert envelope, Gaussian smoothing,
  segment-wise cross-correlation, PCHIP warping, epidermal z-score fingerprint,
  and depth-wise regression subtraction.
- `research_code/matlab/extract_five_depth_windows_original_style.m` organizes
  the five-window extraction step with automatic site-constrained first-peak
  anchoring and manual anchor override.
- `research_code/python/tft_subjectwise_pipeline_skeleton.py` documents the
  subject-wise Darts workflow without including private subject folders.

## Reviewer-Facing Interpretation

A reviewer can use this repository to understand how Bio-PAC works and to verify
that the public implementation runs. However, the repository should be described
as a **minimal reproducibility and method-interpretation package**, not as a
complete release of the clinical study dataset or the full experimental
pipeline.

Recommended wording:

> Code for Bio-PAC preprocessing, feature construction, prediction aggregation,
> and evaluation is publicly available. The clinical OCT-glucose data are not
> publicly released because of institutional ethics and privacy restrictions.

## Usage Notes

Run the Bio-PAC demo:

```bash
python scripts/run_demo.py
```

Run the prediction-metric demo:

```bash
python scripts/evaluate_predictions.py examples/demo_predictions.csv
```

For private clinical use, keep raw OCT volumes, subject folders, model weights,
and intermediate predictions outside the public repository.

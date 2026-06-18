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
   - Public Python code uses a lightweight smoothed structural envelope based on
     `abs(scan - median(scan))` to avoid requiring SciPy for the demo.

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
     `dej_index`, `offsets`, and `half_width`.

## Differences From the Original Darts Training Pipeline

The original internal training code is a full Darts workflow with private
subject folders, subject-wise splitting, four models, long-horizon overlapping
prediction, prediction export, and plotting scripts.

The public repository provides only:

- the expected clinical CSV column format,
- a Darts training template,
- optional dependencies for Darts/XGBoost,
- evaluation utilities for prediction CSV files.

This is sufficient to explain the modeling design, but it is not a full
clinical-result reproduction package.

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

# Calibration-Free OCT Glucose Monitoring via Bio-PAC and TFT

This repository provides a minimal reproducibility package for the manuscript:

**Calibration-Free and Generalizable OCT-Based Non-Invasive Glucose Monitoring via Bio-PAC and Temporal Fusion Transformer**

The code implements the core computational components used in the paper:

- Biologically Informed Physical Alignment and Compensation (Bio-PAC)
- DEJ-anchored dynamic OCT depth-window feature construction
- distribution-aware autoregressive aggregation
- Clarke error grid and regression-metric evaluation

The full clinical OCT-glucose dataset is not included because it contains human
measurements under institutional ethics and privacy restrictions. A small
anonymized/synthetic example is included to verify the code path.

## Repository Structure

```text
src/biopac_tft_oct/
  biopac.py          # morphology alignment and epidermis-referenced decoupling
  features.py        # DEJ-anchored depth-window features
  aggregation.py     # dense candidate averaging for overlapping predictions
  metrics.py         # Clarke zones and regression metrics
scripts/
  run_demo.py        # runnable demo using anonymized/synthetic data
  evaluate_predictions.py
examples/
  demo_anonymized_sample.csv
  demo_predictions.csv
docs/
  assets/figure3_biopac.jpg
  biopac_workflow.md
  data_availability.md
  implementation_notes.md
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The core Bio-PAC and evaluation demo requires only NumPy, pandas, and
matplotlib. Darts, PyTorch, and XGBoost are optional dependencies for full model
training and comparison workflows.

Optional training dependencies can be installed with:

```bash
pip install -r requirements-optional.txt
```

## Quick Demo

```bash
python scripts/run_demo.py
```

Expected output includes the corrected OCT matrix shape, DEJ window ranges,
feature matrix shape, regression metrics, and Clarke zone percentages.

## Data Format

For the clinical workflow, each session is represented as a minute-level time
series with columns similar to:

```text
date, glucose, slopmean1, slopmean2, slopmean3, slopmean4, slopmean5,
gender, age, diabetic_healthy, diabetic_t1, diabetic_t2, finger, arm
```

The five `slopmean` columns correspond to Bio-PAC-processed dynamic OCT
depth-window features. Static covariates include age, sex, diabetes status, and
measurement site.

## Bio-PAC Summary

Bio-PAC processes the depth-time OCT signal `I(t, z)` in two stages:

1. **Morphology-aware refractive-distortion alignment** estimates segment-wise
   shifts by cross-correlating whole envelope segments and interpolates them
   into a continuous depth-warping field.
2. **Epidermis-referenced optical decoupling** averages superficial depth
   pixels into a standardized epidermal fingerprint `P(t)` and removes the
   depth-specific fitted component `alpha_z P(t)`.

This implementation follows the formula logic described in the manuscript.

The manuscript Figure 3 and a detailed step-by-step explanation of the Bio-PAC
computational workflow are available in
[`docs/biopac_workflow.md`](docs/biopac_workflow.md).

![Figure 3. Bio-PAC implementation path](docs/assets/figure3_biopac.jpg)

## Evaluation

For prediction CSV files with columns

```text
actual_glucose,tft_global_pred,transformer_global_pred,xgboost_global_pred,tide_global_pred
```

run:

```bash
python scripts/evaluate_predictions.py path\to\predictions.csv
```

Glucose values are assumed to be in mmol/L. Clarke error grid classification is
computed after conversion to mg/dL.

A small synthetic prediction example is provided:

```bash
python scripts/evaluate_predictions.py examples\demo_predictions.csv
```

Important implementation differences between the original internal research
code and this public de-identified package are summarized in
[`docs/implementation_notes.md`](docs/implementation_notes.md).

## Data Availability

The clinical OCT-glucose data are not publicly released because of institutional
ethics and privacy restrictions. See `docs/data_availability.md`.

## Citation

If you use this code, please cite the associated manuscript once available.

## License

This project is released under the MIT License.

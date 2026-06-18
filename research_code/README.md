# Organized Research Code

This folder contains cleaned, publication-facing versions of the internal
research scripts. The goal is to preserve the original computational logic while
removing private clinical data, absolute workstation paths, generated figures,
model weights, and cache files.

## Contents

```text
matlab/
  biopac_preprocess_original_style.m
  extract_five_depth_windows_original_style.m
python/
  tft_subjectwise_pipeline_skeleton.py
```

## Relation to the Manuscript

- `biopac_preprocess_original_style.m` follows the internal MATLAB Bio-PAC
  workflow: 200-pixel depth crop, Hilbert-envelope morphology alignment,
  segment-wise cross-correlation, PCHIP coordinate warping, epidermis-referenced
  z-score fingerprinting, and depth-wise first-order regression decoupling.
- `extract_five_depth_windows_original_style.m` organizes the five-window
  feature extraction step. It supports the manuscript first-peak/DEJ anchor
  strategy with site-specific search ranges and a manual anchor override.
- `tft_subjectwise_pipeline_skeleton.py` summarizes the Darts workflow used for
  subject-wise model comparison without publishing private data folders.

The runnable lightweight implementation used by the public demo remains under
`src/biopac_tft_oct/` and `scripts/`.

## Data Policy

The clinical OCT volumes, subject-level CSV files, trained checkpoints, and
generated prediction outputs are intentionally excluded. To use the scripts on
private data, place files in an ignored local directory such as
`research_code/data/`.

# Bio-PAC Computational Workflow

This document explains the Bio-PAC preprocessing logic used in the manuscript
figure below. The goal is to make the signal path readable without exposing the
private clinical OCT-reference-blood-glucose dataset.

![Figure 3. Bio-PAC implementation path](assets/figure3_biopac.jpg)

## Input and Output

**Input.** Bio-PAC starts from raw OCT volumetric measurements `V_t` collected at
successive time points. In the manuscript experiments, one time point
corresponds to one approximately 1-min acquisition.

After B-scan preprocessing, surface flattening, lateral cropping, and averaging,
each time point is represented as one depth-intensity A-scan:

```text
I(t, z)
```

where `t` is the acquisition index along the time axis and `z` is the OCT depth
pixel. The value of `I(t, z)` is the backscattered OCT intensity at that time
and depth. In Figure 3, the depth axis is expressed in pixels; one pixel
corresponds to approximately 5.8574 micrometers.

**Output.** The output is a Bio-PAC-corrected depth-time OCT signal and the
derived DEJ-anchored dynamic OCT features used as covariates for TFT.

## Step 1: Preprocess OCT Volumes into Depth-Time Signals

For each acquisition time point, the OCT volume is flattened at the skin surface
and laterally averaged to suppress unstable lateral texture. This converts the
3-D OCT volume into a 1-D depth-intensity A-scan. The A-scans from all
acquisition times are then stacked along the time axis to form `I(t, z)`.

This representation keeps two important dimensions explicit:

- Along the **depth axis**, each curve is an OCT intensity profile through skin
  layers.
- Along the **time axis**, each fixed-depth curve describes how OCT intensity at
  the same depth pixel changes during the OGTT session.

## Step 2: Morphology-Aware Refractive-Distortion Alignment

The OCT depth coordinate is an apparent optical depth. During glucose dynamics
and probe-skin interaction, refractive-index and contact-state changes can make
the same anatomical layer appear at slightly different depth pixels over time.
If not corrected, a fixed pixel index may mix different tissue layers.

Bio-PAC aligns the depth morphology before optical compensation:

1. Each A-scan `I(t, z)` is compared with a reference A-scan.
2. The depth axis is divided into segments.
3. For each segment, Bio-PAC computes the shift that maximizes the correlation
   between the whole structural envelope of the current segment and the
   corresponding reference segment. The correlation is therefore computed over
   the whole segment curve, not only at isolated local extrema.
4. Segment shifts are interpolated into a continuous depth-warping field
   `Delta_t(z)`.
5. The A-scan is resampled using this field to obtain `I_align(t, z)`.

The warping changes the depth coordinate, not the raw intensity values
themselves. When one segment is shifted, neighboring intervals are smoothly
stretched or compressed so that the full depth axis remains continuous.
This alignment is causal with respect to OCT time: the current A-scan is aligned
to the initial OCT morphology reference independently, so no future OCT frame is
needed to align the current frame.

Implementation entry point:

```python
from biopac_tft_oct.biopac import morphology_align

aligned, shifts = morphology_align(
    oct_signal,
    n_segments=10,
    max_shift=30,
)
```

## Step 3: Epidermis-Referenced Optical Decoupling

Superficial epidermal OCT intensity often contains common-mode fluctuations
caused by contact pressure, temperature drift, surface coupling, and motion.
These components may look correlated with glucose during OGTT but are not
specific glucose responses.

Bio-PAC uses the epidermis as a reference pseudo-signal:

1. The superficial depth range is averaged along depth to obtain an epidermal
   temporal fingerprint:

   ```text
   N_epi(t) = mean_z I_align(t, z), z in superficial epidermal depths
   ```

2. In the sequential setting, `N_epi(t)` is formed from the available OCT prefix
   up to the current acquisition. The prefix fingerprint is standardized into a
   dimensionless reference pattern `P(t)`. In the released implementation this
   is z-score standardization, meaning zero mean and unit standard deviation,
   rather than min-max scaling.

3. For each depth pixel `z`, Bio-PAC fits a one-dimensional regression along the
   time axis:

   ```text
   I_align(t, z) = alpha_z P(t) + beta_z + epsilon_z(t)
   ```

   Here, `alpha_z P(t)` is the depth-specific contribution of the epidermal
   pseudo-signal, `beta_z` is the depth baseline, and `epsilon_z(t)` is the
   residual depth-specific temporal signal after removing the common superficial
   drift.

4. Bio-PAC subtracts only the fitted epidermal component:

   ```text
   I_corr(t, z) = I_align(t, z) - alpha_z P(t)
   ```

Because `alpha_z` is estimated separately for each depth, the same epidermal
fingerprint can be removed with depth-specific strength. This avoids applying a
single global subtraction coefficient to all tissue layers.

For a causal no-future-OCT workflow, `alpha_z` at time `t` is estimated from the
available prefix `I_align(1:t, z)`. The corrected value at the current time is
therefore:

```text
I_corr(t, z) = I_align(t, z) - alpha_z^(t) P_t(t)
```

where `P_t` and `alpha_z^(t)` are computed from frames already acquired by time
`t`. Future OCT frames and future reference glucose values are not used for this
calculation. The batch function remains useful for offline inspection of saved
sessions, but the manuscript inference boundary corresponds to the causal
prefix function.

Implementation entry point:

```python
from biopac_tft_oct.biopac import epidermis_referenced_decoupling

corrected, fingerprint, alpha = epidermis_referenced_decoupling(
    aligned,
    epidermis_depth=10,
)
```

Sequential implementation entry point:

```python
from biopac_tft_oct.biopac import causal_biopac_process

result = causal_biopac_process(
    oct_signal,
    n_segments=10,
    max_shift=30,
    epidermis_depth=10,
    min_history=10,
)
```

## Step 4: DEJ-Guided Dynamic Signal-Extraction Rule

After Bio-PAC correction, the depth-time OCT signal is summarized into five
dynamic OCT features using a DEJ-guided signal-extraction rule. This rule is
defined in a separate 28-session, 21-participant rule-discovery cohort, not
selected from the prediction cohort. Reference blood glucose trajectories are
used only during that discovery analysis to establish fixed, site-specific
relationships between OCT-derived DEJ depth and the five window centres.

For every prediction session, the first depth-axis intensity peak is determined
from OCT morphology after skipping the first 10 pixels. To avoid selecting an
unrelated structural peak, the search is restricted to the expected
site-specific DEJ range:

```text
arm/wrist: 27-45 pixels
finger:    55-75 pixels
```

The prediction-session DEJ depth is then inserted into the frozen mapping for
its measurement site:

```text
c_j,m = round(a_j,m * T_m + b_j,m),  j = 1, ..., 5
W_j = [c_j - 5, c_j + 5]
```

Here, `T_m` is the OCT-derived DEJ depth for site `m`; `(a_j,m, b_j,m)` are
locked discovery-cohort coefficients. Thus, the five centres move with tissue
morphology while reference blood glucose in the prediction cohort is neither
used for window selection nor for adaptation.

Implementation entry point:

```python
from biopac_tft_oct.features import dej_guided_dynamic_signal_features
from biopac_tft_oct.rule_discovery import fit_dej_guided_dynamic_signal_extraction_rule

rule = fit_dej_guided_dynamic_signal_extraction_rule(discovery_table)
features, windows, dej_depth = dej_guided_dynamic_signal_features(
    corrected_oct=corrected,
    rule=rule,
    site="arm",
)
```

The discovery table contains one row per discovery session, with `site`,
`dej_depth`, and five `window_center_1` to `window_center_5` columns. It is
not constructed from the prediction cohort.

The resulting feature matrix has shape:

```text
(time points, 5 OCT windows)
```

## Step 5: TFT-Based Conditional Prediction

The Bio-PAC-derived dynamic OCT features are combined with static subject
information, such as age, sex, diabetes status, and measurement site. The
DEJ-guided rule therefore links physically stabilized OCT profiles to a common
set of features before TFT prediction. A short `L0 = 10` initial reference blood
glucose history is used only to initialize the temporal state; it does not fit a
subject-specific OCT regression equation. It is distinct from the model history
`W = 50`, which determines why forecasts begin only after 50 observed points.

This distinction is central to the calibration-equation-free setting:

- Traditional calibration fits an individual model or regression equation for
  each subject.
- The proposed framework uses a shared model and uses the initial reference only
  as a starting temporal state for subsequent OCT-conditioned prediction.

## Run the Demo

The demo uses anonymized/synthetic data and verifies the public code path:

```bash
python scripts/run_demo.py
```

Expected outputs include:

- corrected OCT matrix shape
- DEJ window ranges
- five-window feature matrix shape
- regression metrics
- Clarke zone percentages

For private clinical data, keep raw OCT volumes and identifiable subject
information outside the public repository. Convert each session into the
documented CSV feature format before model training or evaluation.

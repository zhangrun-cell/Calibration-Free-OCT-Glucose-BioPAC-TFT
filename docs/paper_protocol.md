# Manuscript-to-code contract

The recommended entry point is `run_tft.py`, backed by
`scripts/run_paper_darts.py` and `scripts/tune_subjectwise_fivefold.py`.
Older demos and `research_code/` are retained for reference, not used by this entry point.

| Manuscript component | Executable implementation |
| --- | --- |
| 100 sessions / 62 participants | Optional cohort manifest checked by `pipeline.validate_cohorts`; counts never invented from filenames |
| Discovery: 28 sessions / 21 participants | `fit_dej_rule.py` checks discovery membership and separation from prediction participants |
| Prediction: 72 sessions / 41 participants | Only `subject-*/*.csv` prediction sessions enter TFT; `--require-paper-counts` verifies membership/counts with the manifest |
| Morphology alignment | Hilbert envelope, segment correlation, PCHIP displacement field, intensity-coordinate resampling |
| Optical decoupling | Prefix-only epidermal standardization/regression; subtraction of alpha times fingerprint preserves the depth baseline |
| DEJ-guided dynamic signal-extraction rule | Per-acquisition epidermal and DEJ peaks give thickness T; five centres use frozen site-specific aT+b coefficients |
| Five 11-pixel windows | Exact full width enforced; out-of-range windows fail rather than silently shrink |
| Initial reference L0=10 | Only first 10 real reference values enter the inference target history; subsequent unobserved context values are scaled-zero padding or generated predictions |
| W=50 / H=10 | First target is row 50 (zero-based); rows 0-49 have no predictions |
| Five-fold subject-wise cross-validation | Every participant is tested once; the next fold is validation and the other three are training; sessions remain independent |
| Candidate selection | Candidates ranked on autoregressive validation RMSE inside the outer fold; held-out test errors are not used to select parameters |
| No future OCT | Five OCT channels are past covariates, ending at the forecast origin; only deterministic Time extends into the horizon |
| Distribution-aware aggregation | Median quantile from each overlapping horizon; mean of the narrowest contiguous floor(0.30K) candidates, at least one |
| Nominal synchronization | Main evaluation offset is zero; -30 to +30 min are global sensitivity values, never individually fitted |
| Clinical/statistical evaluation | Finite/common locked mask, session/participant summaries, participant-cluster bootstrap and Clarke zones |
| Fig. 7 input sources | `--input-source full/time_only/time_plus_oct/time_plus_static/oct_only`; removed dynamic channels are constant-zero, removed static channels are absent |
| Zero-reference | `--zero-reference` replaces the first ten scaled reference values with the training-only population mean |
| Fig. 7d | `time-shift` changes explicit Time only; OCT/reference pairs and order remain unchanged |
| Fig. 7e | `paired-shuffle` / `paired-circular` reorder OCT and reference together; explicit Time and static covariates remain unchanged |

## Explicit boundaries

- **Forecast origin is not target time.** A forecast made after rows 0-49 uses
  OCT rows 0-49 to predict targets 50-59. On the next acquisition, the origin
  advances by one. This preserves the supplied Darts runner's W-to-H forecasting
  semantics; it does not expose future OCT as a known covariate.
- Training uses supervised target windows from training participants. Validation
  for candidate selection and held-out inference mask reference values after L0.
  Scalers are fitted exclusively on the three training folds.
- Paired stress reorders the entire physiological trajectory, including the
  reference segment used to initialize it. It is not an OCT-only ablation and
  does not break the OCT-reference pairing. It is diagnostic, not prospective
  real-life accuracy evidence.
- Exact repeated timestamp/OCT trajectories are kept in the same fold without
  merging participant identities. This extra guard is logged in the fold configuration.
- CSV prediction inputs are already processed OCT features. Their prospective
  provenance cannot be established from those five columns alone. Regenerating
  them requires acquisition-specific morphology settings and the actual frozen
  discovery coefficients/table. No example coefficients are represented as the
  manuscript's clinical coefficients.
- `fit_dej_rule.py` accepts the discovery analysis table of identified five peaks.
  It does not replace the upstream supervised peak-identification analysis.
- Device-specific volume parsing, cropping and surface detection precede the
  depth-profile extraction entry point and are not reconstructed from clinical CSVs.
- The historical manuscript denominator is 5567. The runner reports its actual
  valid count; it never truncates predictions or selects a mask to reach that number.
  A separately locked `--evaluation-mask` may be supplied to reproduce the
  original evaluated physical points, not to optimize the error.
- Clean source does not imply reproduced clinical RMSE. A complete clinical
  reproduction additionally requires the locked inputs, discovery coefficients,
  cohort/fold identities, evaluation mask, selected configuration and checkpoints.

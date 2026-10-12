# Implementation Notes

The recommended executable workflow is `run_tft.py`, not the older
`research_code/python/tft_subjectwise_pipeline_skeleton.py` or training template.

## Current implementation

- Bio-PAC uses a Hilbert structural envelope, segment-wise correlation, PCHIP
  displacement interpolation and intensity-coordinate resampling.
- Sequential compensation estimates epidermal standardization and regression
  coefficients using only each acquired prefix. The offline batch function is
  retained for inspection and is not the online inference path.
- `pipeline.sequential_signal_features` detects epidermal/DEJ peaks per
  acquisition and applies the frozen site-specific mapping to five 11-pixel
  windows. Whole-session average anchors and illustrative fixed-offset windows
  in earlier demo helpers are not used by this sequential entry point.
- The predictor consumes five precomputed channels as observed past covariates.
  Their prospective preprocessing provenance must be checked upstream.
- TFT runs five outer participant folds; validation-only selection, training-only
  scaling, ten-reference masking, fifty-point context, ten-point horizon,
  autoregressive feedback and concentrated-candidate aggregation are executable.
- Time perturbation retains the supplied explicit Time channel rather than
  silently regenerating its original values.
- Paired trajectory stress preserves OCT/reference pairing and leaves Time and
  static covariates unchanged.

The full manuscript-to-code contract is in
[paper_protocol.md](paper_protocol.md).

## Intentionally separate inputs

No raw clinical OCT volumes, private participant tables, fitted clinical DEJ
coefficients, evaluation mask or trained checkpoints are embedded in this source
release. Volume cropping, boundary detection and surface flattening are
device-specific upstream operations. The DEJ fitting entry point consumes the
supervised discovery peak table rather than claiming to recreate its original
peak-identification analysis from five-channel prediction CSVs.

The supplied Darts source informed the runner and parameter candidates; recovered
plot predictions are not substituted for newly generated forecasts. No code
forces the number of evaluated points or tunes against archived manuscript RMSE.

Older MATLAB examples, Python skeletons, demos and templates are kept for
historical reference and are not the recommended reproduction entry points.
They may use offline processing or illustrative parameters. Do not use their
output as evidence that the sequential clinical inference boundary was verified.

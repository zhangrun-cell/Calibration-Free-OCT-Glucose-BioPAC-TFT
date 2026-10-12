# Calibration-equation-free OCT-based blood glucose monitoring

Manuscript-aligned source for Bio-PAC, the DEJ-guided dynamic signal-extraction
rule, and Temporal Fusion Transformer (TFT) prediction. The recommended runner
is now executable, not only a Darts template. Only TFT is trained by this entry
point. Clinical data, weights and generated results are not part of the source release.

## Fixed Prediction Protocol

- L0 = 10 reference blood glucose points initialize the target history.
- W = 50 input points; H = 10 forecast points. The first 50 rows have no prediction.
- Five-fold participant-grouped evaluation: each participant is held out once,
  with every session of that participant kept together.
- OCT is an observed past covariate, never a known future input.
- Overlapping horizon candidates are aggregated using the most concentrated 30%.
- Hyperparameters are selected on validation participants, not the outer test set.
- No participant-specific optical calibration equation or fitted delay is used.

The study has two separate roles: discovery (28 sessions / 21 participants) and
prediction (72 sessions / 41 participants). Only prediction sessions enter TFT.
A cohort manifest can verify the separation and the full 100-session / 62-participant
allocation. The runner does not infer discovery membership from the prediction CSVs.

## Install

Use Python 3.12. For optical utilities and lightweight tests:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For TFT training:

```bash
pip install -r requirements-training.txt
```

Darts 0.39.0 and Torch 2.9.1 are pinned. Install the appropriate CPU/CUDA Torch
build for your hardware. Training is local; it does not submit data to a service.

## Check and Train

One private CSV per session, grouped under `subject-*/`. See
[DATA_FORMAT.md](DATA_FORMAT.md) for columns, identities and the locked-mask format.

```bash
python run_tft.py --mode check --data-dir ../private_data --output-dir ../checks/new_check --accelerator cpu
python run_tft.py --mode train --data-dir ../private_data --output-dir ../runs/new_tft --accelerator gpu
python run_tft.py --mode audit --data-dir ../private_data --output-dir ../runs/new_tft
```

The training command evaluates the candidates in `config/tft_candidates.json`
within each outer fold, then tests its validation-selected winner once.
Use a new output directory; source CSVs are never overwritten.

For an author-verified cohort manifest or a previously locked evaluation mask,
use the underlying runner:

```bash
python scripts/run_paper_darts.py --data-dir ../private_data --output-dir ../runs/fixed_config --cohort-manifest ../cohorts.csv --require-paper-counts --evaluation-mask ../locked_mask.csv --accelerator gpu
```

This command trains the fixed configuration once per fold. It does not search
test metrics for a configuration that matches a reported number.

## Bio-PAC and DEJ Extraction

`scripts/fit_dej_rule.py` fits and freezes the site-specific mapping from an
independent discovery analysis table and verifies the cohort manifest.

`scripts/extract_features.py` processes already surface-aligned, laterally
averaged OCT profiles using causal Bio-PAC and per-acquisition morphology peaks.
The five windows use the frozen mapping, are 11 pixels wide, and never receive
prediction-cohort reference blood glucose as an extraction input.

Device-specific raw-volume parsing/surface detection, clinical discovery
coefficients and restricted source data are not fabricated or embedded here.
Precomputed five-channel CSVs alone cannot verify their upstream causal provenance.

## Controls and Evaluation

Input-source, zero-reference, Time-only shift, and paired physiological-trajectory
stress controls are explicit options/helpers. Paired stress reorders OCT and its
reference together while Time stays fixed; it is not an OCT-reference disruption.
See [docs/paper_protocol.md](docs/paper_protocol.md) for exact semantics and boundaries.

Reports are generated only in the chosen output folder: predictions, fold/session/
participant summaries, Clarke zones, participant-cluster bootstrap and fixed global
offset sensitivity. The valid denominator is measured, never forced to the
historical manuscript value of 5567.

Evaluate stress controls using the same validation-selected full-input models:

```bash
python scripts/run_tft_controls.py --run-dir ../runs/new_tft --data-dir ../private_data --output-dir ../runs/new_controls --accelerator gpu
```

Input-source controls require separately trained models, using the underlying
runner's `--input-source` option. Zero-ref Time combines `time_only` with
`--zero-reference`. Main protocol lengths remain fixed in these TFT utilities;
they do not claim to rerun the separate historical four-model/control analyses.

## Clean Source Package

```bash
python scripts/build_clean_source.py --output ../OCT_TFT_source_only.zip
```

An explicit allowlist excludes data, clinical/plot predictions, checkpoints,
logs, caches, synthetic CSV fixtures and earlier research scripts. Original
private files and earlier packages are not modified or deleted.

## Source Layout

```text
run_tft.py                      check / train / audit
config/tft_candidates.json      validation-only TFT search space
scripts/run_paper_darts.py      fixed-protocol TFT runner
scripts/tune_subjectwise_fivefold.py
scripts/fit_dej_rule.py
scripts/extract_features.py
scripts/run_tft_controls.py     stress tests using saved full-input models
src/biopac_tft_oct/             optical, extraction, protocol and metric functions
tests/                         executable protocol and causality checks
docs/paper_protocol.md          manuscript-to-code contract
```

Earlier `research_code/` and demo/template scripts remain for reference but are
not called by the recommended runner. Existing example CSVs are synthetic demo
fixtures, not clinical results. A source-only ZIP excludes these fixtures and all
run outputs.

## Data and Reproduction

Clinical health information is restricted by ethics/privacy requirements.
See [docs/data_availability.md](docs/data_availability.md).
Passing software tests verifies the implemented logic; it does not establish that
clinical RMSE or the published denominator has been reproduced. Exact numerical
reproduction requires the actual frozen inputs, identities, coefficients and mask.

Released under the repository's existing MIT License.

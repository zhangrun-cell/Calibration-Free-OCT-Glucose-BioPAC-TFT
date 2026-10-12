# Private input format

Prediction sessions are separate UTF-8 CSV files under:

```text
private_data/
  subject-1/session_a.csv
  subject-1/session_b.csv
  subject-2/session_c.csv
```

Required columns:

```text
date,glucose,slopmean1,slopmean2,slopmean3,slopmean4,slopmean5,age,gender,diabetic_healthy,diabetic_t1,diabetic_t2,finger,arm
```

- `glucose`: reference blood glucose trajectory in mmol/L. Only the first ten
  values are model inputs at inference; later values are for scoring only.
- `slopmean1`-`slopmean5`: five locked DEJ-guided Bio-PAC feature channels,
  preserving acquisition order. Original column names are retained for compatibility.
- `date`: minute-level timestamp, or an Excel date serial.
- Static covariates must be constant within a session. Binary/category coding
  must be consistent across participants. Age is divided by 100 before modelling.
- Prediction/export columns already in a CSV are ignored, never used as inputs.
- Missing OCT can be forward-filled only from acquired values. Missing starting
  OCT values fail; no backward fill is permitted.
- Regular minute gaps are explicitly recorded. Reference interpolation across
  newly inserted minutes is for supervised/evaluation bookkeeping only;
  inserted references are not scored, and cannot supply the ten initial references.

Optional participant map: `session,subject`, one row for every prediction file,
using paths such as `subject-1/session_a.csv`. An author-verified folder identity
is sufficient when no map is supplied.

Cohort manifest: `cohort,subject,session`, where cohort is `discovery` or
`prediction`. IDs must be globally consistent: the same person cannot have
different IDs in the two cohorts. Prediction session IDs match relative CSV paths.
Discovery session IDs are distinct. With `--require-paper-counts`, expected
counts are discovery 28/21 and prediction 72/41.

Discovery analysis table:

```text
subject,session,site,thickness_scale,window_center_1,window_center_2,window_center_3,window_center_4,window_center_5
```

`thickness_scale` is DEJ-peak minus epidermal-peak depth in pixels, using the
same surface-aligned coordinate system as the five absolute-correlation peak
centres. Discovery reference glucose may identify these peaks; prediction
reference glucose cannot determine or adapt them. Site is `arm` or `finger`.

Raw-feature extraction input is a finite NumPy `(time, depth)` array of
surface-aligned, laterally averaged OCT intensity profiles. Search ranges for
the two morphology peaks are acquisition-specific, explicitly supplied, and
end-exclusive. The extractor outputs optical features and morphology diagnostics;
timestamps/reference/static fields are joined separately to form prediction CSVs.

Optional locked evaluation mask: `session,timestamp,included`, where `included`
is true/false or 1/0. It is applied after aggregation without changing predictions.
Do not derive this mask from residuals or choose it to match a reported RMSE.

# Changelog

## Unreleased (2026-09-21)

### Added — comparison methods (separate from ComCAT)

- `combat_family.py`: NumPy port of the R package
  [ComBatFamily](https://github.com/andy1764/ComBatFamily) v0.2.2, with all
  features fitted at once instead of one model per feature:
  - `combat_eb()`: ComBat with empirical Bayes (`comfam(model = lm)`)
  - `covbat()`: CovBat (`covfam(model = lm)`; Chen et al., 2022)
  - `combatls()`: ComBatLS (`combatls()`, i.e. `comfam(model = gamlss, family = NO())`;
    Gardner et al., 2025)
- `combat_family_ui.py`: `covbat_ui()`, `combatls_ui()`, `combat_eb_ui()` and a CLI.
  File I/O is imported from `comcat_ui.py`, so inputs are the same. Outputs go to
  `covbat_*`, `combatls_*` and `combateb_*` folders; `nuisance` is accepted but
  ignored, with a warning.
- `run_combat_family_from_files.py`: template running all three methods on the
  samples of `run_comcat_from_files.py` that have site labels.
- `tests/test_combat_family.py` and `tests/combat_family_reference.R`: checks of
  known properties and agreement with the R package: ~1e-13 for ComBat and CovBat;
  ~5e-7 for ComBatLS, where `gamlss` itself stops about 1e-7 from the
  maximum-likelihood solution.

### Added — optional ComCAT extensions (`comcat.py`, `comcat_ui.py`)

All are off by default. With the defaults, output is bit-identical to the previous
version. This was verified against the previous commit on ON-Harmony data for:
sites + nuisance + preserve, nuisance only, sites only, `mean_only`, `ref_batch`,
`comcat_from_training()`, and the `comcat_ui()` MAT output and estimate files.

- `preserve_df` / `preserve_bounds`: B-spline expansion of continuous preserve
  covariates. Folder suffix `_preserve<n>_gam<df>`.
- `residual_delta`: site variances δ estimated after removing the additive site and
  nuisance effects, as in ComBat. Folder suffix `_resdelta`.
- `nuisance_scale`: removes nuisance-dependent variance with a per-feature log-linear
  variance model (site + nuisance + preserve), and returns a per-feature
  likelihood-ratio test of nuisance effects on the variance (`scale_lr`, `scale_p`,
  `scale_lr_df` in the estimates; saved as maps by `comcat_ui(save_estimates=True)`).
  Implies `residual_delta`. Folder suffix `_scale`.
- `comcat_from_training()` applies all three options to new data with the
  training estimates. Estimate dicts without the new keys (e.g. from
  `decentralized_comcat.py`) still work.
- `comcat_ui()`: new keyword arguments and CLI flags `--preserve-df`,
  `--nuisance-scale` and `--residual-delta`. The log `.mat` has the fields
  `preserve_df`, `nuisance_scale` and `residual_delta`.
- `run_comcat_from_files.py`: `PRESERVE_DF`, `NUISANCE_SCALE` and `RESIDUAL_DELTA`
  settings.
- `tests/test_comcat_options.py`: simulations with known effects for each option.

### Documentation

- `ComCat-Theory.md`: new section *Optional extensions* with the model and the
  estimators. Added a note that the default δ contains nuisance-explained variance.
  Corrected the knot placement (interior knots at quantiles, not equally spaced).
- `README.md`: sections *Optional model extensions* and *Comparison Methods*, a
  testing overview, and updated signatures.

### Known issues (not changed)

- The default δ (as in `comcat.m`) contains nuisance-explained variance, which shrinks
  the residual variance of harmonized data by about `1 − R²` of the within-site
  nuisance fit. This also applies to single-site runs through `comcat_ui()`, which
  codes a missing `batch` as one site. See `residual_delta`.
- `comcat_ui._load_mat()` uses `scipy.io.loadmat` without `mat_dtype=True`.
  Integer-valued double variables (e.g. `age`, `label`) are therefore written back as
  `uint8`, and arithmetic on them in MATLAB saturates.

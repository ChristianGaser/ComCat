# Changelog

## Unreleased (2026-10-28)

### Added — empirical Bayes estimation of site effects (on by default)

- `comcat()`: new argument `eb=True`. With at least two sites, the additive (γ) and
  multiplicative (δ) site effects are estimated with the parametric empirical Bayes
  method of ComBat (Johnson et al., 2007): per-site normal and inverse-gamma priors are
  estimated over all features by the method of moments, and the posterior means are
  found with the iteration and stopping rule of `sva`/ComBatFamily. The update is written
  in terms of the sufficient statistics (γ̂, δ̂, nᵢ). It therefore applies to the default
  δ, `residual_delta` and `nuisance_scale` alike.
- ComCAT's site coefficients are levels at the zero of the nuisance basis and share a
  per-feature offset from the nuisance model. They are shrunk as differences to the
  size-weighted site mean, or to `ref_batch`, which are ComBat's site effects. Shrinking
  them directly would leave a site-dependent part of the offset in the data. In a
  simulation with a strong IQM effect, it also lost almost all of the gain
  (location error 0.30 → 0.30 instead of 0.30 → 0.23).
- The reference site and sites with a single subject keep their L/S estimates. With
  `mean_only`, the posterior of γ still uses the estimated site variance. This is the
  behaviour of `combat_family.combat_eb()`; `sva`/neuroCombat use n = 1, δ = 1 instead.
- Without nuisance covariates, ComCAT now equals ComBat with EB:
  `combat_family.combat_eb()` to ~1e-13, with and without EB and `mean_only`.
- Estimates: the returned `gamma_hat` / `delta_hat` are the effects that were applied
  (EB estimates when `eb=True`). The estimates dict adds `eb`, `gamma_star_masked`,
  `delta_star_masked` and `eb_priors`. `comcat_from_training()` uses the EB estimates
  when present; older dicts work as before.
- `decentralized_comcat.py`: `decentralized_fit(..., eb=True)` runs the EB step on the
  aggregator. No further data leave the sites, and the result matches centralized
  `comcat()` to ~4e-14 with and without EB.
- `comcat_ui()`: argument `eb=True`, CLI flag `--no-eb`, field `eb` in the log `.mat`.
  Output folders get `_eb` after `_sites` (e.g. `comcat_sites_eb_preserve1_nuisance7_gam5`),
  so EB results never overwrite results of earlier versions.
- `run_comcat_from_files.py`: setting `EB`.
- `tests/test_comcat_eb.py`: equality with ComBat, single site, a simulation with small
  sites and an IQM where EB brings site effects and harmonized data closer to the truth
  (fails if the site effects are not centred), `ref_batch`, `comcat_from_training()`,
  single-subject sites.

### Changed

- **Default output with ≥ 2 sites.** EB is on by default, so harmonized data differ from
  earlier versions. On real data (gray matter, 8 mm) the RMS change per site was
  ≤ 0.05 SD for sites with > 50 subjects, 0.03–0.16 SD for 11–50, 0.06–0.23 SD for 6–10
  and 0.09–0.7 SD for 2–5. `eb=False` is bit-identical to the previous version. This was
  checked for 16 combinations of `nuisance`, `residual_delta`, `nuisance_scale`,
  `mean_only`, `ref_batch` and `preserve_df`. Single-site runs are unchanged.
- `tests/test_comcat_py.py` runs the multi-site MATLAB comparisons with `eb=False`
  (`comcat.m` has no EB step). `tests/poc_decentralized_comcat.py` compares with
  `comcat(..., eb=False)`.

### Evaluation (real data; details in ComCat-Theory.md)

- Split halves of ON-Harmony, Buchert, ABIDE, NormSample and ADNI, 10 splits: EB reduced
  the error of the site variances by 26–87% in every sample. It reduced the error of the
  site locations by 9–72% without IQMs.
- Held-out traveling subjects, sites only:
  - Buchert, one scan per scanner held out: the spread of the held-out scans was 5,364×
    raw with L/S and 0.55× with EB.
  - ON-Harmony, leave one subject out: within-subject / total variance was 0.112 with L/S
    and 0.083 with EB (raw 0.074).
- ADNI with IQMs, in-sample: the 99th percentile of the harmonized values at a site with
  two subjects was 17.5 SD from the voxel mean with L/S, and 1.5 SD with EB.
- Limitation: with IQMs that vary almost only between sites (`res_RMS`), the site levels
  at fixed IQM are not identified (rank-deficient or near-singular designs in all five
  samples). The location shrinkage then has almost no effect: 0% error reduction in
  Buchert, ABIDE and NormSample, 17% in ON-Harmony, 26% in ADNI. The variance shrinkage is
  unaffected.

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
  likelihood-ratio test of nuisance effects on the variance, studentized for
  non-normal residuals (`scale_lr`, `scale_p`, `scale_kurtosis`, `scale_lr_df` and
  `scale_resid_df` in the estimates; the first three are saved as maps by
  `comcat_ui(save_estimates=True)`). Implies `residual_delta`. Folder suffix `_scale`.
- `comcat_from_training()` applies all three options to new data with the
  training estimates. Estimate dicts without the new keys (e.g. from
  `decentralized_comcat.py`) still work.
- `comcat_ui()`: new keyword arguments and CLI flags `--preserve-df`,
  `--nuisance-scale` and `--residual-delta`. The log `.mat` has the fields
  `preserve_df`, `nuisance_scale` and `residual_delta`.
- `run_comcat_from_files.py`: `PRESERVE_DF`, `NUISANCE_SCALE` and `RESIDUAL_DELTA`
  settings.
- `tests/test_comcat_options.py`: simulations with known effects for each option.

### Fixed

- `ref_batch` together with `nuisance`: the harmonized sites were aligned to the
  intercept of the reference site, i.e. to its level at the zero of the nuisance
  spline basis instead of at its own nuisance values, which left an offset against
  the untouched reference data. When the nuisance splines are collinear with the site
  indicators (e.g. an IQM with many tied values, such as `res_RMS`, gives repeated
  knots), this intercept is not identified and the harmonized data reached ±1e11.
  The target level is now the mean fitted site + nuisance level of the reference
  site. Without `ref_batch`, or with `ref_batch` but no nuisance, the output is
  unchanged (identical, or within 1e-13). Test: `test_ref_batch_with_nuisance`.
- `ref_batch`: the other sites were scaled to the pooled residual variance of all
  sites instead of the variance of the reference site, so a variance difference to the
  untouched reference data remained (noise SD 1, 2 and 0.5 with the SD-2 site as
  reference gave 1.32, 1.99 and 1.31). They are now scaled to the δ of the reference
  site, as with the reference batch of neuroCombat, in `comcat()` and
  `comcat_from_training()`. With `residual_delta=True` the residual variance of the
  harmonized sites equals that of the reference site. Without `ref_batch`, or with
  `mean_only`, the output is bit-identical (checked on ON-Harmony). Test:
  `test_ref_batch_variance`.
- `nuisance_scale` test: the likelihood-ratio test assumed normal residuals and was
  too liberal for skewed or heavy-tailed data. In gray matter at 8 mm of the lifespan
  sample (2,241 subjects, real residual shapes, three random IQMs without effect on
  the variance), 12.2% of the voxels had `scale_p` < 0.05, and 37% of the voxels with
  |skewness| > 1. The statistic is now studentized by the kurtosis of the residuals of
  the null variance model (Koenker, 1981) and compared with an F distribution: 5.0%
  (4.9–5.2% in all skewness ranges). The harmonized data do not change. Test:
  `test_scale_lr_test` with t5 and chi-square noise.
- `comcat.py` could not be imported with Python 3.8 or 3.9, which `pyproject.toml`
  supports: annotations such as `np.ndarray | None` were evaluated when the functions
  were defined. Added `from __future__ import annotations`; the tests pass with
  Python 3.9.13.
- `tests/test_combat_family.py` failed instead of skipping the comparison with R when
  ComBatFamily was not installed and R did not run in English, because it searched
  the translated error message. `combat_family_reference.R` now checks the packages
  with `requireNamespace()` and exits with status 3, which the test skips.

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
- `nuisance_scale` in small samples with many nuisance columns: the residuals of a
  large mean model contain fewer than n independent values, so the variance test is
  liberal even for normal residuals, and the IQM effect on the variance is
  underestimated. In simulations with the design of ON-Harmony (80 scans, 6 sites,
  7 IQMs, `gam_df=5`, i.e. 42 mean-model columns), about 14% of the features had
  p < 0.05 under H0 (8.5% with `gam_df=4`), and a true effect of 0.25 on log SD was
  estimated as 0.11 (0.20 with 240 scans). This was the same before the
  studentization; dividing the residuals by `sqrt(1 - leverage)` did not help.
- `comcat_ui._load_mat()` uses `scipy.io.loadmat` without `mat_dtype=True`.
  Integer-valued double variables (e.g. `age`, `label`) are therefore written back as
  `uint8`, and arithmetic on them in MATLAB saturates.

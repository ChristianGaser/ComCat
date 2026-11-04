# Changelog

## Unreleased (2026-11-04)

### Changed — sites with few subjects: location only

- `comcat()` corrects sites with fewer than `min_scale_n` subjects in the fit (default 4,
  at least two) in location only, without the scale step, and so do
  `comcat_from_training()` and `comcat_crossfit()`. A site variance from two or three
  subjects is so noisy that the scale step added more held-out variance than it removed:
  in Buchert (116 scanners, 2-3 training scans per scanner in a 3-fold cross-fit) the
  held-out variance across scans was 0.47-0.49 with the scale step vs. 0.30-0.33 without
  on scanners with 2-3 training scans, and 0.38 vs. 0.66 on those with 4 or more. In
  smoothed data the noise of these few subjects is shared across features, so the
  empirical Bayes pooling over features does not average it out. Not applied with
  `combat_compatible=True` (ComBat); `min_scale_n=2` restores the earlier behaviour.

### Added — `comcat_mat.py --min-scale-n`

- `--min-scale-n N` sets `min_scale_n` (default 4). For NeuroGAMLSS z-maps use 2: their
  site adaptation has one common scale, so the scale step of small sites matters more
  than its noise (Buchert, held-out log density +64 with 2 vs -52 with 4, 10^-3 nat
  relative to the adaptation alone; ADNI unchanged).

### Added — `clip`

- `comcat()`, `comcat_from_training()` and `comcat_crossfit()` take `clip=True`;
  `clip=False` returns the affine harmonization without setting negative values to 0.
  The clipped transform is not invertible at 0, so a density of the data by the change
  of variables (NeuroGAMLSS `harmonization_comparison.py`) needs the affine one.

### Changed — non-negative data stay non-negative

- `comcat()` and `comcat_from_training()` (and with them `comcat_crossfit()`,
  `comcat_ui.py` and `comcat_mat.py`) set negative harmonized values to 0 if the input
  has no negative values, e.g. tissue densities. Before, location and scale corrections
  pushed low values below 0: 0.13-0.45% of the values of six s6mwp1 gray matter test
  samples, mostly at voxels with little gray matter, which normative models of raw data
  turned into extreme z-scores (minimum -116 in ADNI, -17 after clipping; the controls'
  z SD unchanged). Signed data (e.g. z-maps) are not changed; data that are signed by
  nature but positive in the sample would be clipped too.

### Added — `comcat_mat.py`

- Command line for the data `Y` of BA_data2mat mat-files (v5 and v7.3), fitted on the
  controls and cross-fitted with `comcat_crossfit()` (`--folds 0`: one model of the
  controls for all scans), before BrainAGE or the normative models of NeuroGAMLSS. The
  output is a copy of the mat-file with `Y` replaced. Tests: `tests/test_comcat_mat.py`.

## Unreleased (2026-11-03)

### Added — cross-sectional model for repeated scans

- `comcat_crossfit(..., subject=labels)` keeps all scans of a subject in one fold and,
  by default (`subject_terms=False`), estimates the effects across subjects, as for
  single scans; `subject_terms=True` adds subject terms (effects within subjects, the
  behaviour of `subject=` before). `folds=k` without subject labels splits the scans of
  a subject between the folds. The README recommends the cross-sectional model for
  repeated scans, with subject terms as a reference (and for few subjects scanned
  unevenly across sites), and its cross-fitting example now uses the df-consistent site
  variances instead of `mean_only=True`.
- A warning when subject terms cannot identify site effects: sites that are not linked
  by subjects scanned at several of them (e.g. every subject at one site).

### Fixed — sites with a single subject

- The location of such a site is that subject's whole deviation. In-sample, the subject
  was therefore replaced by its predicted values (both settings of `combat_compatible`),
  and every other scan of the site, e.g. patients when fitted on the controls, had that
  subject's anatomy subtracted. With `combat_compatible=True`, the site variance was
  also undefined (NaN), and new scans of the site were replaced by their predicted
  values. Such sites now get no site correction, as unseen sites: the size-weighted
  mean level of the other sites and variance 1 (warning). ADNI fitted on the controls
  has 4 sites with one control (25 subjects).
- `comcat_crossfit`: held-out samples of a site without fit samples in the other folds
  were harmonized with the model of all fit samples, i.e. in-sample (for a site with one
  subject: replaced by its predicted values). They now get the fold's model without site
  correction (warning). With `ref_batch`, the reference site is in every fit and keeps
  its data (before: an error if it had a single unit).

### Fixed — features and missing values

- Features without variance were returned as 0; they keep their values (in
  `comcat_from_training()`, features without variance in the training keep the new
  values).
- `comcat_from_training()` replaced missing values of the data by predicted values; they
  stay missing. `comcat_crossfit()` sets features with missing values to missing for
  all samples, as `comcat()` does.
- Missing or non-finite nuisance or preserved values raise an error (before, one
  missing IQM value silently dropped the whole column, with a misleading warning).
- `comcat_from_training()`: with `ref_batch`, samples of unseen sites are scaled to the
  variance of the reference site, as the known sites.
- The estimates store `var_scale` (pooled variance of the standardization relative to
  the unbiased one), needed to express df-consistent site variances in data units.

### Tests and documentation

- `tests/test_comcat_edge_cases.py` (new); `test_single_subject_site` checks the
  location as well; `test_defaults_unchanged` checks the documented defaults (before, it
  compared two identical calls); the EB test expresses site variances with `var_scale`;
  `check()` in `tests/test_comcat_nuisance_models.py` raises, as in the other files.
- README and ComCat-Theory.md: default P-splines and residual site variances described
  as the defaults, corrected claims of bit-identity with earlier versions (they need
  `combat_compatible=True` and `nuisance_model='spline'`), folder names with `_dfc`,
  full signatures, the decentralized version (matches only with those two settings).
- README: `tests/test_comcat_py.py` (comparison with the MATLAB `comcat.m`) agrees only
  without nuisance variables since the B-spline nuisance model (2026-06; `comcat.m`
  uses polynomials); stated, generator in `deprecated/gen_test_data.m`.

## Unreleased (2026-11-02)

### Changed — df-consistent site variances (default)

- The scale step of ComBat estimates every site variance with n_i − 1 degrees of
  freedom but the pooled variance with N, and its EB posterior divides by n_i. With
  small sites the site variances are therefore too large on average, and the scale step
  shrinks the harmonized residuals, also of new subjects, although the sites do not
  differ in variance (held out, to 0.85 of a location-only correction with 3 subjects
  per site in the fit, 0.92 with 5, 0.97 with 10, 1.00 with 40).
- By default, `comcat()` now uses df-consistent site variances: the residual sum of
  squares of a site over its residual df (n_i minus the summed leverage of its subjects
  in the full penalized model), relative to RSS / (N − tr H), always from the residuals
  after the nuisance effects. The EB posterior treats them as sums of squares with these
  df, and the prior is estimated net of their sampling variance. Sites of equal variance
  get δ² ≈ 1; sites with fewer than 0.5 residual df keep 1. The location posterior uses
  the residual variances too, so location-only corrections with EB also change.
- `combat_compatible=True` (`comcat_ui`, CLI `--combat-compatible`,
  `run_comcat_from_files.py` `COMBAT_COMPATIBLE`) gives the earlier estimates
  bit-identically and, without nuisance covariates, ComBat with EB. `residual_delta`
  now matters only together with it (otherwise the site variances always come from the
  residuals). `comcat_ui` folders of the new default get `_dfc`.
- `decentralized_comcat.py` keeps ComBat's site variances (`combat_compatible=True`);
  the df-consistent ones would need every site's leverages.
- Tests: `tests/test_comcat_scale.py`; the EB, option and MATLAB comparison tests pin
  `combat_compatible=True` where they compare with ComBat or earlier results.

## Unreleased (2026-10-28)

### Added — nuisance models, subject terms, cross-fitting (opt-in)

- `comcat(..., nuisance_model='spline'|'linear'|'pspline')`. `'spline'` (default) is the
  unpenalized B-spline basis of earlier versions. `'linear'` enters every nuisance
  column linearly. `'pspline'` uses penalized cubic B-splines (Eilers & Marx): one
  unpenalized linear term and `gam_df - 2` nonlinear terms per column, shrunk by a
  second-order difference penalty with one smoothing parameter for all columns,
  estimated from all features by pooled REML (default) or GCV (`nuisance_penalty`), or
  fixed. This is an empirical Bayes estimate of the size of the nonlinear nuisance
  effects across features; large penalties give the linear model. New data are
  evaluated with the nonlinear part at the training range (the linear part continues).
  With both new models, nuisance columns that do not vary within the sites (and
  subjects) are dropped with a warning, and columns with fewer than `gam_df + 2`
  distinct values stay linear. Estimates add `nuisance_model`, `nuisance_keep`,
  `nuisance_setup`, `nuisance_lambda` and `nuisance_edf`.
- `comcat(..., subject=labels)`: centred subject indicators are added to the preserved
  covariates, so the site and nuisance effects are estimated from the variation within
  subjects (traveling-subject, test–retest and motion designs). Implies `mean_only`.
  `comcat_from_training()` treats its subjects as new subjects.
- `comcat_crossfit()`: every subject is harmonized with a model fitted without it
  (k-fold over subjects, stratified by site, or given folds), optionally fitted on a
  subset such as the controls and applied to all others.
- `comcat_from_training(..., unseen_batch='mean')`: sites not seen in training get the
  size-weighted mean site level and the pooled variance (no site correction; the
  nuisance effects are removed), e.g. patients of a site without controls.
- `comcat_ui`/CLI: `nuisance_model` (`--nuisance-model`), `nuisance_penalty`; folder
  suffix `_nuisance<q>_linear` or `_nuisance<q>_pspline` instead of `_gam<df>`.
- Tests: `tests/test_comcat_nuisance_models.py`.

### Changed — default nuisance model: penalized B-splines

- `nuisance_model='pspline'` is now the default of `comcat()`, `comcat_crossfit()`,
  `comcat_ui()` and the CLI (`gam_df` defaults to 10 basis functions; REML sets the
  smoothness). With the unpenalized B-splines of earlier versions, several IQMs gave
  rank-deficient designs that overfit and did not generalize to new scans
  (`HeldOut-Evaluation.md`; held-out evaluation of the paper revision in
  `Harmonization-Oxford/revision/`). `nuisance_model='spline'` reproduces earlier
  results; `comcat_ui` folders of the new default end in `_nuisance<q>_pspline`.
- Pinned to `'spline'` so that their documented numbers reproduce: `decentralized_comcat.py`
  (whose decentralized fit implements that basis), `simulate_comcat.py`, the scripts in
  `evaluation/` and the MATLAB comparison `tests/test_comcat_py.py`. The option and EB
  tests run with the new default.

### Changed — one site: location correction only

- With a single site, `comcat()` now corrects the location only, also when the site is
  given explicitly, as `comcat_ui` does when no batch is given. The variance step with
  one site divides every residual by its SD over all subjects, which shrinks all
  deviations by about sqrt(1 - R²) of the nuisance fit, biological variance included
  (MR-ART: between-subject variance 0.78 → 0.64 of raw in-sample; it also flattered the
  single-subject BrainAGE results). `single_site_scale=True` (`--single-site-scale`)
  reproduces earlier results. `comcat_ui` folders of such runs get `_meanonly`, so
  earlier results are not overwritten.
- Fix: `import warnings` inside `comcat()` made `warnings` a local name.

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
- Scripts: `evaluation/eb_splithalf.py`, `eb_traveling.py`, `eb_impact.py`.

### Added — held-out evaluation

- `HeldOut-Evaluation.md`: the IQM nuisance model on held-out traveling-subject and
  test–retest data (Tohoku, Buchert, MR-ART, ON-Harmony). It covers the causes, tested
  alternatives (truncated SVD, ridge penalty chosen by GCV, IQM subsets, linear terms,
  principal components, IQM effects estimated within subjects, transfer between datasets)
  and suggested fixes.
- `evaluation/`: the scripts behind it and behind the empirical Bayes results, with data
  paths set by environment variables.

### Known issues

- **The IQM nuisance model does not generalize to held-out data.** With the seven CAT12
  IQMs and the default settings, in-sample vs held-out:
  - Tohoku: variance 0.09 vs 0.21 of raw.
  - Buchert: 0.17 vs 348 (median of 10 draws).
  - ON-Harmony: within-subject variance ×4.35 vs ×163; between-subject variance ×0.29
    in-sample.
  - MR-ART: between-subject variance ×0.64 in-sample.

  There are three causes:
  - an unpenalized spline basis of partly redundant IQMs (SIQR ≈ res_ECR; IQR ≈
    NCR, ICR, res_RMS), which is rank-deficient or near-singular;
  - IQM effects estimated across subjects that partly reflect anatomy;
  - the variance step, which in single-site `comcat_ui` runs rescales all residuals by
    about 1 − R² of the in-sample fit.

  On held-out data, two things helped. Ridge-penalized or linear IQM terms remove the
  blow-ups. Estimating the IQM effects within subjects (subject terms, location only) was
  the only variant that reduced technical variance without removing biology: MR-ART
  motion bias −87%, ON-Harmony W/T 0.068 vs 0.074 raw. The numbers are for the earlier
  defaults (`nuisance_model='spline'`; one site with `single_site_scale=True`). The
  changes above target the first and third cause (penalized splines by default,
  location only with one site), and the second where subjects were scanned repeatedly
  (`subject` terms); in cross-sectional data the second cause remains. See
  `HeldOut-Evaluation.md`.

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

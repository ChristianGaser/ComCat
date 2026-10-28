> [!WARNING]
> This project is **currently under construction** and might contain bugs. **If you experience any issues, please [let me know](https://github.com/ChristianGaser/ComCat/issues)!**

> [!CAUTION]
> **The IQM nuisance correction does not generalize to held-out data.** With the seven
> CAT12 IQMs, the nuisance model overfits, and the IQM effects it estimates partly reflect
> anatomy. In traveling-subject and test–retest data, it adds technical variance to
> held-out scans and removes between-subject (biological) variance, even in-sample.
> [HeldOut-Evaluation.md](HeldOut-Evaluation.md) has the evidence and suggested fixes.

> [!CAUTION]
> **Current scope is limited to BrainAGE and normative modelling workflows.**  
> For general GLM-based group analyses, harmonizing data with ComCAT and then running a standard GLM inflates degrees of freedom and leads to inflated false-positive rates. The two-step correction proposed by Li et al. that addresses this problem is **not yet implemented** in the production pipeline (`comcat_ui.py`). It is only available inside the simulation framework (`simulate_comcat.py`) for evaluation purposes. Until the Li et al. correction is integrated, avoid using ComCAT-harmonized data as input to group-comparison GLMs.

# ComCAT — Combating CovariATe Effects

ComCAT is a Python toolkit for **harmonizing multi-site neuroimaging data**. It removes scanner/site batch effects and unwanted nuisance covariate effects from high-dimensional data (e.g., voxel- or vertex-wise brain maps) while preserving biological signals of interest such as age or group membership. ComCAT is a heavily extended Python port of the original MATLAB-based ComBat/ComCAT implementation.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [File Overview](#file-overview)
- [Core API — `comcat.py`](#core-api--comcatpy)
  - [`comcat()` — harmonize data](#comcat--harmonize-data)
  - [`comcat_from_training()` — apply to new data](#comcat_from_training--apply-to-new-data)
  - [Nuisance modelling: B-spline GAM](#nuisance-modelling-b-spline-gam)
  - [Empirical Bayes (default)](#empirical-bayes-default)
  - [Optional model extensions](#optional-model-extensions)
- [File I/O Interface — `comcat_ui.py`](#file-io-interface--comcat_uipy)
  - [Python usage](#python-usage)
  - [Command-line usage](#command-line-usage)
  - [Supported file formats](#supported-file-formats)
  - [Output naming](#output-naming)
- [Batch Processing — `run_comcat_from_files.py`](#batch-processing--run_comcat_from_filespy)
- [Comparison Methods — CovBat, ComBatLS, ComBat (EB)](#comparison-methods--covbat-combatls-combat-eb)
- [Simulation Tools](#simulation-tools)
  - [`simulate_comcat.py`](#simulate_comcatpy)
  - [`simulate_comcat_ui.py`](#simulate_comcat_uipy)
- [Testing](#testing)
- [Common Workflows](#common-workflows)
- [Parameter Reference](#parameter-reference)

---

## Overview

ComCAT implements a harmonization model that:

1. **Removes additive and multiplicative batch effects** (scanner / site differences).
2. **Regresses out nuisance covariates** (e.g., image quality metrics, total intracranial volume) using penalized B-splines (smoothness by REML).
3. **Preserves biological covariates of interest** (e.g., age, group) so they are not inadvertently removed.

When only batch correction is requested, ComCAT is ComBat with empirical Bayes: it agrees with `combat_family.combat_eb()`, the port of the R package ComBatFamily, to ~1e-13. With nuisance covariates it extends ComBat with flexible nuisance modelling, and the empirical Bayes step is applied to the site effects.

If you are interested in ComCat's theoretical background, you can read about it [here](ComCat-Theory.md).

---

## Installation

ComCAT has no mandatory installation step — clone the repository and install dependencies. Requires **Python ≥ 3.8**.

**Install required dependencies only:**
```bash
pip install -r requirements.txt
```

| Dependency | Required | Purpose |
|---|---|---|
| `numpy>=1.22` | Yes | Array operations |
| `scipy>=1.8` | Yes | `.mat` file I/O (v5–v7.2), stats |
| `nibabel>=4.0` | Yes | NIfTI / GIFTI file I/O |
| `statsmodels>=0.13` | Yes | B-spline GAM for nuisance modelling |
| `h5py>=3.0` | Yes | MATLAB v7.3 `.mat` files |
| `matplotlib>=3.5` | Yes | Plots in simulation tools |

Add the repository directory to your Python path or pass it via `sys.path.insert` (see `run_comcat_from_files.py` for an example).

---

## File Overview

| File | Purpose |
|---|---|
| `comcat.py` | Core harmonization algorithm (low-level API) |
| `comcat_ui.py` | High-level file I/O interface + CLI entry point |
| `run_comcat_from_files.py` | Example/template script for processing multiple datasets |
| `simulate_comcat.py` | Monte-Carlo simulation comparing ComCAT vs GLM AnCova |
| `simulate_comcat_ui.py` | Parameter sweep over simulation conditions |
| `combat_family.py` | CovBat, ComBatLS and ComBat with empirical Bayes (port of the R package ComBatFamily) |
| `combat_family_ui.py` | Same file interface as `comcat_ui.py` for these methods + CLI |
| `run_combat_family_from_files.py` | Template script running them on the ComCAT inputs |
| `decentralized_comcat.py` | Experimental decentralized/federated harmonization (sites never pool raw data) — see **[Decentralized.md](Decentralized.md)** |
| `HeldOut-Evaluation.md` | Known issue: behaviour of the IQM nuisance model on held-out data, causes and suggested fixes |
| `evaluation/` | Real-data evaluation scripts (held-out IQM model, empirical Bayes) |
| `tests/test_comcat_py.py` | Numerical validation against MATLAB reference output |
| `tests/test_combat_family.py` | Tests for `combat_family.py`, incl. comparison with R ComBatFamily |

> **Decentralized / federated use:** to harmonize data split across sites that
> cannot share raw data, see **[Decentralized.md](Decentralized.md)**. The result
> matches centralized `comcat()` to machine precision.

---

## Core API — `comcat.py`

### `comcat()` — harmonize data

```python
from comcat import comcat

Y_harmonized, beta_hat, gamma_hat, delta_hat = comcat(
    Y,           # (n_features, n_subjects) data matrix
    batch,       # (n_subjects,) site/scanner labels
    nuisance,    # (n_subjects, n_nuisance) variables to remove  — or None
    preserve,    # (n_subjects, n_preserve) variables to keep    — or None
    mean_only=False,
    verbose=True,
)
```

**Returns**

| Value | Shape | Description |
|---|---|---|
| `Y_harmonized` | `(n_features, n_subjects)` | Harmonized data |
| `beta_hat` | `(n_params, n_features)` | Full design-matrix coefficients |
| `gamma_hat` | `(n_batch+n_Z, n_features)` | Additive batch/nuisance effects removed from the data (empirical Bayes estimates in the site rows when `eb=True`) |
| `delta_hat` | `(n_batch, n_features)` | Multiplicative batch effects removed from the data (empirical Bayes estimates when `eb=True`) |

**Key parameters**

| Parameter | Default | Description |
|---|---|---|
| `mean_only` | `False` | Adjust mean only; skip variance scaling |
| `eb` | `True` | Empirical Bayes estimation of the site effects, as in ComBat (used with ≥ 2 sites); `False` gives the L/S estimates of earlier versions |
| `ref_batch` | `None` | Site label to treat as reference (its data is left untouched; the other sites are harmonized to its mean and variance) |
| `return_estimates` | `False` | Return a 5th element (dict) for use with `comcat_from_training()` |
| `smooth_term_bounds` | `None` | Explicit boundary knots for the nuisance splines |
| `gam_df` | `None` | B-spline basis dimension per nuisance column (auto-selected if `None`) |
| `verbose` | `False` | Print progress messages |

**Single-site / nuisance-only mode**

Passing `batch=None` (or an empty array) treats all subjects as one site.  
ComCAT then only regresses out nuisance covariates while optionally preserving others — no scanner correction is applied.

### `comcat_from_training()` — apply to new data

```python
from comcat import comcat, comcat_from_training

# Step 1: fit on training data and save estimates
Y_train_harm, _, _, _, estimates = comcat(
    Y_train, batch_train, nuisance_train, preserve_train,
    return_estimates=True,
    smooth_term_bounds=(lo, hi),  # fix bounds for consistent knot placement
)

# Step 2: apply to new (test) data
Y_test_harm = comcat_from_training(
    Y_test, batch_test, nuisance_test, preserve_test,
    estimates=estimates,
)
```

> [!IMPORTANT]
> When using train/test workflows, always specify `smooth_term_bounds` so knot positions are identical between training and new data.

### Nuisance modelling: B-spline GAM

Every nuisance column is modelled as a smooth function. By default
(`nuisance_model='pspline'`) it is a penalized cubic B-spline with `gam_df = 10` basis
functions: a linear term plus nonlinear terms shrunk by a second-order difference
penalty, with one smoothing parameter for all columns estimated by REML across all
features. Large penalties give the linear model, so the data choose between linear and
nonlinear effects. `nuisance_model='linear'` uses linear terms only.
`nuisance_model='spline'` is the unpenalized B-spline of earlier versions; for it,
`gam_df` is the smoothness, and the guidance below applies. `smooth_term_bounds` sets
the knot bounds (by default the training range; new data are evaluated with the
nonlinear part at the boundary).

**Recommended `gam_df` values by covariate type (unpenalized `'spline'` only)**

| Covariate | Recommended `gam_df` |
|---|---|
| Age (20–90 yr) | 6 – 8 |
| TIV / ICV (cm³) | 5 – 7 |
| Continuous score | 5 – 6 |
| General rule | `max(5, n // 30)`, capped at 10 |

When `gam_df=None` (default), it is 10 for `'pspline'` and `min(10, max(5, n // 30))` for `'spline'`.

### Empirical Bayes (default)

With at least two sites, the additive (γ) and multiplicative (δ) site effects are
estimated with the parametric empirical Bayes method of ComBat (Johnson et al., 2007).
For every site, the effects of all features are assumed to come from common priors,
and each feature's estimate is shrunk towards them. The shrinkage is strong when a site
has few subjects or its estimates are noisy, and negligible for large sites. The maths
is in [ComCat-Theory.md](ComCat-Theory.md#empirical-bayes-estimation-of-site-effects),
and the tests are in `tests/test_comcat_eb.py`.

- **On by default** (`eb=True`). `eb=False` reproduces the results of earlier versions
  bit for bit. With a single site (or `batch=None`), EB is not used.
- **What is shrunk.** ComCAT's site coefficients contain an offset from the nuisance
  model. They are therefore shrunk as differences to the size-weighted mean of the
  sites (or to `ref_batch`), which are ComBat's site effects. A reference site keeps its
  estimates, and sites with one subject are left out.
- **Without nuisance covariates** the result equals ComBat with EB
  (`combat_family.combat_eb()`) to ~1e-13.
- **Estimates.** `comcat()` returns the applied (EB) `gamma_hat` / `delta_hat`. With
  `return_estimates=True`, the dict holds the L/S estimates (`gamma_hat_masked`,
  `delta_hat_masked`), the EB estimates (`gamma_star_masked`, `delta_star_masked`)
  and the priors (`eb_priors`). `comcat_from_training()` applies the EB estimates.
- **Output folder.** `comcat_ui()` adds `_eb` after `_sites`, for example
  `comcat_sites_eb_preserve1_nuisance7_gam5`. EB results therefore never overwrite
  results of earlier versions. CLI: `--no-eb`.

Why it matters: with small sites, the L/S variance δ is estimated from a handful of
values, and dividing by it can blow up single features. On real data (gray matter,
8 mm), EB gave these results:

| Test | L/S | EB |
|---|---|---|
| Buchert, one scan per scanner held out, sites only: spread of held-out scans vs raw | 5,364× (training sites with 2–3 scans) | 0.55× |
| ON-Harmony, one subject held out, sites only: within-subject / total variance (raw 0.074) | 0.112 | 0.083 |
| ADNI with IQMs, in-sample: 99th percentile of harmonized values at a site with two subjects | 17.5 SD | 1.5 SD |
| Split halves, 5 samples: error of the site variances | — | 26–87% lower |
| Split halves, without IQMs: error of the site locations | — | 9–72% lower |

Switching EB on changes the in-sample harmonized data (RMS, in SD units of each feature)
by ≤ 0.05 for sites with more than 50 subjects, 0.03–0.16 for 11–50 subjects, 0.06–0.23
for 6–10 subjects, and 0.09–0.7 for 2–5 subjects, where the L/S estimates are unreliable.

Limitation: some IQMs vary almost only between sites, for example `res_RMS`, which is set
by the protocol. With such IQMs, the site levels at fixed IQM are not identified, so the
location shrinkage has almost no effect. The variance shrinkage still applies.

### Optional model extensions

Three options extend the model. All are **off by default**: without them and with
`eb=False`, `comcat()` and `comcat_ui()` give bit-identical results to earlier versions.
The maths is in [ComCat-Theory.md](ComCat-Theory.md#optional-extensions), and the
behaviour is tested in `tests/test_comcat_options.py`.

| Option | Default | What it does | Output folder suffix |
|---|---|---|---|
| `preserve_df` | `None` (linear) | B-spline expansion of continuous preserve covariates (int, or `'same'` = `gam_df`). Columns with fewer than `preserve_df` + 2 distinct values (group, sex) stay linear. | `_preserve<n>_gam<df>` |
| `preserve_bounds` | `None` | Boundary knots for the preserve splines, like `smooth_term_bounds` but per preserve column. Required by `comcat_from_training()` when new data exceed the training range. | — |
| `residual_delta` | `False` | Estimate site variances δ after removing the additive site **and** nuisance effects, as ComBat does. | `_resdelta` |
| `nuisance_scale` | `False` | Also remove nuisance-dependent variance (heteroscedasticity) with a log-linear variance model per feature; implies `residual_delta`. | `_scale` |

**`preserve_df` — preserve non-linear effects of interest.** Nuisance columns are
spline-expanded, and preserve columns are linear by default. When an IQM depends
non-linearly on age (e.g. more motion in children and older adults), the IQM splines
can represent the non-linear part of the age effect, and ComCAT then removes it. In a
simulated lifespan setting with seven such IQMs, ~70–80% of the age curve was lost
with a linear age term, against ~5% with `preserve_df='same'`. This matters most for
wide age ranges (e.g. the normative sample) and hardly at all for narrow ones.

**`residual_delta` — site variances from residuals.** By default (as in the MATLAB
implementation), δ is the within-site variance of data that still contain the
nuisance effects. The adjusted residuals are divided by it, so their variance shrinks
to about `1 − R²` of the within-site nuisance fit. On ON-Harmony (8 mm, 7 IQMs) the
residual variance of the harmonized data is 42% of the model residual variance at
`gam_df=5`, and 13% at `gam_df=10`. On Tohoku (one subject, 121 scans) IQM removal
alone leaves 23% of the scan-to-scan variance, and the δ step reduces this further to
5%. Because the shrinkage grows with `gam_df`, it is worth checking results that
depend on individual variance (e.g. BrainAGE MAE) with `residual_delta=True`. Only
`mean_only=True` avoids the effect, and `comcat_ui()` treats a missing `batch` as one
site, so single-site runs are affected too.

**`nuisance_scale` — remove quality-dependent noise.** The residuals of every feature
are modelled as `log sd = site + IQMs + preserve` (raw columns, z-scored, linear) by
maximum likelihood. Each subject's residual is then rescaled to the noise level at the
mean IQM value. Site and preserve terms stay in the variance model: site variance
differences are removed afterwards by δ, and biological effects on the variance
(e.g. of age) are preserved. The fit also yields a per-feature likelihood-ratio test
of IQM effects on the variance (`estimates['scale_lr']`, `estimates['scale_p']`).
With `save_estimates=True`, `comcat_ui()` saves these as `scale_lr` / `scale_p` maps,
together with `scale_kurtosis`, and prints the share of features with p < 0.05. This
is a direct test of the heteroscedasticity assumption.

The plain likelihood-ratio test assumes normal residuals. With skewed or heavy-tailed
residuals it is too liberal: on gray matter at 8 mm (2,241 subjects, real residual
shapes, three random IQMs without any effect on the variance) it gave p < 0.05 for 12%
of the voxels, and for 37% of the voxels with |skewness| > 1. The statistic is
therefore studentized by the kurtosis of the residuals (`estimates['scale_kurtosis']`),
as in Koenker's version of the Breusch–Pagan test, and compared with an F distribution
(`scale_lr / scale_lr_df` with `estimates['scale_lr_df']` and
`estimates['scale_resid_df']` df). In the same data this gave 5.0%. For normal
residuals the kurtosis is about 3 and the statistic stays as before.

In small samples with many nuisance columns, the test is too liberal even for normal
residuals, because the residuals of a large mean model are fewer independent values
than subjects. In simulations with the design of ON-Harmony (80 scans, 6 sites,
7 IQMs, `gam_df=5`, i.e. 42 mean-model columns), about 14% of the features had
p < 0.05 under the null hypothesis (8.5% with `gam_df=4`), and the estimated IQM
effect on the log SD was less than half of the true one. Use the test as a guide in
such samples.

```python
Y_harm, *_, est = comcat(Y, batch, nuisance=iqm, preserve=age[:, None],
                         preserve_df='same', nuisance_scale=True,
                         return_estimates=True)
print(np.nanmean(est['scale_p'] < 0.05))   # share of features with IQM-dependent variance
```

These options are not available in `decentralized_comcat.py`. Empirical Bayes is: it
runs on the aggregator and needs no additional data from the sites.

---

### Nuisance models, subject terms and cross-fitting

The default nuisance model is penalized B-splines (`nuisance_model='pspline'`) with one
smoothing parameter estimated by REML across all features. The unpenalized B-splines of
earlier versions (`'spline'`) overfit when several IQMs are used and do not generalize to
new scans ([HeldOut-Evaluation.md](HeldOut-Evaluation.md)). For BrainAGE, the revision of
the ComCat paper uses (site variances off, i.e. location correction only):

```python
from comcat import comcat_crossfit

H = comcat_crossfit(
    Y, site, iqm[:, [ICR, NCR, res_ECR, contrastr]], preserve,   # no IQR/SIQR
    folds=5,                     # every subject harmonized by a model fitted without it
    fit=controls,                # optional: estimate on controls, apply to everybody
    subject=subject_ids,         # repeated scans: effects estimated within subjects
    nuisance_model='pspline',    # penalized splines, smoothing by REML (or 'linear')
    mean_only=True,              # location correction of the IQM effects
)
```

- **IQMs.** IQR is a deterministic function of NCR and res_RMS (CAT12:
  `rms([NCR res_RMS], 8)`), and SIQR correlates 0.96–1.00 with res_ECR. res_RMS is a
  property of the protocol: use it only where the resolution varies within the sites.
- **Shape.** With `'pspline'`, REML chooses between linear and nonlinear effects; in
  most samples it chose nearly linear ones.
- **One site.** Location correction only (default); the variance step would rescale all
  residuals by about sqrt(1 - R²) of the nuisance fit.
- **Preserved covariates** matter only for the estimation in location-only mode: the
  harmonized data are `Y - nuisance effect`, so the preserved fit cannot leak into
  BrainAGE. With a variance step, every scan is pulled toward the preserved fit at its
  own age.

## File I/O Interface — `comcat_ui.py`

`comcat_ui.py` wraps `comcat()` with file loading/saving logic, so you can work directly with neuroimaging files or data matrices stored on disk.

### Python usage

```python
from comcat_ui import comcat_ui
import numpy as np

batch    = np.loadtxt("batch.txt").astype(int)
nuisance = np.loadtxt("nuisance.txt")   # (n_subjects, n_nuisance)
preserve = np.loadtxt("preserve.txt")   # (n_subjects, n_preserve)

Y_harmonized, gamma_hat, delta_hat = comcat_ui(
    files          = ["subj_001.nii.gz", "subj_002.nii.gz", ...],
    batch          = batch,
    nuisance       = nuisance,
    preserve       = preserve,
    mean_only      = False,
    save_estimates = True,
    verbose        = True,
)
```

### Command-line usage

```bash
python comcat_ui.py --help
```

### Supported file formats

| Format | Extension(s) | Notes |
|---|---|---|
| NIfTI | `.nii`, `.nii.gz` | One file per subject; includes CIFTI |
| GIFTI | `.func.gii`, `.shape.gii`, `.gii` | Surface data |
| MATLAB | `.mat` | Single file containing variable `Y` of shape `(n_features × n_subjects)`, i.e. created by [BA_data2mat.m](https://github.com/ChristianGaser/BrainAGE); supports v5–v7.3 (requires `h5py` for v7.3) |
| Plain text / CSV | `.txt`, `.csv` | Single file; shape auto-detected |

### Output naming

Harmonized files are saved into an auto-generated subfolder next to the input files. The folder name encodes the harmonization settings, for example:

```
comcat_sites_eb_preserve1_nuisance1_gam6/   # sites + nuisance, empirical Bayes (default)
comcat_sites_preserve1_nuisance1_gam6/      # the same with eb=False (earlier versions)
combat_sites_eb/
comcat_nuisance2_gam6/                      # single site: no empirical Bayes
```

Override the folder name with the `subfolder` parameter. For MAT/TXT files, the output file is placed inside the subfolder with the same file name.

---

## Batch Processing — `run_comcat_from_files.py`

`run_comcat_from_files.py` is a ready-to-use template script for processing multiple datasets. Edit the `SAMPLES` dictionary and configuration section at the top of the file, then run:

```bash
python run_comcat_from_files.py
```

**Input file formats expected**

`batch.txt` — one site label per line (integer or string):
```
1
1
2
2
```

`nuisance.txt` — space- or tab-delimited, one row per subject, no header:
```
25.3  1450.2  1
31.7  1380.5  0
```

`preserve.txt` — same format as `nuisance.txt`.

`data.mat` — MATLAB file containing variable `Y` of shape `(n_features × n_subjects)`, i.e created by [BA_data2mat.m](https://github.com/ChristianGaser/BrainAGE).

**Setting the ComCAT directory**

The script resolves `comcat_ui.py` via the `COMCAT_DIR` variable. Either edit it directly or set the environment variable before running:

```bash
export COMCAT_DIR=/path/to/ComCat
python run_comcat_from_files.py
```

---

## Comparison Methods — CovBat, ComBatLS, ComBat (EB)

For comparisons with established harmonization methods, `combat_family.py`
implements three members of the ComBat family. The code is a NumPy port of
the reference R package [ComBatFamily](https://github.com/andy1764/ComBatFamily)
(v0.2.2) that fits all features at once instead of looping over them:

| Function | R equivalent | Method |
|---|---|---|
| `combat_eb()` | `comfam(model = lm)` | ComBat with empirical Bayes (Johnson et al., 2007; Fortin et al., 2018) |
| `covbat()` | `covfam(model = lm)` | CovBat: ComBat followed by harmonization of PC scores, which removes site effects in the covariance between features (Chen et al., 2022, *Hum Brain Mapp* 43:1179) |
| `combatls()` | `combatls(..., sigma.formula = ~ covariates)` | ComBatLS: covariate effects on the mean **and** the variance are modelled (normal GAMLSS) and preserved (Gardner et al., 2025, *Hum Brain Mapp* 46:e70197) |

`tests/test_combat_family.py` compares the output with the R package when
`Rscript` and ComBatFamily are available; the harmonized data agree to
~1e-13 for ComBat and CovBat and to ~5e-7 for ComBatLS, where `gamlss` stops
on a deviance criterion slightly before the maximum-likelihood solution.

`combat_family_ui.py` uses the file I/O of `comcat_ui.py`, so the same inputs
can be passed unchanged:

```python
from combat_family_ui import covbat_ui, combatls_ui, combat_eb_ui

covbat_ui(["data.mat"], batch=site, preserve=age)       # PCs explaining 95% of variance
combatls_ui(["data.mat"], batch=site, preserve=age)     # log(sigma) ~ age
combat_eb_ui(["data.mat"], batch=site, preserve=age)
```

```bash
python combat_family_ui.py covbat data.mat --batch scanner.txt --preserve age.txt
python combat_family_ui.py --help
```

Each call returns `(Y_harmonized, gamma_star, delta_star)` and saves the result
in a subfolder named `covbat_*`, `combatls_*` or `combateb_*`, so ComCAT's
`comcat_*` / `combat_*` results are never overwritten.

Differences from `comcat_ui()`:

- **At least two sites are required.** These methods only remove site effects.
- **`nuisance` is ignored (with a warning).** Every covariate in these models is
  preserved; none of them can remove covariate effects such as IQMs.
- **Preserved covariates enter linearly**, as in ComCAT. ComBatLS models
  log(sigma) with the preserved covariates by default (`scale=` to change);
  batch is not part of the variance model, as in the ComBatLS paper.
- Empirical Bayes is on by default (`eb=False` gives the plain L/S model).

`run_combat_family_from_files.py` runs all three methods on the samples of
`run_comcat_from_files.py` that have site labels.

---

## Simulation Tools

### `simulate_comcat.py`

Monte-Carlo simulation that compares the statistical performance of **ComCAT** against a standard **GLM AnCova** approach. Use it to understand how nuisance amplitude and covariate–effect-of-interest correlation affect false-positive rates and effect-size recovery.

```python
from simulate_comcat import simulate_comcat

avgD, FPR = simulate_comcat(
    a=[1.0, 0.2, 0.0, 0.5],  # [EoI_amp, nuisance_amp, mult_amp, nuisance–EoI covariance]
    no_preserving=False,
    n=1000,
    n_sim=500,
    n_nuisance=1,
    mean_only=True,
    gam_df=6,       # B-spline basis dimension per nuisance term (GAM always on)
)
# avgD / FPR shape: (2,) → [AnCova, ComCAT (GAM)]
```

The simulation applies an optional **Zhao et al. two-step correction** (`apply_2step_correction=True` by default) to pre-whiten harmonized data and correct for degrees-of-freedom inflation introduced by ComCAT.

### `simulate_comcat_ui.py`

Runs a full parameter sweep over nuisance amplitude, nuisance–effect-of-interest covariance, and number of nuisance covariates. Results are saved to a `.mat` file for further analysis.

```bash
python simulate_comcat_ui.py                              # default sweep
python simulate_comcat_ui.py --n 1000 --n-sim 5000 --no-fig
python simulate_comcat_ui.py --mean-only --output my_results.mat
```

**Sweep dimensions**

| Dimension | Default values |
|---|---|
| Nuisance amplitude `a2` | 0, 0.05, …, 0.30 |
| Nuisance–EoI covariance `a4` | 0, 0.05, …, 0.50 |
| Number of nuisance covariates | 1, 2, 5, 10 |

Results are stored as arrays of shape `(n_a2, n_a4, n_nuisance, n_methods)` for both mean Cohen's D and false-positive rate.

---

## Testing

| Script | Needs | Tests |
|---|---|---|
| `tests/test_comcat_py.py` | MATLAB reference files | Python port vs `comcat.m` (below) |
| `tests/test_comcat_options.py` | — | Optional extensions on simulated data with known effects; defaults unchanged; `comcat_from_training()` reproduces the in-sample result |
| `tests/test_comcat_eb.py` | — | Empirical Bayes: equals ComBat (`combat_eb`) without nuisance; site effects and harmonized data closer to the truth than L/S in a simulation with small sites and IQMs; `ref_batch`; `comcat_from_training()` |
| `tests/test_combat_family.py` | R + ComBatFamily (optional) | CovBat / ComBatLS / ComBat-EB properties and agreement with the R package |

### `test_comcat_py.py`

Validates that `comcat.py` produces results numerically identical to the MATLAB reference implementation within float32 tolerances (`atol=1e-4`, `rtol=1e-4`).

**Step 1** — Generate test data in MATLAB and place the `test_case*.mat` files in the `tests/` directory (the test script looks for them next to itself):
```matlab
gen_test_data   % produces test_case1.mat, test_case2.mat, test_case3.mat
```

**Step 2** — Run the Python tests:
```bash
python tests/test_comcat_py.py
```

Three test cases are covered:

| Case | Configuration |
|---|---|
| Case 1 | Multi-site, linear nuisance + preserve |
| Case 2 | Single site, linear nuisance, `mean_only=True` |
| Case 3 | Multi-site, B-spline GAM nuisance (requires `statsmodels`) |

---

## Common Workflows

### 1. Multi-site harmonization with age preservation

```python
from comcat import comcat
import numpy as np

# Y: (n_voxels, n_subjects), batch: site IDs, age: variable to preserve
Y_harm, *_ = comcat(Y, batch, nuisance=iqm, preserve=age[:, None])
```

### 2. Nuisance regression without site correction

```python
# Pass batch=None for single-site nuisance removal
Y_harm, *_ = comcat(Y, batch=None, nuisance=iqm, preserve=age[:, None])
```

### 3. Processing NIfTI files via the UI

```python
from comcat_ui import comcat_ui
import numpy as np

comcat_ui(
    files    = sorted(glob("data/*.nii.gz")),
    batch    = np.loadtxt("scanner.txt").astype(int),
    nuisance = np.loadtxt("iqm.txt")[:, None],
    preserve = np.loadtxt("age.txt")[:, None],
    save_estimates=True,
)
```

### 4. Train on one dataset, apply to new subjects

```python
from comcat import comcat, comcat_from_training

# Training
_, _, _, _, estimates = comcat(
    Y_train, batch_train, nuisance_train, preserve_train,
    return_estimates=True,
    smooth_term_bounds=(18.0, 90.0),  # age range covering both datasets
)

# New data
Y_new_harm = comcat_from_training(
    Y_new, batch_new, nuisance_new, preserve_new,
    estimates=estimates,
)
```

---

## Parameter Reference

### `comcat()` full signature

```python
comcat(
    Y,                          # (n_features, n_subjects)
    batch,                      # (n_subjects,) — site labels; None for single site
    nuisance       = None,      # (n_subjects, n_nuisance)
    preserve       = None,      # (n_subjects, n_preserve)
    mean_only      = False,     # True → mean correction only
    verbose        = False,
    ref_batch      = None,      # reference site label (left untouched)
    return_estimates = False,   # return fitted parameter dict
    smooth_term_bounds = None,  # (lo, hi) or [(lo0,hi0), ...] or None
    gam_df         = None,      # B-spline df per nuisance column; None = auto
    preserve_df    = None,      # B-spline df for continuous preserve columns; 'same' = gam_df
    preserve_bounds = None,     # (lo, hi) or [(lo0,hi0), ...] per preserve column
    nuisance_scale = False,     # also remove nuisance-dependent variance
    residual_delta = False,     # site variances from residuals (as ComBat)
    eb             = True,      # empirical Bayes estimation of site effects (as ComBat)
    nuisance_model = 'pspline', # 'pspline' (REML, default), 'linear', 'spline' (earlier)
    nuisance_penalty = None,    # 'pspline': None/'reml', 'gcv' or a fixed penalty
    subject        = None,      # subject labels: effects estimated within subjects
    single_site_scale = False,  # one site: True = variance step of earlier versions
)
```

`comcat_crossfit(Y, batch, nuisance, preserve, folds=5, fit=None, subject=None, **options)`
harmonizes every subject with a model fitted without it (`fit`: e.g. the controls; all
other subjects are harmonized with the model of all `fit` subjects).

### `comcat_ui()` full signature

```python
comcat_ui(
    files,                      # list of file paths
    batch          = None,      # (n_subjects,)
    nuisance       = None,      # (n_subjects, n_nuisance)
    preserve       = None,      # (n_subjects, n_preserve)
    mean_only      = False,
    subfolder      = None,      # override auto-generated output folder name
    save_estimates = False,     # save gamma/delta estimates alongside data
    verbose        = True,
    smooth_term_bounds = None,
    gam_df         = None,
    preserve_df    = None,
    preserve_bounds = None,
    nuisance_scale = False,
    residual_delta = False,
    eb             = True,
)
```

---

## Citation / Origin

ComCAT is based on the ComBat algorithm originally developed by W. Evan Johnson and Cheng Li, and extended in the neuroimaging context by Jean-Philippe Fortin. The MATLAB implementation and extensions are by Christian Gaser (University of Jena).

For issues or feature requests, please use the [GitHub issue tracker](https://github.com/ChristianGaser/ComCat/issues).

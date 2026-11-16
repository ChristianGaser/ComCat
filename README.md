> [!WARNING]
> This project is **under construction** and might contain bugs. **If you experience any issues, please [let me know](https://github.com/ChristianGaser/ComCat/issues)!**

> [!CAUTION]
> **Current scope is limited to BrainAGE and normative modelling workflows.**  
> For general GLM-based group analyses, harmonizing data with ComCAT and then running a standard GLM inflates degrees of freedom and leads to inflated false-positive rates. The two-step correction proposed by Li et al. that addresses this problem is **not yet implemented** in the production pipeline (`comcat_ui.py`). It is only available inside the simulation framework (`simulate_comcat.py`) for evaluation purposes. Until the Li et al. correction is integrated, avoid using ComCAT-harmonized data as input to group-comparison GLMs.

# ComCAT — Combating CovariATe Effects

ComCAT harmonizes multi-site neuroimaging data (voxel- or vertex-wise maps, regional
values). It is **ComBat with continuous nuisance covariates**:

1. **Sites** are corrected in location and scale exactly as in ComBat (Johnson et al.,
   2007), with empirical Bayes. Without nuisance covariates, ComCAT *is* ComBat.
2. **Continuous nuisance covariates**, such as image quality measures (IQMs), are removed
   in location as penalized splines whose smoothness is estimated from the data.
3. **Covariates of interest** (age, sex, diagnosis) are preserved.

```python
from comcat import comcat, comcat_crossfit

# in-sample: fitted on the data it is applied to
Y_h, *_ = comcat(Y, site, iqm, preserve)        # Y: (features, subjects)

# cross-fitted: every subject harmonized by a model fitted without it
Y_h = comcat_crossfit(Y, site, iqm, preserve, folds=5)
```

---

## Which design?

| Your data | Remove | Preserve |
|---|---|---|
| Site labels, about 70 subjects or more | sites + 4 IQMs | age, sex, and every effect you will analyze (diagnosis) |
| One site, or no site labels | 4 IQMs | the same |
| Fewer than about 70 subjects | sites only (this is ComBat) | the same |
| Sites with fewer than about 10 subjects | add `mean_only=True` | the same |

- **Fit on all subjects and preserve the diagnosis.** Fitting on the controls only
  (`comcat_crossfit(..., fit=controls)`) is a possible sensitivity analysis, but it
  corrects the sites of the patients less well.
- **Cross-fit** (`comcat_crossfit`, `comcat_mat.py`) whenever the harmonized data are
  scored by a model trained elsewhere (BrainAGE, normative models) or harmonization
  methods are compared. In-sample harmonization is optimistic.
- **IQMs:** ICR, NCR, res_ECR and contrastr of CAT12 (columns 1, 2, 3 and 5 of its IQM
  table). Do not use the composites IQR and SIQR, nor res_RMS, which is a property of the
  protocol. IQMs from other tools work alike; ComCAT warns about columns that do not vary
  within the sites.
- **Wide age ranges** (children to old age): `preserve_df='same'` preserves nonlinear age
  effects.
- **Voxel-wise group tests:** put the site and IQM terms into the GLM instead of testing
  harmonized data; a test after harmonization overstates the degrees of freedom.

---

## Installation

Clone the repository and install the dependencies. Python 3.9 or later.

```bash
pip install -r requirements.txt
```

`comcat.py` needs numpy and scipy only; nibabel is needed for NIfTI/GIFTI files, h5py for
MATLAB v7.3 files, statsmodels for `preserve_df`.

---

## Usage

### Python

```python
from comcat import comcat, comcat_from_training, comcat_crossfit

Y_harmonized, beta_hat, gamma_hat, delta_hat = comcat(
    Y,           # (n_features, n_subjects)
    batch,       # (n_subjects,) site labels, or None for one site
    nuisance,    # (n_subjects, n_nuisance) covariates to remove, or None
    preserve,    # (n_subjects, n_preserve) covariates to keep, or None
)
```

| Option | Default | Meaning |
|---|---|---|
| `mean_only` | `False` | correct the sites in location only |
| `ref_batch` | `None` | reference site: its data stay as they are, the others are harmonized to it |
| `nuisance_model` | `'pspline'` | `'linear'` uses linear nuisance terms |
| `preserve_df`, `preserve_bounds` | `None` | B-splines for continuous preserved covariates (`'same'` = 10 basis functions) |
| `subject` | `None` | subject labels of repeated scans: effects estimated within subjects (location only) |
| `clip` | `True` | set negative harmonized values to 0 if the input has none; `False` returns the affine harmonization |
| `return_estimates` | `False` | also return the fitted parameters for `comcat_from_training()` |
| `verbose` | `False` | print progress |

`gamma_hat` and `delta_hat` are the additive and multiplicative effects that were removed
(site rows first; the empirical Bayes estimates).

**New scans.** Fit once, apply to scans of the same sites:

```python
*_, est = comcat(Y_train, site_train, iqm_train, preserve_train, return_estimates=True)
Y_new_h = comcat_from_training(Y_new, site_new, iqm_new, preserve_new, estimates=est,
                               unseen_batch='mean')   # other sites: IQM effects only
```

**Cross-fitting.**

```python
Y_h = comcat_crossfit(Y, site, iqm, preserve,
                      folds=5,              # stratified by site
                      subject=subject_id,   # repeated scans: a subject stays in one fold
                      fit=None)             # or a boolean mask, e.g. the controls
```

With `subject_terms=True`, the effects are estimated within subjects; this is for
traveling-subject or test–retest data with few subjects.

### Command line

**`comcat_mat.py`** — cross-fitted, for mat-files of BA_data2mat (BrainAGE, NeuroGAMLSS).
It writes a copy of the mat-file with `Y` harmonized and all other fields kept.

```bash
python comcat_mat.py s6mwp1_4mm_ABIDE437_CAT12.9.mat --site Scanner437.txt \
    --nuisance IQM437.txt --nuisance-cols 1,2,3,5 --group DX_group437.txt \
    --preserve age sex TIV437.txt --out comcat/s6mwp1_4mm_ABIDE437_CAT12.9.mat
```

| Option | Meaning |
|---|---|
| `--site FILE` | site of every scan, one label per line (default: one site) |
| `--nuisance FILE`, `--nuisance-cols 1,2,3,5` | nuisance variables to remove, e.g. ICR, NCR, res_ECR and contrastr of a CAT12 IQM table |
| `--preserve age sex FILE` | effects to keep: age and sex of the mat-file and the columns of tables (e.g. TIV); only what is listed is kept |
| `--group FILE` | group of every scan (e.g. diagnosis): its differences are preserved |
| `--controls 1:108,150` | instead of `--group`: fit on these scans only, apply their model to all others |
| `--subject FILE`, `--within-subjects` | repeated scans: a subject stays in one fold; with `--within-subjects` the effects are estimated within subjects |
| `--location-only` | no scale correction of the sites |
| `--folds K` | folds of the cross-fit (default 5; 0: in-sample) |

Tables are text files of numbers without header, one row per scan in the order of the
mat-file.

**`comcat_ui.py`** — in-sample, for NIfTI, GIFTI, mat and text files; also a Python
function `comcat_ui(files, batch, nuisance, preserve, ...)`.

```bash
python comcat_ui.py data/*.nii.gz --batch 1 1 2 2 3 3 --nuisance iqm.txt --preserve age_sex.txt
```

The harmonized files go into a subfolder named after what was removed and preserved,
e.g. `comcat_sites_eb_preserve2_nuisance4_pspline/`; `--save-estimates` also writes the
removed effects.

---

## The model in brief

For feature *v* and scan *j*: `Y = site + f(nuisance) + preserved + error`.

- All terms are estimated jointly; the preserved effects are added back unchanged.
- **Sites:** ComBat's empirical Bayes estimates of location and scale; the site variances
  are estimated from the residuals after the site and nuisance effects. With one site,
  nothing is scaled. A site with a single subject gets no site correction.
- **Nuisance:** per covariate a linear term plus a cubic B-spline (10 basis functions)
  with a second-order difference penalty. One smoothing parameter is estimated by REML
  from all features, so nearly linear effects stay linear. Location only.
- **Output:** in the units of the input; negative values of non-negative data are set
  to 0.

Details: [ComCat-Theory.md](ComCat-Theory.md).

---

## Evidence

Held out (cross-fitted) on six datasets, gray matter at 8 mm and BrainAGE; ComBat and
ComCAT with the same design. Analysis code and manuscript: ComCat paper (in preparation).

| Dataset | Measure | Raw | ComBat | ComCAT |
|---|---|---|---|---|
| Tohoku: 1 subject, 121 acquisitions, no site labels | variance across scans | 1 | – | 0.65 (permuted IQMs: 1.05) |
| MR-ART: 148 subjects, 3 motion levels | BrainAGE increase with strong motion (years) | +3.11 | – | +0.09 |
| Buchert: 1 subject, 116 scanners | variance across scans | 1 | 0.46 | 0.39 (permuted IQMs: 0.46; IQMs only: 0.83) |
| ABIDE: 437 subjects, 14 scanners | site R² | 0.088 | 0.007 | 0.006 (IQMs only: 0.052) |
| ABIDE | ASD − TD in BrainAGE (years) | −0.26 | −0.36 | −0.42 |
| ADNI: 835 subjects, 58 sites | AD − CN in BrainAGE (years) | 4.72 | 4.63 | 4.09 (IQMs only: 4.13) |
| ADNI | pMCI − sMCI (no IQM differences) | 2.35 | 2.57 | 2.43 |
| ON-Harmony: 10 subjects, 6 scanners | within-subject variance | 1 | 1.77 | 10.0 |

What ComCAT adds to ComBat:

- **Correction without site labels**, and of quality differences within a site (Tohoku,
  Buchert without labels).
- **Removal of motion bias** (MR-ART), from IQM effects estimated across subjects.
- **Removal of quality-related group differences**: in ADNI, patients' scans are of lower
  quality, and the AD − CN difference fell by 0.54 years (95% CI 0.31 to 0.78) relative
  to ComBat, while a contrast without IQM differences did not change significantly.

What it does not add: with good site labels, the IQMs improve neither site removal nor
age prediction (ABIDE: r = 0.690 vs. 0.688).

## Known limits

- **Number of subjects.** IQM effects estimated across few subjects take up anatomy. With
  10 subjects (ON-Harmony), every IQM model increased the within-subject variance; in
  MR-ART, about 70 training subjects were needed for a neutral result. With fewer, use the
  sites only.
- **Small sites.** ComBat's scale step shrinks the residuals of small sites, also when
  the sites do not differ in variance: held out, to about 0.84 of the residual variance
  with 3 subjects per site, 0.92 with 5, 0.97 with 10, and 1.00 with 40. ComBat and ComCAT
  are affected alike, and effect sizes are not; `mean_only=True` avoids it.
- **Variance.** Nuisance effects are removed in location only. Where image quality also
  changes the noise level (strong motion), that part stays in the data.
- **In-sample use is optimistic.** The MAE of BrainAGE was 3–36% lower in-sample than
  held out, and with many site terms effect sizes rose by about 13% (ComBat: 11%). Group
  differences in years changed by less than 0.1.
- **IQMs are not purely technical.** ICR and contrastr vary mostly between subjects, and
  IQM differences between groups may partly reflect disease. Report results with and
  without the IQMs.

---

## Files

| | |
|---|---|
| **Core** | |
| `comcat.py` | `comcat()`, `comcat_from_training()`, `comcat_crossfit()` |
| `comcat_mat.py` | command line for BA_data2mat mat-files, cross-fitted |
| `comcat_ui.py` | file interface and command line for NIfTI, GIFTI, mat, text; in-sample |
| `get_IQM_CAT.py`, `get_TIV_CAT.py` | IQM and TIV tables from CAT12 reports |
| `run_comcat_from_files.py` | template script for several datasets |
| **Comparison methods** | |
| `combat_family.py`, `combat_family_ui.py`, `run_combat_family_from_files.py` | ComBat (EB), CovBat and ComBatLS, ported from the R package ComBatFamily |
| **Evaluation and experimental** | |
| `evaluation/comcat_legacy.py` | ComCAT before 2026-11-16 with all earlier options, frozen |
| `evaluation/` | real-data evaluations of the earlier model ([HeldOut-Evaluation.md](HeldOut-Evaluation.md)) |
| `decentralized_comcat.py` | decentralized harmonization of the earlier model ([Decentralized.md](Decentralized.md)) |
| `simulate_comcat.py`, `simulate_comcat_ui.py` | simulation of group tests after harmonization (earlier model) |
| `deprecated/` | the MATLAB implementation |

## Tests

```bash
for t in tests/test_comcat_{vs_legacy,eb,scale,nuisance_models,options,edge_cases,mat}.py \
         tests/test_combat_family.py; do python "$t" || break; done
```

| Script | Tests |
|---|---|
| `test_comcat_vs_legacy.py` | the current model equals the earlier one with ComBat's site variances from the residuals, for every option that was kept; equals ComBat without nuisance covariates |
| `test_comcat_eb.py` | empirical Bayes: ComBat without nuisance covariates; closer to the truth than L/S with small sites |
| `test_comcat_scale.py` | scale step: equalizes unequal sites; pins the known shrinkage with small sites |
| `test_comcat_nuisance_models.py` | penalized and linear nuisance terms, subject terms, cross-fitting, diagnosis preserved or fitted on controls |
| `test_comcat_options.py` | `preserve_df`, `ref_batch`, new data |
| `test_comcat_edge_cases.py` | constant and missing features, missing covariates, unseen sites, clipping |
| `test_comcat_mat.py` | `comcat_mat.py` |
| `test_combat_family.py` | comparison methods, incl. agreement with the R package if installed |
| `test_comcat_py.py` | earlier model against the MATLAB code (needs reference files) |

## Earlier versions

Versions before 2026-11-16 had 18 options (unpenalized splines with `gam_df`, `eb`,
`combat_compatible`, `min_scale_n`, `residual_delta`, `single_site_scale`,
`nuisance_scale`, `nuisance_penalty`, `smooth_term_bounds`). An ablation on the six
datasets showed that they either reproduced errors of the first version or changed
nothing outside one extreme dataset, and they were removed ([CHANGELOG.md](CHANGELOG.md)).
Their names are still accepted when they ask for what this version does; otherwise an
error points to `evaluation/comcat_legacy.py`, where the earlier code is kept for the
evaluations of the ComCat paper.

## Citation / Origin

ComCAT is based on the ComBat algorithm of W. Evan Johnson and Cheng Li, extended to
neuroimaging by Jean-Philippe Fortin. ComCAT and its MATLAB predecessor are by Christian
Gaser (University of Jena).

For issues or feature requests, please use the [GitHub issue tracker](https://github.com/ChristianGaser/ComCat/issues).

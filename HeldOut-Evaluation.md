# ComCAT on held-out data: known issue and suggested fixes

> **Update (2026-10-28, later):** the fixes suggested below are now options of
> `comcat()`: `nuisance_model='linear'|'pspline'` (penalized splines, smoothing by
> REML), `subject=` (effects within subjects), `comcat_crossfit()` (held-out
> harmonization, optionally fitted on controls), and one site is now corrected for
> location only (`single_site_scale=True` reproduces earlier results). The evaluation
> on all test samples, with BrainAGE, is in the paper folder
> (`Harmonization-Oxford/revision/`, see its README). The default `nuisance_model`
> is still the unpenalized spline, so earlier results can be reproduced.

**Status:** known issue of the default nuisance model (found 2026-10-28). It bears on reviewer issues B
(in-sample model flexibility) and C (IQMs carry biology), and on revision tasks
T1.1–T1.4 (permutation control, out-of-sample application, feature-level validation,
variance decomposition). All numbers can be reproduced with the scripts in
[evaluation/](evaluation/).

## Summary

With the seven CAT12 image quality metrics (IQMs) as nuisance covariates, ComCAT's
nuisance model does not generalize to scans or subjects that were not used to fit it. In
four datasets where the answer is known, because the same subjects were scanned
repeatedly, the default model behaves as follows:

- **In-sample** it reduces technical variance, largely by overfitting.
- **On held-out data** it *adds* variance. Within-subject variance rises up to 163-fold
  (ON-Harmony). The scanner spread of held-out scans reaches 348 times that of the raw
  data (Buchert, median of 10 draws). Single values lie more than 1,000 SD from the input
  (section 2).
- **Between-subject variance**, which is biology in these designs, is removed already
  in-sample: 71% in ON-Harmony and 36% in MR-ART.

The causes can be separated:

1. **Instability and overfitting.** There are seven partly redundant IQMs, each expanded
   into 5–10 unpenalized B-spline columns. The resulting designs are rank-deficient or
   near-singular, so the fit cancels on the training data but not for new IQM values.
   This alone explains the failure in single-subject data, where no biology can be removed.
2. **Confounding with anatomy.** IQMs differ between subjects, so a model fitted across
   subjects learns part of each subject's anatomy as an IQM effect. It removes biology,
   and on new data it adds within-subject variance.
3. **The variance step.** Site variances come from few or unbalanced scans, and they make
   held-out results worse. In single-site runs through `comcat_ui`, the step also rescales
   all residuals by about 1 − R² of the in-sample fit, which flatters single-subject
   metrics.

What helped:

- Penalized (ridge) or linear IQM terms remove the blow-ups.
- Estimating the IQM effects within subjects, i.e. with subject terms in the model and a
  location correction only, was the only variant that reduced technical variance on
  held-out data without removing biology.

What did not help: IQM effects estimated within subjects on one dataset did not transfer
to another.

## Data and design

| Dataset | Design | Held out | Metric on the held-out harmonized data |
|---|---|---|---|
| Tohoku | 1 subject, 121 scans (several sequences and sessions), no site labels | 10 folds of scans | var(H) / var(raw) across scans |
| Buchert | 1 subject, 531 scans on 116 scanners (3–11 per scanner) | 1 scan per scanner; 10 random draws, median | var(H) / var(raw) across the held-out scans |
| MR-ART | 148 subjects × 3 acquisitions (standard and two head-motion levels), 1 scanner | 5 folds of subjects | W/W₀, B/B₀, motion bias |
| ON-Harmony | 10 subjects × 6 scanners; 4 subjects with 5 rescans on one scanner | 1 subject at a time | W/W₀, B/B₀, W/T |

Common settings:

- Gray matter at 8 mm (s6rp1); voxels with mean ≥ 0.05.
- Preserved covariate: age.
- IQMs: ICR, NCR, res_ECR, res_RMS, contrastr, SIQR, IQR. res_RMS is constant in MR-ART
  and was dropped there.
- `gam_df` is chosen automatically from the training n.
- `smooth_term_bounds` is the range over all scans, so held-out IQMs lie inside the knots.
- Empirical Bayes is on.
- ComCAT is fitted on the training scans with `comcat()`. The held-out scans are
  harmonized with `comcat_from_training()`.

The metrics are computed per voxel and averaged over voxels:

- **W** is the within-subject sum of squares (deviations from each subject's mean).
- **B** is the between-subject sum of squares (subject means around the grand mean).
- **W₀, B₀** are the same for the raw data.

In these designs, W is technical variance plus scan–rescan noise and should go down. B is
biology and should stay at about 1. Motion bias (MR-ART) is the mean over voxels of
|mean(high motion) − mean(standard)| / SD.

## 1. The current model with default settings

Multi-site data were run with the variance step (default δ). Single-site data were run as
`comcat_ui` runs them: a missing batch is coded as one site, so the variance step is
applied.

| Dataset | In-sample | Held-out |
|---|---|---|
| Tohoku: var / raw | 0.09 | 0.21 |
| Buchert: var / raw | 0.17 | 348 (range 0.41 – 2.9·10⁶) |
| MR-ART: W/W₀ · B/B₀ | 1.40 · 0.64 | 7.96 · 1.19 |
| ON-Harmony: W/W₀ · B/B₀ (W/T; raw 0.074) | 4.35 · 0.29 (0.300) | 163 · 1.25 (0.649) |

In-sample, the result looks like successful harmonization only in the single-subject
data, where W cannot be separated from B. In the multi-subject data the default model
damages even the in-sample data: within-subject variance goes up, and between-subject
variance goes down.

## 2. Location model variants (without the variance step)

The protocols are the same, but with `mean_only=True`, so the harmonized data are the
input minus the estimated site and IQM effects. Variants that are not ComCAT options were
injected for this test ([evaluation/heldout.py](evaluation/heldout.py)). The table shows
held-out results. For Tohoku and Buchert, lower is better. For MR-ART and ON-Harmony,
lower W, B near 1 and lower motion bias are better.

| Variant | Tohoku var/raw | Buchert var/raw | MR-ART W/W₀ · B/B₀ · motion bias | ON-Harmony W/W₀ · B/B₀ |
|---|---|---|---|---|
| raw | 1 | 1 | 1 · 1 · 0.190 | 1 · 1 |
| no IQMs (sites only) | 1 | 0.39 | 1 · 1 · 0.190 | 1.91 · 1.00 |
| 7 IQMs, splines (current) | 0.77 | 1,631 | 10.1 · 1.53 · 0.100 | 483 · 3.62 |
| … truncated SVD (rcond 1e-3, scaled columns) | 0.85 | 6.3 | 4.0 · 1.19 · 0.069 | 297 · 2.13 |
| … ridge penalty on the spline block, λ by GCV | **0.67** | 0.50 | 1.04 · 0.99 · 0.069 | 8.8 · 1.13 |
| 5 IQMs (without SIQR, IQR), splines | 0.73 | 1,313 | 2.6 · 1.09 · 0.058 | 51 · 1.47 |
| IQR only, splines | 0.83 | 0.56 | 1.7 · 1.04 · 0.092 | 25 · 1.10 |
| 7 IQMs, linear | 0.77 | 0.43 | 1.06 · 1.00 · 0.062 | 13.5 · 1.21 |
| 5 IQMs, linear | 0.76 | 0.41 | 0.99 · 1.00 · 0.067 | 11.2 · 1.20 |
| 2 principal components, splines | 0.75 | 0.98 | 1.9 · 1.04 · 0.068 | 12.7 · 1.11 |
| 2 principal components, linear | 0.77 | 0.39 | 0.96 · 0.99 · 0.078 | 5.5 · 1.04 |
| **5 IQMs, linear, estimated within subjects** | — | — | **0.92 · 1.05 · 0.024** | **0.99 · 1.02** |
| 5 IQMs, splines, estimated within subjects | — | — | 1.26 · 1.11 · 0.022 | 1.97 · 1.05 |

A dash means one subject only: every effect there is already estimated within subject.
The script also prints the in-sample values. W/T of ON-Harmony is 0.074 raw and 0.068
with the within-subject linear model.

### Cause 1: instability and overfitting

Single-subject data contain no biology, so failures there are purely statistical:

- **Tohoku:** the current model reduces the variance to 0.28 in-sample, but only to 0.77
  held-out.
- **Buchert:** it reaches 0.21 in-sample but 1,631 held-out (median), with single values
  more than 1,000 SD off.

Two properties of the model cause this:

- **The IQMs are partly redundant.** SIQR ≈ res_ECR (r = 0.99–1.00). IQR is largely a
  combination of NCR, ICR and res_RMS (linear R² 0.80–0.86). res_RMS has only 2–63
  distinct values and varies mostly between sites.
- **The splines are unpenalized.** 7 × `gam_df` columns (35 for n = 80, 70 for n = 531)
  give rank-deficient designs in ON-Harmony, ABIDE and NormSample, and near-singular ones
  in Buchert and ADNI (condition 10⁶–10⁷). `pinv` then returns large coefficients that
  cancel in the training data but not for new IQM values.

The fixes compare as follows:

- **Not enough:** a truncated SVD (Buchert 6.3), or removing SIQR and IQR while keeping
  splines (Buchert 1,313).
- **Removes the blow-ups in all four datasets:** a ridge penalty on the spline block (one
  λ, chosen by GCV pooled over voxels), or linear IQM terms. The largest deviations are
  then 1–13 SD, against 51–1,200 SD for the current model in MR-ART, ON-Harmony and
  Buchert. Ridge gives the best Tohoku result (0.67).
- **Buchert:** no IQM model beats site correction alone (0.39). With 116 scanner effects
  in the model, the IQMs only vary within scanner.

### Cause 2: IQM effects estimated across subjects are confounded with anatomy

In ON-Harmony, every IQM model fitted across subjects damages the data, the penalized and
linear models included:

- Held-out, the within-subject (between-scanner) variance rises 5- to 480-fold.
- In-sample, 20–60% of the between-subject variance is removed (B/B₀ 0.40–0.80, without
  the variance step).

The mechanism: IQMs differ between subjects, so a model fitted across subjects learns
part of each subject's anatomy as an IQM effect. Applied to a new subject whose IQMs
differ between scanners, it subtracts a different amount of that "anatomy" on each
scanner. In MR-ART, the models fitted across subjects remove only about two thirds of the
high-motion bias (0.190 → 0.06–0.07).

**The fix that worked:** add centred subject indicators to the preserved covariates. The
IQM effects, and the site effects, are then estimated from within-subject variation only.
This was the only variant that reduced technical variance on held-out data without
touching biology:

- **MR-ART:** the motion bias fell by 87% (0.190 → 0.024) and W by 8%. B rose by 5%,
  because motion artifacts in the subject means were removed.
- **ON-Harmony:** W/T fell below raw (0.068 vs 0.074), with B unchanged.

With splines instead of linear terms, the within-subject model overfits again (W ×1.26
and ×1.97).

In ON-Harmony, site correction alone doubles W (1.91), because the four subjects with
five rescans dominate their scanner's mean. Subject terms in the model remove this
confound as well.

## 3. The variance step

Held-out results with and without the variance step. ON-Harmony shows W/W₀ · B/B₀
(W/T); Buchert shows var/raw (median of 10 draws).

| Variant | ON-H mean only | ON-H default δ | ON-H residual δ | Buchert mean only | Buchert default δ | Buchert residual δ |
|---|---|---|---|---|---|---|
| no IQMs (sites only) | 1.91 · 1.00 (0.080) | 2.02 · 0.90 (0.083) | 2.02 · 0.90 (0.083) | 0.39 | 0.58 | 0.58 |
| 7 IQMs, splines (current) | 483 · 3.62 (0.655) | 163 · 1.25 (0.649) | 491 · 3.23 (0.664) | 1,631 | 348 | 2,970 |
| … ridge, λ by GCV | 8.8 · 1.13 (0.158) | 4.8 · 0.61 (0.165) | 8.9 · 1.03 (0.166) | 0.50 | 0.44 | 0.95 |
| 5 IQMs, linear | 11.2 · 1.20 (0.173) | 7.1 · 0.75 (0.179) | 10.8 · 1.11 (0.176) | 0.41 | 0.54 | 0.52 |
| 5 IQMs, linear, within subjects | 0.99 · 1.02 (0.068) | 6.4 · 0.80 (0.209) | 8.3 · 1.19 (0.189) | — | — | — |

On held-out data:

- **ON-Harmony:** the variance step never improved W/T, because it lowers W and B
  together. For the within-subject model it tripled W/T (0.068 → 0.209). That model looks
  best of all in-sample with the default δ (W/W₀ 0.45), but reaches 6.4 held-out.
- **Buchert:** it made site correction alone and the linear model worse (0.39 → 0.58,
  0.41 → 0.54). It made only the ridge model slightly better (0.50 → 0.44).

The site variances come from few scans (Buchert: 2–10 per scanner in training) or are
distorted by the rescans (ON-Harmony). Meanwhile, a new subject's residual, which
contains its anatomy, is multiplied by a scanner-specific factor. `residual_delta` with a
flexible nuisance model is unreliable, because overfitting shrinks the in-sample residual
variances, and by different amounts at different sites.

**Single-site runs through `comcat_ui`** rescale every residual by about 1 − R² of the
in-sample nuisance fit:

- **Tohoku:** this rescaling alone takes the held-out variance from 0.77 (location
  correction) to 0.21, and the in-sample variance from 0.28 to 0.09.
- **MR-ART:** it lowers in-sample B/B₀ from 0.78 to 0.64.

Metrics on single-subject samples, such as Tohoku's BrainAGE MAE, are therefore flattered
by the variance step, not only by the IQM model.

## 4. Transferring within-subject IQM effects

Linear IQM effects (ICR, NCR, res_ECR, contrastr) were estimated within subjects on
MR-ART. They were then subtracted, unchanged, from the other datasets; nothing was fitted
on the target data.

| Target | Result |
|---|---|
| Tohoku | var/raw 2.10 |
| Buchert | var/raw 4.97 (within-scanner 2.04) |
| ON-Harmony | W/W₀ 3.06, B/B₀ 1.02 (W/T 0.125; raw 0.074) |

The voxel-wise IQM effects depend on the scanner, the sequence and the kind of artifact:
motion in MR-ART, protocol differences elsewhere. They cannot be estimated on one dataset
and applied to another.

## Suggestions

The suggestions are in order of priority. Each names the cause it addresses; the
evidence is in the sections above.

1. **Stabilize the nuisance model** (cause 1; needed in any case).
   - Drop redundant IQMs (SIQR, IQR) and IQMs that are constant in the data.
   - Replace the unpenalized splines by a penalized fit with the penalty chosen by GCV or
     REML (here: one ridge penalty on the spline block, GCV pooled over voxels), or use
     linear IQM terms. This removed all blow-ups.
   - Choose the IQMs, `gam_df` and the penalty by an out-of-sample criterion, not by
     in-sample fit or p-values (reviewer 3).
   - Not implemented yet: this needs new options in `comcat()` (linear terms, penalty).
2. **Estimate the IQM and site effects within subjects wherever subjects have repeated
   scans** (cause 2). This applies to traveling subjects, test–retest data and MR-ART.
   - Add centred subject indicators to the preserved covariates, use linear IQM terms,
     and correct the location only.
   - This was the only model that improved held-out data without removing biology. It
     also turns the traveling-subject datasets into the feature-level validation the
     reviewers asked for (T1.3).
   - With the current code, `preserve` (centred subject indicators, with zeros for new
     subjects in `comcat_from_training()`) and `mean_only=True` already do this, but only
     with splines, which overfit (see suggestion 1).
3. **Use only the location correction for held-out or new data** (cause 3), or estimate
   the site variances much more conservatively.
   - Do not combine `residual_delta` with a flexible nuisance model.
   - Do not apply the variance step in single-site runs. `comcat()` with `batch=None`
     already uses mean only, but `comcat_ui` codes a missing batch as one site. If the
     step is used, report its rescaling explicitly.
4. **Report held-out results.** Harmonize each subject (or fold) with parameters
   estimated without it (cross-fitting). In-sample metrics reward overfitting: for the
   IQM model alone, Tohoku reaches 0.28 in-sample against 0.77 held-out. This is the core
   of reviewer issue B.
5. **Cross-sectional samples (ABIDE, NormSample, ADNI).** There, IQM effects cannot be
   separated from anatomy, and transferring effects from other data failed (section 4).
   - Use the stabilized model and state the confound as a limitation.
   - Quantify the confound: the share of between-subject variance that the IQMs remove
     (variance decomposition, T1.4), and a comparison with permuted IQMs (T1.1).
   - A clean solution needs within-subject information from the same scanners, for
     example a few rescans per site.

## Reproducing

The scripts are in [evaluation/](evaluation/); data paths are set in
[evaluation/data.py](evaluation/data.py) or through environment variables.

| Script | Sections |
|---|---|
| `heldout_run.py` | 2 (location variants, in-sample and held-out) |
| `heldout_delta.py` | 1 (ON-Harmony, Buchert) and 3 |
| `heldout_ui_single.py` | 1 (Tohoku, MR-ART) and 3 (single-site rescaling) |
| `transfer.py` | 4 |

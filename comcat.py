"""
ComCAT: Combating CovariATe effects — core harmonization function.

Usage
-----
from comcat import comcat

Y_harmonized, beta_hat, gamma_hat, delta_hat = comcat(
    Y, batch, nuisance, preserve,
    mean_only=False, verbose=True
)

Parameters
----------
Y           : ndarray, shape (n_features, n_subjects)
batch       : array-like, shape (n_subjects,)  — site/scanner labels
nuisance    : ndarray or None, shape (n_subjects, n_nuisance)  — variables to remove
preserve    : ndarray or None, shape (n_subjects, n_preserve)  — variables to keep
mean_only   : bool   — if True, only adjust mean (no variance scaling)
verbose     : bool

Returns
-------
Y_harmonized : ndarray, shape (n_features, n_subjects)
beta_hat     : ndarray  — full design matrix betas
gamma_hat    : ndarray  — additive batch/nuisance effects removed from the data
                          (full feature space; empirical Bayes estimates in the
                          site rows when eb=True)
delta_hat    : ndarray  — multiplicative batch effects removed from the data
                          (full feature space; empirical Bayes estimates when eb=True)

Additional parameters
---------------------
eb                   : bool (default True)
                       Empirical Bayes estimation of the site effects, as in ComBat
                       (Johnson et al., 2007): the additive (gamma) and multiplicative
                       (delta) site parameters of every feature are shrunk towards
                       priors estimated from all features, which stabilizes them for
                       small sites.  Used only with at least two sites.  eb=False
                       gives the L/S estimates of earlier versions.  Without
                       nuisance covariates the result equals ComBat with empirical
                       Bayes (combat_family.combat_eb).
ref_batch            : label of a site to use as reference (its data is left untouched;
                       all other sites are harmonized to its mean level, including its
                       mean nuisance level, and to its variance delta). Default None.
return_estimates     : if True, a 5th element (dict) with all fitted parameters is
                       returned; pass it to comcat_from_training() for new data.
                       With eb=True it holds the L/S estimates ('gamma_hat_masked',
                       'delta_hat_masked'), the applied empirical Bayes estimates
                       ('gamma_star_masked', 'delta_star_masked') and the priors
                       ('eb_priors').
Every nuisance column is modelled as a penalized B-spline (nuisance_model, below;
statsmodels is needed for the unpenalized 'spline' model). The basis is configured by:

smooth_term_bounds   : boundary knots for the B-spline of each nuisance column.
                       • None  — infer bounds from training data (safe for training only)
                       • (lo, hi) — same bounds for every nuisance column
                       • [(lo0,hi0), (lo1,hi1), ...] — one pair per nuisance column
                       For apply-to-new-data workflows, always specify explicit bounds
                       that cover the full range of training AND test data.
gam_df               : int, B-spline basis dimension per nuisance column (default None:
                       10 for 'pspline', whose penalty sets the smoothness; for the
                       unpenalized 'spline' min(10, max(5, n//30)), where higher values
                       capture finer nonlinearities but overfit).

Optional model extensions (all off by default; with eb=False, the defaults
reproduce earlier results)
-------------------------------------------------------------------------------
preserve_df          : None (default) | int | 'same'
                       B-spline expansion of continuous preserve covariates, so that
                       non-linear effects of interest (e.g. age) are preserved with the
                       same flexibility with which nuisance effects are removed.
                       'same' uses gam_df.  Columns with fewer than preserve_df + 2
                       distinct values (e.g. binary group or sex) stay linear.
preserve_bounds      : boundary knots for the preserve splines, same format as
                       smooth_term_bounds but indexed by preserve column.  Needed for
                       comcat_from_training() when new data exceed the training range.
nuisance_scale       : bool (default False)
                       Also remove nuisance-dependent variance (heteroscedasticity).
                       A log-linear model  log sd = site + nuisance + preserve  is
                       fitted by maximum likelihood to the residuals of every feature;
                       residuals are rescaled to the variance at the mean nuisance
                       level.  Site and preserve variance effects stay in the model,
                       so preserved effects on the variance are kept.  Implies
                       residual_delta=True.  estimates['scale_lr'] holds a per-feature
                       likelihood-ratio test of the nuisance variance effects,
                       studentized by the residual kurtosis (estimates['scale_kurtosis'])
                       so that it also holds for skewed or heavy-tailed residuals.
residual_delta       : bool (default False)
                       Estimate site variances delta from the residuals after the
                       additive site AND nuisance effects are removed, as ComBat does.
                       The default (False, as in the MATLAB implementation) estimates
                       delta before the nuisance effects are removed; delta then also
                       contains nuisance-explained variance, which shrinks the residual
                       variance of the harmonized data by roughly
                       var(residual) / (var(residual) + var(nuisance effect)).

Nuisance model, subjects and cross-fitting
-------------------------------------------------------------------------------
nuisance_model       : 'pspline' (default) | 'linear' | 'spline'
                       'spline'  unpenalized cubic B-splines with gam_df columns per
                                 nuisance column (versions before 2026-10; they
                                 overfit with several IQMs and do not generalize to
                                 new scans, see HeldOut-Evaluation.md).
                       'linear'  every nuisance column enters linearly (z-scored with
                                 the training mean and SD).
                       'pspline' penalized cubic B-splines (P-splines): per nuisance
                                 column one linear term and gam_df - 2 nonlinear terms
                                 that are shrunk by a second-order difference penalty.
                                 One smoothing parameter for all nuisance columns is
                                 estimated from all features (nuisance_penalty), i.e.
                                 an empirical Bayes estimate of the size of the
                                 nonlinear nuisance effects across features.  Large
                                 penalties give the linear model.  gam_df defaults to
                                 10; new data are evaluated with the nonlinear part at
                                 the training range (the linear part continues).
                       With 'linear' and 'pspline', nuisance columns without variation
                       within the sites (and subjects) are dropped with a warning, and
                       columns with fewer than gam_df + 2 distinct values stay linear.
nuisance_penalty     : None | 'reml' | 'gcv' | float  (only 'pspline')
                       None/'reml': restricted maximum likelihood pooled over the
                       standardized features; 'gcv': generalized cross-validation
                       pooled over features; a number fixes the penalty.
subject              : None | array-like (n_subjects,) of subject labels
                       For repeated scans of the same subjects (traveling subjects,
                       test-retest data, scans with and without motion).  Centred
                       subject indicators are added to the preserved covariates, so
                       that the site and nuisance effects are estimated from the
                       variation within subjects and cannot take up anatomy that
                       differs between subjects.  Implies mean_only=True.
                       comcat_from_training() treats its subjects as new subjects.
comcat_crossfit()    : every subject is harmonized with a model that was not fitted
                       on it (k-fold over subjects), optionally fitted on a subset such
                       as the controls and applied to all others.

GAM smoothness recommendations (unpenalized 'spline' model)
--------------------------------------------------------------
With the default 'pspline' model the smoothness is estimated (REML), and gam_df
only sets the size of the basis.  For the unpenalized 'spline' model, the B-spline
basis uses cubic splines (degree=3) with `gam_df` columns
(= n_internal_knots + degree + 1 with intercept in statsmodels convention).

Practical guidelines:
- `gam_df=None` (default) uses the sample-size heuristic min(10, max(5, n//30)):
    n=80  → 5,  n=200 → 6,  n=300 → 10,  n=500 → 10 (capped).
  Pass an explicit integer to override.
- Values above 10 rarely help and inflate the design matrix (slows pinv).
- For small samples (n < 100): keep `gam_df ≤ 6` to avoid near-rank-deficiency.
- Always set `smooth_term_bounds` explicitly in train/test workflows so the
  knot positions are identical between training and new data.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.linalg import pinv
from statsmodels.gam.api import BSplines

_NUISANCE_MODELS = ('spline', 'linear', 'pspline')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def comcat(
    Y: np.ndarray,
    batch: np.ndarray,
    nuisance: np.ndarray | None = None,
    preserve: np.ndarray | None = None,
    mean_only: bool = False,
    verbose: bool = False,
    ref_batch=None,
    return_estimates: bool = False,
    smooth_term_bounds=None,
    gam_df: int | None = None,
    preserve_df: int | str | None = None,
    preserve_bounds=None,
    nuisance_scale: bool = False,
    residual_delta: bool = False,
    eb: bool = True,
    nuisance_model: str = 'pspline',
    nuisance_penalty=None,
    subject=None,
):
    """ComCAT harmonization for sites and nuisance parameters.

    See the module docstring for all parameters, including the empirical
    Bayes option eb, the optional extensions preserve_df, preserve_bounds,
    nuisance_scale and residual_delta, and the nuisance models, subject terms
    and cross-fitting (nuisance_model, nuisance_penalty, subject).
    """

    # ------------------------------------------------------------------ setup
    # Use float64 throughout for numerical precision (MATLAB promotes to double too)
    Y = np.array(Y, dtype=np.float64)

    if nuisance_model not in _NUISANCE_MODELS:
        raise ValueError(f"nuisance_model must be one of {_NUISANCE_MODELS}, "
                         f"not {nuisance_model!r}.")

    # Handle None/empty batch — determine n_subjects from nuisance/preserve if needed
    batch_empty = (batch is None or np.asarray(batch).size == 0)

    if not batch_empty:
        batch = np.asarray(batch).ravel()
        n_subjects = len(batch)
    else:
        n_subjects = None
        for arr in (nuisance, preserve, subject):
            if arr is not None:
                a = np.asarray(arr)
                if a.size > 0:
                    n_subjects = int(max(a.shape))
                    break
        if n_subjects is None:
            # Nothing to harmonize
            return (np.array(Y, dtype=np.float32), np.array([]),
                    np.array([]), np.array([]))
        batch = np.ones(n_subjects, dtype=int)
        mean_only = True

    # recode batch labels to 0-based integers; keep originals for ref_batch lookup
    levels_original, batch = np.unique(batch, return_inverse=True)

    ref_level: int | None = None
    if ref_batch is not None:
        matches = np.where(levels_original == ref_batch)[0]
        if len(matches) == 0:
            raise ValueError(
                f"ref_batch={ref_batch!r} not found. "
                f"Available labels: {levels_original.tolist()}"
            )
        ref_level = int(matches[0])

    nuisance = _to_col_matrix(nuisance, n_subjects)   # (n_subjects, n_Z)
    preserve = _to_col_matrix(preserve, n_subjects)   # (n_subjects, n_X)

    n_Z = nuisance.shape[1]

    # subject terms: effects are estimated within subjects (location only)
    subject_cols = np.empty((n_subjects, 0))
    if subject is not None:
        subject = np.asarray(subject).ravel()
        if len(subject) != n_subjects:
            raise ValueError(f"subject has {len(subject)} labels for {n_subjects} subjects.")
        if nuisance_scale:
            raise ValueError("nuisance_scale cannot be combined with subject terms.")
        subject_cols = _subject_design(subject)
        if verbose:
            print(f"[ComCAT] {subject_cols.shape[1] + 1} subjects: effects estimated "
                  "within subjects, mean only")
        mean_only = True

    # Resolve gam_df from sample size when not set explicitly:
    #   min(10, max(5, n_subjects // 30)); 10 for penalized splines
    if gam_df is None:
        gam_df = 10 if nuisance_model == 'pspline' else min(10, max(5, n_subjects // 30))
        if verbose and nuisance_model != 'linear':
            print(f"[ComCAT] gam_df auto-selected: {gam_df} (n={n_subjects})")

    # optional B-spline expansion of continuous preserve covariates
    preserve_orig = preserve.copy()                   # linear columns, used by the scale model
    if preserve_df == 'same':
        preserve_df = gam_df
    preserve, preserve_splines = _build_preserve_basis(
        preserve, preserve_df, preserve_bounds, verbose
    )
    preserve = np.hstack([preserve, subject_cols])    # subject terms are the last columns
    n_X = preserve.shape[1]

    # Y must be (n_features, n_subjects)
    transp = False
    if Y.shape[1] != n_subjects:
        if Y.shape[0] == n_subjects:
            Y = Y.T
            transp = True
        else:
            raise ValueError(
                f"Shape mismatch: Y {Y.shape}, n_subjects={n_subjects}"
            )

    n_features, _ = Y.shape

    # ------------------------------------------------------------------ mask
    sd0 = np.std(Y, axis=1, ddof=1)
    ind_mask = (sd0 > 0) & np.isfinite(sd0)
    ind_nan = np.isnan(sd0)

    Ym = Y[ind_mask, :]              # (n_valid, n_subjects)

    # ------------------------------------------------ nuisance basis expansion
    n_nuisance_orig = n_Z            # columns before expansion (needed for from_training)
    nuisance_orig = nuisance.copy()  # keep original columns for confounding diagnostics
    nuisance_keep = np.arange(n_Z)   # columns used (all of them for 'spline')
    nuisance_setup = None            # basis of 'linear' and 'pspline' (for from_training)
    pen_nuisance = np.zeros(0, dtype=bool)   # penalized columns of the nuisance basis
    spline_constructors = {}
    if n_Z > 0 and nuisance_model == 'spline':
        if verbose:
            print(f"[ComCAT] GAM (B-spline, df={gam_df}) for all {n_Z} nuisance column(s)")
        nuisance, spline_constructors = _build_nuisance_basis(
            nuisance, smooth_term_bounds, gam_df, verbose
        )
        n_Z = nuisance.shape[1]
    elif n_Z > 0:
        # nuisance columns must vary within the sites (and subjects)
        units = np.hstack([(batch[:, None] == np.unique(batch)[None, :]).astype(float),
                           subject_cols])
        nuisance_keep = np.flatnonzero(_screen_nuisance(nuisance, units))
        if nuisance_keep.size < n_Z:
            warnings.warn(
                f"Nuisance column(s) {sorted(set(range(n_Z)) - set(nuisance_keep))} do "
                "not vary within the sites (and subjects) and are not used.",
                RuntimeWarning, stacklevel=2,
            )
        nuisance_setup = [
            _nuisance_column_setup(nuisance[:, j], nuisance_model, gam_df,
                                   _column_bounds(smooth_term_bounds, j))
            for j in nuisance_keep
        ]
        if nuisance_keep.size:
            nuisance, pen_nuisance = _nuisance_design(nuisance[:, nuisance_keep],
                                                      nuisance_setup)
        else:
            nuisance = np.empty((n_subjects, 0))
        n_Z = nuisance.shape[1]
        if verbose:
            kinds = [s['kind'] for s in nuisance_setup]
            print(f"[ComCAT] Nuisance model '{nuisance_model}': {kinds.count('linear')} "
                  f"linear, {kinds.count('pspline')} penalized spline column(s) "
                  f"({n_Z} terms)")

    # --------------------------------------------------- batch / design matrix
    levels = np.unique(batch)
    n_batch = len(levels)
    batchmod = (batch[:, None] == levels[None, :]).astype(float)  # one-hot (n_subjects, n_batch)

    batches = [np.where(batch == lvl)[0] for lvl in levels]
    n_batches = np.array([len(b) for b in batches])

    if verbose and n_batch > 1:
        print(f"[ComCAT] Found {n_batch} different sites")

    ind_batch    = slice(0, n_batch)
    ind_nuisance = slice(n_batch, n_batch + n_Z)
    ind_preserve = slice(n_batch + n_Z, n_batch + n_Z + n_X)

    # full design matrix: [batch_onehot | nuisance | preserve]
    parts = [batchmod]
    if n_Z > 0:
        parts.append(nuisance)
    if n_X > 0:
        parts.append(preserve)
        if verbose:
            print(f"[ComCAT] Preserving {n_X} covariate(s)")
    XZ = np.hstack(parts)    # (n_subjects, n_batch + n_Z + n_X)

    # confounding check — warn but continue (pinv handles rank deficiency).
    # Subject terms are collinear with covariates that do not vary within
    # subjects (e.g. age); only the site and nuisance part is checked then.
    XZ_check = XZ if subject is None else XZ[:, :n_batch + n_Z]
    if np.linalg.matrix_rank(XZ_check) < XZ_check.shape[1]:
        # Identify which *original* nuisance columns are most confounded with
        # batch by computing the R² of regressing each original column onto
        # the batch one-hot matrix.
        confounded = []
        if nuisance_orig.shape[1] > 0:
            for col_idx in range(nuisance_orig.shape[1]):
                col = nuisance_orig[:, col_idx]
                proj = batchmod @ (pinv(batchmod) @ col)
                ss_res = np.sum((col - proj) ** 2)
                ss_tot = np.sum((col - col.mean()) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
                if r2 > 0.95:
                    confounded.append((col_idx, r2))
        if confounded:
            details = ", ".join(
                f"col {i} (R²={r:.3f})" for i, r in confounded
            )
            warnings.warn(
                "Design matrix is rank-deficient: nuisance covariate(s) are "
                f"strongly confounded with batch — {details}. "
                "Proceeding with pseudoinverse; confounded columns will have "
                "reduced or no independent effect.",
                RuntimeWarning, stacklevel=3,
            )
        else:
            warnings.warn(
                "Design matrix is rank-deficient (covariates confounded with "
                "batch). Proceeding with pseudoinverse.",
                RuntimeWarning, stacklevel=3,
            )

    # --------------------------------------------------- standardize
    if verbose:
        print("[ComCAT] Standardizing data across features")

    # penalty of the nonlinear nuisance terms ('pspline'): one smoothing
    # parameter for all nuisance columns, estimated from all features
    pen = None
    nuisance_lambda = nuisance_edf = None
    if pen_nuisance.any():
        pen_cols = np.zeros(XZ.shape[1], dtype=bool)
        pen_cols[n_batch:n_batch + n_Z] = pen_nuisance
        if nuisance_penalty is None or isinstance(nuisance_penalty, str):
            nuisance_lambda, edf_pen = _select_penalty(
                Ym, XZ, pen_cols, method=nuisance_penalty or 'reml')
        else:
            nuisance_lambda = float(nuisance_penalty)
            edf_pen = _penalized_edf(XZ, pen_cols, nuisance_lambda)
        nuisance_edf = edf_pen + int(np.sum(~pen_nuisance))
        pen = np.where(pen_cols, nuisance_lambda, 0.0)
        if verbose:
            print(f"[ComCAT] Nuisance penalty {nuisance_lambda:.3g}: {nuisance_edf:.1f} "
                  f"effective df of {n_Z} nuisance terms")
    elif nuisance_model == 'linear':
        nuisance_edf = float(n_Z)

    beta_hat = _solve(XZ, pen) @ Ym.T   # (n_cols, n_valid)

    XZ_no_preserve = XZ[:, list(range(n_batch)) + list(range(n_batch, n_batch + n_Z))]
    fitted_site = XZ_no_preserve @ beta_hat[:n_batch + n_Z, :]   # site + nuisance part of the fit
    if ref_level is not None:
        # grand mean = mean site + nuisance level of the reference batch.  The
        # intercept of the reference batch alone is its level at the zero of the
        # nuisance basis, and it is not identified when the nuisance splines are
        # collinear with the batch indicators (e.g. tied IQM values); the mean
        # fitted level is identified and matches the untouched reference data.
        grand_mean = np.mean(fitted_site[batches[ref_level]], axis=0)
    else:
        grand_mean = np.mean(fitted_site, axis=0)
    del fitted_site

    residuals = Ym - (XZ @ beta_hat).T   # (n_valid, n_subjects)
    std_pooled = np.sqrt(np.mean(residuals ** 2, axis=1))   # (n_valid,)

    # guard against zero pooled std
    nz = std_pooled > 0
    if not np.all(nz):
        std_pooled[~nz] = np.median(std_pooled[nz]) if np.any(nz) else 1.0

    # subtract grand mean and preserve-covariate contribution, then scale
    if n_X > 0:
        preserve_contrib = (XZ[:, ind_preserve] @ beta_hat[ind_preserve, :]).T   # (n_valid, n_subjects)
    else:
        preserve_contrib = 0.0

    Ym = (Ym - grand_mean[:, None] - preserve_contrib) / std_pooled[:, None]

    # --------------------------------------------------- fit L/S model
    if verbose:
        print("[ComCAT] Fitting L/S model")

    X_nuisance = np.hstack([batchmod] + ([nuisance] if n_Z > 0 else []))   # (n_subjects, n_batch + n_Z)
    # with the same penalty as above, the estimates equal the site and nuisance
    # rows of beta_hat (in standardized units)
    gamma_hat_masked = _solve(X_nuisance, None if pen is None else pen[:n_batch + n_Z]) @ Ym.T

    # Residuals after removing additive site and nuisance effects; needed when
    # site variances are estimated from residuals or nuisance variance is removed
    scale_model = None
    if nuisance_scale and n_Z == 0:
        if verbose:
            print("[ComCAT] nuisance_scale ignored: no nuisance covariates")
        nuisance_scale = False
    if nuisance_scale or residual_delta:
        resid_std = Ym - (X_nuisance @ gamma_hat_masked).T   # (n_valid, n_subjects)
        if nuisance_scale:
            if verbose:
                print(f"[ComCAT] Fitting log-variance model (site + "
                      f"{nuisance_orig.shape[1]} nuisance + {preserve_orig.shape[1]} "
                      "preserve covariate(s), linear)")
            scale_model = _fit_nuisance_scale(
                resid_std, batchmod, nuisance_orig, preserve_orig
            )
            resid_std /= _nuisance_scale_factor(nuisance_orig, scale_model)
        var_source = resid_std
        if verbose:
            print("[ComCAT] Site variances estimated from residuals")
    else:
        resid_std = None
        var_source = Ym

    delta_hat_masked = np.zeros((n_batch + n_Z, Ym.shape[0]), dtype=np.float64)
    for i in range(n_batch):
        idx = batches[i]
        if mean_only:
            delta_hat_masked[i, :] = 1.0
        else:
            delta_hat_masked[i, :] = np.var(var_source[:, idx], axis=1, ddof=1)

    for i in range(n_batch, n_batch + n_Z):
        if mean_only:
            delta_hat_masked[i, :] = 1.0
        else:
            delta_hat_masked[i, :] = np.var(Ym, axis=1, ddof=1)

    # --------------------------------------------------- empirical Bayes
    use_eb = eb and n_batch > 1 and Ym.shape[0] > 1
    if use_eb:
        if verbose:
            print("[ComCAT] Empirical Bayes estimation of site effects")
        # mean_only still needs the site variances for the posterior of gamma
        site_var = (np.vstack([np.var(var_source[:, idx], axis=1, ddof=1) for idx in batches])
                    if mean_only else delta_hat_masked[:n_batch])
        gamma_star_masked, delta_star_sites, eb_priors = _eb_site_effects(
            gamma_hat_masked, site_var, n_batches, ref_level
        )
        delta_star_masked = delta_hat_masked.copy()
        if not mean_only:
            delta_star_masked[:n_batch] = delta_star_sites
    else:
        gamma_star_masked, delta_star_masked, eb_priors = gamma_hat_masked, delta_hat_masked, None

    # --------------------------------------------------- adjust data
    if verbose:
        print("[ComCAT] Adjusting the data")
        if ref_level is not None:
            print(f"[ComCAT] Reference batch: index {ref_level} "
                  f"({levels_original[ref_level]!r}) — left unchanged")

    # with a reference batch, the other batches are scaled to its variance
    # instead of the pooled one, so that they match the untouched reference data
    # (the reference batch is not shrunk by empirical Bayes)
    delta_ref = delta_star_masked[ref_level, :] if ref_level is not None else None

    for i in range(n_batch):
        if ref_level is not None and i == ref_level:
            continue  # reference batch is not adjusted
        idx = batches[i]
        delta_i = delta_star_masked[i, :] if delta_ref is None else delta_star_masked[i, :] / delta_ref
        denom = np.sqrt(delta_i)[:, None] * np.ones((1, n_batches[i]))
        if resid_std is not None:
            numer = resid_std[:, idx]
            if use_eb:
                # resid_std removed the L/S site effect; remove the EB one instead
                shift = (gamma_hat_masked[i] - gamma_star_masked[i])[:, None]
                if scale_model is not None:
                    shift = shift / _nuisance_scale_factor(nuisance_orig[idx], scale_model)
                numer = numer + shift
        else:
            numer = Ym[:, idx] - (X_nuisance[idx, :] @ gamma_star_masked).T
        Ym[:, idx] = numer / denom
    del resid_std

    Ym = np.where(np.isfinite(Ym), Ym, 0.0)

    # --------------------------------------------------- reconstruct
    Y_harmonized = np.zeros((n_features, n_subjects), dtype=np.float64)
    for i in range(n_subjects):
        pc_i = (XZ[i, ind_preserve] @ beta_hat[ind_preserve, :]) if n_X > 0 else 0.0
        Y_harmonized[ind_mask, i] = Ym[:, i] * std_pooled + grand_mean + pc_i
    Y_harmonized[ind_nan, :] = np.nan

    # restore reference batch to original values
    if ref_level is not None:
        ref_idx = batches[ref_level]
        Y_harmonized[:, ref_idx] = Y[:, ref_idx]

    # --------------------------------------------------- full-space outputs
    # effects that were removed (empirical Bayes estimates with eb)
    n_gamma = gamma_star_masked.shape[0]
    gamma_hat = np.zeros((n_gamma, n_features), dtype=np.float64)
    for i in range(n_gamma):
        gamma_hat[i, ind_mask] = gamma_star_masked[i, :]

    n_delta = delta_star_masked.shape[0]
    delta_hat = np.zeros((n_delta, n_features), dtype=np.float64)
    for i in range(n_delta):
        delta_hat[i, ind_mask] = delta_star_masked[i, :]

    beta_hat_full = np.zeros((XZ.shape[1], n_features), dtype=np.float64)
    beta_hat_full[:, ind_mask] = beta_hat

    if transp:
        Y_harmonized = Y_harmonized.T

    if not return_estimates:
        return Y_harmonized, beta_hat_full, gamma_hat, delta_hat

    estimates = {
        'grand_mean':          grand_mean,
        'std_pooled':          std_pooled,
        'gamma_hat_masked':    gamma_hat_masked,
        'delta_hat_masked':    delta_hat_masked,
        'beta_hat_preserve':   beta_hat[ind_preserve, :].copy() if n_X > 0 else None,
        'ind_mask':            ind_mask,
        'ind_nan':             ind_nan,
        'batch_levels':        levels_original,
        'n_batch':             n_batch,
        'n_batches':           n_batches,
        'n_nuisance_orig':     n_nuisance_orig,
        'n_Z':                 n_Z,
        'n_X':                 n_X,
        'mean_only':           mean_only,
        'ref_level':           ref_level,
        # GAM parameters
        'smooth_term_bounds':  smooth_term_bounds,
        'gam_df':              gam_df,
        'spline_constructors': spline_constructors,
        # optional extensions
        'preserve_splines':    preserve_splines,
        'residual_delta':      bool(residual_delta or nuisance_scale),
        'nuisance_scale':      scale_model,
        # empirical Bayes: applied site effects (None without eb) and priors
        'eb':                  use_eb,
        'gamma_star_masked':   gamma_star_masked if use_eb else None,
        'delta_star_masked':   delta_star_masked if use_eb else None,
        'eb_priors':           eb_priors,
        # nuisance model ('linear', 'pspline') and subject terms
        'nuisance_model':      nuisance_model,
        'nuisance_keep':       nuisance_keep,
        'nuisance_setup':      nuisance_setup,
        'nuisance_lambda':     nuisance_lambda,
        'nuisance_edf':        nuisance_edf,
        'n_subject_cols':      subject_cols.shape[1],
    }
    if scale_model is not None:
        # per-feature test of nuisance effects on the variance, full feature space
        for key in ('scale_lr', 'scale_p', 'scale_kurtosis'):
            full = np.full(n_features, np.nan)
            full[ind_mask] = scale_model[key]
            estimates[key] = full
        estimates['scale_lr_df'] = scale_model['lr_df']
        estimates['scale_resid_df'] = scale_model['resid_df']
    return Y_harmonized, beta_hat_full, gamma_hat, delta_hat, estimates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_col_matrix(arr, n: int) -> np.ndarray:
    """Return a 2-D column matrix (n, k), handling None / 1-D / transposed inputs."""
    if arr is None or (hasattr(arr, '__len__') and len(arr) == 0):
        return np.empty((n, 0), dtype=np.float64)
    arr = np.array(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != n:
        arr = arr.T
    return arr


def _build_nuisance_basis(
    nuisance: np.ndarray,
    smooth_term_bounds,
    gam_df: int,
    verbose: bool = False,
    spline_constructors: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Expand every nuisance column into a B-spline GAM basis.

    Each column is modelled with a statsmodels BSplines basis (no linear or
    polynomial option).

    Parameters
    ----------
    nuisance            : (n_subjects, n_cols) raw nuisance array
    smooth_term_bounds  : None | (lo, hi) | [(lo0,hi0), ...]  (per column)
    gam_df              : B-spline degrees of freedom per column
    verbose             : print warnings
    spline_constructors : pre-fitted BSplines objects keyed by column index
                          (from estimates dict); when provided, `.transform()` is
                          called instead of fitting new knots.  Pass {} to fit fresh.

    Returns
    -------
    expanded  : (n_subjects, n_expanded_cols)
    new_constructors : dict  {col_idx: BSplines}  (populated only when fitting fresh)
    """
    n_cols = nuisance.shape[1]
    new_constructors: dict = {}

    if n_cols == 0:
        return np.empty((nuisance.shape[0], 0), dtype=np.float64), new_constructors

    parts = []
    for i in range(n_cols):
        col = nuisance[:, i:i + 1].astype(float)  # keep 2-D

        if spline_constructors and i in spline_constructors:
            # apply training knots to new data
            bs = spline_constructors[i]
            basis = bs.transform(col)
        else:
            # fit new B-spline basis
            if isinstance(smooth_term_bounds, list):
                lo, hi = smooth_term_bounds[i]
            elif isinstance(smooth_term_bounds, tuple) and smooth_term_bounds != (None, None):
                lo, hi = smooth_term_bounds
            else:
                lo, hi = None, None
            knot_kwds = [{'lower_bound': lo, 'upper_bound': hi}]
            bs = BSplines(col, df=gam_df, degree=3, knot_kwds=knot_kwds)
            new_constructors[i] = bs
            basis = bs.basis
        parts.append(basis)

    return np.hstack(parts), new_constructors


def _build_preserve_basis(
    preserve: np.ndarray,
    preserve_df: int | None,
    preserve_bounds=None,
    verbose: bool = False,
    preserve_splines: dict | None = None,
) -> tuple[np.ndarray, dict | None]:
    """Optionally expand continuous preserve columns into B-spline bases.

    A column is expanded when it has at least preserve_df + 2 distinct values;
    others (binary group, sex, ...) stay linear.  The basis has no intercept
    column (the site indicators provide it), so linear trends are included.

    Parameters
    ----------
    preserve         : (n_subjects, n_preserve) raw preserve covariates
    preserve_df      : B-spline basis dimension, or None for no expansion
    preserve_bounds  : None | (lo, hi) for all columns | list of (lo, hi) or
                       None per preserve column
    preserve_splines : dict returned by a previous (training) call; its knots
                       are applied to `preserve` (new data)

    Returns
    -------
    expanded         : (n_subjects, n_expanded_cols)
    preserve_splines : {'df', 'cols' (expanded column indices),
                        'constructors' {col: BSplines}} or None
    """
    if preserve_splines is not None:
        cols = preserve_splines['cols']
        cons = preserve_splines['constructors']
        parts = [cons[c].transform(preserve[:, c:c + 1]) if c in cols
                 else preserve[:, c:c + 1] for c in range(preserve.shape[1])]
        return (np.hstack(parts) if parts else preserve), preserve_splines

    if preserve_df is None or preserve.shape[1] == 0:
        return preserve, None

    df = int(preserve_df)
    parts, cols, constructors = [], [], {}
    for c in range(preserve.shape[1]):
        col = preserve[:, c:c + 1]
        n_unique = len(np.unique(col))
        if n_unique < df + 2:
            if verbose:
                print(f"[ComCAT] Preserve column {c}: {n_unique} distinct values, kept linear")
            parts.append(col)
            continue
        bounds = preserve_bounds[c] if isinstance(preserve_bounds, list) else preserve_bounds
        basis, cons = _build_nuisance_basis(col, bounds, df)
        parts.append(basis)
        cols.append(c)
        constructors[c] = cons[0]

    if verbose and cols:
        print(f"[ComCAT] GAM (B-spline, df={df}) for preserve column(s) {cols}")
    return np.hstack(parts), {'df': df, 'cols': cols, 'constructors': constructors}


def _subject_design(subject) -> np.ndarray:
    """Centred subject indicators without the last subject, (n, n_subjects - 1).

    Added to the preserved covariates, they restrict the site and nuisance
    estimates to the variation within subjects (subject fixed effects).
    """
    _, codes = np.unique(subject, return_inverse=True)
    D = (codes[:, None] == np.arange(codes.max() + 1)[None, :]).astype(float)
    return (D - D.mean(axis=0))[:, :-1]


def _screen_nuisance(nuisance, units, tol=1e-6) -> np.ndarray:
    """Columns of nuisance (n, q) that vary within the units (n, k) — the site
    indicators and subject terms: their sum of squares after regression on the
    units must exceed tol times the total sum of squares."""
    U, s, _ = np.linalg.svd(units, full_matrices=False)
    Q = U[:, s > s[0] * max(units.shape) * np.finfo(float).eps]
    keep = np.zeros(nuisance.shape[1], dtype=bool)
    for j in range(nuisance.shape[1]):
        z = nuisance[:, j] - nuisance[:, j].mean()
        within = np.sum((z - Q @ (Q.T @ z)) ** 2)
        keep[j] = within > tol * np.sum(z ** 2) and np.ptp(nuisance[:, j]) > 0
    return keep


def _column_bounds(smooth_term_bounds, j):
    """Boundary knots of nuisance column j from smooth_term_bounds, or None."""
    if isinstance(smooth_term_bounds, list):
        return smooth_term_bounds[j]
    if isinstance(smooth_term_bounds, tuple) and smooth_term_bounds != (None, None):
        return smooth_term_bounds
    return None


def _bspline_basis(x, knots, degree):
    """Cubic B-spline basis (n, len(knots) - degree - 1) at x within the knots."""
    from scipy.interpolate import BSpline
    return BSpline.design_matrix(np.asarray(x, float), knots, degree).toarray()


def _nuisance_column_setup(x, model, df, bounds=None, degree=3) -> dict:
    """Basis of one nuisance column for 'linear' and 'pspline'.

    'linear': z-scored column.  'pspline' (Eilers & Marx, 1996): the z-scored
    column (unpenalized) plus the nonlinear part of a cubic B-spline basis with
    df functions on equally spaced knots over [lo, hi] (training range or
    bounds).  The nonlinear part is the basis minus its least-squares fit by
    intercept and slope in the training data, reparametrized so that the
    second-order difference penalty becomes a ridge penalty on its df - 2
    coefficients.  Outside [lo, hi] it is evaluated at the boundary, so the
    fitted effect continues linearly there.  Columns with fewer than df + 2
    distinct values stay linear.
    """
    x = np.asarray(x, float)
    sd = x.std()
    setup = {'kind': 'linear', 'mean': float(x.mean()), 'sd': float(sd) if sd > 0 else 1.0}
    if model == 'linear' or len(np.unique(x)) < df + 2:
        return setup
    lo, hi = (float(x.min()), float(x.max())) if bounds is None else map(float, bounds)
    n_seg = df - degree
    if n_seg < 1 or not hi > lo:
        return setup
    inner = np.linspace(lo, hi, n_seg + 1)          # exact boundary knots
    dx = inner[1] - inner[0]
    knots = np.concatenate([lo - dx * np.arange(degree, 0, -1), inner,
                            hi + dx * np.arange(1, degree + 1)])
    c = np.clip(x, lo, hi)
    B = _bspline_basis(c, knots, degree)
    C = np.linalg.lstsq(np.column_stack([np.ones_like(c), c]), B, rcond=None)[0]
    D = np.diff(np.eye(df), n=2, axis=0)
    ev, V = np.linalg.eigh(D.T @ D)
    pos = ev > ev.max() * 1e-10
    setup.update(kind='pspline', lo=lo, hi=hi, knots=knots, degree=degree, C=C,
                 T=V[:, pos] / np.sqrt(ev[pos]))
    return setup


def _nuisance_design(nuisance, setups):
    """Nuisance basis (n, n_Z) of the columns of nuisance (n, q) with their
    setups, and the penalized columns (n_Z,) bool."""
    parts, pen = [], []
    for j, st in enumerate(setups):
        x = np.asarray(nuisance[:, j], float)
        parts.append(((x - st['mean']) / st['sd'])[:, None])
        pen.append(False)
        if st['kind'] == 'pspline':
            c = np.clip(x, st['lo'], st['hi'])
            N = (_bspline_basis(c, st['knots'], st['degree'])
                 - np.column_stack([np.ones_like(c), c]) @ st['C'])
            parts.append(N @ st['T'])
            pen.extend([True] * st['T'].shape[1])
    return np.hstack(parts), np.array(pen, dtype=bool)


def _solve(X, pen=None):
    """Least-squares operator (k, n) of X (n, k): pinv(X), or with a ridge
    penalty pen (k,) on some coefficients the minimizer of
    ||y - X b||^2 + sum(pen * b^2), via the augmented system."""
    if pen is None or not np.any(pen):
        return pinv(X)
    A = np.vstack([X, np.diag(np.sqrt(pen))])
    return pinv(A)[:, :X.shape[0]]


def _penalty_spectrum(X, pen):
    """Rank of the unpenalized columns, their orthonormal basis Q, and the
    singular values and left vectors of the penalized columns after
    projecting out the unpenalized ones."""
    U = X[:, ~pen]
    Uu, su, _ = np.linalg.svd(U, full_matrices=False)
    r_u = int(np.sum(su > su[0] * max(U.shape) * np.finfo(float).eps)) if su.size else 0
    Q = Uu[:, :r_u]
    P = X[:, pen]
    P = P - Q @ (Q.T @ P)
    W, s, _ = np.linalg.svd(P, full_matrices=False)
    ok = s > (s[0] if s.size else 0) * max(P.shape) * np.finfo(float).eps
    return r_u, Q, W[:, ok], s[ok]


def _penalized_edf(X, pen, lam):
    """Effective df of the penalized columns pen of X for the ridge penalty lam."""
    _, _, _, s = _penalty_spectrum(X, pen)
    return float(np.sum(s ** 2 / (s ** 2 + lam)))


def _select_penalty(Ym, X, pen, method='reml', n_grid=141, chunk=20000):
    """Common ridge penalty of the penalized columns pen of X, estimated from
    all features of Ym (n_valid, n_subjects), each standardized.

    'reml': restricted log-likelihood summed over features, with a variance per
    feature: y = U a + P b + e, b ~ N(0, sigma^2 / lam), e ~ N(0, sigma^2), U
    the unpenalized columns.  'gcv': n * sum(RSS) / (n - edf)^2 pooled over
    features.  The penalty is searched on a log grid of 14 decades around the
    median squared singular value of the penalized columns.

    Returns (lambda, effective df of the penalized columns).
    """
    if method not in ('reml', 'gcv'):
        raise ValueError(f"nuisance_penalty must be 'reml', 'gcv' or a number, not {method!r}.")
    n = X.shape[0]
    r_u, Q, W, s = _penalty_spectrum(X, pen)
    if s.size == 0:
        return np.inf, 0.0
    n_eff = n - r_u
    s2 = s ** 2
    # projections of the standardized features, chunked over features
    rss_perp, C2 = [], []
    for start in range(0, Ym.shape[0], chunk):
        y = Ym[start:start + chunk]
        sd = y.std(axis=1, keepdims=True)
        y = (y - y.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)
        qy, cy = y @ Q, y @ W
        rss_perp.append(np.sum(y ** 2, axis=1) - np.sum(qy ** 2, axis=1) - np.sum(cy ** 2, axis=1))
        C2.append(cy ** 2)
    rss_perp = np.maximum(np.concatenate(rss_perp), 0.0)
    C2 = np.vstack(C2)
    grid = np.median(s2) * 10.0 ** np.linspace(-6, 8, n_grid)
    scores = np.empty(n_grid)
    for i, lam in enumerate(grid):
        f = lam / (s2 + lam)              # share of a component left in the residual
        if method == 'gcv':
            rss = np.sum(rss_perp) + np.sum(C2 @ (f ** 2))
            edf = r_u + np.sum(s2 / (s2 + lam))
            scores[i] = n * rss / max(n - edf, 1e-9) ** 2
        else:
            sig2 = (rss_perp + C2 @ f) / n_eff
            scores[i] = (n_eff * np.sum(np.log(np.maximum(sig2, 1e-300)))
                         + len(sig2) * np.sum(np.log1p(s2 / lam)))
    lam = float(grid[int(np.argmin(scores))])
    return lam, float(np.sum(s2 / (s2 + lam)))


def _standardize_columns(A: np.ndarray, mean=None, sd=None):
    """z-score columns (constant columns become 0); returns (A_std, mean, sd)."""
    if mean is None:
        mean = A.mean(axis=0)
        sd = A.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
    return (A - mean) / sd, mean, sd


def _fit_log_scale(E, W, max_iter=100, tol=1e-8, max_elements=20_000_000):
    """ML fit of  E_ij ~ N(0, exp(W_i theta_j)^2)  for every feature j.

    Fisher scoring for a log-linear variance model (working weights 2, as in
    gamlss for the sigma parameter of a normal distribution), with step
    halving whenever the deviance increases.  W must contain the site
    indicators (or an intercept).

    Parameters
    ----------
    E : (n_features, n_subjects) residuals
    W : (n_subjects, k) design of log(sd)

    Returns
    -------
    theta     : (k, n_features)
    deviance  : (n_features,) -2 log-likelihood without the constant n*log(2*pi)
    converged : (n_features,) bool
    """
    p, n = E.shape
    W_pinv = np.linalg.pinv(W)
    theta = np.empty((W.shape[1], p))
    deviance = np.empty(p)
    converged = np.zeros(p, dtype=bool)

    def dev(e2, eta):
        return np.sum(2 * eta + e2 * np.exp(-2 * eta), axis=0)

    step = max(1, max_elements // n)
    for start in range(0, p, step):
        e2 = E[start:start + step].T ** 2          # (n, pc)
        # start: constant sd per feature
        eta0 = 0.5 * np.log(np.maximum(e2.mean(axis=0), np.finfo(float).tiny))
        th = W_pinv @ np.broadcast_to(eta0, e2.shape)
        d = dev(e2, W @ th)

        active = np.arange(e2.shape[1])
        for _ in range(max_iter):
            e2a, tha = e2[:, active], th[:, active]
            eta = np.clip(W @ tha, -300, 300)
            d_th = W_pinv @ (0.5 * (e2a * np.exp(-2 * eta) - 1))
            th_new = tha + d_th
            d_new = dev(e2a, np.clip(W @ th_new, -300, 300))
            for _ in range(20):                    # step halving
                worse = d_new > d[active]
                if not np.any(worse):
                    break
                d_th[:, worse] *= 0.5
                th_new[:, worse] = tha[:, worse] + d_th[:, worse]
                d_new[worse] = dev(e2a[:, worse],
                                   np.clip(W @ th_new[:, worse], -300, 300))
            th[:, active] = th_new
            d[active] = d_new
            done = np.max(np.abs(d_th), axis=0) < tol
            converged[start + active[done]] = True
            active = active[~done]
            if active.size == 0:
                break

        theta[:, start:start + step] = th
        deviance[start:start + step] = d

    return theta, deviance, converged


def _pearson_dispersion(E, W, theta, df, max_elements=20_000_000):
    """Pearson estimate sum((u^2 - 1)^2) / df of Var(u^2) for the standardized
    residuals u = E / exp(W theta) of every feature (2 for normal residuals).

    E (n_features, n_subjects), W (n_subjects, k), theta (k, n_features);
    features are processed in chunks as in _fit_log_scale.
    """
    p, n = E.shape
    phi = np.empty(p)
    step = max(1, max_elements // n)
    for start in range(0, p, step):
        cols = slice(start, start + step)
        u2 = E[cols].T ** 2 * np.exp(-2 * np.clip(W @ theta[:, cols], -300, 300))
        phi[cols] = np.sum((u2 - 1) ** 2, axis=0) / df
    return phi


def _fit_nuisance_scale(resid, batchmod, nuisance_raw, preserve_raw):
    """Nuisance-dependent variance of the residuals (nuisance_scale option).

    Model per feature:  log sd_ij = site_i + z_i theta_z + x_i theta_x
    with the raw (not spline-expanded) nuisance z and preserve x columns,
    z-scored and entered linearly.  Site and preserve terms are in the model
    so that variance differences they explain are not attributed to the
    nuisance covariates.  The reduced model without z gives a
    likelihood-ratio test of nuisance effects on the variance.

    The deviance difference assumes Var(u^2) = 2 for the standardized
    residuals u, which holds only for normal residuals; in general it is
    Var(u^2) / 2 times a chi-square variable (kurtosis - 1 = Var(u^2)).  It is
    therefore divided by the Pearson estimate of Var(u^2) / 2 under H0, from
    the residuals standardized by the reduced model, as in the studentized
    Breusch-Pagan test (Koenker, 1981), and compared with an F distribution
    (quasi-likelihood F test).  For normal residuals the statistic is about
    the deviance difference.

    Returns a dict with theta_z (n_z, n_valid), the z-scoring parameters,
    the per-feature studentized LR statistic 'scale_lr', its p-value 'scale_p'
    (F distribution of scale_lr / lr_df with lr_df and resid_df degrees of
    freedom), the residual kurtosis 1 + Var(u^2) 'scale_kurtosis' and 'converged'.
    """
    from scipy.stats import f as f_dist

    Zs, z_mean, z_sd = _standardize_columns(nuisance_raw)
    Xs = _standardize_columns(preserve_raw)[0]
    W1 = np.hstack([batchmod, Zs, Xs])
    W0 = np.hstack([batchmod, Xs])

    theta1, dev1, conv1 = _fit_log_scale(resid, W1)
    theta0, dev0, conv0 = _fit_log_scale(resid, W0)
    converged = conv1 & conv0
    if not np.all(converged):
        warnings.warn(
            f"nuisance_scale: log-variance model did not converge for "
            f"{int(np.sum(~converged))} feature(s).", RuntimeWarning, stacklevel=3,
        )

    rank0 = int(np.linalg.matrix_rank(W0))
    lr_df = int(np.linalg.matrix_rank(W1)) - rank0
    resid_df = max(resid.shape[1] - rank0, 1)
    phi = _pearson_dispersion(resid, W0, theta0, resid_df)
    lr = np.maximum(dev0 - dev1, 0.0) * 2 / np.maximum(phi, 1e-12)
    n_b = batchmod.shape[1]
    return {
        'theta_z':   theta1[n_b:n_b + Zs.shape[1]],
        'z_mean':    z_mean,
        'z_sd':      z_sd,
        'scale_lr':  lr,
        'scale_p':   (f_dist.sf(lr / lr_df, lr_df, resid_df) if lr_df > 0
                      else np.ones_like(lr)),
        'scale_kurtosis': 1 + phi,
        'lr_df':     lr_df,
        'resid_df':  resid_df,
        'converged': converged,
    }


def _nuisance_scale_factor(nuisance_raw, scale_model):
    """sd ratio exp(z_i theta_z) of each subject relative to the mean nuisance level.

    Returns (n_valid, n_subjects); dividing residuals by it removes the
    nuisance-dependent part of the variance.  z is z-scored with the training
    mean, so the geometric mean of the factor over training subjects is 1.
    """
    Zs = _standardize_columns(nuisance_raw, scale_model['z_mean'], scale_model['z_sd'])[0]
    return np.exp(np.clip(Zs @ scale_model['theta_z'], -300, 300)).T


# ComBat's stopping rule (sva::it.sol conv = 1e-4; ComBatFamily: 10e-5 == 1e-4);
# the iteration limit only guards against non-termination
_EB_CONV = 1e-4
_EB_MAX_ITER = 1000


def _eb_site_effects(gamma_masked, site_var, n_batches, ref_level=None):
    """Empirical Bayes estimates of the site effects (eb option).

    The site rows of gamma_masked are the site coefficients at the zero of
    the nuisance basis minus the grand mean, so they share a per-feature
    offset that belongs to the nuisance and grand-mean model.  They are
    therefore shrunk as differences to the size-weighted mean of the sites
    (or to the reference site), which are the site effects of ComBat, and
    the offset is added back unchanged.  The reference site and sites with
    fewer than two subjects keep their L/S estimates.

    Parameters
    ----------
    gamma_masked : (n_batch + n_Z, n_valid) L/S estimates [site | nuisance]
    site_var     : (n_batch, n_valid) site variances delta_hat
    n_batches    : (n_batch,) subjects per site
    ref_level    : index of the reference site, or None

    Returns gamma_star (gamma_masked with the site rows replaced),
    delta_star (n_batch, n_valid) and the priors (see _eb_estimates).
    """
    n_batch = len(n_batches)
    gamma_site = gamma_masked[:n_batch]
    if ref_level is None:
        center = (n_batches / n_batches.sum()) @ gamma_site
    else:
        center = gamma_site[ref_level]
    skip = [i for i in range(n_batch) if i == ref_level or n_batches[i] < 2]
    g_star, delta_star, priors = _eb_estimates(gamma_site - center, site_var,
                                               n_batches, skip)
    gamma_star = gamma_masked.copy()
    gamma_star[:n_batch] = g_star + center
    return gamma_star, delta_star, priors


def _eb_estimates(gamma_hat, delta_hat, n_batches, skip=()):
    """Parametric empirical Bayes of Johnson et al. (2007), per site.

    Priors gamma_ig ~ N(gamma_bar_i, t2_i) and delta_ig ~ InvGamma(a_i, b_i)
    are estimated by the method of moments from all features, and the
    posterior means are found with the iteration and stopping rule of
    sva::it.sol and ComBatFamily::comfam (as in combat_family._eb_estimates).
    Only sufficient statistics are needed: the sum of squares of the
    standardized data of site i around gamma is
    (n_i - 1) delta_hat + n_i (gamma_hat - gamma)^2.

    gamma_hat, delta_hat : (n_batch, n_valid);  n_batches : (n_batch,)
    skip : sites that keep gamma_hat and delta_hat

    Returns gamma_star, delta_star and a dict of per-site hyperparameters
    'gamma_bar', 't2', 'a', 'b' (NaN for skipped sites).
    """
    gamma_star, delta_star = gamma_hat.copy(), delta_hat.copy()
    priors = {k: np.full(len(n_batches), np.nan) for k in ('gamma_bar', 't2', 'a', 'b')}

    for i, n_b in enumerate(n_batches):
        if i in skip:
            continue
        g_hat, d_hat = gamma_hat[i], delta_hat[i]

        # method of moments estimates of the priors
        g_bar, t2 = np.mean(g_hat), np.var(g_hat, ddof=1)
        d_bar, d_var = np.mean(d_hat), np.var(d_hat, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            a = (2 * d_var + d_bar ** 2) / d_var
            b = (d_bar * d_var + d_bar ** 3) / d_var
        ss = (n_b - 1) * d_hat

        g_old, d_old = g_hat, d_hat
        change_old = 1.0
        with np.errstate(divide='ignore', invalid='ignore'):
            for count in range(_EB_MAX_ITER):
                g_new = (n_b * t2 * g_hat + d_old * g_bar) / (n_b * t2 + d_old)
                d_new = ((ss + n_b * (g_hat - g_new) ** 2) / 2 + b) / (n_b / 2 + a - 1)

                # R: max(abs(g_new - g_old)/g_old, abs(d_new - d_old)/d_old)
                rel = np.concatenate([np.abs(g_new - g_old) / g_old,
                                      np.abs(d_new - d_old) / d_old])
                change = np.nanmax(rel) if np.any(~np.isnan(rel)) else 0.0
                if count > 30 and change > change_old:
                    warnings.warn(
                        "Empirical Bayes step failed to converge after 30 "
                        "iterations, using estimate before change between "
                        "iterations increases.", RuntimeWarning, stacklevel=4,
                    )
                    break
                g_old, d_old = g_new, d_new
                change_old = change
                if not change > _EB_CONV:
                    break

        # degenerate priors (e.g. identical delta_hat in all features): keep L/S
        ok = np.isfinite(g_new) & np.isfinite(d_new)
        gamma_star[i] = np.where(ok, g_new, g_hat)
        delta_star[i] = np.where(ok, d_new, d_hat)
        for key, val in zip(('gamma_bar', 't2', 'a', 'b'), (g_bar, t2, a, b)):
            priors[key][i] = val

    return gamma_star, delta_star, priors


# ---------------------------------------------------------------------------
# Apply pre-trained estimates to new data
# ---------------------------------------------------------------------------

def comcat_from_training(
    Y: np.ndarray,
    batch: np.ndarray,
    nuisance: np.ndarray | None = None,
    preserve: np.ndarray | None = None,
    estimates: dict = None,
    verbose: bool = False,
    unseen_batch: str = 'error',
) -> np.ndarray:
    """Apply pre-fitted ComCAT estimates to new data.

    Parameters
    ----------
    Y         : (n_features, n_subjects_new)  — new data matrix
    batch     : (n_subjects_new,) — site labels; must be a subset of the
                labels seen during training (estimates['batch_levels'])
    nuisance  : (n_subjects_new, n_nuisance_orig) — same variables as in training
    preserve  : (n_subjects_new, n_preserve) — same raw variables as in training
                (spline expansion from preserve_df is reapplied with the training knots)
    estimates : dict returned by comcat(..., return_estimates=True)
    verbose   : print progress
    unseen_batch : 'error' (default) raises for site labels not seen in training;
                'mean' gives them the size-weighted mean site level and the pooled
                variance (no site correction, nuisance effects are removed), e.g.
                for patients of a site without controls when fitted on controls

    The optional extensions used in training (preserve_df, nuisance_scale,
    residual_delta) are applied with the training estimates, and so are the
    empirical Bayes site effects when the training used eb.

    Returns
    -------
    Y_harmonized : (n_features, n_subjects_new)
    """
    if estimates is None:
        raise ValueError("estimates dict is required. "
                         "Obtain it via comcat(..., return_estimates=True).")

    Y = np.array(Y, dtype=np.float64)
    batch = np.asarray(batch).ravel()
    n_subjects = len(batch)

    # Unpack estimates; the site effects applied in training are the empirical
    # Bayes estimates when present (older dicts only have the L/S estimates)
    grand_mean        = estimates['grand_mean']
    std_pooled        = estimates['std_pooled']
    gamma_masked      = estimates.get('gamma_star_masked')
    if gamma_masked is None:
        gamma_masked = estimates['gamma_hat_masked']
    delta_masked      = estimates.get('delta_star_masked')
    if delta_masked is None:
        delta_masked = estimates['delta_hat_masked']
    beta_hat_preserve = estimates['beta_hat_preserve']
    ind_mask          = estimates['ind_mask']
    ind_nan           = estimates['ind_nan']
    batch_levels      = estimates['batch_levels']
    n_batch           = estimates['n_batch']
    n_nuisance_orig   = estimates['n_nuisance_orig']
    n_X               = estimates['n_X']
    ref_level         = estimates['ref_level']

    # Map new batch labels to training indices (-1: not seen in training)
    lookup = {b: i for i, b in enumerate(batch_levels.tolist())}
    batch_idx = np.array([lookup.get(b, -1) for b in batch.tolist()], dtype=int)
    if np.any(batch_idx < 0) and unseen_batch != 'mean':
        missing = set(np.asarray(batch)[batch_idx < 0].tolist())
        raise ValueError(
            f"Batch labels {missing} were not seen during training. "
            f"Known labels: {batch_levels.tolist()}"
        )

    nuisance = _to_col_matrix(nuisance, n_subjects)
    preserve = _to_col_matrix(preserve, n_subjects)
    nuisance_raw = nuisance

    # Preserve spline expansion (preserve_df) with the training knots
    preserve_splines = estimates.get('preserve_splines')
    if preserve_splines is not None:
        preserve, _ = _build_preserve_basis(preserve, None,
                                            preserve_splines=preserve_splines)
    # subject terms of the training: new subjects have none
    n_subject_cols = estimates.get('n_subject_cols', 0)
    if n_subject_cols:
        preserve = np.hstack([preserve, np.zeros((n_subjects, n_subject_cols))])

    # Nuisance basis expansion — same configuration as training
    smooth_term_bounds_ft = estimates.get('smooth_term_bounds')
    gam_df_ft             = estimates.get('gam_df')
    spline_constructors   = estimates.get('spline_constructors', {})
    if n_nuisance_orig > 0 and estimates.get('nuisance_model', 'spline') == 'spline':
        nuisance, _ = _build_nuisance_basis(
            nuisance, smooth_term_bounds_ft,
            gam_df_ft, verbose, spline_constructors=spline_constructors
        )
    elif n_nuisance_orig > 0:
        keep = estimates['nuisance_keep']
        nuisance = (_nuisance_design(nuisance[:, keep], estimates['nuisance_setup'])[0]
                    if len(keep) else np.empty((n_subjects, 0)))
    n_Z = nuisance.shape[1]

    # Transpose Y if needed
    transp = False
    if Y.shape[1] != n_subjects:
        if Y.shape[0] == n_subjects:
            Y = Y.T
            transp = True
        else:
            raise ValueError(f"Shape mismatch: Y {Y.shape}, n_subjects={n_subjects}")

    n_features = Y.shape[0]
    Ym = Y[ind_mask, :]   # (n_valid, n_subjects)

    # Preserve contribution from training betas
    if n_X > 0 and beta_hat_preserve is not None:
        preserve_contrib = (preserve @ beta_hat_preserve).T   # (n_valid, n_subjects)
    else:
        preserve_contrib = 0.0

    # Standardize using training parameters
    Ym_std = (Ym - grand_mean[:, None] - preserve_contrib) / std_pooled[:, None]

    # Build nuisance design for new subjects (batch one-hot + nuisance); unseen
    # sites get the size-weighted mean of the site levels
    batchmod_new = (batch_idx[:, None] == np.arange(n_batch)[None, :]).astype(float)
    unseen = batch_idx < 0
    if np.any(unseen):
        nb = estimates.get('n_batches')
        w = np.ones(n_batch) if nb is None else np.asarray(nb, float)
        batchmod_new[unseen] = w / w.sum()
    X_nuisance_new = np.hstack([batchmod_new] + ([nuisance] if n_Z > 0 else []))

    # Apply saved batch effects
    if verbose:
        print("[ComCAT from training] Applying pre-fitted estimates")

    # Nuisance-dependent variance (nuisance_scale) with the training model
    scale_model = estimates.get('nuisance_scale')
    scale_factor = (_nuisance_scale_factor(nuisance_raw, scale_model)
                    if scale_model is not None else None)

    batches_new = [np.where(batch_idx == i)[0] for i in range(n_batch)]
    delta_ref = delta_masked[ref_level, :] if ref_level is not None else None
    Ym_adj = Ym_std.copy()
    for i in range(n_batch):
        idx = batches_new[i]
        if len(idx) == 0:
            continue
        if ref_level is not None and i == ref_level:
            continue
        delta_i = delta_masked[i, :] if delta_ref is None else delta_masked[i, :] / delta_ref
        denom = np.sqrt(delta_i)[:, None] * np.ones((1, len(idx)))
        numer = Ym_std[:, idx] - (X_nuisance_new[idx, :] @ gamma_masked).T
        if scale_factor is not None:
            numer /= scale_factor[:, idx]
        Ym_adj[:, idx] = numer / denom

    if np.any(unseen):                      # location only, pooled variance
        idx = np.flatnonzero(unseen)
        numer = Ym_std[:, idx] - (X_nuisance_new[idx, :] @ gamma_masked).T
        if scale_factor is not None:
            numer /= scale_factor[:, idx]
        Ym_adj[:, idx] = numer

    Ym_adj = np.where(np.isfinite(Ym_adj), Ym_adj, 0.0)

    # Reconstruct
    Y_harmonized = np.zeros((n_features, n_subjects), dtype=np.float64)
    for i in range(n_subjects):
        pc_i = (preserve[i, :] @ beta_hat_preserve
                if (n_X > 0 and beta_hat_preserve is not None) else 0.0)
        Y_harmonized[ind_mask, i] = Ym_adj[:, i] * std_pooled + grand_mean + pc_i
    Y_harmonized[ind_nan, :] = np.nan

    if ref_level is not None:
        ref_idx = np.concatenate([batches_new[ref_level]])
        if len(ref_idx):
            Y_harmonized[:, ref_idx] = Y[:, ref_idx]

    if transp:
        Y_harmonized = Y_harmonized.T

    return Y_harmonized


# ---------------------------------------------------------------------------
# Cross-fitted harmonization
# ---------------------------------------------------------------------------

def comcat_crossfit(
    Y: np.ndarray,
    batch: np.ndarray | None,
    nuisance: np.ndarray | None = None,
    preserve: np.ndarray | None = None,
    folds=5,
    fit=None,
    subject=None,
    seed: int = 0,
    return_folds: bool = False,
    **kwargs,
):
    """ComCAT with every subject harmonized by a model that was not fitted on it.

    The samples on which models are estimated (fit, default all) are split into
    folds; the samples of every fold are harmonized with comcat_from_training()
    using a model fitted on the other folds.  All other samples (e.g. patients
    when fit marks the controls) are harmonized with the model fitted on all fit
    samples, so a disease effect cannot enter the estimated nuisance effects.
    Samples of a site that has no fit sample in the other folds are also
    harmonized with that model (with a warning).

    Parameters
    ----------
    Y, batch, nuisance, preserve : as for comcat(); batch None = one site
    folds    : int k (k folds over subjects, or over samples without subject
               labels, stratified by site), or an array (n_subjects,) of fold labels
    fit      : bool array (n_subjects,) of the samples used for estimation
    subject  : subject labels; all scans of a subject fall into one fold, and
               they are passed on to comcat() (effects within subjects)
    seed     : random seed of the folds
    kwargs   : further options of comcat()

    Returns Y_harmonized (orientation as Y), and with return_folds the folds.
    """
    Y = np.array(Y, dtype=np.float64)
    n = None
    for arr in (batch, subject):
        if arr is not None and np.asarray(arr).size:
            n = np.asarray(arr).size
            break
    else:
        for arr in (nuisance, preserve):
            if arr is not None and np.asarray(arr).size:
                n = int(max(np.asarray(arr).shape))
                break
    if n is None:
        raise ValueError("comcat_crossfit needs batch, nuisance, preserve or subject.")
    transp = Y.shape[1] != n
    if transp:
        Y = Y.T
    bvec = np.ones(n, dtype=int) if batch is None else np.asarray(batch).ravel()
    Z = None if nuisance is None else _to_col_matrix(nuisance, n)
    P = None if preserve is None else _to_col_matrix(preserve, n)
    S = None if subject is None else np.asarray(subject).ravel()
    fit = np.ones(n, dtype=bool) if fit is None else np.asarray(fit, dtype=bool).ravel()
    if np.isscalar(folds):
        folds = _make_folds(bvec, S, fit, int(folds), seed)
    folds = np.asarray(folds).ravel()

    # unpenalized B-splines: knots over all samples, so held-out values lie inside
    if (Z is not None and kwargs.get('nuisance_model', 'pspline') == 'spline'
            and kwargs.get('smooth_term_bounds') is None):
        kwargs['smooth_term_bounds'] = [(float(Z[:, j].min()), float(Z[:, j].max()))
                                        for j in range(Z.shape[1])]
    # preserve splines likewise (comcat_from_training cannot extrapolate them)
    if (P is not None and kwargs.get('preserve_df') is not None
            and kwargs.get('preserve_bounds') is None):
        kwargs['preserve_bounds'] = [(float(P[:, j].min()), float(P[:, j].max()))
                                     for j in range(P.shape[1])]
    kwargs['return_estimates'] = True

    def rows(a, m):
        return None if a is None else a[m]

    def fit_on(m):
        return comcat(Y[:, m], None if batch is None else bvec[m], rows(Z, m), rows(P, m),
                      subject=rows(S, m), **kwargs)[4]

    def apply(est, m):
        return comcat_from_training(Y[:, m], bvec[m], rows(Z, m), rows(P, m), estimates=est,
                                    unseen_batch='mean')

    H = np.full(Y.shape, np.nan)
    rest = ~fit
    for k in np.unique(folds[fit]):
        te = fit & (folds == k)
        est = fit_on(fit & (folds != k))
        known = np.isin(bvec, est['batch_levels'])
        if np.any(te & known):
            H[:, te & known] = apply(est, te & known)
        rest |= te & ~known
    if np.any(rest):
        n_unseen = int(np.sum(rest & fit))
        if n_unseen:
            warnings.warn(f"{n_unseen} sample(s) of sites without fit samples in the other "
                          "folds were harmonized with the model of all fit samples.",
                          RuntimeWarning, stacklevel=2)
        H[:, rest] = apply(fit_on(fit), rest)
    H = H.T if transp else H
    return (H, folds) if return_folds else H


def _make_folds(batch, subject, fit, k, seed=0):
    """Fold labels (n,) over subjects (or samples), stratified by site.

    Units are subjects (all their scans in one fold) or single samples.  Within
    every site, its fit units are shuffled and dealt to the folds in turn, the
    start rotating between sites so that small sites do not all fall into the
    first folds.  A subject scanned at several sites is counted at its most
    frequent one.  Samples outside fit get the label -1.
    """
    rng = np.random.default_rng(seed)
    n = len(batch)
    units = np.arange(n) if subject is None else subject
    levels, code = np.unique(units, return_inverse=True)
    unit_fit = np.zeros(len(levels), dtype=bool)
    np.logical_or.at(unit_fit, code, fit)
    unit_site = np.empty(len(levels), dtype=object)
    for u in range(len(levels)):
        b, c = np.unique(batch[code == u], return_counts=True)
        unit_site[u] = b[np.argmax(c)]
    fold_u = np.full(len(levels), -1)
    start = 0
    for s in np.unique(unit_site[unit_fit].astype(str)):
        members = np.flatnonzero(unit_fit & (unit_site.astype(str) == s))
        rng.shuffle(members)
        fold_u[members] = (start + np.arange(len(members))) % k
        start = (start + len(members)) % k
    folds = fold_u[code]
    folds[~fit] = -1
    return folds

"""
ComCAT: Combating CovariATe effects — harmonization of multi-site data.

ComCAT is ComBat (Johnson et al., 2007) with continuous nuisance covariates.  It removes
site effects in location and scale with ComBat's empirical Bayes estimates, removes the
effects of continuous nuisance covariates (e.g. image quality measures) in location, and
keeps the effects of the preserved covariates (e.g. age, sex, diagnosis).  Without
nuisance covariates it is ComBat.

    from comcat import comcat, comcat_from_training, comcat_crossfit

    Y_harmonized, beta_hat, gamma_hat, delta_hat = comcat(Y, batch, nuisance, preserve)

The model (fixed, no options)
-----------------------------
Sites      location and scale, parametric empirical Bayes as in ComBat; the site
           variances are estimated from the residuals after the site and nuisance effects.
           With one site (batch=None) nothing is scaled.  A site with a single subject
           gets no site correction.
Nuisance   location only.  Every column is a penalized cubic B-spline (P-spline, Eilers &
           Marx, 1996) with 10 basis functions: a linear term plus nonlinear terms shrunk
           by a second-order difference penalty.  One smoothing parameter for all columns
           is estimated by REML from all features; large penalties give linear effects.
           Columns that do not vary within the sites are dropped with a warning.
Preserved  linear terms, estimated jointly with the site and nuisance effects and left in
           the data.
Output     in the units of the input; if the input has no negative values (e.g. tissue
           densities), negative harmonized values are set to 0.

Parameters of comcat()
----------------------
Y           : (n_features, n_subjects) data
batch       : (n_subjects,) site labels, or None for one site
nuisance    : (n_subjects, n_nuisance) covariates to remove, or None
preserve    : (n_subjects, n_preserve) covariates to keep, or None
mean_only   : True corrects the sites in location only (e.g. with fewer than about ten
              subjects per site, where ComBat's site variances shrink the residuals)
ref_batch   : label of a reference site: its data stay as they are, the other sites are
              harmonized to its level and variance
nuisance_model : 'pspline' (default) or 'linear' (linear terms only)
preserve_df : None (linear) | int | 'same' (10): B-spline expansion of continuous
              preserved covariates, for wide age ranges (needs statsmodels);
              preserve_bounds gives its boundary knots, (lo, hi) or a list per column
subject     : (n_subjects,) subject labels of repeated scans: adds subject terms, so that
              the site and nuisance effects are estimated within subjects (location
              only).  For traveling-subject or test-retest data with few subjects.
clip        : False returns the affine harmonization also for non-negative data
return_estimates : also return the fitted parameters (dict) for comcat_from_training()
verbose     : print progress

Returns Y_harmonized (as Y), beta_hat (coefficients of the full design), gamma_hat and
delta_hat (the additive and multiplicative effects that were removed; site rows first),
and with return_estimates the dict of estimates.

comcat_from_training() applies the estimates to new scans, and comcat_crossfit()
harmonizes every subject with a model that was not fitted on it.  Use the latter whenever
harmonized data are scored by a model trained elsewhere (BrainAGE, normative models) or
when harmonization methods are compared: in-sample harmonization is optimistic.

Options of earlier versions (unpenalized B-splines with gam_df, eb=False, the
df-consistent site variances with min_scale_n, single_site_scale, residual_delta,
nuisance_scale, nuisance_penalty, smooth_term_bounds) are in evaluation/comcat_legacy.py,
which is kept for the evaluations of the ComCat paper only.  Their names are still
accepted here when they ask for what this version does (e.g. eb=True,
combat_compatible=True, min_scale_n=2) and raise an error otherwise.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.linalg import pinv

_NUISANCE_MODELS = ('pspline', 'linear')
_GAM_DF = 10    # basis functions of a penalized spline; the penalty sets the smoothness

# options of earlier versions with the values that describe this version
_REMOVED = {
    'eb': (True,),
    'combat_compatible': (True,),
    'residual_delta': (True,),
    'min_scale_n': (2,),
    'single_site_scale': (False,),
    'nuisance_scale': (False,),
    'nuisance_penalty': (None, 'reml'),
    'gam_df': (None, _GAM_DF),
    'smooth_term_bounds': (None,),
}
_LEGACY = "evaluation/comcat_legacy.py has the earlier version."


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
    nuisance_model: str = 'pspline',
    preserve_df: int | str | None = None,
    preserve_bounds=None,
    subject=None,
    clip: bool = True,
    **removed,
):
    """ComCAT harmonization of sites and nuisance covariates.

    See the module docstring for the model and all parameters.
    """

    # ------------------------------------------------------------------ setup
    # Use float64 throughout for numerical precision (MATLAB promotes to double too)
    Y = np.array(Y, dtype=np.float64)
    _check_removed(removed)
    _check_nuisance_model(nuisance_model)

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
            if return_estimates:
                raise ValueError("Nothing to estimate: no batch, nuisance, preserve or "
                                 "subject.")
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
    _check_finite(nuisance=nuisance, preserve=preserve)

    n_Z = nuisance.shape[1]

    # subject terms: effects are estimated within subjects (location only)
    subject_cols = np.empty((n_subjects, 0))
    if subject is not None:
        subject = np.asarray(subject).ravel()
        if len(subject) != n_subjects:
            raise ValueError(f"subject has {len(subject)} labels for {n_subjects} subjects.")
        subject_cols = _subject_design(subject)
        if verbose:
            print(f"[ComCAT] {subject_cols.shape[1] + 1} subjects: effects estimated "
                  "within subjects, mean only")
        mean_only = True
        n_groups = _site_groups(batch, subject)
        if n_groups > 1:
            warnings.warn(
                f"With subject terms, the site effects are estimated within subjects, but "
                f"the sites form {n_groups} groups without a subject scanned in two of "
                "them (e.g. every subject scanned at one site): the differences between "
                "these groups are not identified and stay in the data.  Use "
                "comcat_crossfit(..., subject_terms=False) or no subject terms.",
                UserWarning, stacklevel=2)

    # optional B-spline expansion of continuous preserve covariates
    if preserve_df == 'same':
        preserve_df = _GAM_DF
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
    nuisance_keep = np.arange(n_Z)   # columns used
    nuisance_setup = None            # basis of the columns (for from_training)
    pen_nuisance = np.zeros(0, dtype=bool)   # penalized columns of the nuisance basis
    if n_Z > 0:
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
        nuisance_setup = [_nuisance_column_setup(nuisance[:, j], nuisance_model, _GAM_DF)
                          for j in nuisance_keep]
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
    if n_batch == 1 and not mean_only:
        # one site: no site variance to correct; a variance step would only
        # rescale all residuals by about 1 - R^2 of the nuisance fit
        if verbose:
            print("[ComCAT] One site: location correction only")
        mean_only = True

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

    # penalty of the nonlinear nuisance terms: one smoothing parameter for all
    # nuisance columns, estimated by REML from all features
    pen = None
    nuisance_lambda = None
    nuisance_edf = float(n_Z) if n_Z > 0 else None
    if pen_nuisance.any():
        pen_cols = np.zeros(XZ.shape[1], dtype=bool)
        pen_cols[n_batch:n_batch + n_Z] = pen_nuisance
        nuisance_lambda, edf_pen = _select_penalty(Ym, XZ, pen_cols)
        nuisance_edf = edf_pen + int(np.sum(~pen_nuisance))
        pen = np.where(pen_cols, nuisance_lambda, 0.0)
        if verbose:
            print(f"[ComCAT] Nuisance penalty {nuisance_lambda:.3g}: {nuisance_edf:.1f} "
                  f"effective df of {n_Z} nuisance terms")

    beta_hat = _solve(XZ, pen) @ Ym.T   # (n_cols, n_valid)

    fitted_site = XZ[:, :n_batch + n_Z] @ beta_hat[:n_batch + n_Z, :]   # site + nuisance fit
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
    del residuals

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

    # residuals after the additive site and nuisance effects: the site variances are
    # estimated from them, as in ComBat (not from data that still hold the nuisance
    # effects, which would shrink the harmonized residuals)
    resid_std = Ym - (X_nuisance @ gamma_hat_masked).T   # (n_valid, n_subjects)

    # site variances of ComBat; the nuisance rows are not applied
    delta_hat_masked = np.ones((n_batch + n_Z, Ym.shape[0]), dtype=np.float64)
    if not mean_only:
        for i, idx in enumerate(batches):
            if len(idx) > 1:    # the variance of a single subject is undefined
                delta_hat_masked[i, :] = np.var(resid_std[:, idx], axis=1, ddof=1)
        delta_hat_masked[n_batch:, :] = np.var(Ym, axis=1, ddof=1)

    # --------------------------------------------------- empirical Bayes
    use_eb = n_batch > 1 and Ym.shape[0] > 1
    if use_eb:
        if verbose:
            print("[ComCAT] Empirical Bayes estimation of site effects")
        # mean_only still needs the site variances for the posterior of gamma
        site_var = (np.vstack([np.var(resid_std[:, idx], axis=1, ddof=1)
                               if len(idx) > 1 else np.ones(Ym.shape[0])
                               for idx in batches])
                    if mean_only else delta_hat_masked[:n_batch])
        gamma_star_masked, delta_star_sites, eb_priors = _eb_site_effects(
            gamma_hat_masked, site_var, n_batches, ref_level
        )
        delta_star_masked = delta_hat_masked.copy()
        if not mean_only:
            delta_star_masked[:n_batch] = delta_star_sites
    else:
        gamma_star_masked, delta_star_masked, eb_priors = gamma_hat_masked, delta_hat_masked, None

    # A site with a single subject: its site effect cannot be separated from that
    # subject's deviation, and removing it would replace the subject by its predicted
    # values and shift every other scan of the site by its anatomy.  Such sites are
    # corrected like unseen sites instead: the size-weighted mean level of the other
    # sites and the pooled variance.
    single = [i for i in range(n_batch) if n_batches[i] < 2 and i != ref_level]
    if single and n_batch > 1:
        others = [i for i in range(n_batch) if i not in single] or list(range(n_batch))
        w = np.asarray(n_batches, dtype=np.float64)[others]
        level = (w / w.sum()) @ gamma_star_masked[others]
        gamma_star_masked = gamma_star_masked.copy()
        delta_star_masked = delta_star_masked.copy()
        gamma_star_masked[single] = level
        delta_star_masked[single] = 1.0
        warnings.warn(f"{len(single)} site(s) with a single subject get no site correction "
                      "(only the nuisance effects are removed).", UserWarning, stacklevel=2)

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
        numer = resid_std[:, idx]
        if gamma_star_masked is not gamma_hat_masked:
            # resid_std removed the L/S site effect; remove the EB one (or that of a
            # site with a single subject) instead
            numer = numer + (gamma_hat_masked[i] - gamma_star_masked[i])[:, None]
        Ym[:, idx] = numer / np.sqrt(delta_i)[:, None]
    del resid_std

    Ym = np.where(np.isfinite(Ym), Ym, 0.0)

    # --------------------------------------------------- reconstruct
    # features without variance keep their values
    Y_harmonized = np.array(Y, dtype=np.float64)
    for i in range(n_subjects):
        pc_i = (XZ[i, ind_preserve] @ beta_hat[ind_preserve, :]) if n_X > 0 else 0.0
        Y_harmonized[ind_mask, i] = Ym[:, i] * std_pooled + grand_mean + pc_i
    Y_harmonized[ind_nan, :] = np.nan

    # restore reference batch to original values
    if ref_level is not None:
        ref_idx = batches[ref_level]
        Y_harmonized[:, ref_idx] = Y[:, ref_idx]
    nonnegative = _nonnegative(Y)
    if clip and nonnegative:
        np.maximum(Y_harmonized, 0, out=Y_harmonized)

    # --------------------------------------------------- full-space outputs
    # effects that were removed (empirical Bayes estimates with several sites)
    gamma_hat = np.zeros((gamma_star_masked.shape[0], n_features), dtype=np.float64)
    gamma_hat[:, ind_mask] = gamma_star_masked

    delta_hat = np.zeros((delta_star_masked.shape[0], n_features), dtype=np.float64)
    delta_hat[:, ind_mask] = delta_star_masked

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
        'nonnegative':         nonnegative,     # input without negative values: clip
        'batch_levels':        levels_original,
        'n_batch':             n_batch,
        'n_batches':           n_batches,
        'n_nuisance_orig':     n_nuisance_orig,
        'n_Z':                 n_Z,
        'n_X':                 n_X,
        'mean_only':           mean_only,
        'ref_level':           ref_level,
        'preserve_splines':    preserve_splines,
        # applied site effects (None if they are the L/S estimates: one site, and no
        # sites of a single subject) and the empirical Bayes priors
        'eb':                  use_eb,
        'gamma_star_masked':   (gamma_star_masked if gamma_star_masked is not gamma_hat_masked
                                else None),
        'delta_star_masked':   (delta_star_masked if delta_star_masked is not delta_hat_masked
                                else None),
        'eb_priors':           eb_priors,
        # nuisance model and subject terms
        'nuisance_model':      nuisance_model,
        'nuisance_keep':       nuisance_keep,
        'nuisance_setup':      nuisance_setup,
        'nuisance_lambda':     nuisance_lambda,
        'nuisance_edf':        nuisance_edf,
        'n_subject_cols':      subject_cols.shape[1],
    }
    return Y_harmonized, beta_hat_full, gamma_hat, delta_hat, estimates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _same(a, b) -> bool:
    try:
        return a is b or bool(a == b)
    except Exception:
        return False


def _check_removed(options):
    """Options of earlier versions: accepted when they ask for the behaviour of this
    version, an error otherwise."""
    for name, value in options.items():
        if name not in _REMOVED:
            raise TypeError(f"comcat() got an unexpected keyword argument {name!r}")
        if not any(_same(value, v) for v in _REMOVED[name]):
            now = ' or '.join(repr(v) for v in _REMOVED[name])
            raise ValueError(
                f"The option {name}={value!r} was removed (this version behaves as "
                f"{name}={now}). {_LEGACY}")


def _check_nuisance_model(nuisance_model):
    if nuisance_model == 'spline':
        raise ValueError("The unpenalized B-splines (nuisance_model='spline') were removed: "
                         f"they overfit and do not generalize to new scans. {_LEGACY}")
    if nuisance_model not in _NUISANCE_MODELS:
        raise ValueError(f"nuisance_model must be one of {_NUISANCE_MODELS}, "
                         f"not {nuisance_model!r}.")


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

    from statsmodels.gam.api import BSplines     # needed for preserve_df only

    df = int(preserve_df)
    parts, cols, constructors = [], [], {}
    for c in range(preserve.shape[1]):
        col = preserve[:, c:c + 1].astype(float)
        n_unique = len(np.unique(col))
        if n_unique < df + 2:
            if verbose:
                print(f"[ComCAT] Preserve column {c}: {n_unique} distinct values, kept linear")
            parts.append(col)
            continue
        bounds = preserve_bounds[c] if isinstance(preserve_bounds, list) else preserve_bounds
        lo, hi = (None, None) if bounds is None else bounds
        bs = BSplines(col, df=df, degree=3, knot_kwds=[{'lower_bound': lo, 'upper_bound': hi}])
        parts.append(bs.basis)
        cols.append(c)
        constructors[c] = bs

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


def _site_groups(site, subject) -> int:
    """Number of groups of sites that are linked by subjects scanned at several of
    them; with subject terms, only the differences within a group are identified."""
    sites = np.unique(site)
    parent = {s: s for s in sites.tolist()}

    def root(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    for u in np.unique(subject):
        at = np.unique(np.asarray(site)[np.asarray(subject) == u]).tolist()
        for s in at[1:]:
            parent[root(s)] = root(at[0])
    return len({root(s) for s in sites.tolist()})


def _nonnegative(Y):
    """True if Y has no negative values (e.g. tissue densities): their harmonized
    values are then set to 0 where negative (clip)."""
    y = Y[np.isfinite(Y)]
    return bool(y.size) and bool(y.min() >= 0)


def _check_finite(**arrays):
    """Raise for missing or non-finite values of the covariate arrays."""
    for name, a in arrays.items():
        a = np.asarray(a, dtype=np.float64)
        if a.size and not np.all(np.isfinite(a)):
            bad = np.flatnonzero(~np.all(np.isfinite(a.reshape(len(a), -1)), axis=1))
            raise ValueError(
                f"{name} has missing or non-finite values for {bad.size} subject(s) "
                f"(indices {bad[:10].tolist()}{' ...' if bad.size > 10 else ''}): remove "
                "these subjects or impute the values.")


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


def _bspline_basis(x, knots, degree):
    """Cubic B-spline basis (n, len(knots) - degree - 1) at x within the knots."""
    from scipy.interpolate import BSpline
    return BSpline.design_matrix(np.asarray(x, float), knots, degree).toarray()


def _nuisance_column_setup(x, model, df, degree=3) -> dict:
    """Basis of one nuisance column.

    'linear': z-scored column.  'pspline' (Eilers & Marx, 1996): the z-scored
    column (unpenalized) plus the nonlinear part of a cubic B-spline basis with
    df functions on equally spaced knots over the training range [lo, hi].  The
    nonlinear part is the basis minus its least-squares fit by intercept and
    slope in the training data, reparametrized so that the second-order
    difference penalty becomes a ridge penalty on its df - 2 coefficients.
    Outside [lo, hi] it is evaluated at the boundary, so the fitted effect
    continues linearly there.  Columns with fewer than df + 2 distinct values
    stay linear.
    """
    x = np.asarray(x, float)
    sd = x.std()
    setup = {'kind': 'linear', 'mean': float(x.mean()), 'sd': float(sd) if sd > 0 else 1.0}
    if model == 'linear' or len(np.unique(x)) < df + 2:
        return setup
    lo, hi = float(x.min()), float(x.max())
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


def _select_penalty(Ym, X, pen, n_grid=141, chunk=20000):
    """Common ridge penalty of the penalized columns pen of X, estimated by REML
    from all features of Ym (n_valid, n_subjects), each standardized.

    Restricted log-likelihood summed over features, with a variance per
    feature: y = U a + P b + e, b ~ N(0, sigma^2 / lam), e ~ N(0, sigma^2), U
    the unpenalized columns.  The penalty is searched on a log grid of 14
    decades around the median squared singular value of the penalized columns.

    Returns (lambda, effective df of the penalized columns).
    """
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
        sig2 = (rss_perp + C2 @ f) / n_eff
        scores[i] = (n_eff * np.sum(np.log(np.maximum(sig2, 1e-300)))
                     + len(sig2) * np.sum(np.log1p(s2 / lam)))
    lam = float(grid[int(np.argmin(scores))])
    return lam, float(np.sum(s2 / (s2 + lam)))


# ComBat's stopping rule (sva::it.sol conv = 1e-4; ComBatFamily: 10e-5 == 1e-4);
# the iteration limit only guards against non-termination
_EB_CONV = 1e-4
_EB_MAX_ITER = 1000


def _eb_site_effects(gamma_masked, site_var, n_batches, ref_level=None):
    """Empirical Bayes estimates of the site effects.

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
    g_star, delta_star, priors = _eb_estimates(gamma_site - center, site_var, n_batches, skip)
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
    clip: bool = True,
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
    clip      : negative values are set to 0 if the training data had none (stored in
                the estimates), so that new scans are treated alike, one at a time or
                together; signed training data are never clipped

    Nuisance values outside the training range are extrapolated linearly.  Subject
    terms of the training are not applied: new scans are new subjects.

    Returns
    -------
    Y_harmonized : (n_features, n_subjects_new)
    """
    if estimates is None:
        raise ValueError("estimates dict is required. "
                         "Obtain it via comcat(..., return_estimates=True).")
    if (estimates.get('nuisance_scale') is not None
            or (estimates['n_nuisance_orig'] > 0
                and estimates.get('nuisance_model', 'spline') not in _NUISANCE_MODELS)):
        raise ValueError("These estimates come from an option that was removed "
                         f"(unpenalized B-splines or nuisance_scale). {_LEGACY}")

    Y = np.array(Y, dtype=np.float64)
    batch = np.asarray(batch).ravel()
    n_subjects = len(batch)

    # Unpack estimates; the site effects applied in training are the empirical
    # Bayes estimates when present
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
    _check_finite(nuisance=nuisance, preserve=preserve)

    # Preserve spline expansion (preserve_df) with the training knots
    preserve_splines = estimates.get('preserve_splines')
    if preserve_splines is not None:
        preserve, _ = _build_preserve_basis(preserve, None,
                                            preserve_splines=preserve_splines)
    # subject terms of the training: new subjects have none
    n_subject_cols = estimates.get('n_subject_cols', 0)
    if n_subject_cols:
        preserve = np.hstack([preserve, np.zeros((n_subjects, n_subject_cols))])

    # Nuisance basis with the training setup
    if n_nuisance_orig > 0:
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
        numer = Ym_std[:, idx] - (X_nuisance_new[idx, :] @ gamma_masked).T
        Ym_adj[:, idx] = numer / np.sqrt(delta_i)[:, None]

    if np.any(unseen):                      # location only, pooled variance
        idx = np.flatnonzero(unseen)
        numer = Ym_std[:, idx] - (X_nuisance_new[idx, :] @ gamma_masked).T
        # with a reference site, the known sites are scaled to its variance
        Ym_adj[:, idx] = numer if delta_ref is None else numer * np.sqrt(delta_ref)[:, None]

    Ym_adj = np.where(np.isfinite(Ym_adj), Ym_adj, 0.0)

    # Reconstruct; features without variance in the training keep their values, and
    # missing values stay missing
    Y_harmonized = np.array(Y, dtype=np.float64)
    for i in range(n_subjects):
        pc_i = (preserve[i, :] @ beta_hat_preserve
                if (n_X > 0 and beta_hat_preserve is not None) else 0.0)
        Y_harmonized[ind_mask, i] = Ym_adj[:, i] * std_pooled + grand_mean + pc_i
    Y_harmonized[ind_nan, :] = np.nan
    Y_harmonized[~np.isfinite(Y)] = np.nan

    if ref_level is not None:
        ref_idx = np.concatenate([batches_new[ref_level]])
        if len(ref_idx):
            Y_harmonized[:, ref_idx] = Y[:, ref_idx]
    nonnegative = estimates.get('nonnegative')
    if nonnegative is None:                     # estimates without the entry
        nonnegative = _nonnegative(Y)
    if clip and nonnegative:
        np.maximum(Y_harmonized, 0, out=Y_harmonized)

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
    subject_terms: bool = False,
    **kwargs,
):
    """ComCAT with every subject harmonized by a model that was not fitted on it.

    The samples on which models are estimated (fit, default all) are split into
    folds; the samples of every fold are harmonized with comcat_from_training()
    using a model fitted on the other folds.  All other samples (e.g. patients
    when fit marks the controls) are harmonized with the model fitted on all fit
    samples, so a disease effect cannot enter the estimated nuisance effects.
    Held-out samples of a site without fit samples in the other folds get no site
    correction (unseen site; with a warning).  With ref_batch, the samples of the
    reference site are in every fit and keep their values.  Features with missing
    values are missing in the output, as in comcat().

    Parameters
    ----------
    Y, batch, nuisance, preserve : as for comcat(); batch None = one site
    folds    : int k (k folds over subjects, or over samples without subject
               labels, stratified by site), or an array (n_subjects,) of fold labels
    fit      : bool array (n_subjects,) of the samples used for estimation.  The
               default (all samples, with the diagnosis among the preserved
               covariates) corrects the sites of all groups alike; fit=controls is
               the stricter alternative when every site has enough controls.
    subject  : subject labels of repeated scans: all scans of a subject fall into
               one fold.  Without them, int folds split the scans of a subject
               between the folds, so that its other scans enter the model that
               harmonizes it.
    seed     : random seed of the folds
    subject_terms : with subject labels, also add subject terms in comcat(): the
               effects are then estimated within subjects (location only).  The
               default False estimates them across subjects, as for single scans.
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
    if not _nonnegative(Y):         # clipping is decided on all data, not on each fold
        kwargs['clip'] = False
    bvec = np.ones(n, dtype=int) if batch is None else np.asarray(batch).ravel()
    Z = None if nuisance is None else _to_col_matrix(nuisance, n)
    P = None if preserve is None else _to_col_matrix(preserve, n)
    S = None if subject is None else np.asarray(subject).ravel()
    fit = np.ones(n, dtype=bool) if fit is None else np.asarray(fit, dtype=bool).ravel()
    # the reference site keeps its data: its samples are in every fit
    ref = (bvec == kwargs['ref_batch']) if kwargs.get('ref_batch') is not None else None
    if np.isscalar(folds):
        folds = _make_folds(bvec, S, fit if ref is None else fit & ~ref, int(folds), seed)
    folds = np.asarray(folds).ravel()
    missing = ~np.all(np.isfinite(Y), axis=1)          # features with missing values

    # preserve splines: knots over all samples (comcat_from_training cannot
    # extrapolate them)
    if (P is not None and kwargs.get('preserve_df') is not None
            and kwargs.get('preserve_bounds') is None):
        kwargs['preserve_bounds'] = [(float(P[:, j].min()), float(P[:, j].max()))
                                     for j in range(P.shape[1])]
    kwargs['return_estimates'] = True

    def rows(a, m):
        return None if a is None else a[m]

    def fit_on(m):
        return comcat(Y[:, m], None if batch is None else bvec[m], rows(Z, m), rows(P, m),
                      subject=rows(S, m) if subject_terms else None, **kwargs)[4]

    def apply(est, m):
        return comcat_from_training(Y[:, m], bvec[m], rows(Z, m), rows(P, m), estimates=est,
                                    unseen_batch='mean', clip=kwargs.get('clip', True))

    H = np.full(Y.shape, np.nan)
    n_unseen = 0
    for k in np.unique(folds[fit & (folds >= 0)]):
        te = fit & (folds == k)
        est = fit_on(fit & (folds != k))
        n_unseen += int(np.sum(te & ~np.isin(bvec, est['batch_levels'])))
        H[:, te] = apply(est, te)
    if n_unseen:
        warnings.warn(f"{n_unseen} held-out sample(s) of sites without fit samples in the "
                      "other folds get no site correction.", UserWarning, stacklevel=2)
    rest = ~fit | (folds < 0)                       # incl. the reference site (kept)
    if np.any(rest):
        H[:, rest] = apply(fit_on(fit), rest)
    H[missing] = np.nan
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

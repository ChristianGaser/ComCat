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
gamma_hat    : ndarray  — additive batch/nuisance effects (full feature space)
delta_hat    : ndarray  — multiplicative batch effects (full feature space)

Additional parameters
---------------------
ref_batch            : label of a site to use as reference (its data is left untouched;
                       all other sites are harmonized relative to it). Default None.
return_estimates     : if True, a 5th element (dict) with all fitted parameters is
                       returned; pass it to comcat_from_training() for new data.
Every nuisance column is always modelled with a B-spline GAM (requires
statsmodels). The GAM is configured by:

smooth_term_bounds   : boundary knots for the B-spline of each nuisance column.
                       • None  — infer bounds from training data (safe for training only)
                       • (lo, hi) — same bounds for every nuisance column
                       • [(lo0,hi0), (lo1,hi1), ...] — one pair per nuisance column
                       For apply-to-new-data workflows, always specify explicit bounds
                       that cover the full range of training AND test data.
gam_df               : int, B-spline basis dimension per nuisance column (default None).
                       Higher values capture finer nonlinearities but risk overfitting.

Optional model extensions (all off by default; defaults reproduce earlier results)
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
                       likelihood-ratio test of the nuisance variance effects.
residual_delta       : bool (default False)
                       Estimate site variances delta from the residuals after the
                       additive site AND nuisance effects are removed, as ComBat does.
                       The default (False, as in the MATLAB implementation) estimates
                       delta before the nuisance effects are removed; delta then also
                       contains nuisance-explained variance, which shrinks the residual
                       variance of the harmonized data by roughly
                       var(residual) / (var(residual) + var(nuisance effect)).

GAM smoothness recommendations
-------------------------------
The B-spline basis uses cubic splines (degree=3) with `gam_df` columns
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

import warnings

import numpy as np
from numpy.linalg import pinv
from statsmodels.gam.api import BSplines


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
):
    """ComCAT harmonization for sites and nuisance parameters.

    See the module docstring for all parameters, including the optional
    extensions preserve_df, preserve_bounds, nuisance_scale and residual_delta.
    """

    # ------------------------------------------------------------------ setup
    # Use float64 throughout for numerical precision (MATLAB promotes to double too)
    Y = np.array(Y, dtype=np.float64)

    # Handle None/empty batch — determine n_subjects from nuisance/preserve if needed
    batch_empty = (batch is None or np.asarray(batch).size == 0)

    if not batch_empty:
        batch = np.asarray(batch).ravel()
        n_subjects = len(batch)
    else:
        n_subjects = None
        for arr in (nuisance, preserve):
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

    # Resolve gam_df from sample size when not set explicitly:
    #   min(10, max(5, n_subjects // 30))
    if gam_df is None:
        gam_df = min(10, max(5, n_subjects // 30))
        if verbose:
            print(f"[ComCAT] gam_df auto-selected: {gam_df} (n={n_subjects})")

    # optional B-spline expansion of continuous preserve covariates
    preserve_orig = preserve.copy()                   # linear columns, used by the scale model
    if preserve_df == 'same':
        preserve_df = gam_df
    preserve, preserve_splines = _build_preserve_basis(
        preserve, preserve_df, preserve_bounds, verbose
    )
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
    if n_Z > 0:
        if verbose:
            print(f"[ComCAT] GAM (B-spline, df={gam_df}) for all {n_Z} nuisance column(s)")
        nuisance, spline_constructors = _build_nuisance_basis(
            nuisance, smooth_term_bounds, gam_df, verbose
        )
        n_Z = nuisance.shape[1]
    else:
        spline_constructors = {}

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

    # confounding check — warn but continue (pinv handles rank deficiency)
    if np.linalg.matrix_rank(XZ) < XZ.shape[1]:
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
            import warnings
            warnings.warn(
                "Design matrix is rank-deficient: nuisance covariate(s) are "
                f"strongly confounded with batch — {details}. "
                "Proceeding with pseudoinverse; confounded columns will have "
                "reduced or no independent effect.",
                RuntimeWarning, stacklevel=3,
            )
        else:
            import warnings
            warnings.warn(
                "Design matrix is rank-deficient (covariates confounded with "
                "batch). Proceeding with pseudoinverse.",
                RuntimeWarning, stacklevel=3,
            )

    # --------------------------------------------------- standardize
    if verbose:
        print("[ComCAT] Standardizing data across features")

    beta_hat = pinv(XZ) @ Ym.T   # (n_cols, n_valid)

    XZ_no_preserve = XZ[:, list(range(n_batch)) + list(range(n_batch, n_batch + n_Z))]
    if ref_level is not None:
        # grand mean = intercept of the reference batch
        grand_mean = beta_hat[ref_level, :].copy()
    else:
        grand_mean = np.mean(XZ_no_preserve @ beta_hat[:n_batch + n_Z, :], axis=0)

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
    gamma_hat_masked = pinv(X_nuisance) @ Ym.T   # (n_batch+n_Z, n_valid)

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

    # --------------------------------------------------- adjust data
    if verbose:
        print("[ComCAT] Adjusting the data")
        if ref_level is not None:
            print(f"[ComCAT] Reference batch: index {ref_level} "
                  f"({levels_original[ref_level]!r}) — left unchanged")

    for i in range(n_batch):
        if ref_level is not None and i == ref_level:
            continue  # reference batch is not adjusted
        idx = batches[i]
        denom = np.sqrt(delta_hat_masked[i, :])[:, None] * np.ones((1, n_batches[i]))
        if resid_std is not None:
            numer = resid_std[:, idx]
        else:
            numer = Ym[:, idx] - (X_nuisance[idx, :] @ gamma_hat_masked).T
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
    n_gamma = gamma_hat_masked.shape[0]
    gamma_hat = np.zeros((n_gamma, n_features), dtype=np.float64)
    for i in range(n_gamma):
        gamma_hat[i, ind_mask] = gamma_hat_masked[i, :]

    n_delta = delta_hat_masked.shape[0]
    delta_hat = np.zeros((n_delta, n_features), dtype=np.float64)
    for i in range(n_delta):
        delta_hat[i, ind_mask] = delta_hat_masked[i, :]

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
    }
    if scale_model is not None:
        # per-feature test of nuisance effects on the variance, full feature space
        for key in ('scale_lr', 'scale_p'):
            full = np.full(n_features, np.nan)
            full[ind_mask] = scale_model[key]
            estimates[key] = full
        estimates['scale_lr_df'] = scale_model['lr_df']
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


def _fit_nuisance_scale(resid, batchmod, nuisance_raw, preserve_raw):
    """Nuisance-dependent variance of the residuals (nuisance_scale option).

    Model per feature:  log sd_ij = site_i + z_i theta_z + x_i theta_x
    with the raw (not spline-expanded) nuisance z and preserve x columns,
    z-scored and entered linearly.  Site and preserve terms are in the model
    so that variance differences they explain are not attributed to the
    nuisance covariates.  The reduced model without z gives a
    likelihood-ratio test of nuisance effects on the variance.

    Returns a dict with theta_z (n_z, n_valid), the z-scoring parameters,
    the per-feature LR statistic 'scale_lr', its p-value 'scale_p'
    (chi-square with 'lr_df' degrees of freedom) and 'converged'.
    """
    from scipy.stats import chi2

    Zs, z_mean, z_sd = _standardize_columns(nuisance_raw)
    Xs = _standardize_columns(preserve_raw)[0]
    W1 = np.hstack([batchmod, Zs, Xs])
    W0 = np.hstack([batchmod, Xs])

    theta1, dev1, conv1 = _fit_log_scale(resid, W1)
    _, dev0, conv0 = _fit_log_scale(resid, W0)
    converged = conv1 & conv0
    if not np.all(converged):
        warnings.warn(
            f"nuisance_scale: log-variance model did not converge for "
            f"{int(np.sum(~converged))} feature(s).", RuntimeWarning, stacklevel=3,
        )

    lr_df = int(np.linalg.matrix_rank(W1) - np.linalg.matrix_rank(W0))
    lr = np.maximum(dev0 - dev1, 0.0)
    n_b = batchmod.shape[1]
    return {
        'theta_z':   theta1[n_b:n_b + Zs.shape[1]],
        'z_mean':    z_mean,
        'z_sd':      z_sd,
        'scale_lr':  lr,
        'scale_p':   chi2.sf(lr, lr_df) if lr_df > 0 else np.ones_like(lr),
        'lr_df':     lr_df,
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

    The optional extensions used in training (preserve_df, nuisance_scale,
    residual_delta) are applied with the training estimates.

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

    # Unpack estimates
    grand_mean        = estimates['grand_mean']
    std_pooled        = estimates['std_pooled']
    gamma_hat_masked  = estimates['gamma_hat_masked']
    delta_hat_masked  = estimates['delta_hat_masked']
    beta_hat_preserve = estimates['beta_hat_preserve']
    ind_mask          = estimates['ind_mask']
    ind_nan           = estimates['ind_nan']
    batch_levels      = estimates['batch_levels']
    n_batch           = estimates['n_batch']
    n_nuisance_orig   = estimates['n_nuisance_orig']
    n_X               = estimates['n_X']
    ref_level         = estimates['ref_level']

    # Map new batch labels to training indices
    try:
        batch_idx = np.array(
            [int(np.where(batch_levels == b)[0][0]) for b in batch]
        )
    except IndexError as exc:
        missing = set(batch) - set(batch_levels)
        raise ValueError(
            f"Batch labels {missing} were not seen during training. "
            f"Known labels: {batch_levels.tolist()}"
        ) from exc

    nuisance = _to_col_matrix(nuisance, n_subjects)
    preserve = _to_col_matrix(preserve, n_subjects)
    nuisance_raw = nuisance

    # Preserve spline expansion (preserve_df) with the training knots
    preserve_splines = estimates.get('preserve_splines')
    if preserve_splines is not None:
        preserve, _ = _build_preserve_basis(preserve, None,
                                            preserve_splines=preserve_splines)

    # Nuisance basis expansion — same configuration as training
    smooth_term_bounds_ft = estimates.get('smooth_term_bounds')
    gam_df_ft             = estimates.get('gam_df')
    spline_constructors   = estimates.get('spline_constructors', {})
    if n_nuisance_orig > 0:
        nuisance, _ = _build_nuisance_basis(
            nuisance, smooth_term_bounds_ft,
            gam_df_ft, verbose, spline_constructors=spline_constructors
        )
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

    # Build nuisance design for new subjects (batch one-hot + nuisance)
    batchmod_new = (batch_idx[:, None] == np.arange(n_batch)[None, :]).astype(float)
    X_nuisance_new = np.hstack([batchmod_new] + ([nuisance] if n_Z > 0 else []))

    # Apply saved batch effects
    if verbose:
        print("[ComCAT from training] Applying pre-fitted estimates")

    # Nuisance-dependent variance (nuisance_scale) with the training model
    scale_model = estimates.get('nuisance_scale')
    scale_factor = (_nuisance_scale_factor(nuisance_raw, scale_model)
                    if scale_model is not None else None)

    batches_new = [np.where(batch_idx == i)[0] for i in range(n_batch)]
    Ym_adj = Ym_std.copy()
    for i in range(n_batch):
        idx = batches_new[i]
        if len(idx) == 0:
            continue
        if ref_level is not None and i == ref_level:
            continue
        denom = np.sqrt(delta_hat_masked[i, :])[:, None] * np.ones((1, len(idx)))
        numer = Ym_std[:, idx] - (X_nuisance_new[idx, :] @ gamma_hat_masked).T
        if scale_factor is not None:
            numer /= scale_factor[:, idx]
        Ym_adj[:, idx] = numer / denom

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

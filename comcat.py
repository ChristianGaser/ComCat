"""
ComCAT: Combating CovariATe effects — core harmonization function.

MIT License
Copyright (c) 2020 Jean-Philippe Fortin
Heavily modified and extended by Christian Gaser, Robert Dahnke
Python port: 2026

Usage
-----
from comcat import comcat

Y_harmonized, beta_hat, gamma_hat, delta_hat = comcat(
    Y, batch, nuisance, preserve,
    mean_only=False, poly_degree=2, verbose=True
)

Parameters
----------
Y           : ndarray, shape (n_features, n_subjects)
batch       : array-like, shape (n_subjects,)  — site/scanner labels
nuisance    : ndarray or None, shape (n_subjects, n_nuisance)  — variables to remove
preserve    : ndarray or None, shape (n_subjects, n_preserve)  — variables to keep
mean_only   : bool   — if True, only adjust mean (no variance scaling)
poly_degree : int    — polynomial expansion degree for nuisance (default 2)
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
smooth_terms         : which nuisance columns to model with B-spline GAM.
                       • 'all' (default) — apply GAM to every nuisance column
                       • list of 0-based indices, e.g. [0, 2] — GAM for those columns,
                         polynomial for the rest
                       • None — polynomial expansion for all columns (no GAM)
                       Requires statsmodels when set to 'all' or a non-empty list.
                       Falls back to polynomial silently if statsmodels is missing.
smooth_term_bounds   : boundary knots for each smooth term.
                       • None  — infer bounds from training data (safe for training only)
                       • (lo, hi) — same bounds for all smooth terms
                       • [(lo0,hi0), (lo1,hi1), ...] — one pair per entry in smooth_terms
                       For apply-to-new-data workflows, always specify explicit bounds
                       that cover the full range of training AND test data.
gam_df               : int, B-spline basis dimension per smooth term (default 6).
                       Higher values capture finer nonlinearities but risk overfitting.

GAM smoothness recommendations
-------------------------------
The B-spline basis uses cubic splines (degree=3) with `gam_df` columns
(= n_internal_knots + degree + 1 with intercept in statsmodels convention).

Typical `gam_df` choices by covariate type:

| Covariate             | Recommended gam_df | Notes                              |
|-----------------------|--------------------|------------------------------------|
| Age (20–90 yr)        | 6 – 8              | Gentle non-linear growth curves    |
| TIV / ICV (cm³)       | 5 – 7              | Moderate curvature expected        |
| Continuous score      | 5 – 6              | Unless strong curvature suspected  |
| Cortical thickness    | 6 – 8              | Similar to age                     |
| General rule          | max(5, n // 30)    | Cap at 15 for any sample size      |

Practical guidelines:
- `gam_df=None` (default) uses the sample-size heuristic min(15, max(5, n//30)):
    n=80  → 5,  n=200 → 6,  n=300 → 10,  n=500 → 15 (capped).
  Pass an explicit integer to override.
- Values above 15 rarely help and inflate the design matrix (slows pinv).
- For small samples (n < 100): keep `gam_df ≤ 6` to avoid near-rank-deficiency.
- Always set `smooth_term_bounds` explicitly in train/test workflows so the
  knot positions are identical between training and new data.
- Combining `smooth_terms` with `poly_degree > 1` for other nuisance columns
  is supported (hybrid: some columns B-spline, others polynomial).
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import pinv


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def comcat(
    Y: np.ndarray,
    batch: np.ndarray,
    nuisance: np.ndarray | None = None,
    preserve: np.ndarray | None = None,
    mean_only: bool = False,
    poly_degree: int = 2,
    verbose: bool = False,
    ref_batch=None,
    return_estimates: bool = False,
    smooth_terms: list[int] | str | None = 'all',
    smooth_term_bounds=None,
    gam_df: int | None = None,
):
    """ComCAT harmonization for sites and nuisance parameters."""

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
    n_X = preserve.shape[1]

   # Resolve gam_df from sample size when not set explicitly:
    #   min(15, max(5, n_subjects // 30))
    if gam_df is None:
        gam_df = min(15, max(5, n_subjects // 30))
        if verbose:
            print(f"[ComCAT] gam_df auto-selected: {gam_df} (n={n_subjects})")

    # Resolve 'all' sentinel: apply GAM to every nuisance column
    if smooth_terms == 'all':
        smooth_terms = list(range(n_Z)) if n_Z > 0 else None

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
    #avg = np.mean(Y, axis=1)
    #ind_mask = (sd0 > np.max(avg)/100) & np.isfinite(sd0)
    ind_nan = np.isnan(sd0)

    Ym = Y[ind_mask, :]          # (n_valid, n_subjects)

    # ------------------------------------------------ nuisance basis expansion
    n_nuisance_orig = n_Z          # columns before expansion (needed for from_training)
    nuisance_orig = nuisance.copy()  # keep original columns for confounding diagnostics
    if n_Z > 0:
        if verbose:
            has_gam = smooth_terms and len(smooth_terms) > 0
            poly_cols = [i for i in range(n_Z) if not (smooth_terms and i in smooth_terms)]
            if has_gam:
                print(f"[ComCAT] GAM (B-spline, df={gam_df}) for nuisance col(s): {smooth_terms}")
            if poly_cols and poly_degree > 1:
                print(f"[ComCAT] Polynomial extension (degree {poly_degree}) for nuisance col(s): {poly_cols}")
        nuisance, spline_constructors = _build_nuisance_basis(
            nuisance, poly_degree, smooth_terms, smooth_term_bounds, gam_df, verbose
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

    delta_hat_masked = np.zeros((n_batch + n_Z, Ym.shape[0]), dtype=np.float64)
    for i in range(n_batch):
        idx = batches[i]
        if mean_only:
            delta_hat_masked[i, :] = 1.0
        else:
            delta_hat_masked[i, :] = np.var(Ym[:, idx], axis=1, ddof=1)

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
        numer = Ym[:, idx] - (X_nuisance[idx, :] @ gamma_hat_masked).T
        Ym[:, idx] = numer / denom

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
        'poly_degree':         poly_degree,
        'mean_only':           mean_only,
        'ref_level':           ref_level,
        # GAM parameters (None if not used)
        'smooth_terms':        smooth_terms,
        'smooth_term_bounds':  smooth_term_bounds,
        'gam_df':              gam_df,
        'spline_constructors': spline_constructors,
    }
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


def _polynomial(x: np.ndarray, p: int) -> np.ndarray:
    """Polynomial expansion and Gram-Schmidt orthogonalisation of x up to degree p.

    Returns columns [x, x^2, ..., x^p] orthogonalised against all lower-degree
    columns (mirrors SPM's spm_detrend / the MATLAB polynomial() helper).
    """
    x = np.array(x, dtype=np.float64).ravel()
    x = x - np.mean(x)   # detrend (mean removal)

    cols = [x]
    V = np.zeros((len(x), p + 1))

    for j in range(p + 1):
        xj = x ** j
        # orthogonalise against previously accumulated V
        if j > 0:
            xj = xj - V @ (pinv(V) @ xj)
        V[:, j] = xj
        if j >= 2:
            cols.append(xj)

    return np.column_stack(cols) if len(cols) > 1 else cols[0][:, None]


def _build_nuisance_basis(
    nuisance: np.ndarray,
    poly_degree: int,
    smooth_terms: list[int] | None,
    smooth_term_bounds,
    gam_df: int,
    verbose: bool = False,
    spline_constructors: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Expand each nuisance column into a basis for the design matrix.

    For columns in `smooth_terms`: B-spline basis via statsmodels BSplines.
    For all other columns:          polynomial expansion (degree `poly_degree`).

    Parameters
    ----------
    nuisance            : (n_subjects, n_cols) raw nuisance array
    poly_degree         : degree for polynomial columns
    smooth_terms        : list of 0-based column indices to model with B-splines
    smooth_term_bounds  : None | (lo, hi) | [(lo0,hi0), ...]
    gam_df              : B-spline degrees of freedom per smooth term
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
    smooth_set = set(smooth_terms) if smooth_terms else set()
    new_constructors: dict = {}

    if smooth_set:
        try:
            from statsmodels.gam.api import BSplines
        except ImportError as exc:
            raise ImportError(
                "statsmodels is required for GAM smoothing. "
                "Install with:  pip install statsmodels"
            ) from exc

    parts = []
    for i in range(n_cols):
        col = nuisance[:, i:i + 1].astype(float)  # keep 2-D

        if i in smooth_set:
            if spline_constructors and i in spline_constructors:
                # apply training knots to new data
                bs = spline_constructors[i]
                basis = bs.transform(col)
            else:
                # fit new B-spline basis
                if isinstance(smooth_term_bounds, list):
                    idx_in_list = sorted(smooth_set).index(i)
                    lo, hi = smooth_term_bounds[idx_in_list]
                elif isinstance(smooth_term_bounds, tuple) and smooth_term_bounds != (None, None):
                    lo, hi = smooth_term_bounds
                else:
                    lo, hi = None, None
                knot_kwds = [{'lower_bound': lo, 'upper_bound': hi}]
                bs = BSplines(col, df=gam_df, degree=3, knot_kwds=knot_kwds)
                new_constructors[i] = bs
                basis = bs.basis
            parts.append(basis)
        else:
            if poly_degree > 1:
                parts.append(_polynomial(nuisance[:, i], poly_degree))
            else:
                parts.append(col)

    if not parts:
        return np.empty((nuisance.shape[0], 0), dtype=np.float64), new_constructors

    return np.hstack(parts), new_constructors


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
    preserve  : (n_subjects_new, n_X) — same variables as in training
    estimates : dict returned by comcat(..., return_estimates=True)
    verbose   : print progress

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
    poly_degree       = estimates['poly_degree']
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

    # Nuisance basis expansion — same configuration as training
    smooth_terms_ft      = estimates.get('smooth_terms')
    smooth_term_bounds_ft = estimates.get('smooth_term_bounds')
    gam_df_ft            = estimates.get('gam_df', poly_degree)
    spline_constructors  = estimates.get('spline_constructors', {})
    if n_nuisance_orig > 0:
        nuisance, _ = _build_nuisance_basis(
            nuisance, poly_degree, smooth_terms_ft, smooth_term_bounds_ft,
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

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
ref_batch       : label of a site to use as reference (its data is left untouched;
                  all other sites are harmonized relative to it). Default None.
return_estimates: if True, a 5th element (dict) with all fitted parameters is
                  returned; pass it to comcat_from_training() for new data.
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

    Ym = Y[ind_mask, :]          # (n_valid, n_subjects)

    # --------------------------------------------------------- polynomial ext
    n_nuisance_orig = n_Z          # columns before expansion (needed for from_training)
    if n_Z > 0 and poly_degree > 1:
        if verbose:
            print(f"[ComCAT] Polynomial extension of nuisance with degree {poly_degree}")
        parts = [_polynomial(nuisance[:, i], poly_degree) for i in range(n_Z)]
        nuisance = np.hstack(parts)
        n_Z = nuisance.shape[1]

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

    # confounding check
    if np.linalg.matrix_rank(XZ) < XZ.shape[1]:
        raise ValueError(
            "Design matrix is rank-deficient (covariates confounded with batch). "
            "Please remove confounded covariates and rerun ComCAT."
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

    # remove additive nuisance before estimating scales
    Ym_for_delta = Ym.copy()
    if n_Z > 0:
        Ym_for_delta = Ym_for_delta - (nuisance @ gamma_hat_masked[n_batch:, :]).T

    delta_hat_masked = np.zeros((n_batch + n_Z, Ym.shape[0]), dtype=np.float64)
    for i in range(n_batch):
        idx = batches[i]
        if mean_only:
            delta_hat_masked[i, :] = 1.0
        else:
            delta_hat_masked[i, :] = np.var(Ym_for_delta[:, idx], axis=1, ddof=1)

    for i in range(n_batch, n_batch + n_Z):
        if mean_only:
            delta_hat_masked[i, :] = 1.0
        else:
            delta_hat_masked[i, :] = np.var(Ym_for_delta, axis=1, ddof=1)

    del Ym_for_delta

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
        'grand_mean':       grand_mean,
        'std_pooled':       std_pooled,
        'gamma_hat_masked': gamma_hat_masked,
        'delta_hat_masked': delta_hat_masked,
        'beta_hat_preserve': beta_hat[ind_preserve, :].copy() if n_X > 0 else None,
        'ind_mask':         ind_mask,
        'ind_nan':          ind_nan,
        'batch_levels':     levels_original,
        'n_batch':          n_batch,
        'n_nuisance_orig':  n_nuisance_orig,
        'n_Z':              n_Z,
        'n_X':              n_X,
        'poly_degree':      poly_degree,
        'mean_only':        mean_only,
        'ref_level':        ref_level,
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

    # Polynomial expansion of nuisance — same as training
    if n_nuisance_orig > 0 and poly_degree > 1:
        parts = [_polynomial(nuisance[:, i], poly_degree)
                 for i in range(n_nuisance_orig)]
        nuisance = np.hstack(parts)
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

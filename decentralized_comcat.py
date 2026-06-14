"""
Decentralized ComCAT (Mode A: shared unique covariate values) — GAM B-spline path.

Reproduces a centralized `comcat()` run across sites that never pool their raw
imaging data `Y`.  Only aggregate statistics leave a site:
  - feature first/second moments (for the common mask)
  - the *unique values* of each smooth covariate column (for B-spline knots)
  - design-matrix Gram blocks  X_iᵀX_i  and cross terms  X_iᵀY_i
  - per-site residual sums and per-batch variances

Design notes
------------
* comcat.py is NOT modified.  Equivalence is achieved purely by constructing
  `statsmodels` BSplines objects with *injected* knots and handing them to the
  existing `spline_constructors` hook that `comcat_from_training` already honours.
* The B-spline basis is reproduced **bitwise** (knot vector is reconstructed
  exactly from the pooled unique covariate values; basis evaluation is
  deterministic).  The regression/standardization steps match centralized output
  to machine precision (~1e-13), because decentralized regression uses the
  normal-equations form  pinv(ΣXᵢᵀXᵢ)·ΣXᵢᵀYᵢ  (identity: pinv(X)=pinv(XᵀX)Xᵀ).
* Assumes the federated topology of Bostami et al.: **each site holds exactly one
  batch** (so per-batch variance is computed locally).  Polynomial nuisance is
  intentionally unsupported — ComCAT uses GAM B-splines.

See tests/DECENTRALIZED-GAM-DESIGN.md for the full design.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import pinv
from statsmodels.gam.api import BSplines

from comcat import comcat_from_training, _to_col_matrix


# ---------------------------------------------------------------------------
# B-spline basis: spec built globally, constructor rebuilt identically anywhere
# ---------------------------------------------------------------------------

def _gam_df(n_total: int) -> int:
    """Mirror comcat's sample-size heuristic for gam_df."""
    return min(10, max(5, n_total // 30))


def bspline_spec_from_pooled_unique(pooled_unique: np.ndarray, df: int,
                                    degree: int = 3) -> dict:
    """Reconstruct the exact knot spec a centralized BSplines fit would produce.

    statsmodels computes inner knots from ``unique(x within [min,max])`` via its
    R-compatible quantile rule.  Passing the pooled unique values (bounds = their
    min/max) yields a knot vector bitwise-identical to a fit on the pooled raw
    column, because the constructor itself takes ``unique`` internally.
    """
    col = np.asarray(pooled_unique, dtype=np.float64).reshape(-1, 1)
    bs = BSplines(col, df=df, degree=degree,
                  knot_kwds=[{'lower_bound': None, 'upper_bound': None}])
    order = degree + 1
    knots = bs.smoothers[0].knots
    return {
        'df': df,
        'degree': degree,
        'lower_bound': float(knots[0]),
        'upper_bound': float(knots[-1]),
        'inner_knots': np.asarray(knots[order:-order], dtype=np.float64),
    }


def build_constructor(spec: dict) -> BSplines:
    """Rebuild a BSplines with injected knots (identical anywhere given the spec)."""
    # x is irrelevant once knots are injected (.transform recomputes from knots);
    # a 2-point dummy spanning the bounds is enough to construct the object.
    dummy = np.array([[spec['lower_bound']], [spec['upper_bound']]], dtype=np.float64)
    return BSplines(
        dummy, df=spec['df'], degree=spec['degree'],
        knot_kwds=[{'knots': spec['inner_knots'],
                    'lower_bound': spec['lower_bound'],
                    'upper_bound': spec['upper_bound']}],
    )


def expand_nuisance(nuisance_site: np.ndarray, constructors: dict) -> np.ndarray:
    """Locally expand every (smooth) nuisance column via its shared constructor."""
    nz = _to_col_matrix(nuisance_site, len(nuisance_site) if nuisance_site is not None else 0)
    if nz.shape[1] == 0:
        return np.empty((nz.shape[0], 0), dtype=np.float64)
    parts = [constructors[i].transform(nz[:, i:i + 1]) for i in range(nz.shape[1])]
    return np.hstack(parts)


# ---------------------------------------------------------------------------
# Per-site design block: [ batch one-hot | expanded nuisance | preserve ]
# ---------------------------------------------------------------------------

def _design(batch_idx, nuis_exp, preserve_site, n_batch, n_i):
    batchmod = np.zeros((n_i, n_batch), dtype=np.float64)
    batchmod[:, batch_idx] = 1.0
    pres = _to_col_matrix(preserve_site, n_i)
    return np.hstack([batchmod, nuis_exp, pres])


# ---------------------------------------------------------------------------
# Round 0 — common feature mask (pooled moments)
# ---------------------------------------------------------------------------

def site_round0(Y_site):
    Y = np.asarray(Y_site, dtype=np.float64)
    return Y.shape[1], Y.sum(axis=1), (Y ** 2).sum(axis=1)


def aggregate_mask(stats0):
    n = sum(s[0] for s in stats0)
    sumY = sum(s[1] for s in stats0)
    sumY2 = sum(s[2] for s in stats0)
    var = (sumY2 - sumY ** 2 / n) / (n - 1)
    sd0 = np.sqrt(var)
    return (sd0 > 0) & np.isfinite(sd0), np.isnan(sd0), n


# ---------------------------------------------------------------------------
# Round B — B-spline knot specs from pooled unique covariate values (Mode A)
# ---------------------------------------------------------------------------

def site_basis_stats(nuisance_site, smooth_cols):
    nz = _to_col_matrix(nuisance_site, len(nuisance_site))
    return {c: np.unique(nz[:, c]) for c in smooth_cols}


def aggregate_basis_specs(basis_stats, smooth_cols, n_total, degree=3, gam_df=None):
    df = gam_df if gam_df is not None else _gam_df(n_total)
    specs = {}
    for c in smooth_cols:
        pooled_unique = np.unique(np.concatenate([bs[c] for bs in basis_stats]))
        specs[c] = bspline_spec_from_pooled_unique(pooled_unique, df, degree)
    return specs


# ---------------------------------------------------------------------------
# Round 1 — decentralized regression (full design) → beta
# ---------------------------------------------------------------------------

def site_round1(Y_site, batch_idx, nuis_exp, preserve_site, ind_mask, n_batch):
    Ym = np.asarray(Y_site, dtype=np.float64)[ind_mask]      # (n_valid, n_i)
    X = _design(batch_idx, nuis_exp, preserve_site, n_batch, Ym.shape[1])
    D = Ym.T
    return X.T @ X, X.T @ D


def aggregate_beta(stats1):
    G = sum(s[0] for s in stats1)
    b = sum(s[1] for s in stats1)
    return pinv(G) @ b


# ---------------------------------------------------------------------------
# Round 2 — pooled std + grand mean   (grand mean uses [batch | nuisance])
# ---------------------------------------------------------------------------

def site_round2(Y_site, batch_idx, nuis_exp, preserve_site, ind_mask, n_batch, n_Z, beta):
    Ym = np.asarray(Y_site, dtype=np.float64)[ind_mask]
    X = _design(batch_idx, nuis_exp, preserve_site, n_batch, Ym.shape[1])
    resid = Ym.T - X @ beta
    SS = (resid ** 2).sum(axis=0)
    Xnp = X[:, :n_batch + n_Z]                               # XZ_no_preserve
    GMpart = (Xnp @ beta[:n_batch + n_Z]).sum(axis=0)
    return SS, GMpart


def aggregate_std_grandmean(stats2, n):
    SS = sum(s[0] for s in stats2)
    GMpart = sum(s[1] for s in stats2)
    std_pooled = np.sqrt(SS / n)
    nz = std_pooled > 0
    if not np.all(nz):
        std_pooled[~nz] = np.median(std_pooled[nz]) if np.any(nz) else 1.0
    return std_pooled, GMpart / n


# ---------------------------------------------------------------------------
# Round 3 — L/S model: gamma (decentralized) + delta (batch=local, nuisance=pooled)
# ---------------------------------------------------------------------------

def site_round3(Y_site, batch_idx, nuis_exp, preserve_site, ind_mask, n_batch,
                n_Z, beta_preserve, grand_mean, std_pooled):
    Ym = np.asarray(Y_site, dtype=np.float64)[ind_mask]
    X = _design(batch_idx, nuis_exp, preserve_site, n_batch, Ym.shape[1])
    pc = (X[:, n_batch + n_Z:] @ beta_preserve).T if beta_preserve.shape[0] else 0.0
    Ym_std = (Ym - grand_mean[:, None] - pc) / std_pooled[:, None]
    Xn = X[:, :n_batch + n_Z]                                # [batch | nuisance]
    A2 = Xn.T @ Xn
    B2 = Xn.T @ Ym_std.T
    delta_batch = np.var(Ym_std, axis=1, ddof=1)            # this site's batch row
    ss1 = Ym_std.sum(axis=1)
    ss2 = (Ym_std ** 2).sum(axis=1)
    return A2, B2, (batch_idx, delta_batch), ss1, ss2


def aggregate_gamma_delta(stats3, n_batch, n_Z, n_valid, n, mean_only=False):
    G2 = sum(s[0] for s in stats3)
    b2 = sum(s[1] for s in stats3)
    gamma = pinv(G2) @ b2
    delta = np.zeros((n_batch + n_Z, n_valid), dtype=np.float64)
    if mean_only:
        delta[:] = 1.0
        return gamma, delta
    for _, _, (bidx, dvar), _, _ in stats3:
        delta[bidx, :] = dvar                               # batch rows (local)
    if n_Z > 0:                                             # nuisance rows = pooled var
        S1 = sum(s[3] for s in stats3)
        S2 = sum(s[4] for s in stats3)
        var_pooled = (S2 - S1 ** 2 / n) / (n - 1)
        delta[n_batch:, :] = var_pooled[None, :]
    return gamma, delta


# ---------------------------------------------------------------------------
# Orchestration — runs the rounds and assembles a comcat-compatible estimates dict
# ---------------------------------------------------------------------------

def decentralized_fit(site_Y, site_batch_idx, site_nuisance, site_preserve,
                      batch_levels, smooth_cols=None, gam_df=None, degree=3,
                      mean_only=False):
    n_sites = len(site_Y)
    n_batch = len(batch_levels)
    n_nuis_orig = 0 if site_nuisance[0] is None else _to_col_matrix(
        site_nuisance[0], site_Y[0].shape[1]).shape[1]
    if smooth_cols is None:
        smooth_cols = list(range(n_nuis_orig))              # Mode A default: GAM all cols

    # Round 0 ----------------------------------------------------------------
    s0 = [site_round0(Y) for Y in site_Y]
    ind_mask, ind_nan, n = aggregate_mask(s0)
    n_valid = int(ind_mask.sum())

    # Round B (knots) --------------------------------------------------------
    if smooth_cols:
        bstats = [site_basis_stats(site_nuisance[k], smooth_cols) for k in range(n_sites)]
        specs = aggregate_basis_specs(bstats, smooth_cols, n, degree, gam_df)
        constructors = {c: build_constructor(specs[c]) for c in smooth_cols}
    else:
        specs, constructors = {}, {}

    nuis_exp = [expand_nuisance(site_nuisance[k], constructors) for k in range(n_sites)]
    n_Z = nuis_exp[0].shape[1] if n_nuis_orig else 0

    # Round 1 (beta) ---------------------------------------------------------
    s1 = [site_round1(site_Y[k], site_batch_idx[k], nuis_exp[k], site_preserve[k],
                      ind_mask, n_batch) for k in range(n_sites)]
    beta = aggregate_beta(s1)
    beta_preserve = beta[n_batch + n_Z:, :]

    # Round 2 (std, grand mean) ---------------------------------------------
    s2 = [site_round2(site_Y[k], site_batch_idx[k], nuis_exp[k], site_preserve[k],
                      ind_mask, n_batch, n_Z, beta) for k in range(n_sites)]
    std_pooled, grand_mean = aggregate_std_grandmean(s2, n)

    # Round 3 (gamma, delta) -------------------------------------------------
    s3 = [site_round3(site_Y[k], site_batch_idx[k], nuis_exp[k], site_preserve[k],
                      ind_mask, n_batch, n_Z, beta_preserve, grand_mean, std_pooled)
          for k in range(n_sites)]
    gamma, delta = aggregate_gamma_delta(s3, n_batch, n_Z, n_valid, n, mean_only)

    estimates = {
        'grand_mean': grand_mean,
        'std_pooled': std_pooled,
        'gamma_hat_masked': gamma,
        'delta_hat_masked': delta,
        'beta_hat_preserve': beta_preserve if beta_preserve.shape[0] else None,
        'ind_mask': ind_mask,
        'ind_nan': ind_nan,
        'batch_levels': np.asarray(batch_levels),
        'n_batch': n_batch,
        'n_nuisance_orig': n_nuis_orig,
        'n_X': beta_preserve.shape[0],
        'poly_degree': 2,                                   # unused (no poly path)
        'mean_only': mean_only,
        'ref_level': None,
        'smooth_terms': list(smooth_cols),
        'smooth_term_bounds': None,                         # constructors override
        'gam_df': specs[smooth_cols[0]]['df'] if smooth_cols else 5,
        'spline_constructors': constructors,
    }
    return estimates


def decentralized_harmonize(site_Y, site_batch_idx, site_nuisance, site_preserve,
                            batch_levels, **kw):
    """Full pipeline: fit globally, then harmonize each site locally."""
    est = decentralized_fit(site_Y, site_batch_idx, site_nuisance, site_preserve,
                            batch_levels, **kw)
    out = []
    for k in range(len(site_Y)):
        lbl = np.full(site_Y[k].shape[1], batch_levels[site_batch_idx[k]])
        out.append(comcat_from_training(
            site_Y[k], lbl, nuisance=site_nuisance[k],
            preserve=site_preserve[k], estimates=est))
    return out, est


# ---------------------------------------------------------------------------
# Validation against centralized comcat() (with GAM)
# ---------------------------------------------------------------------------

def main():
    from comcat import comcat

    rng = np.random.default_rng(7)
    n_features = 180
    sizes = [96, 74, 60]                                    # 3 sites / 3 batches
    n = sum(sizes)
    batch = np.concatenate([[k] * sizes[k] for k in range(len(sizes))])

    age = rng.uniform(20, 80, n)                            # smooth nuisance (GAM)
    score = rng.normal(0, 1, n)                             # preserve covariate

    base = rng.normal(0, 1, (n_features, 1))
    Y = (base
         + rng.normal(0, 0.04, (n_features, 1)) * age[None, :]
         + rng.normal(0, 0.03, (n_features, 1)) * (age[None, :] ** 2) / 100
         + rng.normal(0, 0.05, (n_features, 1)) * score[None, :]
         + rng.normal(0, 0.5, (n_features, n)))
    for k in range(1, len(sizes)):                          # site additive/multiplicative effects
        idx = batch == k
        Y[:, idx] += rng.normal(0.3 * k, 0.1, (n_features, 1))
        Y[:, idx] *= rng.normal(1.0 + 0.2 * k, 0.1, (n_features, 1))

    # centralized reference (default smooth_terms='all' -> GAM on age)
    Y_central, *_ = comcat(Y, batch, nuisance=age, preserve=score,
                           mean_only=False, smooth_terms='all', verbose=False)

    # decentralized: partition by site, never pool raw Y
    idxs = [np.where(batch == k)[0] for k in range(len(sizes))]
    site_Y = [Y[:, ix] for ix in idxs]
    site_nuis = [age[ix] for ix in idxs]
    site_pres = [score[ix] for ix in idxs]
    site_batch_idx = list(range(len(sizes)))
    batch_levels = list(range(len(sizes)))

    out, est = decentralized_harmonize(site_Y, site_batch_idx, site_nuis, site_pres,
                                       batch_levels)
    Y_dec = np.zeros_like(Y_central)
    for k, ix in enumerate(idxs):
        Y_dec[:, ix] = out[k]

    # --- bitwise check of the GAM basis itself --------------------------------
    from comcat import _build_nuisance_basis
    central_basis, _ = _build_nuisance_basis(age[:, None], 2, [0], None,
                                             est['gam_df'], False)
    dec_basis = est['spline_constructors'][0].transform(age[:, None])
    basis_bitwise = np.array_equal(central_basis, dec_basis)

    finite = np.isfinite(Y_central) & np.isfinite(Y_dec)
    max_abs = np.abs(Y_central[finite] - Y_dec[finite]).max()
    rel = max_abs / (np.abs(Y_central[finite]).max() + 1e-30)

    print("=" * 66)
    print("Decentralized ComCAT — GAM (Mode A), 3 sites, preserve=score")
    print("=" * 66)
    print(f"  features / subjects     : {n_features} / {n}  (sites {sizes})")
    print(f"  gam_df (from total n)   : {est['gam_df']}")
    print(f"  GAM basis bitwise-equal : {basis_bitwise}")
    print(f"  max |central - decentral|: {max_abs:.3e}")
    print(f"  relative to data scale   : {rel:.3e}")
    print(f"  nan layout matches       : {np.array_equal(np.isnan(Y_central), np.isnan(Y_dec))}")
    print("-" * 66)
    ok = basis_bitwise and max_abs < 1e-8
    print("  RESULT: " + ("PASS — basis bitwise, output within machine precision"
                          if ok else "FAIL"))
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

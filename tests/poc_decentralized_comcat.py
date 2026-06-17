"""
POC: Decentralized ComCAT — proof that a sufficient-statistics protocol
reproduces centralized `comcat()` output without pooling raw subject data.

Scenario (deliberately kept simple, NO GAM):
  - 2 sites, each site holds all subjects of exactly one batch
  - one global `preserve` covariate (e.g. age) -> makes the design non-block-
    diagonal so beta genuinely requires cross-site aggregation
  - no nuisance (preserve only), mean_only=False, ref_batch=None

Protocol (mirrors Bostami et al. 2022 pseudo-algorithm):
  Round 0  sites -> aggregator:  n_i, sum(Y), sum(Y^2)      => common mask + n
  Round 1  sites -> aggregator:  X_i^T X_i, X_i^T D_i        => beta (pinv of gram)
  Round 2  sites -> aggregator:  resid SS, grand-mean parts  => std_pooled, grand_mean
  Round 3  sites -> aggregator:  Xn_i^T Xn_i, Xn_i^T Zstd_i,
                                 per-batch var               => gamma, delta
  Apply    local on each site (reuses comcat_from_training)

Only aggregate statistics ever leave a site; individual subjects never do.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from numpy.linalg import pinv

# comcat.py lives in the repo root, one level up from this tests/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comcat import comcat, comcat_from_training, _to_col_matrix


# ---------------------------------------------------------------------------
# Site-local design builder (full global column layout: [batch | preserve])
# ---------------------------------------------------------------------------

def _build_design(batch_idx: int, preserve_site, n_batch: int, n_i: int) -> np.ndarray:
    """Full design row block for a site whose subjects all belong to `batch_idx`."""
    batchmod = np.zeros((n_i, n_batch), dtype=np.float64)
    batchmod[:, batch_idx] = 1.0
    X = _to_col_matrix(preserve_site, n_i)            # (n_i, n_X)
    return np.hstack([batchmod, X])                   # [batch | preserve]


# ---------------------------------------------------------------------------
# Round 0 — common feature mask from pooled first/second moments
# ---------------------------------------------------------------------------

def site_round0(Y_site):
    Y = np.asarray(Y_site, dtype=np.float64)          # (n_features, n_i)
    return Y.shape[1], Y.sum(axis=1), (Y ** 2).sum(axis=1)


def aggregate_mask(stats0):
    n = sum(s[0] for s in stats0)
    sumY = sum(s[1] for s in stats0)
    sumY2 = sum(s[2] for s in stats0)
    var = (sumY2 - sumY ** 2 / n) / (n - 1)           # pooled ddof=1 variance
    sd0 = np.sqrt(var)
    ind_mask = (sd0 > 0) & np.isfinite(sd0)
    ind_nan = np.isnan(sd0)
    return ind_mask, ind_nan, n


# ---------------------------------------------------------------------------
# Round 1 — decentralized regression for the full design (beta)
# ---------------------------------------------------------------------------

def site_round1(Y_site, batch_idx, preserve_site, ind_mask, n_batch):
    Ym = np.asarray(Y_site, dtype=np.float64)[ind_mask]   # (n_valid, n_i)
    n_i = Ym.shape[1]
    X = _build_design(batch_idx, preserve_site, n_batch, n_i)
    D = Ym.T                                               # (n_i, n_valid)
    return X.T @ X, X.T @ D


def aggregate_beta(stats1):
    G = sum(s[0] for s in stats1)
    b = sum(s[1] for s in stats1)
    return pinv(G) @ b                                    # identity: pinv(X) = pinv(X^TX) X^T


# ---------------------------------------------------------------------------
# Round 2 — pooled residual std + grand mean
# ---------------------------------------------------------------------------

def site_round2(Y_site, batch_idx, preserve_site, ind_mask, n_batch, beta):
    Ym = np.asarray(Y_site, dtype=np.float64)[ind_mask]
    X = _build_design(batch_idx, preserve_site, n_batch, Ym.shape[1])
    resid = Ym.T - X @ beta                                # (n_i, n_valid)
    SS = (resid ** 2).sum(axis=0)                          # (n_valid,)
    Xnp = X[:, :n_batch]                                   # XZ_no_preserve (n_Z=0)
    GMpart = (Xnp @ beta[:n_batch]).sum(axis=0)            # (n_valid,)
    return SS, GMpart


def aggregate_std_grandmean(stats2, n):
    SS = sum(s[0] for s in stats2)
    GMpart = sum(s[1] for s in stats2)
    std_pooled = np.sqrt(SS / n)
    nz = std_pooled > 0                                    # mirror comcat guard
    if not np.all(nz):
        std_pooled[~nz] = np.median(std_pooled[nz]) if np.any(nz) else 1.0
    grand_mean = GMpart / n
    return std_pooled, grand_mean


# ---------------------------------------------------------------------------
# Round 3 — L/S model: gamma (decentralized) + delta (local per batch)
# ---------------------------------------------------------------------------

def site_round3(Y_site, batch_idx, preserve_site, ind_mask, n_batch,
                beta_preserve, grand_mean, std_pooled):
    Ym = np.asarray(Y_site, dtype=np.float64)[ind_mask]   # (n_valid, n_i)
    X = _build_design(batch_idx, preserve_site, n_batch, Ym.shape[1])
    pc = (X[:, n_batch:] @ beta_preserve).T               # (n_valid, n_i)
    Ym_std = (Ym - grand_mean[:, None] - pc) / std_pooled[:, None]
    Xn = X[:, :n_batch]                                    # batchmod
    A2 = Xn.T @ Xn
    B2 = Xn.T @ Ym_std.T
    delta_i = np.var(Ym_std, axis=1, ddof=1)              # this site's batch row
    return A2, B2, (batch_idx, delta_i)


def aggregate_gamma_delta(stats3, n_batch, n_valid):
    G2 = sum(s[0] for s in stats3)
    b2 = sum(s[1] for s in stats3)
    gamma = pinv(G2) @ b2                                  # (n_batch, n_valid)
    delta = np.zeros((n_batch, n_valid), dtype=np.float64)
    for _, _, (bidx, delta_i) in [(None, None, s[2]) for s in stats3]:
        delta[bidx, :] = delta_i
    return gamma, delta


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def decentralized_fit(site_Y, site_batch_idx, site_preserve, batch_levels):
    """Run the multi-round protocol and assemble the `estimates` dict."""
    n_batch = len(batch_levels)

    # Round 0
    s0 = [site_round0(Y) for Y in site_Y]
    ind_mask, ind_nan, n = aggregate_mask(s0)
    n_valid = int(ind_mask.sum())

    # Round 1
    s1 = [site_round1(site_Y[k], site_batch_idx[k], site_preserve[k], ind_mask, n_batch)
          for k in range(len(site_Y))]
    beta = aggregate_beta(s1)
    beta_preserve = beta[n_batch:, :]                     # n_X rows

    # Round 2
    s2 = [site_round2(site_Y[k], site_batch_idx[k], site_preserve[k], ind_mask, n_batch, beta)
          for k in range(len(site_Y))]
    std_pooled, grand_mean = aggregate_std_grandmean(s2, n)

    # Round 3
    s3 = [site_round3(site_Y[k], site_batch_idx[k], site_preserve[k], ind_mask, n_batch,
                      beta_preserve, grand_mean, std_pooled)
          for k in range(len(site_Y))]
    gamma, delta = aggregate_gamma_delta(s3, n_batch, n_valid)

    estimates = {
        'grand_mean': grand_mean,
        'std_pooled': std_pooled,
        'gamma_hat_masked': gamma,
        'delta_hat_masked': delta,
        'beta_hat_preserve': beta_preserve,
        'ind_mask': ind_mask,
        'ind_nan': ind_nan,
        'batch_levels': np.asarray(batch_levels),
        'n_batch': n_batch,
        'n_nuisance_orig': 0,
        'n_X': beta_preserve.shape[0],
        'mean_only': False,
        'ref_level': None,
        'smooth_term_bounds': None,
        'gam_df': 5,
        'spline_constructors': {},
    }
    return estimates


# ---------------------------------------------------------------------------
# Demo / validation
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)

    n_features = 200
    n0, n1 = 96, 74                                       # like the paper's cohorts

    # global covariate to preserve (e.g. age), and a per-site signal + site effects
    age = rng.normal(40, 12, n0 + n1)
    batch = np.array([0] * n0 + [1] * n1)

    base = rng.normal(0, 1, (n_features, 1))
    age_coef = rng.normal(0, 0.05, (n_features, 1))
    Y = base + age_coef * age[None, :] + rng.normal(0, 0.5, (n_features, n0 + n1))
    # additive + multiplicative site effects on site 1
    Y[:, n0:] += rng.normal(0.3, 0.1, (n_features, 1))
    Y[:, n0:] *= rng.normal(1.4, 0.1, (n_features, 1))

    # ---- centralized reference -------------------------------------------
    Y_central, *_ = comcat(
        Y, batch, nuisance=None, preserve=age,
        mean_only=False, verbose=False,
    )

    # ---- decentralized: split by site, never pool raw Y -------------------
    idx0 = np.where(batch == 0)[0]
    idx1 = np.where(batch == 1)[0]
    site_Y = [Y[:, idx0], Y[:, idx1]]
    site_preserve = [age[idx0], age[idx1]]
    site_batch_idx = [0, 1]
    batch_levels = [0, 1]

    estimates = decentralized_fit(site_Y, site_batch_idx, site_preserve, batch_levels)

    # apply locally on each site (reuses existing apply path)
    Y_dec = np.zeros_like(Y_central)
    for k, idx in enumerate([idx0, idx1]):
        Yk = comcat_from_training(
            site_Y[k], np.full(len(idx), batch_levels[k]),
            nuisance=None, preserve=site_preserve[k], estimates=estimates,
        )
        Y_dec[:, idx] = Yk

    # ---- compare ----------------------------------------------------------
    finite = np.isfinite(Y_central) & np.isfinite(Y_dec)
    abs_diff = np.abs(Y_central[finite] - Y_dec[finite])
    max_abs = abs_diff.max()
    rel = max_abs / (np.abs(Y_central[finite]).max() + 1e-30)

    print("=" * 64)
    print("Decentralized ComCAT POC  (2 sites, preserve=age, no GAM)")
    print("=" * 64)
    print(f"  features                : {n_features}")
    print(f"  subjects (site0/site1)  : {n0} / {n1}")
    print(f"  harmonized features     : {int(estimates['ind_mask'].sum())}")
    print(f"  max |central - decentral|: {max_abs:.3e}")
    print(f"  relative to data scale   : {rel:.3e}")
    print(f"  nan layout matches       : {np.array_equal(np.isnan(Y_central), np.isnan(Y_dec))}")
    print("-" * 64)
    ok = max_abs < 1e-8
    print("  RESULT: " + ("PASS — identical to machine precision" if ok
                          else "FAIL — difference exceeds 1e-8"))
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

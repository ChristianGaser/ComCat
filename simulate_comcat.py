"""
simulate_comcat.py — simulation to evaluate ComCAT harmonization vs GLM AnCova.

Python port of simulate_comcat.m.

Usage (programmatic)
--------------------
from simulate_comcat import simulate_comcat

avgD, FPR = simulate_comcat(
    a=[1.0, 0.2, 0.0, 0.5],   # [EoI_amp, nuisance_amp, mult_amp, nuisance–EoI covariance]
    no_preserving=False,
    n=1000,
    n_sim=500,
    n_nuisance=1,
    mean_only=True,
    no_fig=False,
)

Returns
-------
avgD  : (2,) ndarray  — mean Cohen's D for [AnCova, ComCAT]
FPR   : (2,) ndarray  — false-positive rate at alpha=0.05

Two-step correction (Zhao et al.)
----------------------------------
When `apply_2step_correction=True` (default), the harmonized data Y_comcat
are pre-whitened before the GLM to account for the degrees-of-freedom
inflation introduced by ComCAT (see simulate_comcat.m).
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import pinv, matrix_rank, cholesky, inv, eigh
import scipy.stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _spm_orth(X: np.ndarray) -> np.ndarray:
    """Gram-Schmidt orthogonalisation of columns of X (mirrors spm_orth)."""
    Q, _ = np.linalg.qr(X)
    # scale so each column has sum-of-squares = n_rows (matches SPM convention)
    return Q * np.sqrt(X.shape[0])


def _calc_glm(Y: np.ndarray, X: np.ndarray, c: np.ndarray):
    """
    Compute GLM T-statistics.

    Parameters
    ----------
    Y : (n_features, n_subjects)
    X : (n_subjects, n_params) design matrix
    c : (n_params,) contrast vector

    Returns
    -------
    T      : (n_features,) T-values
    trRV   : residual degrees of freedom
    Beta   : (n_features, n_params) estimated parameters
    """
    pKX = pinv(X)                          # (k, n)
    Beta = Y @ pKX.T                       # (n_features, k)
    Y_hat = Beta @ X.T                     # (n_features, n)
    res = Y_hat - Y
    ResSS = np.sum(res ** 2, axis=1)       # (n_features,)

    trRV = X.shape[0] - matrix_rank(X)
    ResMS = ResSS / trRV                   # (n_features,)

    Bcov = pKX @ pKX.T                     # (k, k)
    con = Beta @ c                         # (n_features,)

    T = con / (np.finfo(float).eps + np.sqrt(ResMS * float(c @ Bcov @ c)))
    return T, trRV, Beta


def _two_step_correction(Y_comcat: np.ndarray, X_col: np.ndarray,
                          Z: np.ndarray) -> np.ndarray:
    """
    Zhao et al. two-step correction for degrees-of-freedom inflation.

    Applies a pre-whitening transform S to Y_comcat so that a subsequent
    GLM on [X, ones] has correct residual degrees of freedom.

    Parameters
    ----------
    Y_comcat : (n_features, n_subjects)
    X_col    : (n_subjects, 1)  — covariate of interest as column vector
    Z        : (n_subjects, n_nuisance)

    Returns
    -------
    Y_corrected : (n_features, n_subjects)
    """
    n = X_col.shape[0]
    Id = np.eye(n)

    h1 = X_col @ pinv(X_col.T @ X_col) @ X_col.T
    Z2 = np.column_stack([Z, np.ones(n)])
    h12 = Z2 @ pinv(Z2.T @ (Id - h1) @ Z2) @ Z2.T @ (Id - h1)

    reduce_ = Id - h12
    M = reduce_ @ reduce_.T

    vals, vecs = eigh(M)                   # eigenvalues in ascending order
    noise = 1.0 / n
    err = np.sum(vals) * noise
    vals = np.where(vals < 1e-4, err, vals)

    k_mat = vecs @ np.diag(vals) @ vecs.T
    # s = chol(inv(k))' in MATLAB = lower-triangular Cholesky of inv(k)
    s = cholesky(inv(k_mat))               # lower-triangular L s.t. L @ L.T = inv(k)

    return Y_comcat @ s


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate_comcat(
    a: list[float] | None = None,
    no_preserving: bool = False,
    n: int = 1000,
    n_sim: int = 500,
    n_nuisance: int = 1,
    n_perm: int = 0,
    mean_only: bool = False,
    no_fig: bool = False,
    apply_2step_correction: bool = True,
    seed: int | None = None,
):
    """
    Simulate data and compare ComCAT harmonisation vs GLM AnCova.

    Parameters
    ----------
    a : list of 4 floats  [EoI_amp, nuisance_amp, multiplicative_amp, nuisance_EoI_cov]
        Defaults: [1.0, 0.2, 0.0, 0.5]
    no_preserving          : if True, ComCAT does NOT preserve the EoI
    n                      : total sample size (must be even)
    n_sim                  : number of simulations (columns / features in Y)
    n_nuisance             : number of nuisance parameters
    n_perm                 : number of permutations for null FPR (0 = skip)
    mean_only              : ComCAT mean-only mode
    no_fig                 : suppress matplotlib figures
    apply_2step_correction : apply Zhao et al. DoF correction to ComCAT data
    seed                   : random seed for reproducibility (None = random)

    Returns
    -------
    avgD : (2,) ndarray  — mean Cohen's D for [AnCova, ComCAT]
    FPR  : (2,) ndarray  — false-positive rate at alpha=0.05 for [AnCova, ComCAT]
    """
    # lazy import so this module is usable without comcat installed
    from comcat import comcat as _comcat

    # ---- amplitude defaults ------------------------------------------------
    a1_def, a2_def, a3_def, a4_def = 1.0, 0.2, 0.0, 0.5
    if a is None:
        a = [a1_def, a2_def, a3_def, a4_def]
    a1, a2, a3, a4 = a[0], a[1], a[2], a[3]

    # ---- random number generator -------------------------------------------
    rng = np.random.default_rng(seed)

    # ---- covariate of interest and nuisance --------------------------------
    X0 = rng.standard_normal(n)            # (n,)

    # make n_nuisance columns orthogonal to X0
    raw = np.column_stack([X0, rng.standard_normal((n, n_nuisance))])
    noise_ortho = _spm_orth(raw)[:, 1:n_nuisance + 1]  # (n, n_nuisance)

    D0 = np.column_stack([X0, noise_ortho])  # (n, 1+n_nuisance)

    # covariance structure: nuisance correlated with EoI at strength a4
    A = np.array([[1.0, a4], [a4, 1.0]])
    L = np.linalg.cholesky(A)             # lower-triangular, L @ L.T = A
    # MATLAB: R = chol(A) (upper), Z = D0 @ R  →  Python: Z = D0 @ L.T
    R = L.T                               # upper-triangular (same as MATLAB chol)

    Z = np.empty((n, n_nuisance))
    for i in range(n_nuisance):
        pair = D0[:, [0, i + 1]] @ R      # apply covariance to (X0, noise_i) pair
        Z[:, i] = pair[:, 1]

    X = D0[:, 0]                          # (n,)

    # ---- simulate Y --------------------------------------------------------
    # Build Y as (n, n_sim) first (matches MATLAB column layout), then transpose.
    # This simplifies broadcasting: X and Z are (n,), E is (n, n_sim).
    E = rng.standard_normal((n, n_sim))

    # base signal: EoI + multiplicative noise from first nuisance
    Y0_T = a1 * X[:, None] + np.exp(a3 * Z[:, 0:1]) * E  # (n, n_sim)
    Y0 = Y0_T.T                          # (n_sim, n)

    Y_T = Y0_T.copy()
    for i in range(n_nuisance):
        Y_T += a2 * Z[:, i:i+1]         # broadcast nuisance over n_sim columns
    Y = Y_T.T                            # (n_sim, n)

    # ---- ComCAT harmonisation ----------------------------------------------
    if no_preserving:
        Y_comcat, *_ = _comcat(Y, None, Z, None,
                               mean_only=mean_only, poly_degree=1, verbose=False)
    else:
        Y_comcat, *_ = _comcat(Y, None, Z, X,
                               mean_only=mean_only, poly_degree=1, verbose=False)

    # ---- residualised adjustment -------------------------------------------
    if no_preserving:
        Beta_adj = Y @ pinv(np.column_stack([Z, np.ones(n)]).T)
        Y_adjusted = Y - Beta_adj[:, :n_nuisance] @ Z.T
    else:
        XZ_adj = np.column_stack([Z, X, np.ones(n)])
        Beta_adj = Y @ pinv(XZ_adj.T)
        Y_adjusted = Y - Beta_adj[:, :n_nuisance] @ Z.T

    # ---- Zhao two-step correction of ComCAT output -------------------------
    if apply_2step_correction:
        Y_comcat = _two_step_correction(Y_comcat, X[:, None], Z)

    # ---- GLM: AnCova (test X given Z as nuisance) --------------------------
    c_ancova = np.zeros(n_nuisance + 2)
    c_ancova[n_nuisance] = 1.0             # contrast on X

    XZ_full = np.column_stack([Z, X, np.ones(n)])
    T_ancova, trRV_ancova, Beta_ancova = _calc_glm(Y, XZ_full, c_ancova)

    D_ancova = 2.0 * T_ancova / np.sqrt(trRV_ancova)
    Thresh_ancova = scipy.stats.t.ppf(1 - 0.05, trRV_ancova)
    FPR_ancova = np.sum(T_ancova > Thresh_ancova) / n_sim

    beta_hat_ancova = Beta_ancova[:, n_nuisance]   # coefficient for X

    # ---- GLM: ComCAT harmonised data (test X in simple design) -------------
    c_comcat = np.array([1.0, 0.0])
    XZ_simple = np.column_stack([X, np.ones(n)])

    T_comcat, trRV_comcat, Beta_comcat = _calc_glm(Y_comcat, XZ_simple, c_comcat)

    D_comcat = 2.0 * T_comcat / np.sqrt(trRV_comcat)
    Thresh_comcat = scipy.stats.t.ppf(1 - 0.05, trRV_comcat)
    FPR_comcat = np.sum(T_comcat > Thresh_comcat) / n_sim

    beta_hat_comcat = Beta_comcat[:, 0]

    # ---- collect results ---------------------------------------------------
    avgD = np.array([np.mean(D_ancova), np.mean(D_comcat)])
    FPR  = np.array([FPR_ancova,        FPR_comcat])

    str_data = ['AnCova (GLM)', 'GLM ComCat harmonized']

    # ---- print summary -------------------------------------------------------
    cc = np.corrcoef(np.column_stack([X, Z]).T)
    print(f"\nReal effects (scaling={a1}) / (nuisance={a2}/mult={a3})")
    print(f"Correlation of nuisance with EoI: "
          f"{float(np.mean(np.abs(cc[0, 1:n_nuisance + 1]))):.3f}")

    if n_sim < 5000:
        ZY  = np.corrcoef(np.vstack([Z.T, Y]))
        ZYc = np.corrcoef(np.vstack([Z.T, Y_comcat]))
        for j in range(n_nuisance):
            r0 = float(np.mean(np.abs(ZY [j, n_nuisance:])))
            r1 = float(np.mean(np.abs(ZYc[j, n_nuisance:])))
            print(f"  Mean |corr(Z[{j}], Y)|: {str_data[0]}={r0:.4f}  "
                  f"{str_data[1]}={r1:.4f}")

    print()
    for i, (d, label) in enumerate(zip(avgD, str_data)):
        print(f"Mean of Effect size D = {d:.5f}  {label}")

    if a1 == 0:
        for i, (fpr, label) in enumerate(zip(FPR, str_data)):
            print(f"Rejection rate FP = {fpr:.4f} for {label}")

    # ---- plot ---------------------------------------------------------------
    if not no_fig:
        _plot_results(
            Y0, Y, Y_comcat,
            a2, Z,
            beta_hat_ancova, beta_hat_comcat,
            [D_ancova, D_comcat],
            str_data,
        )

    return avgD, FPR


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(Y0, Y, Y_comcat, a2, Z,
                  beta_hat_ancova, beta_hat_comcat,
                  D_list, str_data):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    COLORS = [
        "#BC3C29",   # red
        "#0072B5",   # blue
        "#E18727",   # orange
        "#20854E",   # green
    ]

    # Figure 1: signal traces
    fig1, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(Y0[0]);        axes[0].set_ylim(-5, 5); axes[0].set_title("Signal with added noise")
    axes[1].plot(Y[0]);         axes[1].set_ylim(-5, 5); axes[1].set_title("Signal with nuisance effects")
    axes[2].plot(a2 * Z[:, 0]); axes[2].set_ylim(-2, 2); axes[2].set_title("Nuisance effects")
    axes[3].plot(Y_comcat[0]);  axes[3].set_ylim(-5, 5); axes[3].set_title("Harmonized signal")
    fig1.tight_layout()

    # Figure 2: beta histograms
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(beta_hat_ancova, bins=40, alpha=0.7, label='Beta AnCova',  color=COLORS[0])
    ax2.hist(beta_hat_comcat, bins=40, alpha=0.7, label='Beta ComCat',  color=COLORS[1])
    ax2.legend()
    ax2.set_title("Beta coefficient distributions")
    ax2.set_xlabel("Beta")
    fig2.tight_layout()

    # Figure 3: effect-size distributions
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    all_d = np.concatenate(D_list)
    mn, mx = 0.9 * all_d.min(), all_d.max()
    x_grid = np.linspace(mn, mx, 200)
    for i, (d, label) in enumerate(zip(D_list, str_data)):
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(d)
            ax3.plot(x_grid, kde(x_grid), lw=2, color=COLORS[i], label=label)
        except Exception:
            ax3.hist(d, bins=40, density=True, alpha=0.6, color=COLORS[i], label=label)
    ax3.legend()
    ax3.set_title("Effect size D distribution")
    ax3.set_xlabel("Cohen's D")
    fig3.tight_layout()

    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate ComCAT harmonization vs GLM AnCova.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--a1", type=float, default=1.0,
                        help="Amplitude of effect of interest (EoI).")
    parser.add_argument("--a2", type=float, default=0.2,
                        help="Amplitude of nuisance effect.")
    parser.add_argument("--a3", type=float, default=0.0,
                        help="Amplitude of multiplicative nuisance effect.")
    parser.add_argument("--a4", type=float, default=0.5,
                        help="Covariance between nuisance and EoI.")
    parser.add_argument("--no-preserving", action="store_true",
                        help="Do not preserve EoI during harmonization.")
    parser.add_argument("--n-subjects", type=int, default=1000,
                        help="Total sample size (must be even).")
    parser.add_argument("--n-sim", type=int, default=500,
                        help="Number of simulations (columns of Y).")
    parser.add_argument("--n-nuisance", type=int, default=1,
                        help="Number of nuisance parameters.")
    parser.add_argument("--n-perm", type=int, default=0,
                        help="Number of permutations for null FPR (0 = skip).")
    parser.add_argument("--mean-only", action="store_true",
                        help="ComCAT mean-only mode (no variance scaling).")
    parser.add_argument("--no-fig", action="store_true",
                        help="Suppress matplotlib figures.")
    parser.add_argument("--no-2step", action="store_true",
                        help="Disable Zhao et al. two-step DoF correction.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()

    simulate_comcat(
        a=[args.a1, args.a2, args.a3, args.a4],
        no_preserving=args.no_preserving,
        n=args.n_subjects,
        n_sim=args.n_sim,
        n_nuisance=args.n_nuisance,
        n_perm=args.n_perm,
        mean_only=args.mean_only,
        no_fig=args.no_fig,
        apply_2step_correction=not args.no_2step,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

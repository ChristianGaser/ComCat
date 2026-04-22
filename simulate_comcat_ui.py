"""
simulate_comcat_ui.py — parameter sweep over ComCAT simulation conditions.

Python port of simulate_comcat_ui.m.

Runs simulate_comcat() across a grid of:
  a2        — nuisance amplitude (0, 0.05, …, 0.30)
  a4        — nuisance–EoI covariance (0, 0.05, …, 0.50)
  n_nuisance — 1, 2, 5, 10

Results (mean Cohen's D and false-positive rate) are saved to a .mat file
and, optionally, summarised as plots.

Usage
-----
python simulate_comcat_ui.py                          # default sweep
python simulate_comcat_ui.py --n 1000 --n-sim 5000 --no-fig
python simulate_comcat_ui.py --mean-only --output my_results.mat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io

from simulate_comcat import simulate_comcat


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_sweep(
    a1: float = 0.0,
    a2_values: list[float] | None = None,
    a4_values: list[float] | None = None,
    no_preserving: bool = False,
    n: int = 1000,
    n_sim: int = 20_000,
    n_nuisance_values: list[int] | None = None,
    mean_only: bool = True,
    apply_2step_correction: bool = True,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Sweep over nuisance amplitude, nuisance–EoI covariance, and number of
    nuisance parameters and collect effect-size D and FPR for each condition.

    Parameters
    ----------
    a1            : EoI amplitude (default 0 → null / FPR mode)
    a2_values     : nuisance amplitudes to sweep  (default 0..0.30 step 0.05)
    a4_values     : covariance values to sweep    (default 0..0.50 step 0.05)
    n_nuisance_values : list of n_nuisance values (default [1, 2, 5, 10])
    n             : sample size per simulation
    n_sim         : number of simulation columns in Y
    mean_only     : ComCAT mean-only mode
    apply_2step_correction : apply Zhao correction to ComCAT output
    seed          : base random seed (each cell gets seed + cell_index)
    verbose       : print progress

    Returns
    -------
    results : dict with keys
        'D'   : ndarray shape (n_a2, n_a4, n_nuis, 2)  — mean Cohen's D
        'FPR' : ndarray shape (n_a2, n_a4, n_nuis, 2)  — false-positive rate
        'a2_values', 'a4_values', 'n_nuisance_values', 'a1', 'n', 'n_sim',
        'mean_only', 'apply_2step_correction'
    """
    if a2_values is None:
        a2_values = list(np.arange(0.0, 0.31, 0.05))
    if a4_values is None:
        a4_values = list(np.arange(0.0, 0.51, 0.05))
    if n_nuisance_values is None:
        n_nuisance_values = [1, 2, 5, 10]

    n_a2   = len(a2_values)
    n_a4   = len(a4_values)
    n_nuis = len(n_nuisance_values)

    D   = np.zeros((n_a2, n_a4, n_nuis, 2))
    FPR = np.zeros((n_a2, n_a4, n_nuis, 2))

    cell_idx = 0
    total = n_a2 * n_a4 * n_nuis

    for j, a2 in enumerate(a2_values):
        for k, a4 in enumerate(a4_values):
            for m, n_nuis_m in enumerate(n_nuisance_values):
                cell_idx += 1
                if verbose:
                    print(f"\n{'─'*66}")
                    print(f"[{cell_idx}/{total}]  a2={a2:.2f}  a4={a4:.2f}  "
                          f"n_nuisance={n_nuis_m}", flush=True)

                cell_seed = None if seed is None else seed + cell_idx

                avgD, fpr = simulate_comcat(
                    a=[a1, a2, 0.0, a4],
                    no_preserving=no_preserving,
                    n=n,
                    n_sim=n_sim,
                    n_nuisance=n_nuis_m,
                    mean_only=mean_only,
                    no_fig=True,
                    apply_2step_correction=apply_2step_correction,
                    seed=cell_seed,
                )
                D  [j, k, m, :] = avgD
                FPR[j, k, m, :] = fpr

    return dict(
        D=D,
        FPR=FPR,
        a2_values=np.array(a2_values),
        a4_values=np.array(a4_values),
        n_nuisance_values=np.array(n_nuisance_values),
        a1=float(a1),
        n=int(n),
        n_sim=int(n_sim),
        mean_only=int(mean_only),
        apply_2step_correction=int(apply_2step_correction),
    )


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_results(results: dict, path: str | Path) -> None:
    """Save sweep results to a MATLAB-compatible .mat file."""
    scipy.io.savemat(str(path), results)
    print(f"Results saved to: {path}")


def load_results(path: str | Path) -> dict:
    """Load sweep results from a .mat file."""
    return scipy.io.loadmat(str(path), squeeze_me=True)


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------

def plot_summary(results: dict) -> None:
    """
    Plot FPR and mean Cohen's D as heat-maps over (a2 × a4) for each
    n_nuisance value.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    D   = results['D']                          # (n_a2, n_a4, n_nuis, 2)
    FPR = results['FPR']
    a2v = np.asarray(results['a2_values']).ravel()
    a4v = np.asarray(results['a4_values']).ravel()
    n_nuisance_values = np.asarray(results['n_nuisance_values']).ravel()

    method_labels = ['AnCova (GLM)', 'ComCAT harmonized']
    n_nuis = len(n_nuisance_values)

    for mi, label in enumerate(method_labels):
        fig, axes = plt.subplots(
            2, n_nuis,
            figsize=(4 * n_nuis, 8),
            squeeze=False,
        )
        fig.suptitle(label, fontsize=13)

        for m in range(n_nuis):
            # FPR heatmap
            ax_fpr = axes[0, m]
            im = ax_fpr.imshow(
                FPR[:, :, m, mi].T,
                origin='lower', aspect='auto',
                extent=[a2v[0], a2v[-1], a4v[0], a4v[-1]],
                vmin=0, vmax=0.2,
                cmap='RdYlGn_r',
            )
            ax_fpr.axhline(0.05, color='white', ls='--', lw=0.8)
            ax_fpr.set_title(f"FPR  (n_nuisance={n_nuisance_values[m]})")
            ax_fpr.set_xlabel("Nuisance amplitude (a2)")
            ax_fpr.set_ylabel("Nuisance–EoI cov (a4)")
            plt.colorbar(im, ax=ax_fpr)

            # Effect size heatmap
            ax_d = axes[1, m]
            im2 = ax_d.imshow(
                D[:, :, m, mi].T,
                origin='lower', aspect='auto',
                extent=[a2v[0], a2v[-1], a4v[0], a4v[-1]],
                cmap='viridis',
            )
            ax_d.set_title(f"Mean D  (n_nuisance={n_nuisance_values[m]})")
            ax_d.set_xlabel("Nuisance amplitude (a2)")
            ax_d.set_ylabel("Nuisance–EoI cov (a4)")
            plt.colorbar(im2, ax=ax_d)

        fig.tight_layout()

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep over ComCAT simulation conditions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--a1", type=float, default=0.0,
                        help="EoI amplitude (0 = null/FPR mode).")
    parser.add_argument("--a2-max", type=float, default=0.30,
                        help="Maximum nuisance amplitude (swept 0..a2-max in 0.05 steps).")
    parser.add_argument("--a4-max", type=float, default=0.50,
                        help="Maximum nuisance–EoI covariance (swept 0..a4-max in 0.05 steps).")
    parser.add_argument("--n-nuisance", type=int, nargs="+", default=[1, 2, 5, 10],
                        metavar="K", help="List of nuisance-parameter counts.")
    parser.add_argument("--no-preserving", action="store_true",
                        help="Do not preserve EoI during harmonization.")
    parser.add_argument("-n", type=int, default=1000,
                        help="Total sample size per simulation.")
    parser.add_argument("--n-sim", type=int, default=20_000,
                        help="Number of simulation columns in Y.")
    parser.add_argument("--mean-only", action="store_true", default=False,
                        help="ComCAT mean-only mode.")
    parser.add_argument("--no-2step", action="store_true",
                        help="Disable Zhao et al. two-step DoF correction.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed for reproducibility.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .mat filename (auto-generated if omitted).")
    parser.add_argument("--no-fig", action="store_true",
                        help="Suppress summary plots.")

    args = parser.parse_args()

    a2_values = list(np.round(np.arange(0.0, args.a2_max + 1e-9, 0.05), 4))
    a4_values = list(np.round(np.arange(0.0, args.a4_max + 1e-9, 0.05), 4))

    results = run_sweep(
        a1=args.a1,
        a2_values=a2_values,
        a4_values=a4_values,
        no_preserving=args.no_preserving,
        n=args.n,
        n_sim=args.n_sim,
        n_nuisance_values=args.n_nuisance,
        mean_only=args.mean_only,
        apply_2step_correction=not args.no_2step,
        seed=args.seed,
    )

    # determine output filename
    if args.output:
        out_path = args.output
    else:
        suffix = "_mean_only" if args.mean_only else ""
        out_path = f"D_FPR_comcat_n{args.n}{suffix}_nocorr.mat"

    save_results(results, out_path)

    if not args.no_fig:
        plot_summary(results)


if __name__ == "__main__":
    main()

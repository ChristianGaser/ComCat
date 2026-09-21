"""
combat_family_ui.py — file interface for ComBat (EB), CovBat and ComBatLS.

Same file handling as comcat_ui.py (NIfTI, GIFTI, MAT, TXT/CSV; the I/O
helpers are imported from there), so existing ComCAT inputs can be passed
unchanged.  The algorithms live in combat_family.py.

Usage (Python)
--------------
    from combat_family_ui import covbat_ui, combatls_ui, combat_eb_ui

    covbat_ui(files, batch=site, preserve=age)
    combatls_ui(files, batch=site, preserve=age)       # log(sigma) ~ age
    combat_eb_ui(files, batch=site, preserve=age)      # ComBat with EB

Each function returns (Y_harmonized, gamma_star, delta_star), like comcat_ui().

Usage (command line)
--------------------
    python combat_family_ui.py covbat data.mat --batch scanner.txt --preserve age.txt
    python combat_family_ui.py --help

Differences from comcat_ui
--------------------------
• batch is required: these methods need at least two sites.
• nuisance is accepted for call compatibility but ignored (with a warning):
  CovBat and ComBatLS preserve every covariate in their model and have no
  step that removes covariate effects.
• Output subfolders are named combateb_*, covbat_* and combatls_*, so they
  never overwrite ComCAT's comcat_* / combat_* results.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np

from combat_family import combat_eb, combatls, covbat
from comcat_ui import (
    _detect_filetype,
    _load_gifti,
    _load_mat,
    _load_nifti,
    _output_path_nifti,
    _save_estimates_nifti,
    _save_gifti,
    _save_mat,
    _save_nifti,
)

METHODS = {
    'combateb': combat_eb,
    'covbat': covbat,
    'combatls': combatls,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def combat_eb_ui(
    files: list[str],
    batch: np.ndarray | None = None,
    nuisance: np.ndarray | None = None,
    preserve: np.ndarray | None = None,
    mean_only: bool = False,
    subfolder: str | None = None,
    save_estimates: bool = False,
    verbose: bool = True,
    eb: bool = True,
):
    """ComBat with empirical Bayes (ComBatFamily::comfam with lm) on files.

    Parameters as in comcat_ui(); eb=False gives the plain L/S model.
    """
    return harmonize_files(
        'combateb', files, batch, nuisance, preserve, mean_only, subfolder,
        save_estimates, verbose, eb=eb,
    )


def covbat_ui(
    files: list[str],
    batch: np.ndarray | None = None,
    nuisance: np.ndarray | None = None,
    preserve: np.ndarray | None = None,
    mean_only: bool = False,
    subfolder: str | None = None,
    save_estimates: bool = False,
    verbose: bool = True,
    eb: bool = True,
    percent_var: float = 0.95,
    n_pc: int | None = None,
    std_var: bool = True,
    score_eb: bool = False,
):
    """CovBat (Chen et al., 2022) on files.

    Parameters as in comcat_ui(), plus:
    eb          : empirical Bayes in the ComBat step (default True)
    percent_var : harmonize the PCs explaining this proportion of variance (0.95)
    n_pc        : harmonize exactly this many PCs (overrides percent_var)
    std_var     : standardize features before the PCA (default True)
    score_eb    : empirical Bayes when harmonizing PC scores (default False)
    """
    return harmonize_files(
        'covbat', files, batch, nuisance, preserve, mean_only, subfolder,
        save_estimates, verbose, eb=eb, percent_var=percent_var, n_pc=n_pc,
        std_var=std_var, score_eb=score_eb,
    )


def combatls_ui(
    files: list[str],
    batch: np.ndarray | None = None,
    nuisance: np.ndarray | None = None,
    preserve: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    mean_only: bool = False,
    subfolder: str | None = None,
    save_estimates: bool = False,
    verbose: bool = True,
    eb: bool = True,
):
    """ComBatLS (Gardner et al., 2025) on files.

    Parameters as in comcat_ui(), plus:
    scale : (n_subjects, n_scale) covariates of log(sigma).  None (default)
            uses the preserve covariates, as in the ComBatLS paper.
    eb    : empirical Bayes estimation of site effects (default True)
    """
    return harmonize_files(
        'combatls', files, batch, nuisance, preserve, mean_only, subfolder,
        save_estimates, verbose, eb=eb, scale=scale,
    )


def harmonize_files(
    method: str,
    files: list[str],
    batch: np.ndarray | None = None,
    nuisance: np.ndarray | None = None,
    preserve: np.ndarray | None = None,
    mean_only: bool = False,
    subfolder: str | None = None,
    save_estimates: bool = False,
    verbose: bool = True,
    **options,
):
    """Load files, run `method` ('combateb', 'covbat', 'combatls') and save.

    Returns (Y_harmonized, gamma_star, delta_star); gamma_star and delta_star
    have shape (n_sites, n_features).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(METHODS)}.")
    if not files:
        raise ValueError("No input files provided.")
    if batch is None or np.asarray(batch).size == 0:
        raise ValueError(f"{method} needs site labels (batch).")

    filetype = _detect_filetype(files[0])
    if filetype == 'unknown':
        raise ValueError(f"Unsupported file type: {files[0]}")

    # ------------------------------------------------------------------ load
    if filetype == 'nifti':
        Y, meta = _load_nifti(files)
    elif filetype == 'gifti':
        Y, meta = _load_gifti(files)
    elif filetype == 'mat':
        if len(files) > 1:
            print("Only one MAT file supported. Ignoring extras.")
        Y, meta = _load_mat(files[0])
    else:
        if len(files) > 1:
            print("Only one TXT/CSV file supported. Ignoring extras.")
        Y = np.loadtxt(files[0])
        meta = None

    batch = np.asarray(batch).ravel()
    n_subjects = batch.size

    # MAT/TXT may be stored as (subjects x features); columns must be subjects
    Y_was_transposed = False
    if filetype in ('mat', 'txt') and Y.ndim == 2 and Y.shape[1] != n_subjects:
        if Y.shape[0] == n_subjects:
            Y = Y.T
            Y_was_transposed = True
        else:
            raise ValueError(
                f"Y shape {Y.shape} is incompatible with {n_subjects} site "
                f"labels. Expected one dimension to equal {n_subjects}."
            )

    if nuisance is not None and np.asarray(nuisance).size > 0:
        warnings.warn(
            f"{method}: nuisance covariates are ignored. CovBat/ComBatLS/ComBat "
            "preserve all modelled covariates and cannot remove them; use "
            "ComCAT for nuisance removal.", UserWarning, stacklevel=3,
        )

    preserve = _as_2d(preserve, n_subjects)
    if 'scale' in options and options['scale'] is not None:
        options['scale'] = _as_2d(options['scale'], n_subjects)

    # --------------------------------------------------------- harmonize
    Y_adj, est = METHODS[method](
        Y, batch, preserve, mean_only=mean_only, verbose=verbose, **options,
    )

    # same safeguard as comcat_ui: revert features whose SD grew > 10-fold
    sd0 = np.std(Y, axis=1, ddof=1)
    sd1 = np.std(Y_adj, axis=1, ddof=1)
    extreme = (sd1 / (sd0 + np.finfo(float).eps)) > 10
    if np.any(extreme):
        print(f"Reverting {int(np.sum(extreme))} feature(s) with extreme variance change.")
        Y_adj[extreme, :] = Y[extreme, :]

    gamma_star, delta_star = est['gamma_star'], est['delta_star']

    # --------------------------------------------------------- save
    if subfolder is None:
        subfolder = _build_subfolder(method, preserve.shape[1], mean_only, est, options)
    pth = str(Path(files[0]).parent)
    log = _log_dict(method, batch, preserve, mean_only, est, options)

    if filetype in ('nifti', 'gifti'):
        out_paths = [_output_path_nifti(f, subfolder) for f in files]
        if verbose:
            print(f'Saving harmonized data to subfolder "{subfolder}"')
        if filetype == 'nifti':
            _save_nifti(Y_adj, meta, out_paths)
        else:
            _save_gifti(Y_adj, meta, out_paths)
        if save_estimates:
            _save_estimates_nifti(gamma_star, delta_star, str(Path(pth) / subfolder),
                                  meta[0], filetype == 'gifti')
        _save_log_mat(pth, subfolder, log)

    else:
        out_dir = Path(pth) / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = out_dir / Path(files[0]).name
        Y_save = Y_adj.T if Y_was_transposed else Y_adj
        if filetype == 'mat':
            _save_mat(str(out_name), {**(meta or {}), 'Y': Y_save})
        else:
            np.savetxt(str(out_name), Y_save, fmt='%g')
        if verbose:
            print(f'Saving harmonized data to subfolder "{subfolder}"')
            print(f"Saved harmonized {filetype.upper()} file: {out_name}")
        if save_estimates:
            ext = '.txt' if filetype == 'mat' else Path(files[0]).suffix
            for name, arr in (('gamma', gamma_star), ('delta', delta_star)):
                for i in range(arr.shape[0]):
                    np.savetxt(str(out_dir / f"{name}{i + 1:02d}{ext}"),
                               arr[i][None, :], fmt='%g')
        _save_log_mat(str(out_dir), subfolder, log)

    if verbose:
        print()

    return Y_adj, gamma_star, delta_star


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_2d(arr, n: int) -> np.ndarray:
    if arr is None:
        return np.empty((n, 0))
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    if arr.shape[0] != n:
        arr = arr.T
    return arr


def _build_subfolder(method, n_preserve, mean_only, est, options) -> str:
    sf = method + '_sites'
    if n_preserve > 0:
        sf += f'_preserve{n_preserve}'
    if method == 'combatls':
        scale = options.get('scale')
        if scale is not None:
            sf += f'_scale{scale.shape[1]}'
    if mean_only:
        sf += '_meanonly'
    if not options.get('eb', True):
        sf += '_noeb'
    if method == 'covbat':
        if options.get('n_pc') is not None:
            sf += f"_npc{est['n_pc']}"
        else:
            sf += f"_pc{round(100 * options.get('percent_var', 0.95))}"
    return sf


def _log_dict(method, batch, preserve, mean_only, est, options) -> dict:
    log = {
        'method':    method,
        'batch':     np.asarray(batch),
        'preserve':  preserve,
        'mean_only': int(mean_only),
        'eb':        int(options.get('eb', True)),
    }
    if method == 'covbat':
        log.update(n_pc=int(est['n_pc']), pc_var=est['pc_var'],
                   percent_var=float(options.get('percent_var', 0.95)),
                   std_var=int(options.get('std_var', True)),
                   score_eb=int(options.get('score_eb', False)))
    if method == 'combatls':
        scale = options.get('scale')
        log['scale'] = preserve if scale is None else scale
        log['n_not_converged'] = int(np.sum(~est['converged']))
    return log


def _save_log_mat(pth, subfolder, log):
    """Save the settings as <subfolder>.mat (struct 'Harmonization')."""
    from scipy.io import savemat
    savemat(os.path.join(pth, subfolder + '.mat'), {'Harmonization': log})


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _load_labels(values):
    """--batch accepts a text file (one label per line) or the labels themselves."""
    if len(values) == 1 and os.path.isfile(values[0]):
        return np.loadtxt(values[0], dtype=str)
    return np.asarray(values)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ComBat (EB), CovBat or ComBatLS harmonization for NIfTI, "
                    "GIFTI, MAT, or TXT data."
    )
    p.add_argument('method', choices=list(METHODS), help='Harmonization method.')
    p.add_argument('files', nargs='+', help='Input data files.')
    p.add_argument('--batch', nargs='+', required=True, metavar='FILE|LABEL',
                   help='Text file with one site label per subject, or the '
                        'labels themselves (space-separated).')
    p.add_argument('--preserve', metavar='FILE',
                   help='Text file with covariates to preserve (n_subjects × n_covariates).')
    p.add_argument('--nuisance', metavar='FILE',
                   help='Accepted for compatibility with comcat_ui; ignored.')
    p.add_argument('--scale', metavar='FILE',
                   help='combatls: covariates of log(sigma) (default: preserve covariates).')
    p.add_argument('--no-eb', action='store_true', help='Disable empirical Bayes.')
    p.add_argument('--mean-only', action='store_true',
                   help='Adjust site means only (no variance scaling).')
    p.add_argument('--percent-var', type=float, default=0.95,
                   help='covbat: proportion of variance of harmonized PCs (default 0.95).')
    p.add_argument('--n-pc', type=int, default=None,
                   help='covbat: number of harmonized PCs (overrides --percent-var).')
    p.add_argument('--no-std-var', action='store_true',
                   help='covbat: do not standardize features before PCA.')
    p.add_argument('--score-eb', action='store_true',
                   help='covbat: empirical Bayes when harmonizing PC scores.')
    p.add_argument('--subfolder', default=None,
                   help='Override auto-generated output subfolder name.')
    p.add_argument('--save-estimates', action='store_true',
                   help='Save gamma (additive) and delta (multiplicative) site effects.')
    p.add_argument('--quiet', action='store_true', help='Suppress progress output.')
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    common = dict(
        files=args.files,
        batch=_load_labels(args.batch),
        nuisance=np.loadtxt(args.nuisance) if args.nuisance else None,
        preserve=np.loadtxt(args.preserve) if args.preserve else None,
        mean_only=args.mean_only,
        subfolder=args.subfolder,
        save_estimates=args.save_estimates,
        verbose=not args.quiet,
        eb=not args.no_eb,
    )
    if args.method == 'covbat':
        covbat_ui(**common, percent_var=args.percent_var, n_pc=args.n_pc,
                  std_var=not args.no_std_var, score_eb=args.score_eb)
    elif args.method == 'combatls':
        combatls_ui(**common, scale=np.loadtxt(args.scale) if args.scale else None)
    else:
        combat_eb_ui(**common)


if __name__ == '__main__':
    main()

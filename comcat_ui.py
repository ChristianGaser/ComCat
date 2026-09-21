"""
comcat_ui.py — Command-line / scripted interface for ComCAT harmonization.

Supports:
  • NIfTI files   (.nii, .nii.gz)  via nibabel
  • CIFTI/GIFTI   (.dscalar.nii, .func.gii)  via nibabel
  • MATLAB files  (.mat)           via scipy.io / h5py
  • Plain text    (.txt, .csv)     via numpy

Harmonized data are saved to a subfolder (NIfTI/GIFTI) or with a leading
prefix character (MAT/TXT), mirroring the behaviour of cat_stat_comcat.m.

Usage (command line)
--------------------
    python comcat_ui.py --help

Usage (Python)
--------------
    from comcat_ui import comcat_ui
    comcat_ui(files, batch=batch_vec, nuisance=age_vec, preserve=group_vec,
              mean_only=False, save_estimates=True)

Dependencies
------------
    numpy, scipy, nibabel
    (h5py is optional — needed for MATLAB v7.3 .mat files)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from comcat import comcat


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _detect_filetype(path: str) -> str:
    p = path.lower()
    if p.endswith('.mat'):
        return 'mat'
    if p.endswith('.txt') or p.endswith('.csv'):
        return 'txt'
    # GIFTI
    if p.endswith('.func.gii') or p.endswith('.shape.gii') or p.endswith('.gii'):
        return 'gifti'
    # NIfTI (including CIFTI)
    if p.endswith('.nii') or p.endswith('.nii.gz'):
        return 'nifti'
    return 'unknown'


def _load_nifti(files: list[str]) -> tuple[np.ndarray, object]:
    """Return (Y, headers) where Y is (n_voxels, n_subjects)."""
    import nibabel as nib
    imgs = [nib.load(f) for f in files]
    Y = np.column_stack([img.get_fdata(dtype=np.float32).ravel() for img in imgs])
    return Y, imgs


def _save_nifti(Y_harmonized: np.ndarray, ref_imgs: list, out_paths: list[str]):
    import nibabel as nib
    for i, (img, out_path) in enumerate(zip(ref_imgs, out_paths)):
        new_data = Y_harmonized[:, i].reshape(img.shape)
        new_img = img.__class__(new_data.astype(np.float32), img.affine, img.header)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        nib.save(new_img, out_path)


def _load_gifti(files: list[str]) -> tuple[np.ndarray, object]:
    import nibabel as nib
    imgs = [nib.load(f) for f in files]
    Y = np.column_stack([img.darrays[0].data.ravel() for img in imgs])
    return Y, imgs


def _save_gifti(Y_harmonized: np.ndarray, ref_imgs: list, out_paths: list[str]):
    import nibabel as nib
    import nibabel.gifti as ngi
    for i, (img, out_path) in enumerate(zip(ref_imgs, out_paths)):
        new_data = Y_harmonized[:, i].reshape(img.darrays[0].data.shape)
        new_darray = ngi.GiftiDataArray(new_data.astype(np.float32))
        new_img = ngi.GiftiImage(darrays=[new_darray])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        nib.save(new_img, out_path)


def _load_mat(path: str) -> tuple[np.ndarray, dict]:
    """Load a .mat file; return (Y, extra_fields) where extra_fields holds all
    non-private variables other than 'Y' so they can be written back."""
    _SKIP = {'__header__', '__version__', '__globals__'}
    try:
        from scipy.io import loadmat
        data = loadmat(path)
        if 'Y' not in data:
            raise KeyError("Mat-file does not contain a 'Y' field.")
        Y = np.array(data['Y'], dtype=np.float64)
        extra = {k: v for k, v in data.items() if k not in _SKIP and k != 'Y'}
        return Y, extra
    except Exception as e:
        # Try HDF5 / v7.3
        try:
            import h5py
            with h5py.File(path, 'r') as f:
                if 'Y' not in f:
                    raise KeyError("Mat-file does not contain a 'Y' field.")
                Y = np.array(f['Y'], dtype=np.float64)
                extra = {k: np.array(f[k]) for k in f.keys() if k != 'Y'}
            return Y, extra
        except ImportError:
            raise RuntimeError(
                "Could not read .mat file. "
                "Install scipy (for v5-v7.2) or h5py (for v7.3)."
            ) from e


def _save_mat(path: str, data: dict):
    from scipy.io import savemat
    savemat(path, data)


# ---------------------------------------------------------------------------
# Subfolder / output path helpers
# ---------------------------------------------------------------------------

def _build_subfolder(
    n_sites: int,
    n_nuisance: int,
    n_preserve: int,
    mean_only: bool,
    gam_df: int | None = None,
    preserve_df: int | None = None,
    nuisance_scale: bool = False,
    residual_delta: bool = False,
) -> str:
    """Folder name encoding the settings; defaults give the names used before
    the optional extensions existed."""
    if n_sites > 1 and n_nuisance == 0:
        sf = 'combat'
    else:
        sf = 'comcat'
    if n_sites > 1:
        sf += f'_sites'
    if n_preserve > 0:
        sf += f'_preserve{n_preserve}'
        if preserve_df is not None:
            sf += f'_gam{preserve_df}'
    if mean_only:
        sf += '_meanonly'
    if n_nuisance > 0:
        sf += f'_nuisance{n_nuisance}_gam{gam_df}'
        if nuisance_scale:
            sf += '_scale'                 # implies residual site variances
        elif residual_delta and not mean_only:
            sf += '_resdelta'
    return sf


def _output_path_nifti(src_path: str, subfolder: str) -> str:
    p = Path(src_path)
    return str(p.parent / subfolder / p.name)


# ---------------------------------------------------------------------------
# Main UI function
# ---------------------------------------------------------------------------

def comcat_ui(
    files: list[str],
    batch: np.ndarray | None = None,
    nuisance: np.ndarray | None = None,
    preserve: np.ndarray | None = None,
    mean_only: bool = False,
    subfolder: str | None = None,
    save_estimates: bool = False,
    verbose: bool = True,
    smooth_term_bounds=None,
    gam_df: int | None = None,
    preserve_df: int | str | None = None,
    preserve_bounds=None,
    nuisance_scale: bool = False,
    residual_delta: bool = False,
):
    """
    Run ComCAT on a list of image/data files and save harmonized results.

    Every nuisance column is modelled with a B-spline GAM (requires statsmodels).

    Parameters
    ----------
    files         : list of file paths (NIfTI, GIFTI, MAT, or TXT/CSV)
                    For MAT/TXT, only a single file is supported (contains all subjects).
    batch         : 1-D array of site labels per subject (None → single site)
    nuisance      : 2-D array (n_subjects × n_nuisance) to remove
    preserve      : 2-D array (n_subjects × n_preserve) to keep
    mean_only     : adjust mean only (no variance scaling)
    subfolder     : override auto-generated subfolder name
    save_estimates: save gamma/delta LS estimates alongside data
    verbose       : print progress
    smooth_term_bounds : boundary knots; None infers from data (fine for single-dataset use).
    gam_df        : B-spline basis dimension per nuisance column.
                    None (default) — auto-selected from sample size: min(10, max(5, n//30)).

    Optional extensions (off by default; see comcat.py for details)
    preserve_df   : B-spline df for continuous preserve covariates; 'same' = gam_df.
                    Output folder gets '_preserve<n>_gam<df>'.
    preserve_bounds : boundary knots for the preserve splines (per preserve column).
    nuisance_scale: also remove nuisance-dependent variance.  Folder suffix '_scale'.
                    With save_estimates, the per-feature likelihood-ratio test of
                    nuisance effects on the variance is saved as scale_lr / scale_p.
    residual_delta: estimate site variances after removing nuisance effects (as in
                    ComBat).  Folder suffix '_resdelta'.
    """
    if not files:
        raise ValueError("No input files provided.")

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
    else:  # txt / csv
        if len(files) > 1:
            print("Only one TXT/CSV file supported. Ignoring extras.")
        Y = np.loadtxt(files[0])
        meta = None

    n_subjects = Y.shape[1] if Y.ndim == 2 else Y.shape[0]

    # For MAT/TXT files, auto-detect orientation by comparing with the length
    # of any supplied covariate vector.  If Y is stored as (subjects × features)
    # instead of (features × subjects), transpose it so that columns = subjects.
    # Track the flip so we can restore the original orientation on save.
    _Y_was_transposed = False
    if filetype in ('mat', 'txt') and Y.ndim == 2:
        ref_len = None
        for cov in (batch, nuisance, preserve):
            if cov is not None:
                ref_len = np.asarray(cov).shape[0]
                break
        if ref_len is not None and n_subjects != ref_len:
            if Y.shape[0] == ref_len:
                Y = Y.T
                n_subjects = Y.shape[1]
                _Y_was_transposed = True
            else:
                raise ValueError(
                    f"Y shape {Y.shape} is incompatible with covariate length "
                    f"{ref_len}. Expected one dimension to equal {ref_len}."
                )

    # ------------------------------------------------------------------ batch
    if batch is None:
        batch = np.ones(n_subjects, dtype=int)

    batch = np.asarray(batch).ravel()
    _, batch_coded = np.unique(batch, return_inverse=True)
    n_sites = len(np.unique(batch_coded))

    if nuisance is None:
        nuisance = np.empty((n_subjects, 0))
    nuisance = np.atleast_2d(np.asarray(nuisance, dtype=float))
    if nuisance.shape[0] != n_subjects:
        nuisance = nuisance.T
    n_nuisance_cols = nuisance.shape[1]

    if preserve is None:
        preserve = np.empty((n_subjects, 0))
    preserve = np.atleast_2d(np.asarray(preserve, dtype=float))
    if preserve.shape[0] != n_subjects:
        preserve = preserve.T
    n_preserve_cols = preserve.shape[1]

    # ----------------------------------------------------------------- verbose
    if verbose and n_nuisance_cols > 0 and n_preserve_cols > 0:
        cc = np.corrcoef(
            np.hstack([preserve, nuisance]).T
        )
        off_diag = cc[:n_preserve_cols, n_preserve_cols:]
        print(f"Correlation of nuisance with preserving variable(s): "
              f"{np.mean(np.abs(off_diag)):.3f}")

    # --------------------------------------------------------- harmonize
    # Resolve gam_df here so subfolder name and log mat are consistent
    if gam_df is None:
        gam_df = min(10, max(5, n_subjects // 30))

    if preserve_df == 'same':
        preserve_df = gam_df

    Y_adj, beta_hat, gamma_hat, delta_hat, estimates = comcat(
        Y, batch_coded, nuisance, preserve,
        mean_only=mean_only, verbose=verbose,
        smooth_term_bounds=smooth_term_bounds,
        gam_df=gam_df,
        preserve_df=preserve_df,
        preserve_bounds=preserve_bounds,
        nuisance_scale=nuisance_scale,
        residual_delta=residual_delta,
        return_estimates=True,
    )
    nuisance_scale = estimates['nuisance_scale'] is not None   # False without nuisance
    options = dict(preserve_df=preserve_df, nuisance_scale=nuisance_scale,
                   residual_delta=residual_delta)

    # guard against extreme variance changes (factor > 10)
    sd0 = np.std(Y, axis=1, ddof=1) if Y.ndim == 2 else np.std(Y, axis=0, ddof=1)
    sd1 = np.std(Y_adj, axis=1, ddof=1) if Y_adj.ndim == 2 else np.std(Y_adj, axis=0, ddof=1)
    extreme = (sd1 / (sd0 + np.finfo(float).eps)) > 10
    if np.any(extreme):
        n_ext = int(np.sum(extreme))
        print(f"Reverting {n_ext} feature(s) with extreme variance change.")
        if Y_adj.ndim == 2:
            Y_adj[extreme, :] = Y[extreme, :]
        else:
            Y_adj[extreme] = Y[extreme]

    # --------------------------------------------------------- subfolder / path
    if subfolder is None:
        subfolder = _build_subfolder(
            n_sites, n_nuisance_cols, n_preserve_cols, mean_only, gam_df=gam_df,
            **options,
        )

    pth = str(Path(files[0]).parent)

    # --------------------------------------------------------- save
    if filetype in ('nifti', 'gifti'):
        out_paths = [_output_path_nifti(f, subfolder) for f in files]
        if verbose:
            print(f'Saving harmonized data to subfolder "{subfolder}"')

        if filetype == 'nifti':
            _save_nifti(Y_adj, meta, out_paths)
        else:
            _save_gifti(Y_adj, meta, out_paths)

        if save_estimates:
            _save_estimates_nifti(
                gamma_hat, delta_hat, pth, meta[0],
                filetype == 'gifti'
            )

        if save_estimates and nuisance_scale:
            _save_scale_test(estimates, str(Path(pth) / subfolder), filetype,
                             meta[0], verbose)

        # save log mat
        _save_log_mat(pth, subfolder, batch, nuisance, preserve, gam_df, **options)

    elif filetype == 'mat':
        out_dir = Path(pth) / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = out_dir / Path(files[0]).name
        # restore original orientation if Y was transposed on load
        Y_save = Y_adj.T if _Y_was_transposed else Y_adj
        # preserve all original fields; replace Y with harmonized version
        save_dict = {**(meta or {}), 'Y': Y_save}
        _save_mat(str(out_name), save_dict)
        if verbose:
            print(f'Saving harmonized data to subfolder "{subfolder}"')
            print(f"Saved harmonized MAT file: {out_name}")

        if save_estimates:
            for i in range(gamma_hat.shape[0]):
                gname = out_dir / f"gamma{i+1:02d}.txt"
                np.savetxt(str(gname), gamma_hat[i, :][None, :], fmt='%g')
                if verbose:
                    print(f"Saved {gname} (additive effects)")
            for i in range(delta_hat.shape[0]):
                dname = out_dir / f"delta{i+1:02d}.txt"
                np.savetxt(str(dname), delta_hat[i, :][None, :], fmt='%g')
                if verbose:
                    print(f"Saved {dname} (multiplicative effects)")

        if save_estimates and nuisance_scale:
            _save_scale_test(estimates, str(out_dir), filetype, None, verbose)

        _save_log_mat(str(out_dir), subfolder, batch, nuisance, preserve, gam_df,
                      **options)

    else:  # txt / csv
        out_dir = Path(pth) / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = out_dir / Path(files[0]).name
        # restore original orientation if Y was transposed on load
        Y_save = Y_adj.T if _Y_was_transposed else Y_adj
        np.savetxt(str(out_name), Y_save, fmt='%g')
        if verbose:
            print(f'Saving harmonized data to subfolder "{subfolder}"')
            print(f"Saved harmonized TXT file: {out_name}")

        if save_estimates:
            ext = Path(files[0]).suffix
            for i in range(gamma_hat.shape[0]):
                gname = out_dir / f"gamma{i+1:02d}{ext}"
                np.savetxt(str(gname), gamma_hat[i, :][None, :], fmt='%g')
            for i in range(delta_hat.shape[0]):
                dname = out_dir / f"delta{i+1:02d}{ext}"
                np.savetxt(str(dname), delta_hat[i, :][None, :], fmt='%g')

        if save_estimates and nuisance_scale:
            _save_scale_test(estimates, str(out_dir), filetype, None, verbose)

        _save_log_mat(str(out_dir), subfolder, batch, nuisance, preserve, gam_df,
                      **options)

    if verbose:
        print()

    return Y_adj, gamma_hat, delta_hat


# ---------------------------------------------------------------------------
# Estimate saving helpers
# ---------------------------------------------------------------------------

def _save_estimates_nifti(gamma_hat, delta_hat, pth, ref_img, is_gifti: bool):
    import nibabel as nib
    for i in range(gamma_hat.shape[0]):
        fname = os.path.join(pth, f"gamma{i+1:02d}.nii.gz")
        if is_gifti:
            import nibabel.gifti as ngi
            arr = ngi.GiftiDataArray(gamma_hat[i].astype(np.float32))
            img = ngi.GiftiImage(darrays=[arr])
        else:
            data = gamma_hat[i].reshape(ref_img.shape)
            img = ref_img.__class__(data.astype(np.float32), ref_img.affine, ref_img.header)
        nib.save(img, fname)

    for i in range(delta_hat.shape[0]):
        fname = os.path.join(pth, f"delta{i+1:02d}.nii.gz")
        if is_gifti:
            import nibabel.gifti as ngi
            arr = ngi.GiftiDataArray(delta_hat[i].astype(np.float32))
            img = ngi.GiftiImage(darrays=[arr])
        else:
            data = delta_hat[i].reshape(ref_img.shape)
            img = ref_img.__class__(data.astype(np.float32), ref_img.affine, ref_img.header)
        nib.save(img, fname)


def _save_scale_test(estimates, out_dir, filetype, ref_img, verbose):
    """Save the likelihood-ratio test of nuisance effects on the variance
    (nuisance_scale option): scale_lr = chi-square statistic with
    estimates['scale_lr_df'] degrees of freedom, scale_p = p-value."""
    import nibabel as nib
    for key in ('scale_lr', 'scale_p'):
        arr = estimates[key]
        if filetype == 'nifti':
            fname = os.path.join(out_dir, f"{key}.nii.gz")
            nib.save(ref_img.__class__(arr.reshape(ref_img.shape).astype(np.float32),
                                       ref_img.affine, ref_img.header), fname)
        elif filetype == 'gifti':
            import nibabel.gifti as ngi
            fname = os.path.join(out_dir, f"{key}.gii")
            nib.save(ngi.GiftiImage(darrays=[ngi.GiftiDataArray(arr.astype(np.float32))]),
                     fname)
        else:
            fname = os.path.join(out_dir, f"{key}.txt")
            np.savetxt(fname, arr[None, :], fmt='%g')
        if verbose:
            print(f"Saved {fname}")
    if verbose:
        frac = np.nanmean(estimates['scale_p'] < 0.05)
        print(f"Nuisance effect on variance: p < 0.05 (uncorrected) in "
              f"{100 * frac:.1f}% of features (chi2, df={estimates['scale_lr_df']})")


def _save_log_mat(pth, subfolder, batch, nuisance, preserve, gam_df=6,
                  preserve_df=None, nuisance_scale=False, residual_delta=False):
    """Save a .mat log file with ComCAT parameters (mirrors MATLAB behaviour)."""
    try:
        from scipy.io import savemat
        log = {
            'batch':      np.array(batch),
            'nuisance':   np.array(nuisance),
            'preserve':   np.array(preserve),
            'gam_df':     int(gam_df),
            'preserve_df':    -1 if preserve_df is None else int(preserve_df),
            'nuisance_scale': int(nuisance_scale),
            'residual_delta': int(residual_delta or nuisance_scale),
        }
        savemat(os.path.join(pth, subfolder + '.mat'), {'Comcat': log})
    except ImportError:
        pass  # scipy not available; skip log


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ComCAT harmonization for NIfTI, GIFTI, MAT, or TXT data."
    )
    p.add_argument('files', nargs='+', help='Input data files.')
    p.add_argument(
        '--batch', nargs='+', type=float, metavar='LABEL',
        help='Site/scanner labels, one per subject (space-separated). '
             'If omitted, all subjects are treated as one site.',
    )
    p.add_argument(
        '--nuisance', metavar='FILE',
        help='Path to a text file with nuisance regressors (n_subjects × n_regressors).',
    )
    p.add_argument(
        '--preserve', metavar='FILE',
        help='Path to a text file with covariates to preserve (n_subjects × n_covariates).',
    )
    p.add_argument('--mean-only', action='store_true',
                   help='Adjust mean only (no variance scaling).')
    p.add_argument('--gam-df', type=int, default=6, metavar='N',
                   help='B-spline basis dimension per nuisance term (default: 6).')
    p.add_argument('--preserve-df', default=None, metavar='N|same',
                   help='B-spline df for continuous preserve covariates '
                        "('same' = gam-df; default: linear).")
    p.add_argument('--nuisance-scale', action='store_true',
                   help='Also remove nuisance-dependent variance.')
    p.add_argument('--residual-delta', action='store_true',
                   help='Estimate site variances after removing nuisance effects.')
    p.add_argument('--subfolder', default=None,
                   help='Override auto-generated output subfolder name.')
    p.add_argument('--save-estimates', action='store_true',
                   help='Save gamma (additive) and delta (multiplicative) estimates.')
    p.add_argument('--quiet', action='store_true', help='Suppress progress output.')
    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    batch = np.array(args.batch) if args.batch else None
    nuisance = np.loadtxt(args.nuisance) if args.nuisance else None
    preserve = np.loadtxt(args.preserve) if args.preserve else None

    comcat_ui(
        files=args.files,
        batch=batch,
        nuisance=nuisance,
        preserve=preserve,
        mean_only=args.mean_only,
        subfolder=args.subfolder,
        save_estimates=args.save_estimates,
        verbose=not args.quiet,
        gam_df=args.gam_df,
        preserve_df=(args.preserve_df if args.preserve_df in (None, 'same')
                     else int(args.preserve_df)),
        nuisance_scale=args.nuisance_scale,
        residual_delta=args.residual_delta,
    )


if __name__ == '__main__':
    main()

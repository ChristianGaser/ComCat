#!/usr/bin/env python3
"""ComCAT for the data Y of a BA_data2mat mat-file, cross-fitted, before BrainAGE or
the normative models of NeuroGAMLSS.

Removes site effects and nuisance effects (e.g. image quality) from the test data,
which are then scored by models trained on raw data.  Every subject is harmonized by
a model that was not fitted on it (comcat_crossfit): the subjects to fit on
(--controls, default all) are split into --folds folds, stratified by site; every
fold is harmonized by the model of the other folds, all other subjects (e.g.
patients) by the model of all controls.  --folds 0 skips the cross-fit: one model of
the controls for all subjects, the controls in-sample (optimistic for them).  The
output is a copy of the mat-file with Y harmonized and all other fields kept.

    python comcat_mat.py s6mwp1_4mm_ABIDE437_CAT12.9.mat --site Scanner437.txt \\
        --controls "$TD" --nuisance IQM437.txt --nuisance-cols 1,2,3,5 \\
        --preserve age sex TIV437.txt --out comcat/s6mwp1_4mm_ABIDE437_CAT12.9.mat

Tables are text files of numbers without header, one row per subject in the order
of the mat-file; columns 1, 2, 3 and 5 of a CAT12 IQM table are ICR, NCR, res_ECR
and contrastr.  Only what --preserve lists is preserved.
"""

import argparse
import os
import shutil

import numpy as np

from comcat import comcat, comcat_crossfit, comcat_from_training


def read_mat(path):
    """Y (n_features, n_subjects), age, male (None if missing) and whether the file
    is v7.3 (HDF5, which stores the MATLAB arrays transposed)."""
    try:
        from scipy.io import loadmat
        m = loadmat(path, variable_names=['Y', 'age', 'male'])
        v73 = False
    except NotImplementedError:
        import h5py
        with h5py.File(path, 'r') as h:
            m = {k: h[k][()].T for k in ('Y', 'age', 'male') if k in h}
        v73 = True
    age = np.ravel(m['age']).astype(float)
    male = np.ravel(m['male']).astype(float) if 'male' in m else None
    Y = np.asarray(m['Y'], dtype=np.float64)
    return (Y.T if Y.shape[0] == age.size else Y), age, male, v73


def write_mat(src, dst, H, v73):
    """Copy of src at dst with Y replaced by H (n_features, n_subjects), in the
    orientation and type of the stored Y."""
    if v73:
        import h5py
        shutil.copyfile(src, dst)
        with h5py.File(dst, 'r+') as h:
            ds = h['Y']
            ds[...] = (H if ds.shape == H.shape else H.T).astype(ds.dtype)
        return
    from scipy.io import loadmat, savemat
    m = {k: v for k, v in loadmat(src, mat_dtype=True).items() if not k.startswith('__')}
    m['Y'] = (H.T if m['Y'].shape == H.T.shape else H).astype(m['Y'].dtype)
    savemat(dst, m, do_compression=True)


def read_table(path, n, cols=None):
    """Numeric table (n, k) without header; cols: 1-based column numbers '1,2,3,5'."""
    with open(path) as f:
        sep = ',' if ',' in f.readline() else None
    T = np.loadtxt(path, delimiter=sep, ndmin=2)
    if len(T) != n:
        raise SystemExit(f"{path}: {len(T)} rows for {n} subjects.")
    return T if cols is None else T[:, [int(c) - 1 for c in cols.split(',')]]


def read_labels(path, n):
    """One label per line (site, subject or group); numbers are compared as numbers."""
    lab = np.loadtxt(path, dtype=str, ndmin=1)
    if len(lab) != n:
        raise SystemExit(f"{path}: {len(lab)} labels for {n} subjects.")
    try:
        return lab.astype(float)
    except ValueError:
        return lab


def parse_index(spec, n):
    """1-based indices as for neurogamlss.py --adjust ('1:108,150') -> boolean mask."""
    mask = np.zeros(n, dtype=bool)
    for part in spec.split(','):
        a, _, b = part.partition(':')
        a, b = int(a), int(b or a)
        if not 1 <= a <= b <= n:
            raise SystemExit(f"--controls: {part} is not within 1..{n}.")
        mask[a - 1:b] = True
    return mask


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mat', help="mat-file of BA_data2mat (v5 or v7.3) with Y, age (and male)")
    p.add_argument('--out', required=True, help="harmonized copy of the mat-file")
    p.add_argument('--site', help="site of every subject, one label per line (default: one "
                   "site)")
    p.add_argument('--nuisance', help="table of nuisance variables to remove, e.g. a CAT12 "
                   "IQM table (penalized splines)")
    p.add_argument('--nuisance-cols', help="1-based columns of --nuisance, e.g. 1,2,3,5 "
                   "(default: all)")
    p.add_argument('--preserve', nargs='+', default=[], metavar='ITEM',
                   help="effects to keep: age and sex of the mat-file and tables (all their "
                        "columns), e.g. 'age sex TIV.txt' (default: nothing)")
    p.add_argument('--controls', help="1-based indices of the subjects to fit on, e.g. "
                   "1:108,150 (default: all)")
    p.add_argument('--group', help="group of every subject (e.g. diagnosis): fit on all "
                   "subjects with the group differences preserved, instead of --controls")
    p.add_argument('--subject', help="subject of every scan (repeated scans): the scans of a "
                   "subject stay in one fold")
    p.add_argument('--within-subjects', action='store_true',
                   help="with --subject: effects estimated within subjects (subject terms, "
                        "location only)")
    p.add_argument('--location-only', action='store_true', help="no scale correction of "
                   "the sites")
    p.add_argument('--folds', type=int, default=5, help="folds of the cross-fit (default 5); "
                   "0: no cross-fit, fitted on the controls and applied to all, the controls "
                   "in-sample (optimistic for them)")
    a = p.parse_args(argv)
    if not (a.site or a.nuisance):
        p.error("Nothing to remove: give --site, --nuisance or both.")
    if a.group and a.controls:
        p.error("--group fits on all subjects: leave out --controls.")
    if a.within_subjects and not a.subject:
        p.error("--within-subjects needs --subject.")
    if os.path.abspath(a.out) == os.path.abspath(a.mat):
        p.error("--out would overwrite the input.")

    Y, age, male, v73 = read_mat(a.mat)
    n = age.size
    # sites as codes 0..k-1 in the order of their labels, as in the evaluation (the
    # folds depend on how the labels are written)
    site = np.unique(read_labels(a.site, n), return_inverse=True)[1] if a.site else None
    Q = read_table(a.nuisance, n, a.nuisance_cols) if a.nuisance else None
    keep = []
    for item in a.preserve:
        if item == 'age':
            keep.append(age[:, None])
        elif item == 'sex':
            if male is None:
                p.error(f"{a.mat} has no sex: give it as a table instead of 'sex'.")
            keep.append(male[:, None])
        else:
            keep.append(read_table(item, n))
    if a.group:
        g = read_labels(a.group, n)
        keep += [(g == lv)[:, None].astype(float) for lv in np.unique(g)[1:]]
    subject = read_labels(a.subject, n) if a.subject else None
    fit = parse_index(a.controls, n) if a.controls else None

    P = np.hstack(keep) if keep else None
    if a.folds == 0:            # no cross-fit: one model of the controls, applied to all
        f = np.ones(n, dtype=bool) if fit is None else fit

        def rows(x):
            return None if x is None else x[f]
        est = comcat(Y[:, f], rows(site), rows(Q), rows(P), mean_only=a.location_only,
                     subject=rows(subject) if a.within_subjects else None,
                     return_estimates=True)[4]
        H = comcat_from_training(Y, np.ones(n, dtype=int) if site is None else site, Q, P,
                                 estimates=est, unseen_batch='mean')
    else:
        H = comcat_crossfit(Y, site, Q, P, folds=a.folds, fit=fit, subject=subject,
                            subject_terms=a.within_subjects, mean_only=a.location_only)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_mat(a.mat, a.out, H, v73)
    print(f"{a.out}: {n} subjects, fitted on {n if fit is None else int(fit.sum())}, "
          f"{1 if site is None else np.unique(site).size} site(s), "
          f"{0 if Q is None else Q.shape[1]} nuisance and "
          f"{sum(k.shape[1] for k in keep)} preserved column(s), "
          f"{f'{a.folds} folds' if a.folds else 'no cross-fit (in-sample)'}")


if __name__ == '__main__':
    main()

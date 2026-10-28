"""E1: split-half accuracy of the site parameters on real data.

Each site's subjects are split at random into halves A and B (sites with >= 4
subjects).  ComCAT is fitted on A (L/S and EB) and on B (L/S).  Because B is
independent of A and its L/S estimates are unbiased, for an estimator E on A
    MSE(E_A, LS_B) = err(E)^2 + err(LS)^2
and by symmetry of the halves MSE(LS_A, LS_B) = 2 err(LS)^2, so the error of
each estimator against the population site parameters can be estimated.
Location: site differences to the size-weighted site mean, in units of the
voxel's SD.  Scale: within-site variance (delta * sigma^2) in units of the
voxel's variance.  Default delta (unbiased within-site sample variance).
"""
import os, sys, time, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comcat import comcat
from data import load
warnings.simplefilter('ignore', RuntimeWarning)

R = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def params(Y, site, iqm, age, eb, bounds, w):
    est = comcat(Y, site, iqm, age, eb=eb, smooth_term_bounds=bounds, return_estimates=True, nuisance_model='spline')[4]
    g = est['gamma_star_masked'] if est['eb'] else est['gamma_hat_masked']
    d = est['delta_star_masked'] if est['eb'] else est['delta_hat_masked']
    K = len(w)
    loc = g[:K] * est['std_pooled']
    loc = loc - w @ loc
    var = d[:K] * est['std_pooled'] ** 2
    return loc, var, est['ind_mask']


print(f"{'sample':11s} {'nuis':4s} {'sites':>5s} {'n/site/half':>11s} | "
      f"{'loc err LS':>10s} {'EB':>6s} {'reduction':>9s} | {'var err LS':>10s} {'EB':>6s} {'reduction':>9s}")
for name in ('ON-Harmony', 'Buchert', 'ABIDE', 'NormSample', 'ADNI'):
    d = load(name)
    labels, codes = np.unique(d['site'], return_inverse=True)
    sizes = np.bincount(codes)
    keep_site = sizes >= 4
    keep = keep_site[codes]
    Y, iqm, age = d['Y'][:, keep], d['iqm'][keep], d['age'][keep]
    _, codes = np.unique(codes[keep], return_inverse=True)
    K = codes.max() + 1
    w = np.bincount(codes) / len(codes)
    sdv = Y.std(1)
    bounds = [(iqm[:, j].min(), iqm[:, j].max()) for j in range(iqm.shape[1])]
    for nuis_label, use_nuis in (("none", False), ("IQM", True)):
        acc = {k: [] for k in ('ls_ab', 'eb_ab', 'vls_ab', 'veb_ab')}
        t0 = time.time()
        for r in range(R):
            rng = np.random.default_rng(100 + r)
            A = np.zeros(len(codes), bool)
            for k in range(K):
                idx = rng.permutation(np.flatnonzero(codes == k))
                A[idx[: len(idx) // 2]] = True
            Bm = ~A
            nz = (lambda m: iqm[m]) if use_nuis else (lambda m: None)
            ls_A, vls_A, mA = params(Y[:, A], codes[A], nz(A), age[A], False, bounds, w)
            eb_A, veb_A, _ = params(Y[:, A], codes[A], nz(A), age[A], True, bounds, w)
            ls_B, vls_B, mB = params(Y[:, Bm], codes[Bm], nz(Bm), age[Bm], False, bounds, w)
            # features valid in both halves
            vA, vB = np.flatnonzero(mA), np.flatnonzero(mB)
            common = np.intersect1d(vA, vB)
            ia, ib = np.searchsorted(vA, common), np.searchsorted(vB, common)
            s = sdv[common]
            acc['ls_ab'].append(np.mean(((ls_A[:, ia] - ls_B[:, ib]) / s) ** 2))
            acc['eb_ab'].append(np.mean(((eb_A[:, ia] - ls_B[:, ib]) / s) ** 2))
            acc['vls_ab'].append(np.mean(((vls_A[:, ia] - vls_B[:, ib]) / s ** 2) ** 2))
            acc['veb_ab'].append(np.mean(((veb_A[:, ia] - vls_B[:, ib]) / s ** 2) ** 2))
        m = {k: np.mean(v) for k, v in acc.items()}
        e_ls = m['ls_ab'] / 2;  e_eb = m['eb_ab'] - e_ls
        v_ls = m['vls_ab'] / 2; v_eb = m['veb_ab'] - v_ls
        # split-to-split SE of the reduction (from the per-split values)
        red = 1 - (np.array(acc['eb_ab']) - np.array(acc['ls_ab']) / 2) / (np.array(acc['ls_ab']) / 2)
        vred = 1 - (np.array(acc['veb_ab']) - np.array(acc['vls_ab']) / 2) / (np.array(acc['vls_ab']) / 2)
        print(f"{name:11s} {nuis_label:4s} {K:5d} {np.median(np.bincount(codes)) / 2:11.1f} | "
              f"{np.sqrt(e_ls):10.3f} {np.sqrt(max(e_eb, 0)):6.3f} {100 * (1 - e_eb / e_ls):6.0f}±{100 * red.std(ddof=1) / np.sqrt(R):.0f}% | "
              f"{np.sqrt(v_ls):10.3f} {np.sqrt(max(v_eb, 0)):6.3f} {100 * (1 - v_eb / v_ls):6.0f}±{100 * vred.std(ddof=1) / np.sqrt(R):.0f}%"
              f"   [{time.time() - t0:.0f}s]", flush=True)

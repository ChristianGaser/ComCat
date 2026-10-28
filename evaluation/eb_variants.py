"""EB location variants for the evaluation (monkeypatched into comcat).

'i'   : current implementation — shrink site coefficients at fixed IQM (ComBat-like)
'ii'  : shrink the site mean of the fitted site + nuisance part (always identified)
'iii' : like 'i', but the IQM effect is used only in directions identified by
        within-site variation (relative singular value > TOL); the rest is site effect
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import comcat as cc  # noqa: E402

_orig = cc._eb_site_effects
CTX = {}          # set before each comcat call: 'Zt' (n x n_Z basis), 'codes', 'variant', 'tol'


def _site_eff_shift(gamma_masked, site_var, n_batches, ref_level, extra):
    K = len(n_batches)
    g = gamma_masked[:K] + extra                       # site effect used for EB
    center = (n_batches / n_batches.sum()) @ g if ref_level is None else g[ref_level]
    skip = [i for i in range(K) if i == ref_level or n_batches[i] < 2]
    g_star, d_star, pri = cc._eb_estimates(g - center, site_var, n_batches, skip)
    out = gamma_masked.copy()
    out[:K] = gamma_masked[:K] - ((g - center) - g_star)  # subtract the EB shrinkage shift
    return out, d_star, pri


def patched(gamma_masked, site_var, n_batches, ref_level=None):
    v = CTX.get('variant', 'i')
    K = len(n_batches)
    gZ = gamma_masked[K:]
    if v == 'i' or gZ.shape[0] == 0:
        return _orig(gamma_masked, site_var, n_batches, ref_level)
    Zt, codes = CTX['Zt'], CTX['codes']
    Bm = (codes[:, None] == np.arange(K)).astype(float)
    M = (Bm.T @ Zt) / n_batches[:, None]               # site means of the basis (K x n_Z)
    if v == 'ii':
        return _site_eff_shift(gamma_masked, site_var, n_batches, ref_level, M @ gZ)
    if v == 'iii':
        Zw = Zt - Bm @ M                               # within-site centered basis
        _, s, Vt = np.linalg.svd(Zw, full_matrices=False)
        r = int(np.sum(s > CTX.get('tol', 1e-3) * s[0]))
        Vr = Vt[:r].T
        gZ_unid = gZ - Vr @ (Vr.T @ gZ)                # IQM effect in weakly identified directions
        return _site_eff_shift(gamma_masked, site_var, n_batches, ref_level, M @ gZ_unid)
    raise ValueError(v)


cc._eb_site_effects = patched

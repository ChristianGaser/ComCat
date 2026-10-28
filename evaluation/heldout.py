"""Held-out evaluation of ComCAT's IQM (nuisance) model and candidate fixes.

The variants (truncated SVD, ridge penalty, linear IQM terms, PCs, IQM effects
estimated within subjects) are not ComCAT options: they are injected here by
patching comcat internals (pinv, nuisance basis).  The harmonization itself is
comcat() / comcat_from_training().  See HeldOut-Evaluation.md.
"""
import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import comcat as cc  # noqa: E402
from comcat import comcat, comcat_from_training  # noqa: E402

warnings.simplefilter('ignore', RuntimeWarning)
_orig_pinv, _orig_basis = cc.pinv, cc._build_nuisance_basis
CTX = {}
NAMES = ["ICR", "NCR", "res_ECR", "res_RMS", "contrastr", "SIQR", "IQR"]
NONRED = [0, 1, 2, 3, 4]                     # drop SIQR (~res_ECR) and IQR (composite)


def _pinv(A):
    mode = CTX.get('solver', 'pinv')
    if mode == 'pinv':
        return _orig_pinv(A)
    if mode == 'trunc':                       # column-scaled truncated SVD
        s = np.linalg.norm(A, axis=0); s[s == 0] = 1
        return np.linalg.pinv(A / s, rcond=CTX['rcond']) / s[:, None]
    if mode == 'ridge':                       # ridge on the nuisance block only
        nb, nz = CTX['n_batch'], CTX['n_Z']
        if nz == 0 or A.shape[1] < nb + nz:
            return _orig_pinv(A)
        D = np.zeros(A.shape[1]); D[nb:nb + nz] = CTX['lam']
        return np.linalg.pinv(A.T @ A + np.diag(D)) @ A.T
    raise ValueError(mode)


def _basis(nuisance, bounds, gam_df, verbose=False, spline_constructors=None):
    if CTX.get('linear'):
        return np.asarray(nuisance, float), {}
    return _orig_basis(nuisance, bounds, gam_df, verbose, spline_constructors)


cc.pinv, cc._build_nuisance_basis = _pinv, _basis


def gcv_lambda(Y, Xu, Xp, grid=np.logspace(-3, 4, 36)):
    """Common ridge penalty for the nuisance block by GCV pooled over features
    (features standardized); unpenalized columns Xu are projected out."""
    Ys = (Y - Y.mean(1, keepdims=True)) / np.where(Y.std(1, keepdims=True) > 0, Y.std(1, keepdims=True), 1)
    U, s, _ = np.linalg.svd(Xu, full_matrices=False)
    Q = U[:, s > 1e-10 * s[0]]
    Xt = Xp - Q @ (Q.T @ Xp)
    yt = Ys.T - Q @ (Q.T @ Ys.T)
    U2, s2, _ = np.linalg.svd(Xt, full_matrices=False)
    keep = s2 > 1e-10 * s2[0]
    U2, s2 = U2[:, keep], s2[keep]
    c = U2.T @ yt
    rss0 = np.sum(yt ** 2) - np.sum(c ** 2)
    n = Y.shape[1]
    best = None
    for lam in grid:
        f = lam / (s2 ** 2 + lam)
        rss = rss0 + np.sum((f[:, None] * c) ** 2)
        df = Q.shape[1] + np.sum(s2 ** 2 / (s2 ** 2 + lam))
        g = n * rss / max(n - df, 1e-9) ** 2
        if best is None or g < best[0]:
            best = (g, lam, df)
    return best[1], best[2]


def prepare_iqm(iqm, tr, variant):
    """IQM columns for the variant: returns (Z_all, bounds, linear) with Z_all
    transformed with training parameters only."""
    cols = variant.get('cols', list(range(7)))
    Z = iqm[:, cols].astype(float)
    Z = Z[:, Z[tr].std(0) > 0]                       # drop columns constant in training
    kind = variant.get('basis', 'spline')
    if kind in ('linear', 'pca2', 'pca2lin'):
        mu, sd = Z[tr].mean(0), Z[tr].std(0)
        Z = (Z - mu) / sd
        if kind.startswith('pca2'):
            _, _, Vt = np.linalg.svd(Z[tr], full_matrices=False)
            Z = Z @ Vt[:2].T
    linear = kind in ('linear', 'pca2lin')
    bounds = [(Z[:, j].min(), Z[:, j].max()) for j in range(Z.shape[1])]
    return Z, bounds, linear


def harmonize(Y, codes, iqm, age, tr, te, variant, subj=None):
    """Fit on tr, harmonize tr (in-sample) and te (held-out).  Returns H_tr, H_te."""
    n_tr = int(tr.sum())
    gam_df = variant.get('gam_df') or min(10, max(5, n_tr // 30))
    multi = len(np.unique(codes[tr])) > 1
    batch_tr = codes[tr] if multi else None
    if variant.get('ui_single') and not multi:       # comcat_ui codes a missing batch as one site
        batch_tr = np.ones(n_tr, int)
    opts = dict(mean_only=variant.get('mean_only', True), eb=True,
                residual_delta=variant.get('residual_delta', False))
    use_iqm = variant.get('iqm', True)
    P_tr, P_te = age[tr][:, None], age[te][:, None]
    if variant.get('subjfe'):                        # IQM effects estimated within subjects
        lv = np.unique(subj[tr])
        D = (subj[tr][:, None] == lv[None, :]).astype(float)
        P_tr = np.hstack([P_tr, D - D.mean(0)])
        P_te = np.hstack([P_te, np.zeros((int(te.sum()), len(lv)))])
    CTX.clear()
    if use_iqm:
        Z, bounds, linear = prepare_iqm(iqm, tr, variant)
        CTX['linear'] = linear
        if variant.get('solver') == 'ridge':
            Xp = Z[tr] if linear else _orig_basis(Z[tr], bounds, gam_df)[0]
            nb = len(np.unique(codes[tr])) if multi else 1
            B = (np.unique(codes[tr], return_inverse=True)[1][:, None] == np.arange(nb)).astype(float)
            lam, _ = gcv_lambda(Y[:, tr], np.hstack([B, P_tr]), Xp)
            CTX.update(solver='ridge', lam=lam, n_batch=nb, n_Z=Xp.shape[1])
        elif variant.get('solver') == 'trunc':
            CTX.update(solver='trunc', rcond=variant['rcond'])
        Ztr, Zte = Z[tr], Z[te]
    else:
        Ztr = Zte = bounds = None
    Htr, *_, est = comcat(Y[:, tr], batch_tr, Ztr, P_tr, gam_df=gam_df, nuisance_model='spline',
                          smooth_term_bounds=bounds, return_estimates=True, **opts)
    bte = codes[te] if multi else np.ones(int(te.sum()), int)
    Hte = comcat_from_training(Y[:, te], bte, Zte, P_te, estimates=est)
    CTX.clear()
    return Htr, Hte


def wb(H, subj):
    """Per-feature within-subject and between-subject sums of squares."""
    W = np.zeros(H.shape[0]); B = np.zeros(H.shape[0])
    gm = H.mean(1)
    for s in np.unique(subj):
        h = H[:, subj == s]
        W += np.sum((h - h.mean(1, keepdims=True)) ** 2, axis=1)
        B += h.shape[1] * (h.mean(1) - gm) ** 2
    return W, B

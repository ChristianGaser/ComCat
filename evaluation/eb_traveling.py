"""E2: held-out traveling-subject consistency.

ON-Harmony: leave one subject out (10 folds); train on 9 subjects (6 scanners),
harmonize the held-out subject's scans with comcat_from_training.  Metric over
the 80 held-out harmonized scans: within-subject / total variance, mean over
voxels (lower = better; 1 - ICC).
Buchert: one subject on 116 scanners.  Hold out one random scan per scanner,
train on the rest, harmonize the held-out scans.  Metric: variance across the
116 held-out scans relative to the raw held-out data, mean over voxels
(lower = better).  5 random hold-out draws.
"""
import os, sys, time, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eb_variants                                  # noqa: F401 (patches comcat)
from eb_variants import CTX
from comcat import comcat, comcat_from_training, _build_nuisance_basis
from data import load, subject_ids
warnings.simplefilter('ignore', RuntimeWarning)

METHODS = (('L/S', False, 'i'), ('EB (i)', True, 'i'), ('EB (ii)', True, 'ii'), ('EB (iii)', True, 'iii'))
CONFIGS = (("sites only", False, {}), ("sites+IQM", True, {}),
           ("sites+IQM resdelta", True, dict(residual_delta=True)))


def fit_apply(Y, codes, iqm, age, tr, te, use_iqm, opts, eb, variant):
    bounds = [(iqm[:, j].min(), iqm[:, j].max()) for j in range(iqm.shape[1])]
    n_tr = tr.sum()
    gam_df = min(10, max(5, n_tr // 30))
    _, cod_tr = np.unique(codes[tr], return_inverse=True)
    CTX.update(variant=variant, codes=cod_tr,
               Zt=_build_nuisance_basis(iqm[tr], bounds, gam_df)[0] if use_iqm else None)
    nz = (lambda m: iqm[m]) if use_iqm else (lambda m: None)
    est = comcat(Y[:, tr], codes[tr], nz(tr), age[tr], gam_df=gam_df, eb=eb,
                 smooth_term_bounds=bounds, return_estimates=True, **opts)[4]
    return comcat_from_training(Y[:, te], codes[te], nz(te), age[te], estimates=est)


def onharmony():
    d = load('ON-Harmony')
    Y, iqm, age = d['Y'], d['iqm'], d['age']
    codes = np.unique(d['site'], return_inverse=True)[1]
    sid = subject_ids(d['files'])
    subs = np.unique(sid)

    def ratio(H):
        W = np.zeros(H.shape[0]); 
        for s in subs:
            h = H[:, sid == s]
            W += np.sum((h - h.mean(1, keepdims=True)) ** 2, axis=1)
        T = np.sum((H - H.mean(1, keepdims=True)) ** 2, axis=1)
        return np.mean(W / T)

    print(f"ON-Harmony (leave-one-subject-out): within-subject / total variance; raw = {ratio(Y):.3f}")
    for cfg, use_iqm, opts in CONFIGS:
        row = []
        for label, eb, variant in METHODS:
            if not use_iqm and variant != 'i':
                continue
            H = np.zeros_like(Y)
            for s in subs:
                te = sid == s
                H[:, te] = fit_apply(Y, codes, iqm, age, ~te, te, use_iqm, opts, eb, variant)
            row.append(f"{label} {ratio(H):.3f}")
        print(f"   {cfg:20s} " + "   ".join(row), flush=True)


def buchert(n_draw=5):
    d = load('Buchert')
    Y, iqm, age = d['Y'], d['iqm'], d['age']
    codes = np.unique(d['site'], return_inverse=True)[1]
    K = codes.max() + 1
    print(f"Buchert (one held-out scan per scanner, {n_draw} draws): variance across held-out scans / raw")
    res = {}
    for r in range(n_draw):
        rng = np.random.default_rng(200 + r)
        te = np.zeros(len(codes), bool)
        for k in range(K):
            te[rng.choice(np.flatnonzero(codes == k))] = True
        raw = Y[:, te].var(1)
        for cfg, use_iqm, opts in CONFIGS:
            for label, eb, variant in METHODS:
                if not use_iqm and variant != 'i':
                    continue
                H = fit_apply(Y, codes, iqm, age, ~te, te, use_iqm, opts, eb, variant)
                res.setdefault((cfg, label), []).append(np.mean(H.var(1) / raw))
    for cfg, use_iqm, _ in CONFIGS:
        row = [f"{label} {np.mean(res[(cfg, label)]):.3f}±{np.std(res[(cfg, label)], ddof=1) / np.sqrt(n_draw):.3f}"
               for label, _, variant in METHODS if (cfg, label) in res]
        print(f"   {cfg:20s} " + "   ".join(row), flush=True)


t0 = time.time()
onharmony()
buchert()
print(f"[{time.time() - t0:.0f}s]")

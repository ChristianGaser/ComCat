"""Pass 1: location model only (mean_only).  In-sample vs held-out.

Tohoku   (1 subject, 121 scans, no sites)    10-fold over scans      var(H)/var(raw)
Buchert  (1 subject, 116 scanners)           1 scan/scanner held out var(H)/var(raw), median of 10 draws
MR-ART   (148 subjects x 3 motion levels)    5-fold over subjects    W/W0, B/B0, motion bias
ON-Harm. (10 subjects x 6 scanners)          leave one subject out   W/W0, B/B0, W/T
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heldout import harmonize, wb, NONRED
from data import load, subject_ids

V = [
    ('no IQM (sites only)',      dict(iqm=False)),
    ('7 IQM spline (current)',   dict()),
    ('7 IQM spline trunc 1e-3',  dict(solver='trunc', rcond=1e-3)),
    ('7 IQM spline ridge-GCV',   dict(solver='ridge')),
    ('5 IQM spline',             dict(cols=NONRED)),
    ('IQR spline',               dict(cols=[6])),
    ('7 IQM linear',             dict(basis='linear')),
    ('5 IQM linear',             dict(cols=NONRED, basis='linear')),
    ('2 PCs spline',             dict(basis='pca2')),
    ('2 PCs linear',             dict(basis='pca2lin')),
    ('5 IQM lin, within-subj',   dict(cols=NONRED, basis='linear', subjfe=True)),
    ('5 IQM spline, within-subj', dict(cols=NONRED, subjfe=True)),
]
only = sys.argv[1:] or ['Tohoku', 'Buchert', 'MR-ART', 'ON-Harmony']


def ratio_var(H, Y):
    return np.mean(H.var(1) / Y.var(1))


def maxdev(H, Y):
    return np.max(np.abs(H - Y) / Y.std(1, keepdims=True))


for name in only:
    t0 = time.time()
    d = load(name)
    Y, iqm, age = d['Y'], d['iqm'], d['age']
    codes = np.unique(d['site'], return_inverse=True)[1]
    n = Y.shape[1]
    allm = np.ones(n, bool)
    if name in ('MR-ART', 'ON-Harmony'):
        subj = subject_ids(d['files'])
    else:
        subj = np.zeros(n, int)
    grouped = name in ('MR-ART', 'ON-Harmony')
    variants = [v for v in V if grouped or not v[1].get('subjfe')]

    # folds
    rng = np.random.default_rng(0)
    if name == 'Tohoku':
        folds = np.array_split(rng.permutation(n), 10)
        tests = [np.isin(np.arange(n), f) for f in folds]
    elif name == 'Buchert':
        tests = []
        rng = np.random.default_rng(1)
        for r in range(10):
            te = np.zeros(n, bool)
            for k in range(codes.max() + 1):
                te[rng.choice(np.flatnonzero(codes == k))] = True
            tests.append(te)
    else:
        subs = np.unique(subj)
        if name == 'MR-ART':
            groups = np.array_split(rng.permutation(subs), 5)
        else:
            groups = [[s] for s in subs]
        tests = [np.isin(subj, g) for g in groups]

    if grouped:
        W0, B0 = wb(Y, subj)
        cond = None
        if name == 'MR-ART':
            cond = np.array(['std' if 'standard' in f else ('hm2' if 'headmotion2' in f else 'hm1') for f in d['files']])
        print(f"\n== {name}: n={n}, raw W/T = {np.mean(W0 / (W0 + B0)):.3f}")
        print(f"   {'variant':28s} | {'in-sample W/W0  B/B0   W/T':>27s} | {'held-out W/W0  B/B0   W/T':>27s} | max|dH| " + (" | hm2 bias in/out" if cond is not None else ""))
    else:
        print(f"\n== {name}: n={n}, sites={codes.max() + 1}")
        print(f"   {'variant':28s} | {'in-sample var/raw':>17s} | {'held-out var/raw':>16s} | max|dH|")

    def bias(H):
        if cond is None:
            return np.nan
        return np.mean(np.abs(H[:, cond == 'hm2'].mean(1) - H[:, cond == 'std'].mean(1)) / Y.std(1))

    for label, var in variants:
        Hin, _ = harmonize(Y, codes, iqm, age, allm, allm, var, subj)
        if grouped:
            Hout = np.full_like(Y, np.nan)
            for te in tests:
                Hout[:, te] = harmonize(Y, codes, iqm, age, ~te, te, var, subj)[1]
            Wi, Bi = wb(Hin, subj); Wo, Bo = wb(Hout, subj)
            line = (f"   {label:28s} | {np.mean(Wi / W0):9.2f} {np.mean(Bi / B0):6.2f} {np.mean(Wi / (Wi + Bi)):6.3f} | "
                    f"{np.mean(Wo / W0):9.2f} {np.mean(Bo / B0):6.2f} {np.mean(Wo / (Wo + Bo)):6.3f} | {maxdev(Hout, Y):7.1f}")
            if cond is not None:
                line += f" | {bias(Y):.3f} raw, {bias(Hin):.3f} / {bias(Hout):.3f}"
            print(line, flush=True)
        elif name == 'Tohoku':
            Hout = np.full_like(Y, np.nan)
            for te in tests:
                Hout[:, te] = harmonize(Y, codes, iqm, age, ~te, te, var, subj)[1]
            print(f"   {label:28s} | {ratio_var(Hin, Y):17.3f} | {ratio_var(Hout, Y):16.3f} | {maxdev(Hout, Y):7.1f}", flush=True)
        else:   # Buchert: held-out scans per draw
            r_out, m_out = [], []
            for te in tests:
                Hte = harmonize(Y, codes, iqm, age, ~te, te, var, subj)[1]
                r_out.append(ratio_var(Hte, Y[:, te])); m_out.append(maxdev(Hte, Y[:, te]))
            print(f"   {label:28s} | {ratio_var(Hin, Y):17.3f} | {np.median(r_out):16.3f} | {np.median(m_out):7.1f}", flush=True)
    print(f"   [{time.time() - t0:.0f}s]")

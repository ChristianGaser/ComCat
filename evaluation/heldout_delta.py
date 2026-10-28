"""Pass 2: the variance (delta) step on top of the location model, in-sample and
held-out (HeldOut-Evaluation.md, Tables 1 and 3).

ON-Harmony (leave one subject out): W/W0, B/B0, W/T.
Buchert (one scan per scanner held out, median of 10 draws): var(H)/var(raw).
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heldout import harmonize, wb, NONRED
from data import load, subject_ids

V = [('no IQM (sites only)', dict(iqm=False)),
     ('7 IQM spline (current)', dict()),
     ('7 IQM spline ridge-GCV', dict(solver='ridge')),
     ('5 IQM linear', dict(cols=NONRED, basis='linear')),
     ('5 IQM lin, within-subj', dict(cols=NONRED, basis='linear', subjfe=True))]
MODES = (('mean_only', dict(mean_only=True)), ('default delta', dict(mean_only=False)),
         ('residual_delta', dict(mean_only=False, residual_delta=True)))

for name in (sys.argv[1:] or ['ON-Harmony', 'Buchert']):
    d = load(name)
    Y, iqm, age = d['Y'], d['iqm'], d['age']
    codes = np.unique(d['site'], return_inverse=True)[1]
    n = Y.shape[1]
    allm = np.ones(n, bool)
    if name == 'ON-Harmony':
        subj = subject_ids(d['files'])
        tests = [subj == s for s in np.unique(subj)]
        W0, B0 = wb(Y, subj)
        print(f"\n== {name}: W/W0  B/B0  W/T   (raw W/T {np.mean(W0 / (W0 + B0)):.3f})")
    else:
        subj = np.zeros(n, int)
        rng = np.random.default_rng(1)
        tests = []
        for r in range(10):
            te = np.zeros(n, bool)
            for k in range(codes.max() + 1):
                te[rng.choice(np.flatnonzero(codes == k))] = True
            tests.append(te)
        print(f"\n== {name}: var(H)/var(raw); held-out = median [range] of 10 draws")
    print(f"   {'variant':26s} {'mode':15s} | {'in-sample':>22s} | {'held-out':>28s}")
    for label, var in V:
        if name == 'Buchert' and var.get('subjfe'):
            continue
        for mode, mo in MODES:
            v = {**var, **mo}
            Hin = harmonize(Y, codes, iqm, age, allm, allm, v, subj)[0]
            if name == 'ON-Harmony':
                H = np.full_like(Y, np.nan)
                for te in tests:
                    H[:, te] = harmonize(Y, codes, iqm, age, ~te, te, v, subj)[1]
                Wi, Bi = wb(Hin, subj)
                W, B = wb(H, subj)
                cin = f"{np.mean(Wi / W0):6.2f} {np.mean(Bi / B0):5.2f} {np.mean(Wi / (Wi + Bi)):6.3f}"
                cout = f"{np.mean(W / W0):9.2f} {np.mean(B / B0):5.2f} {np.mean(W / (W + B)):6.3f}"
            else:
                r = np.array([np.mean(harmonize(Y, codes, iqm, age, ~te, te, v, subj)[1].var(1)
                                      / Y[:, te].var(1)) for te in tests])
                cin = f"{np.mean(Hin.var(1) / Y.var(1)):.3f}"
                cout = f"{np.median(r):.3g} [{r.min():.3g} - {r.max():.3g}]"
            print(f"   {label:26s} {mode:15s} | {cin:>22s} | {cout:>28s}", flush=True)

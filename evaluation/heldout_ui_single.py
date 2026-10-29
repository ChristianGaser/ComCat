"""Single-site samples as comcat_ui runs them with single_site_scale=True, the default
of earlier versions (missing batch = one site, so the variance step is applied):
current model, in-sample vs held-out."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heldout import harmonize, wb
from data import load, subject_ids

for name in ('Tohoku', 'MR-ART'):
    d = load(name)
    Y, iqm, age = d['Y'], d['iqm'], d['age']
    codes = np.zeros(Y.shape[1], int)
    n = Y.shape[1]; allm = np.ones(n, bool)
    rng = np.random.default_rng(0)
    if name == 'Tohoku':
        subj = np.zeros(n, int)
        tests = [np.isin(np.arange(n), f) for f in np.array_split(rng.permutation(n), 10)]
    else:
        subj = subject_ids(d['files'])
        tests = [np.isin(subj, g) for g in np.array_split(rng.permutation(np.unique(subj)), 5)]
    for label, var in (("comcat() batch=None (mean only)", dict()),
                       ("comcat_ui (one site, delta)", dict(ui_single=True, mean_only=False))):
        Hin, _ = harmonize(Y, codes, iqm, age, allm, allm, var, subj)
        Hout = np.full_like(Y, np.nan)
        for te in tests:
            Hout[:, te] = harmonize(Y, codes, iqm, age, ~te, te, var, subj)[1]
        if name == 'Tohoku':
            print(f"{name:7s} {label:33s} var/raw in-sample {np.mean(Hin.var(1) / Y.var(1)):.3f}  held-out {np.mean(Hout.var(1) / Y.var(1)):.3f}")
        else:
            W0, B0 = wb(Y, subj); Wi, Bi = wb(Hin, subj); Wo, Bo = wb(Hout, subj)
            print(f"{name:7s} {label:33s} in-sample W/W0 {np.mean(Wi / W0):.2f} B/B0 {np.mean(Bi / B0):.2f} | held-out W/W0 {np.mean(Wo / W0):.2f} B/B0 {np.mean(Bo / B0):.2f}")

"""Transfer test: linear IQM effects estimated WITHIN subjects on MR-ART (subject
fixed effects, 148 subjects x 3 motion levels) are applied unchanged to Tohoku,
Buchert and ON-Harmony: H = Y - (Z - mean Z) beta.  Nothing is fitted on the
target data.  Common voxel mask; IQMs: ICR, NCR, res_ECR, contrastr (res_RMS is
constant in MR-ART; SIQR/IQR are redundant)."""
import os, re, sys
import numpy as np
from scipy.io import loadmat
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import SAMPLES, T, subject_ids
from heldout import wb

COLS = [0, 1, 2, 4]


def raw(name):
    mat, bf, nf, pf = SAMPLES[name]
    d = loadmat(mat, squeeze_me=True, struct_as_record=False)
    files = [str(f) for f in np.ravel(d['sample'].data)]
    site = np.loadtxt(f'{T}/{bf}', dtype=str) if bf else np.full(len(files), '1')
    return np.asarray(d['Y'], float).T, np.loadtxt(f'{T}/{nf}')[:, COLS], site, files


Ym, Zm, _, fm = raw('MR-ART')
subj = np.array([re.search(r'sub-([^_/]+)', f).group(1) for f in fm])
targets = {k: raw(k) for k in ('Tohoku', 'Buchert', 'ON-Harmony')}
mask = (Ym.std(1) > 0) & (Ym.mean(1) >= 0.05)
for Y, *_ in targets.values():
    mask &= (Y.std(1) > 0) & (Y.mean(1) >= 0.05)
Ym = Ym[mask]

# within-subject (fixed-effects) estimator of the IQM slopes, per voxel
def demean(A):
    A = A.copy()
    for s in np.unique(subj):
        A[..., subj == s] -= A[..., subj == s].mean(-1, keepdims=True)
    return A
Zw, Yw = demean(Zm.T).T, demean(Ym)
beta = np.linalg.lstsq(Zw, Yw.T, rcond=None)[0]               # (4, p)
print(f"MR-ART within-subject fit: {mask.sum()} voxels; IQM SDs MR-ART {np.round(Zm.std(0), 3)}")

for name, (Y, Z, site, files) in targets.items():
    Y = Y[mask]
    H = Y - ((Z - Z.mean(0)) @ beta).T
    print(f"\n== {name}: IQM SDs {np.round(Z.std(0), 3)}")
    if name == 'ON-Harmony':
        sj = subject_ids(files)
        W0, B0 = wb(Y, sj); W, B = wb(H, sj)
        print(f"   raw         W/T {np.mean(W0 / (W0 + B0)):.3f}")
        print(f"   transferred W/W0 {np.mean(W / W0):.2f}  B/B0 {np.mean(B / B0):.2f}  W/T {np.mean(W / (W + B)):.3f}")
    else:
        print(f"   transferred var/raw {np.mean(H.var(1) / Y.var(1)):.3f}"
              + (f"   (within-scanner var/raw {np.mean(sum(((H[:, site == s] - H[:, site == s].mean(1, keepdims=True)) ** 2).sum(1) for s in np.unique(site)) / sum(((Y[:, site == s] - Y[:, site == s].mean(1, keepdims=True)) ** 2).sum(1) for s in np.unique(site))):.3f})" if name == 'Buchert' else ""))

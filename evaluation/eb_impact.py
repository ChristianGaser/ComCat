"""E3: in-sample change of the harmonized data when EB is switched on (full samples)."""
import os, sys, time, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comcat import comcat
from data import load
warnings.simplefilter('ignore', RuntimeWarning)

bins = [(2, 5), (6, 10), (11, 20), (21, 50), (51, 10_000)]
print("RMS change of harmonized data (eb on - off) per site, in units of the voxel's SD; median over sites in each size bin")
print(f"{'sample':11s} {'config':18s} " + " ".join(f"n={a}-{b if b < 10000 else '':<4}" for a, b in bins))
for name in ('ON-Harmony', 'Buchert', 'ABIDE', 'NormSample', 'ADNI'):
    d = load(name)
    Y, site, iqm, age = d['Y'], d['site'], d['iqm'], d['age']
    sd = Y.std(1, keepdims=True)
    labels, codes = np.unique(site, return_inverse=True)
    sizes = np.bincount(codes)
    for cfg, nuis, opts in (("sites only", None, {}), ("sites+IQM", iqm, {}),
                            ("sites+IQM resdelta", iqm, dict(residual_delta=True))):
        t0 = time.time()
        y1 = comcat(Y, site, nuis, age, eb=True, **opts)[0]
        y0 = comcat(Y, site, nuis, age, eb=False, **opts)[0]
        rms = np.array([np.sqrt(np.mean(((y1 - y0)[:, codes == k] / sd) ** 2)) for k in range(len(labels))])
        cells = []
        for a, b in bins:
            m = (sizes >= a) & (sizes <= b)
            cells.append(f"{np.median(rms[m]):.3f}({m.sum():2d})" if m.any() else " " * 9)
        print(f"{name:11s} {cfg:18s} " + " ".join(f"{c:9s}" for c in cells) + f"   [{time.time()-t0:.0f}s]")

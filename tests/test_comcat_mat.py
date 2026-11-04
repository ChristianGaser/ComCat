"""
test_comcat_mat.py
==================
comcat_mat.py: ComCAT for the data Y of a BA_data2mat mat-file (v5 and v7.3),
cross-fitted.  The harmonized file must hold what comcat_crossfit() gives for the
same design, keep all other fields, and the v7.3 file must give the same as v5.

Run:
    python tests/test_comcat_mat.py
"""

import os
import sys
import tempfile
import warnings

import numpy as np
from scipy.io import loadmat, savemat

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import comcat_mat  # noqa: E402
from comcat import comcat, comcat_crossfit, comcat_from_training  # noqa: E402

warnings.simplefilter('ignore', RuntimeWarning)


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:52s} {detail}")
    if not cond:
        raise AssertionError(f"{name} {detail}")


def simulate(d, n=90, p=40, seed=0):
    """v5 mat-file (Y subjects x features, age, male, a struct) and its tables in d."""
    rng = np.random.default_rng(seed)
    site = np.repeat([3, 7, 12], n // 3)                 # labels, not 0..k-1
    age = rng.uniform(20, 80, n)
    male = rng.integers(0, 2, n).astype(float)
    tiv = rng.normal(1500, 120, n)
    q = rng.normal(size=(n, 2))
    group = rng.integers(1, 4, n)
    Y = (0.8 - 0.004 * age[:, None] + 0.1 * q[:, :1] + 0.05 * (site[:, None] == 7)
         + 0.03 * rng.standard_normal((n, p)))
    savemat(os.path.join(d, 'test.mat'), dict(Y=Y.astype(np.float32), age=age[:, None],
                                              male=male[:, None], sample=dict(name='sim')))
    for name, x, fmt in (('site.txt', site, '%d'), ('iqm.txt', q, '%.6f'),
                         ('tiv.txt', tiv, '%.3f'), ('group.txt', group, '%d')):
        np.savetxt(os.path.join(d, name), x, fmt=fmt)
    return dict(Y=Y, site=site, age=age, male=male, tiv=np.loadtxt(os.path.join(d, 'tiv.txt')),
                q=np.loadtxt(os.path.join(d, 'iqm.txt')), group=group)


def test_min_scale_n():
    with tempfile.TemporaryDirectory() as d:
        s = simulate(d)
        f = lambda name: os.path.join(d, name)                  # noqa: E731
        code = np.unique(s['site'], return_inverse=True)[1]
        fit = np.zeros(90, bool)
        fit[np.r_[0:3, 30:90]] = True            # site 1: 3 controls, 2 in every fit
        Y = s['Y'].astype(np.float32).astype(float).T
        H = {}
        for n_min in (None, '2'):
            extra = [] if n_min is None else ['--min-scale-n', n_min]
            comcat_mat.main([f('test.mat'), '--site', f('site.txt'), '--folds', '3',
                             '--controls', '1:3,31:90', '--out', f(f'out/m{n_min}.mat')]
                            + extra)
            H[n_min] = loadmat(f(f'out/m{n_min}.mat'))['Y']
        H2 = comcat_crossfit(Y, code, None, None, folds=3, fit=fit, min_scale_n=2).T
        check("--min-scale-n 2 as comcat_crossfit(min_scale_n=2)",
              np.allclose(H['2'], H2, atol=1e-6))
        check("default 4 differs where a site has 2 subjects in the fit",
              not np.allclose(H[None], H['2'], atol=1e-6))


def write_v73(src, dst):
    """MATLAB v7.3 copy (HDF5 with the MATLAB header) of the numeric arrays of src."""
    import h5py
    m = loadmat(src)
    with h5py.File(dst, 'w', userblock_size=512) as h:
        for k in ('Y', 'age', 'male'):
            h.create_dataset(k, data=np.asarray(m[k]).T)
    with open(dst, 'r+b') as f:
        f.write(b'MATLAB 7.3 MAT-file, Platform: test, HDF5 schema 1.00 .'.ljust(116)
                + b' ' * 8 + b'\x00\x02IM')


def test_same_as_comcat_crossfit():
    with tempfile.TemporaryDirectory() as d:
        s = simulate(d)
        f = lambda name: os.path.join(d, name)                  # noqa: E731
        base = [f('test.mat'), '--site', f('site.txt'), '--nuisance', f('iqm.txt'),
                '--nuisance-cols', '1', '--folds', '3']
        comcat_mat.main(base + ['--preserve', 'age', 'sex', f('tiv.txt'), '--controls',
                                '1:20,31:50,61:80', '--out', f('out/h.mat')])
        m = loadmat(f('out/h.mat'))
        code = np.unique(s['site'], return_inverse=True)[1]
        fit = np.zeros(90, bool)
        fit[np.r_[0:20, 30:50, 60:80]] = True
        H = comcat_crossfit(s['Y'].astype(np.float32).astype(float).T, code, s['q'][:, :1],
                            np.c_[s['age'], s['male'], s['tiv']], folds=3, fit=fit).T
        check("Y as comcat_crossfit (v5)", np.allclose(m['Y'], H, atol=1e-6),
              f"max diff {np.max(np.abs(m['Y'] - H)):.1e}")
        check("other fields kept", str(m['sample']['name'][0, 0][0]) == 'sim'
              and m['Y'].dtype == np.float32 and m['Y'].shape == (90, 40))

        comcat_mat.main(base[:-2] + ['--folds', '0', '--controls', '1:60',
                                     '--out', f('out/i.mat')])
        Y = s['Y'].astype(np.float32).astype(float).T
        est = comcat(Y[:, :60], code[:60], s['q'][:60, :1], None, return_estimates=True)[4]
        Hi = comcat_from_training(Y, code, s['q'][:, :1], None, estimates=est,
                                  unseen_batch='mean').T
        check("--folds 0: controls' model applied to all", np.allclose(
            loadmat(f('out/i.mat'))['Y'], Hi, atol=1e-6))

        comcat_mat.main(base + ['--preserve', 'age', '--group', f('group.txt'),
                                '--out', f('out/g.mat')])
        dummies = (s['group'][:, None] == np.array([2, 3])[None]).astype(float)
        Hg = comcat_crossfit(s['Y'].astype(np.float32).astype(float).T, code, s['q'][:, :1],
                             np.c_[s['age'], dummies], folds=3).T
        check("--group: fit on all, groups preserved", np.allclose(loadmat(f('out/g.mat'))['Y'],
                                                                   Hg, atol=1e-6))
        for label, bad in (("--group with --controls", ['--group', f('group.txt'),
                                                        '--controls', '1:30']),
                           ("--within-subjects without --subject", ['--within-subjects'])):
            try:
                comcat_mat.main(base + bad + ['--out', f('out/x.mat')])
                ok = False
            except SystemExit:
                ok = True
            check(f"refused: {label}", ok)


def test_v73_same_as_v5():
    try:
        import h5py
    except ImportError:
        print("  [SKIP] v7.3: h5py not installed")
        return
    with tempfile.TemporaryDirectory() as d:
        simulate(d)
        f = lambda name: os.path.join(d, name)                  # noqa: E731
        write_v73(f('test.mat'), f('test73.mat'))
        args = ['--site', f('site.txt'), '--nuisance', f('iqm.txt'), '--preserve', 'age', 'sex']
        comcat_mat.main([f('test.mat')] + args + ['--out', f('v5.mat')])
        comcat_mat.main([f('test73.mat')] + args + ['--out', f('v73.mat')])
        with h5py.File(f('v73.mat'), 'r') as h:
            Y73 = h['Y'][()].T
        check("v7.3 as v5", np.array_equal(Y73, loadmat(f('v5.mat'))['Y']))


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} test groups passed.")

"""
test_comcat_edge_cases.py
=========================
Edge cases of comcat(), comcat_from_training() and comcat_crossfit(): features
without variance, missing values, missing covariates, subjects nested in sites
(subject terms), held-out samples of sites without fit samples, and a reference
site with a single subject in the cross-fit.

Run:
    python tests/test_comcat_edge_cases.py
"""

import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from comcat import comcat, comcat_from_training, comcat_crossfit  # noqa: E402


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:58s} {detail}")
    if not cond:
        raise AssertionError(f"{name} {detail}")


def simulate(sizes=(12, 12, 12), p=30, seed=0):
    rng = np.random.default_rng(seed)
    site = np.repeat(np.arange(len(sizes)), sizes)
    n = site.size
    age = rng.uniform(20, 70, n)
    Z = rng.normal(size=(n, 2))
    Y = rng.normal(size=(p, n)) + 0.5 * site + (Z @ rng.normal(0, 0.5, (2, p))).T
    return Y, site, Z, age


def test_constant_features():
    print("Features without variance keep their values")
    Y, site, Z, age = simulate()
    Y[0] = 5.0
    check("comcat", np.allclose(comcat(Y, site, Z, age)[0][0], 5.0))
    check("comcat with ref_batch", np.allclose(comcat(Y, site, Z, age, ref_batch=0)[0][0], 5.0))
    est = comcat(Y, site, Z, age, return_estimates=True)[4]
    new = Y[:, :4].copy()
    new[0] = [1.0, 2.0, 3.0, 4.0]
    check("comcat_from_training: new values kept",
          np.allclose(comcat_from_training(new, site[:4], Z[:4], age[:4], estimates=est)[0],
                      [1, 2, 3, 4]))


def test_missing_values():
    print("Missing values stay missing; missing covariates raise")
    Y, site, Z, age = simulate()
    Yn = Y.copy()
    Yn[3, 5] = np.nan
    H = comcat(Yn, site, Z, age)[0]
    check("comcat: the feature is missing for all", np.all(np.isnan(H[3]))
          and np.all(np.isfinite(np.delete(H, 3, axis=0))))
    est = comcat(Y, site, Z, age, return_estimates=True)[4]
    Hn = comcat_from_training(Yn, site, Z, age, estimates=est)
    check("comcat_from_training: only that value is missing",
          np.isnan(Hn[3, 5]) and np.sum(np.isnan(Hn)) == 1)
    Hc = comcat_crossfit(Yn, site, Z, age, folds=3)
    check("comcat_crossfit: as comcat", np.all(np.isnan(Hc[3]))
          and np.all(np.isfinite(np.delete(Hc, 3, axis=0))))
    Zn = Z.copy()
    Zn[2, 0] = np.nan
    for name, call in (("comcat", lambda: comcat(Y, site, Zn, age)),
                       ("comcat_from_training",
                        lambda: comcat_from_training(Y, site, Zn, age, estimates=est))):
        try:
            call()
            raised = False
        except ValueError:
            raised = True
        check(f"{name}: missing nuisance value raises", raised)


def test_nested_subjects():
    print("Subject terms with every subject at one site: warning")
    Y, site, Z, age = simulate()
    subject = np.arange(site.size) // 2               # two scans per subject, one site
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        comcat(Y, site, Z, age, subject=subject)
    check("warning issued", any('not identified' in str(x.message) for x in w))
    crossed = np.arange(site.size) % 12               # every subject at all three sites
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        comcat(Y, site, Z, age, subject=crossed)
    check("no warning when subjects are crossed with sites",
          not any('not identified' in str(x.message) for x in w))


def test_crossfit_unseen_and_reference():
    print("Cross-fit: sites without fit samples in the other folds; reference site")
    Y, site, Z, age = simulate((12, 12, 1))
    last = site == 2
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        H = comcat_crossfit(Y, site, Z, age, folds=3)
    check("warning about held-out samples without site correction",
          any('no site correction' in str(x.message) for x in w))
    Y1 = Y.copy()
    Y1[:, last] += 1.0
    gain = np.mean(comcat_crossfit(Y1, site, Z, age, folds=3)[:, last] - H[:, last])
    check("held out: the scan keeps its deviations (no in-sample model)",
          abs(gain - 1) < 1e-6, f"+1 -> {gain:+.3f}")
    Y, site, Z, age = simulate((1, 12, 12))
    Hr = comcat_crossfit(Y, site, Z, age, folds=3, ref_batch=0)
    check("ref_batch with one reference subject: kept, all finite",
          np.allclose(Hr[:, 0], Y[:, 0]) and np.all(np.isfinite(Hr)))


def test_nonnegative_input_stays_nonnegative():
    print("Input without negative values: no negative harmonized values")
    Y, site, Z, age = simulate()
    Y = np.abs(0.05 * Y)                           # densities near 0, e.g. at the GM border
    Y[0, 0], Y[-1, 0] = np.nan, 0.0
    Ys = Y.copy()
    Ys[-1, 0] = -1e-12                             # signed input: not clipped
    for label, f in (
            ("comcat()", lambda X: comcat(X, site, Z, age)[0]),
            ("comcat_from_training()", lambda X: comcat_from_training(
                X, site, Z, age, estimates=comcat(X, site, Z, age, return_estimates=True)[4])),
            ("comcat_crossfit()", lambda X: comcat_crossfit(X, site, Z, age, folds=3))):
        H, Hs = f(Y), f(Ys)
        check(f"{label}: no negative values, NaN kept",
              np.nanmin(H) >= 0 and np.isnan(H[0, 0]), f"min {np.nanmin(H):.3f}")
        check(f"{label}: signed input unchanged, else clipped at 0",
              np.nanmin(Hs[:-1]) < 0 and np.allclose(H[:-1], np.maximum(Hs[:-1], 0),
                                                     equal_nan=True))
    # clip=False: the affine harmonization (negative values kept)
    est = comcat(Y, site, Z, age, return_estimates=True)[4]
    for label, H, Hu in (
            ("comcat()", comcat(Y, site, Z, age)[0], comcat(Y, site, Z, age, clip=False)[0]),
            ("comcat_from_training()",
             comcat_from_training(Y, site, Z, age, estimates=est),
             comcat_from_training(Y, site, Z, age, estimates=est, clip=False)),
            ("comcat_crossfit()", comcat_crossfit(Y, site, Z, age, folds=3),
             comcat_crossfit(Y, site, Z, age, folds=3, clip=False))):
        check(f"{label}, clip=False: negative values kept, else the same",
              np.nanmin(Hu) < 0 and np.allclose(H, np.maximum(Hu, 0), equal_nan=True),
              f"min {np.nanmin(Hu):.3f}")


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} test groups passed.")

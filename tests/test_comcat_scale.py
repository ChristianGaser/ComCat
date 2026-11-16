"""
test_comcat_scale.py
====================
Tests for the scale step of the sites (comcat.py, several sites, mean_only=False), which
is that of ComBat: site variances from the residuals after the site and nuisance effects,
with ComBat's empirical Bayes posterior.

1. Known property of ComBat's site variances, kept on purpose (ComCAT without nuisance
   covariates is ComBat): every site variance has n_i - 1 degrees of freedom, the pooled
   variance n, and the posterior divides by n_i.  With few subjects per site, the site
   variances are too large on average, and the scale step shrinks the harmonized
   residuals even when the sites do not differ in variance; with 40 subjects per site
   it does not.  mean_only=True avoids it.
2. Unequal site variances: the scale step reduces the differences of the residual SDs
   between the sites held out, and the site variances follow the true ones.
3. comcat_from_training applies the estimates.
4. A site with a single subject gets no site correction.

Run:
    python tests/test_comcat_scale.py
"""

import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from comcat import comcat, comcat_crossfit, comcat_from_training, _make_folds  # noqa: E402

warnings.simplefilter('ignore', RuntimeWarning)


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:60s} {detail}")
    if not cond:
        raise AssertionError(f"{name} {detail}")


def simulate(sizes, seed=0, p=200, scale_sd=0.0):
    """Sites with additive effects and noise SD exp(N(0, scale_sd)) (equal for
    scale_sd=0), two IQMs whose distribution differs between sites with linear and
    non-linear effects, and an age effect.  Returns Y (p x n), site, IQMs, age, the
    true signal (age effect, p x n) and the noise SD of every site."""
    rng = np.random.default_rng(seed)
    sizes = np.asarray(sizes)
    K, n = len(sizes), int(np.sum(sizes))
    site = np.repeat(np.arange(K), sizes)
    Z = rng.normal(0, 1, (n, 2)) + rng.normal(0, 0.5, (K, 2))[site]
    age = rng.uniform(20, 80, n)
    sd = np.exp(rng.normal(0, scale_sd, K))
    signal = np.outer(np.full(p, 0.02), age - 50)
    Y = (signal + rng.normal(0, 1, (p, K))[:, site]
         + (Z[:, :1] + 0.5 * np.tanh(Z[:, 1:])) .T * rng.normal(0.5, 0.2, (p, 1))
         + rng.normal(0, 1, (p, n)) * sd[site])
    return Y, site, Z, age, signal, sd


def site_means(D, site):
    """Mean over features of D (K x p) per site."""
    return np.array([D[i].mean() for i in range(len(np.unique(site)))])


# ---------------------------------------------------------------------------

def test_shrinkage_with_small_sites():
    print("Equal site variances: ComBat's scale step shrinks the residuals of small sites")

    def ratios(sizes, seed):
        Y, site, Z, age, signal, _ = simulate(sizes, seed=seed)
        n = Y.shape[1]

        def resid_var(H):
            return float(np.mean((H - signal).var(axis=1)))

        folds = _make_folds(site, None, np.ones(n, bool), 4, 0)
        v = {}
        for name, opts in (('loc', dict(mean_only=True)), ('ls', dict())):
            v[name] = resid_var(comcat(Y, site, Z, age, **opts)[0])
            v[name + '_cv'] = resid_var(comcat_crossfit(Y, site, Z, age, folds=folds, **opts))
        return v['ls'] / v['loc'], v['ls_cv'] / v['loc_cv']

    r_in, r_cv = ratios([4] * 30, seed=2)
    check("30 sites of 4, held out: scale step / location only < 0.92", r_cv < 0.92,
          f"{r_cv:.3f}")
    check("30 sites of 4, in-sample: scale step / location only < 0.85", r_in < 0.85,
          f"{r_in:.3f}")
    r_in, r_cv = ratios([40] * 6, seed=2)
    check("6 sites of 40, held out: scale step / location only in 0.97-1.03",
          abs(r_cv - 1) < 0.03, f"{r_cv:.3f}")
    check("6 sites of 40, in-sample: in 0.95-1.03", 0.95 < r_in < 1.03, f"{r_in:.3f}")


def test_scale_step_equalizes_sites():
    print("Unequal site variances: the scale step equalizes the sites")
    sizes = [5, 8, 10, 15, 20, 25, 30, 40]
    Y, site, Z, age, signal, sd = simulate(sizes, seed=3, scale_sd=0.4)
    n, K = Y.shape[1], len(sizes)
    folds = _make_folds(site, None, np.ones(n, bool), 5, 0)

    def log_sd_spread(H):
        R = H - signal
        lsd = np.array([np.log(R[:, site == i].std(axis=1)).mean() for i in range(K)])
        return float(np.std(lsd))

    loc = comcat_crossfit(Y, site, Z, age, folds=folds, mean_only=True)
    ls = comcat_crossfit(Y, site, Z, age, folds=folds)
    s_loc, s_ls = log_sd_spread(loc), log_sd_spread(ls)
    check("held out: spread of log site SDs halved", s_ls < 0.5 * s_loc,
          f"{s_loc:.3f} -> {s_ls:.3f}")
    est = comcat(Y, site, Z, age, return_estimates=True)[4]
    d = np.log(site_means(est['delta_star_masked'][:K], site))
    r = np.corrcoef(d, 2 * np.log(sd))[0, 1]
    check("log site variances follow the true ones (r > 0.9)", r > 0.9, f"r = {r:.3f}")


def test_estimates_and_from_training():
    print("comcat_from_training applies the estimates")
    Y, site, Z, age, *_ = simulate([4, 6, 10, 20], seed=4)
    for opts in (dict(), dict(mean_only=True)):
        Yh, *_, est = comcat(Y, site, Z, age, return_estimates=True, **opts)
        err = np.max(np.abs(Yh - comcat_from_training(Y, site, Z, age, estimates=est)))
        check(f"{opts}: from_training == comcat", err < 1e-8, f"max_abs={err:.1e}")
    d = comcat(Y, site, Z, age, return_estimates=True)[4]['delta_star_masked'][:4]
    check("every site with at least two subjects gets a site variance",
          np.all(np.isfinite(d)) and not np.any(np.all(np.isclose(d, 1), axis=1)))


def test_single_subject_site():
    print("A site with one subject gets no site correction (its effect is not identified)")
    Y, site, Z, age, *_ = simulate([1, 10, 20], seed=5)
    one = site == np.unique(site)[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for mean_only in (False, True):
            tag = f"mean_only={mean_only}"
            opts = dict(mean_only=mean_only)
            Yh, *_, est = comcat(Y, site, Z, age, return_estimates=True, **opts)
            d = est['delta_star_masked'][:3]
            check(f"{tag}: finite site variances, 1 for the single subject",
                  np.all(np.isfinite(d)) and np.allclose(d[0], 1))
            # in-sample, the subject keeps its deviations (it is not replaced by its
            # predicted values, as it would be if its residual were the site effect)
            Y1 = Y.copy()
            Y1[:, one] += 1.0
            gain = np.mean(comcat(Y1, site, Z, age, **opts)[0][:, one] - Yh[:, one])
            check(f"{tag}: the subject keeps its deviations", abs(gain - 1) < 0.1,
                  f"+1 on its data -> {gain:+.3f}")
            # new scans of that site: as at an unseen site, i.e. not shifted by the
            # single subject's deviations
            Yn = Y[:, one] + np.linspace(-1, 1, Y.shape[0])[:, None]
            Ha = comcat_from_training(Yn, site[one], Z[one], age[one], estimates=est)
            Hu = comcat_from_training(Yn, np.array([99]), Z[one], age[one], estimates=est,
                                      unseen_batch='mean')
            check(f"{tag}: new scans as at an unseen site", np.allclose(Ha, Hu),
                  f"max_abs={np.max(np.abs(Ha - Hu)):.1e}")


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} test groups passed.")

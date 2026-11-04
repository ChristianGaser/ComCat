"""
test_comcat_scale.py
====================
Tests for the scale step of the sites (comcat.py, several sites, mean_only=False).

ComBat estimates every site variance with n_i - 1 degrees of freedom but the pooled
variance it is compared with with n, and its empirical Bayes posterior divides by n_i.
With few subjects per site, the site variances are then too large on average, and the
scale step shrinks the harmonized residuals even when the sites do not differ in
variance.  By default, ComCAT uses df-consistent site variances (residual df n_i minus
the leverage of the site's subjects, and the matching EB posterior); combat_compatible
=True gives the ComBat estimates.

1. Equal site variances: the df-consistent site variances are about 1 at every site
   size; with many small sites, the ComBat ones are inflated.
2. Equal site variances: the scale step keeps the residual variance of the location-only
   correction, in-sample and held out (cross-fitted); with combat_compatible it shrinks it.
3. Unequal site variances: the scale step equalizes the residual SDs of the sites held
   out, and the site variances follow the true ones.
4. The estimates record the setting, and comcat_from_training applies them.

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

def test_site_variances_unbiased():
    print("Equal site variances: df-consistent site variances ~ 1 at every site size")
    sizes = [3, 4, 6, 10, 20, 40]
    Y, site, Z, age, *_ = simulate(sizes, seed=1)
    for eb in (False, True):
        key = 'delta_star_masked' if eb else 'delta_hat_masked'
        est = comcat(Y, site, Z, age, eb=eb, return_estimates=True)[4]
        m = site_means(est[key][:len(sizes)], site)
        check(f"eb={eb}: mean site variance per site in 0.85-1.15",
              np.all(np.abs(m - 1) < 0.15), np.round(m, 2).tolist())
    # many small sites: ComBat's site variances are inflated by about n / (n - k)
    sizes = [4] * 30
    Y, site, Z, age, *_ = simulate(sizes, seed=5)
    for cc in (False, True):
        est = comcat(Y, site, Z, age, combat_compatible=cc, residual_delta=True,
                     return_estimates=True)[4]
        m = float(np.mean(est['delta_star_masked'][:len(sizes)]))
        if cc:
            check("30 sites of 4, combat_compatible: mean site variance > 1.2", m > 1.2,
                  f"{m:.3f}")
        else:
            check("30 sites of 4, default: mean site variance in 0.9-1.1",
                  abs(m - 1) < 0.1, f"{m:.3f}")


def test_scale_step_keeps_residual_variance():
    print("Equal site variances: the scale step keeps the residual variance")
    sizes = [4] * 30
    Y, site, Z, age, signal, _ = simulate(sizes, seed=2)
    n = Y.shape[1]

    def resid_var(H):
        return float(np.mean((H - signal).var(axis=1)))

    v = {}
    for name, opts in (('loc', dict(mean_only=True)), ('ls', dict()),
                       ('combat', dict(combat_compatible=True, residual_delta=True))):
        v[name] = resid_var(comcat(Y, site, Z, age, **opts)[0])
        folds = _make_folds(site, None, np.ones(n, bool), 4, 0)
        v[name + '_cv'] = resid_var(comcat_crossfit(Y, site, Z, age, folds=folds, **opts))
    r_new, r_old = v['ls_cv'] / v['loc_cv'], v['combat_cv'] / v['loc_cv']
    check("held out: default / location only in 0.95-1.05", abs(r_new - 1) < 0.05,
          f"{r_new:.3f}")
    check("held out: combat_compatible / location only < 0.92", r_old < 0.92,
          f"{r_old:.3f}")
    r_new, r_old = v['ls'] / v['loc'], v['combat'] / v['loc']
    check("in-sample: default / location only in 0.95-1.05", abs(r_new - 1) < 0.05,
          f"{r_new:.3f}")
    check("in-sample: combat_compatible / location only < 0.85", r_old < 0.85,
          f"{r_old:.3f}")


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
    print("Estimates record the setting; comcat_from_training applies them")
    Y, site, Z, age, *_ = simulate([4, 6, 10, 20], seed=4)
    bounds = [(z.min(), z.max()) for z in Z.T]
    for cc in (False, True):
        Yh, *_, est = comcat(Y, site, Z, age, combat_compatible=cc, return_estimates=True,
                             smooth_term_bounds=bounds)
        check(f"combat_compatible={cc}: recorded", est['combat_compatible'] == cc
              and (est['site_df'] is None) == cc)
        err = np.max(np.abs(Yh - comcat_from_training(Y, site, Z, age, estimates=est)))
        check(f"combat_compatible={cc}: from_training == comcat", err < 1e-8,
              f"max_abs={err:.1e}")
    est = comcat(Y, site, Z, age, return_estimates=True)[4]
    check("site df below the site sizes", np.all(est['site_df'] < np.array([4, 6, 10, 20])),
          np.round(est['site_df'], 2).tolist())


def test_single_subject_site():
    print("A site with one subject gets no site correction (its effect is not identified)")
    Y, site, Z, age, *_ = simulate([1, 10, 20], seed=5)
    one = site == np.unique(site)[0]
    for cc in (False, True):
        for mean_only in (False, True):
            tag = f"combat_compatible={cc}, mean_only={mean_only}"
            opts = dict(combat_compatible=cc, mean_only=mean_only)
            Yh, *_, est = comcat(Y, site, Z, age, return_estimates=True, **opts)
            d = est['delta_star_masked'][:3]
            check(f"{tag}: finite site variances, 1 for the single subject",
                  np.all(np.isfinite(d)) and np.allclose(d[0], 1))
            # in-sample, the subject keeps its deviations (before: replaced by its
            # predicted values, as its residual was removed as the site effect)
            Y1 = Y.copy()
            Y1[:, one] += 1.0
            gain = np.mean(comcat(Y1, site, Z, age, **opts)[0][:, one] - Yh[:, one])
            check(f"{tag}: the subject keeps its deviations", abs(gain - 1) < 0.1,
                  f"+1 on its data -> {gain:+.3f}")
            # new scans of that site: as at an unseen site, i.e. not shifted by the
            # single subject's deviations (and finite: before, ComBat's variance was NaN)
            Yn = Y[:, one] + np.linspace(-1, 1, Y.shape[0])[:, None]
            Ha = comcat_from_training(Yn, site[one], Z[one], age[one], estimates=est)
            Hu = comcat_from_training(Yn, np.array([99]), Z[one], age[one], estimates=est,
                                      unseen_batch='mean')
            check(f"{tag}: new scans as at an unseen site", np.allclose(Ha, Hu),
                  f"max_abs={np.max(np.abs(Ha - Hu)):.1e}")


def test_small_sites_location_only():
    print("Sites with fewer than min_scale_n subjects: location only (no scale step)")
    Y, site, Z, age, *_ = simulate([3, 4, 20], seed=6, scale_sd=0.5)
    small = site == np.unique(site)[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for cc in (False, True):
            tag = f"combat_compatible={cc}"
            Yh, *_, est = comcat(Y, site, Z, age, return_estimates=True, combat_compatible=cc)
            Y2 = comcat(Y, site, Z, age, combat_compatible=cc, min_scale_n=2)[0]
            d = est['delta_star_masked'][:3]
            if not cc:
                check(f"{tag}: site variance 1 with 3 subjects, estimated with 4 and 20",
                      np.allclose(d[0], 1) and not np.allclose(d[1], 1)
                      and not np.allclose(d[2], 1))
                check(f"{tag}: the other sites as without the rule",
                      np.allclose(Yh[:, ~small], Y2[:, ~small]))
                check(f"{tag}: the small site differs from its scale step",
                      not np.allclose(Yh[:, small], Y2[:, small]))
            else:
                check(f"{tag}: ComBat unchanged", np.allclose(Yh, Y2) and not np.allclose(d[0], 1))
            Hf = comcat_from_training(Y, site, Z, age, estimates=est)
            check(f"{tag}: comcat_from_training applies the same", np.allclose(Hf, Yh),
                  f"max_abs={np.max(np.abs(Hf - Yh)):.1e}")


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} test groups passed.")

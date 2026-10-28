"""
test_comcat_eb.py
=================
Tests for the empirical Bayes estimation of the site effects in comcat.py
(eb=True, the default).  No MATLAB reference files are needed.

1. Without nuisance covariates, ComCAT equals ComBat: combat_family.combat_eb(),
   the port of the R package ComBatFamily, with and without EB and mean_only.
2. EB is not used with a single site, and eb=False stores no EB estimates.
3. Simulation with known site effects drawn from site-specific priors, small
   sites and an IQM whose distribution differs between sites and whose effect on
   the mean is strong and differs between features: the EB estimates of the site
   location and scale, and the harmonized data, are closer to the truth than the
   L/S estimates.  The site coefficients of ComCAT contain a per-feature offset
   from the nuisance model; shrinking them without first expressing them as site
   differences loses most of the gain in location, which this test detects.
4. ref_batch: the reference site and its variance are not shrunk.
5. comcat_from_training() applies the EB estimates.
6. Sites with a single subject are left out of EB.

Run:
    python tests/test_comcat_eb.py
"""

import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from comcat import comcat, comcat_from_training  # noqa: E402
from combat_family import combat_eb  # noqa: E402

warnings.simplefilter('ignore', RuntimeWarning)


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:52s} {detail}")
    if not cond:
        raise AssertionError(f"{name} {detail}")


def simulate(seed=0, p=600, sizes=(5, 6, 8, 10, 12, 15, 25, 40), iqm_load=2.0,
             scale_effects=True):
    """Site effects drawn from site-specific priors: location
    gamma_ig = mu_i + 0.3 u_ig, noise SD exp(s_i + 0.15 v_ig).  One IQM z with
    site-dependent distribution and a non-linear mean effect tanh(z) * load_g
    (load_g ~ N(iqm_load, (iqm_load / 2)^2)); age effect 0.02 per year.

    Returns Y (p x n), batch, z, age, sizes, true location gamma (K x p),
    true noise SD (K x p) and the site- and IQM-free data Y0 (p x n)."""
    rng = np.random.default_rng(seed)
    sizes = np.asarray(sizes)
    n, K = sizes.sum(), len(sizes)
    batch = np.repeat(np.arange(K), sizes)
    age = rng.uniform(20, 80, n)
    z = rng.normal(size=n) + 0.5 * (batch % 4)
    load = rng.normal(iqm_load, 0.5 * iqm_load, p)
    gamma = rng.normal(0, 0.5, (K, 1)) + 0.3 * rng.normal(size=(K, p))
    if scale_effects:
        sd = np.exp(rng.normal(0, 0.2, (K, 1)) + 0.15 * rng.normal(size=(K, p)))
    else:
        sd = np.ones((K, p))
    eps = rng.normal(size=(n, p))
    Y = (0.02 * age[:, None] + np.outer(np.tanh(z), load) + gamma[batch] + sd[batch] * eps).T
    Y0 = (0.02 * age[:, None] + sd.mean(0) * eps).T
    return Y, batch, z, age, sizes, gamma, sd, Y0


def site_errors(est, sizes, gamma, sd):
    """RMS error over features of the applied site location (as differences to
    the size-weighted mean of the sites) and of the log site variance, in data
    units, per site."""
    use_eb = est['eb']
    g = est['gamma_star_masked'] if use_eb else est['gamma_hat_masked']
    d = est['delta_star_masked'] if use_eb else est['delta_hat_masked']
    K = len(sizes)
    w = sizes / sizes.sum()
    loc = g[:K] * est['std_pooled']
    loc = loc - w @ loc
    loc_err = np.sqrt(np.mean((loc - (gamma - w @ gamma)) ** 2, axis=1))
    var = d[:K] * est['std_pooled'] ** 2
    var_err = np.sqrt(np.mean((np.log(var) - np.log(sd ** 2)) ** 2, axis=1))
    return loc_err, var_err


# ---------------------------------------------------------------------------

def test_reduces_to_combat():
    print("Without nuisance, ComCAT equals ComBat (combat_family.combat_eb)")
    Y, batch, _, age, *_ = simulate(seed=1, p=300)
    sex = np.random.default_rng(1).integers(0, 2, len(batch)).astype(float)
    X = np.column_stack([age, sex])
    for mean_only in (False, True):
        for eb in (True, False):
            a = comcat(Y, batch, None, X, eb=eb, mean_only=mean_only)[0]
            b = combat_eb(Y, batch, X, eb=eb, mean_only=mean_only)[0]
            err = np.max(np.abs(a - b))
            check(f"eb={eb}, mean_only={mean_only}", err < 1e-10, f"max_abs={err:.1e}")


def test_single_site_and_eb_false():
    print("EB is not used with a single site; eb=False stores no EB estimates")
    Y, batch, z, age, *_ = simulate(seed=2, p=200)
    for label, b in (("batch=None", None), ("one site", np.zeros(len(batch), int))):
        r1 = comcat(Y, b, z, age, gam_df=5, return_estimates=True)
        r0 = comcat(Y, b, z, age, gam_df=5, eb=False)
        same = all(np.array_equal(u, v, equal_nan=True) for u, v in zip(r1[:4], r0))
        check(f"{label}: identical to eb=False", same and not r1[4]['eb'])
    est = comcat(Y, batch, z, age, gam_df=5, eb=False, return_estimates=True)[4]
    check("eb=False: no EB estimates",
          not est['eb'] and est['gamma_star_masked'] is None and est['eb_priors'] is None)


def test_eb_closer_to_truth():
    print("EB site effects and harmonized data are closer to the truth (small sites, IQM)")
    acc = {}
    for seed in range(3):
        Y, batch, z, age, sizes, gamma, sd, _ = simulate(seed=seed)
        for eb in (False, True):
            est = comcat(Y, batch, z, age, gam_df=5, residual_delta=True, eb=eb,
                         return_estimates=True)[4]
            loc, var = site_errors(est, sizes, gamma, sd)
            acc.setdefault(('loc', eb), []).append(loc)
            acc.setdefault(('var', eb), []).append(var)
        # harmonized data against the site- and IQM-free data (no scale effects,
        # so that the target scale is common to all sites)
        Y, batch, z, age, sizes, gamma, sd, Y0 = simulate(seed=seed, scale_effects=False)
        for eb in (False, True):
            E = comcat(Y, batch, z, age, gam_df=5, residual_delta=True, eb=eb)[0] - Y0
            E -= E.mean(1, keepdims=True)
            acc.setdefault(('data', eb), []).append(np.sqrt(np.mean(E ** 2)))

    for key, label in (('loc', "site location"), ('var', "log site variance"),
                       ('data', "harmonized data")):
        ls, eb = np.mean(acc[(key, False)]), np.mean(acc[(key, True)])
        limit = 0.7 if key == 'var' else 0.85
        check(f"{label}: EB error < {limit} x L/S error", eb < limit * ls,
              f"RMS error {ls:.3f} -> {eb:.3f}")
    # the gain is largest for the smallest site
    ls, eb = np.mean(acc[('loc', False)], 0), np.mean(acc[('loc', True)], 0)
    check("smallest site gains most (location)",
          (ls - eb)[0] == np.max(ls - eb), f"n=5: {ls[0]:.3f} -> {eb[0]:.3f}, "
          f"n=40: {ls[-1]:.3f} -> {eb[-1]:.3f}")


def test_ref_batch_eb():
    print("ref_batch with EB: reference site and its variance are not shrunk")
    Y, batch, z, age, sizes, *_ = simulate(seed=4, p=300)
    bounds = [(z.min(), z.max())]
    for opts in (dict(), dict(residual_delta=True), dict(nuisance_scale=True)):
        Yh, *_, est = comcat(Y, batch, z, age, gam_df=5, ref_batch=3, return_estimates=True,
                             smooth_term_bounds=bounds, **opts)
        ref = batch == 3
        check(f"{opts}: reference data unchanged",
              np.max(np.abs(Yh[:, ref] - Y[:, ref])) < 1e-10)
        check(f"{opts}: reference delta not shrunk",
              np.array_equal(est['delta_star_masked'][3], est['delta_hat_masked'][3])
              and np.isnan(est['eb_priors']['t2'][3]))
        others = [i for i in range(len(sizes)) if i != 3]
        check(f"{opts}: other sites shrunk",
              not np.allclose(est['gamma_star_masked'][others], est['gamma_hat_masked'][others]))
        err = np.max(np.abs(Yh - comcat_from_training(Y, batch, z, age, estimates=est)))
        check(f"{opts}: from_training == comcat", err < 1e-8, f"max_abs={err:.1e}")


def test_from_training_eb():
    print("comcat_from_training applies the EB estimates")
    Y, batch, z, age, *_ = simulate(seed=5, p=300)
    bounds = [(z.min(), z.max())]
    pbounds = [(age.min(), age.max())]
    for opts in (dict(), dict(mean_only=True), dict(residual_delta=True),
                 dict(nuisance_scale=True), dict(preserve_df='same', nuisance_scale=True)):
        Yh, *_, est = comcat(Y, batch, z, age, gam_df=5, return_estimates=True,
                             smooth_term_bounds=bounds, preserve_bounds=pbounds, **opts)
        Yft = comcat_from_training(Y, batch, z, age, estimates=est)
        err = np.max(np.abs(Yh - Yft))
        check(f"from_training == comcat  {opts}", est['eb'] and err < 1e-8, f"max_abs={err:.1e}")


def test_single_subject_site():
    print("A site with a single subject is left out of EB")
    Y, batch, z, age, *_ = simulate(seed=6, p=200)
    keep = np.ones(len(batch), bool)
    keep[np.flatnonzero(batch == 0)[1:]] = False          # site 0 keeps one subject
    est = comcat(Y[:, keep], batch[keep], z[keep], age[keep], gam_df=5,
                 return_estimates=True)[4]
    pri = est['eb_priors']
    check("single-subject site not shrunk", np.isnan(pri['t2'][0]))
    check("other sites shrunk with finite priors", np.all(np.isfinite(pri['t2'][1:])))


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} test groups passed.")

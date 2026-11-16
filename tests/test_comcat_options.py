"""
test_comcat_options.py
======================
Tests for the options of comcat(): preserve_df, ref_batch, and that the scale step keeps
the residual variance (site variances from the residuals).  No MATLAB reference files are
needed: data are simulated with known effects.

Simulations
-----------
simulate() (from_training, ref_batch):
  3 sites, one image quality metric (IQM) z correlated with age (r ~ 0.6)
  mean:      non-linear (quadratic) age effect; z has NO effect on the mean
  variance:  noise SD grows with z and with age
  sites:     additive and multiplicative effects
The other tests build their own data (see there).

Run:
    python tests/test_comcat_options.py
"""

import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'evaluation'))
from comcat import comcat, comcat_from_training  # noqa: E402

warnings.simplefilter('ignore', RuntimeWarning)


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:52s} {detail}")
    if not cond:
        raise AssertionError(f"{name} {detail}")


def noise(rng, shape, kind='normal'):
    """Noise with unit variance: normal, heavy-tailed ('t5', excess kurtosis 6)
    or skewed ('chi2', 4 df: skewness 1.4, excess kurtosis 3)."""
    if kind == 'normal':
        return rng.normal(size=shape)
    if kind == 't5':
        return rng.standard_t(5, size=shape) / np.sqrt(5 / 3)
    if kind == 'chi2':
        return (rng.chisquare(4, size=shape) - 4) / np.sqrt(8)
    raise ValueError(kind)


def simulate(n_per_site=150, p=300, z_scale=0.35, seed=0, kind='normal'):
    rng = np.random.default_rng(seed)
    n = 3 * n_per_site
    batch = np.repeat([0, 1, 2], n_per_site)
    age = rng.uniform(20, 80, n)
    a = (age - 50) / 15
    z = 0.6 * a + 0.8 * rng.normal(size=n)          # IQM, corr(z, age) ~ 0.6
    zs = (z - z.mean()) / z.std()

    curve = 0.8 * a ** 2 + 0.5 * a                   # true (non-linear) age effect
    load = rng.uniform(0.5, 1.5, p)
    sd = np.exp(z_scale * zs + 0.15 * a)             # nuisance + biological variance
    Y = (curve[:, None] * load
         + sd[:, None] * noise(rng, (n, p), kind)
         + rng.normal(size=(3, p))[batch]            # additive site effects
         )
    site_scale = np.array([0.8, 1.0, 1.3])[batch]
    Y = Y.mean(0) + (Y - Y.mean(0)) * site_scale[:, None]
    return Y.T, batch, z[:, None], age[:, None], curve, load, sd


def residuals_around_age(Yh, age):
    """Residuals after a flexible (cubic) age fit per feature."""
    A = np.column_stack([np.ones(len(age)), age, age ** 2, age ** 3])
    A = (A - A.mean(0)) / np.where(A.std(0) > 0, A.std(0), 1)
    A[:, 0] = 1
    return Yh - (A @ np.linalg.lstsq(A, Yh.T, rcond=None)[0]).T


def test_options_as_documented():
    print("Options: those of the documentation, with their defaults")
    import inspect
    from comcat import comcat_crossfit
    par = inspect.signature(comcat).parameters
    defaults = {k: v.default for k, v in par.items() if v.kind == v.POSITIONAL_OR_KEYWORD}
    documented = dict(nuisance=None, preserve=None, mean_only=False, verbose=False,
                      ref_batch=None, return_estimates=False, nuisance_model='pspline',
                      preserve_df=None, preserve_bounds=None, subject=None, clip=True)
    extra = set(defaults) - set(documented) - {'Y', 'batch'}
    wrong = {k: defaults.get(k) for k, v in documented.items() if defaults.get(k, '?') != v}
    check("options and defaults as documented (README)", not wrong and not extra,
          f"{wrong} {extra}")
    crossfit = inspect.signature(comcat_crossfit).parameters
    check("comcat_crossfit: subject_terms=False, folds=5",
          crossfit['subject_terms'].default is False and crossfit['folds'].default == 5)


def test_preserve_df_keeps_nonlinear_age_effect():
    print("preserve_df: non-linear age effect is preserved")
    # lifespan-like setting: non-linear age curve and 7 IQMs that are U-shaped
    # functions of age (poorer quality in the young and the old) without any
    # effect on the data.  With a linear age term, the IQM splines absorb the
    # non-linear part of the age effect and ComCAT removes it.
    rng = np.random.default_rng(0)
    n, p = 450, 300
    batch = np.repeat([0, 1, 2], n // 3)
    age = rng.uniform(6, 90, n)
    a = (age - age.mean()) / age.std()
    Z = np.column_stack([0.7 * (a ** 2 - 1) * rng.uniform(0.5, 1) + 0.3 * a * rng.normal()
                         + 0.6 * rng.normal(size=n) for _ in range(7)])
    curve = 0.8 * a ** 2 + 0.5 * a - 0.3 * a ** 3
    true = curve[:, None] * rng.uniform(0.5, 1.5, p)
    true -= true.mean(0)
    Y = (true + rng.normal(size=(n, p)) + rng.normal(size=(3, p))[batch]).T

    def curve_loss(**opts):
        # relative error of the fitted age curve of the harmonized data
        Yh = comcat(Y, batch, Z, age[:, None], **opts)[0]
        A = np.column_stack([np.ones(n), a, a ** 2, a ** 3])
        fit = A @ np.linalg.lstsq(A, Yh.T, rcond=None)[0]
        fit -= fit.mean(0)
        return np.mean((fit - true) ** 2) / np.mean(true ** 2)

    loss_lin, loss_spl = curve_loss(), curve_loss(preserve_df=6)
    check("linear age term: age curve largely removed", loss_lin > 0.5, f"{loss_lin:.3f}")
    check("spline age term: age curve preserved", loss_spl < 0.1, f"{loss_spl:.3f}")


def test_scale_step_keeps_residual_variance():
    print("Scale step: the residual (noise) variance is kept")
    # homoscedastic noise with variance 1, additive site effects, and an IQM
    # effect on the mean with variance ~4 that ComCAT removes
    from comcat_legacy import comcat as comcat_earlier
    rng = np.random.default_rng(0)
    n, p = 450, 300
    batch = np.repeat([0, 1, 2], 150)
    age = rng.uniform(20, 80, n)
    a = (age - 50) / 15
    z = 0.6 * a + 0.8 * rng.normal(size=n)
    Y = ((0.8 * a ** 2 + 0.5 * a)[:, None] + 2.0 * (z - z.mean())[:, None]
         + rng.normal(size=(n, p)) + rng.normal(size=(3, p))[batch]).T

    def noise_var(fn=comcat, **opts):
        Yh = fn(Y, batch, z[:, None], age[:, None], preserve_df=6, **opts)[0]
        return np.median(residuals_around_age(Yh, age).var(1))

    v_new = noise_var()
    # in-sample fit removes ~k/n of the noise variance (k ~ 15 parameters)
    check("site variances from the residuals: residual variance ~ true noise (1)",
          0.9 < v_new < 1.02, f"{v_new:.3f}")
    # earlier versions: site variances before the nuisance effects are removed, shrunk
    # by ~ 1 / (1 + var(nuisance effect) / var(noise)) ~ 1/5
    v_old = noise_var(comcat_earlier, combat_compatible=True)
    check("earlier versions (comcat_legacy): residual variance shrunk", v_old < 0.3,
          f"{v_old:.3f}")


def test_from_training_reproduces_in_sample():
    print("comcat_from_training reproduces the in-sample result with all options")
    Y, batch, z, age, *_ = simulate()
    pbounds = [(age.min(), age.max())]
    for opts in (dict(), dict(preserve_df='same'), dict(mean_only=True),
                 dict(nuisance_model='linear'), dict(preserve_df=6, nuisance_model='linear')):
        Yh, *_, est = comcat(Y, batch, z, age, return_estimates=True,
                             preserve_bounds=pbounds, **opts)
        Yft = comcat_from_training(Y, batch, z, age, estimates=est)
        err = np.max(np.abs(Yh - Yft))
        check(f"from_training == comcat  {opts}", err < 1e-8, f"max_abs={err:.1e}")


def test_ref_batch_with_nuisance():
    print("ref_batch with nuisance: harmonized sites match the untouched reference site")
    Y, batch, z, age, *_ = simulate()
    rng = np.random.default_rng(1)
    zs = (z[:, 0] - z.mean()) / z.std()
    Y = Y + 0.5 * np.outer(rng.uniform(0.5, 1.5, Y.shape[0]), zs)   # IQM effect on the mean
    # second IQM with many ties (like res_RMS)
    z2 = np.maximum(rng.normal(size=len(batch)), 0.3)
    Z = np.column_stack([z[:, 0], z2])
    Yh, *_, est = comcat(Y, batch, Z, age, preserve_df=6, ref_batch=0,
                         preserve_bounds=[(age.min(), age.max())], return_estimates=True)
    ref = batch == 0
    check("harmonized data finite and bounded",
          np.all(np.isfinite(Yh)) and np.abs(Yh).max() < 2 * np.abs(Y).max(),
          f"max_abs={np.abs(Yh).max():.3g} (input {np.abs(Y).max():.3g})")
    check("reference site unchanged", np.max(np.abs(Yh[:, ref] - Y[:, ref])) < 1e-10)

    # after removing the common age effect, every site's mean must match the mean
    # of the reference site, which still contains its own nuisance effects
    def site_offset(D):
        R = residuals_around_age(D, age[:, 0])
        return max(np.mean(np.abs(R[:, batch == b].mean(1) - R[:, ref].mean(1))) for b in (1, 2))
    before, after = site_offset(Y), site_offset(Yh)
    check("site means match the reference", after < 0.15 and after < 0.2 * before,
          f"mean |offset| {after:.3f} (before {before:.3f})")
    Yft = comcat_from_training(Y, batch, Z, age, estimates=est)
    err = np.max(np.abs(Yh - Yft))
    check("from_training == comcat", err < 1e-8, f"max_abs={err:.1e}")


def test_ref_batch_variance():
    print("ref_batch: harmonized sites take the variance of the reference site")
    # noise SD 1, 2 and 0.5 at the three sites, reference = site 1; without the
    # reference, all sites would be scaled to the pooled SD of ~1.32
    rng = np.random.default_rng(2)
    n_site, p = 200, 300
    batch = np.repeat([0, 1, 2], n_site)
    n = batch.size
    age = rng.uniform(20, 80, n)
    z = rng.normal(size=n)
    noise_sd = np.array([1.0, 2.0, 0.5])[batch]
    Y = (0.03 * age[:, None] + 0.8 * z[:, None] + rng.normal(size=(3, p))[batch]
         + noise_sd[:, None] * rng.normal(size=(n, p))).T

    def sd_ratio(Yh, covariates):
        # noise SD of every site relative to the reference site; covariates are
        # removed within each site, as the reference site keeps its IQM effect
        sd = []
        for b in range(3):
            A = np.column_stack([np.ones(n_site)] + [c[batch == b] for c in covariates])
            D = Yh[:, batch == b]
            sd.append(np.median((D - (A @ np.linalg.lstsq(A, D.T, rcond=None)[0]).T).std(1)))
        return np.array(sd) / sd[1]

    for label, nuis, opts, cov in (("no nuisance", None, {}, (age,)),
                                   ("nuisance", z[:, None], {}, (age, z))):
        Yh, *_, est = comcat(Y, batch, nuis, age[:, None], ref_batch=1,
                             return_estimates=True, **opts)
        before, after = sd_ratio(Y, cov), sd_ratio(Yh, cov)
        check(f"{label}: SD of sites 0, 2 = reference SD", np.all(np.abs(after - 1) < 0.05),
              f"{after[0]:.3f}, {after[2]:.3f} (before {before[0]:.2f}, {before[2]:.2f})")
        Yft = comcat_from_training(Y, batch, nuis, age[:, None], estimates=est)
        err = np.max(np.abs(Yh - Yft))
        check(f"{label}: from_training == comcat", err < 1e-8, f"max_abs={err:.1e}")


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} test groups passed.")

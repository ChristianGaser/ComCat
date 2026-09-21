"""
test_comcat_options.py
======================
Tests for the optional ComCAT extensions preserve_df, nuisance_scale and
residual_delta (comcat.py).  Unlike test_comcat_py.py, no MATLAB reference
files are needed: data are simulated with known effects.

Simulations
-----------
simulate() (nuisance_scale, LR test, from_training):
  3 sites, one image quality metric (IQM) z correlated with age (r ~ 0.6)
  mean:      non-linear (quadratic) age effect; z has NO effect on the mean
  variance:  noise SD grows with z (nuisance heteroscedasticity, to remove)
             and with age (biological, to preserve)
  sites:     additive and multiplicative effects
preserve_df and residual_delta tests build their own data (see there).

Run:
    python tests/test_comcat_options.py
"""

import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from comcat import comcat, comcat_from_training  # noqa: E402

warnings.simplefilter('ignore', RuntimeWarning)


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:52s} {detail}")
    if not cond:
        raise AssertionError(f"{name} {detail}")


def simulate(n_per_site=150, p=300, z_scale=0.35, seed=0):
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
         + sd[:, None] * rng.normal(size=(n, p))
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


def test_defaults_unchanged():
    print("Defaults: new options off give the same result as before")
    Y, batch, z, age, *_ = simulate()
    ref = comcat(Y, batch, z, age, gam_df=6)
    new = comcat(Y, batch, z, age, gam_df=6, preserve_df=None,
                 nuisance_scale=False, residual_delta=False)
    check("identical output", all(np.array_equal(a, b) for a, b in zip(ref, new)))


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
        Yh = comcat(Y, batch, Z, age[:, None], gam_df=6, residual_delta=True, **opts)[0]
        A = np.column_stack([np.ones(n), a, a ** 2, a ** 3])
        fit = A @ np.linalg.lstsq(A, Yh.T, rcond=None)[0]
        fit -= fit.mean(0)
        return np.mean((fit - true) ** 2) / np.mean(true ** 2)

    loss_lin, loss_spl = curve_loss(), curve_loss(preserve_df='same')
    check("linear age term: age curve largely removed", loss_lin > 0.5, f"{loss_lin:.3f}")
    check("spline age term: age curve preserved", loss_spl < 0.1, f"{loss_spl:.3f}")


def test_residual_delta_keeps_residual_variance():
    print("residual_delta: residual (noise) variance is kept")
    # homoscedastic noise with variance 1, additive site effects, and an IQM
    # effect on the mean with variance ~4 that ComCAT removes
    rng = np.random.default_rng(0)
    n, p = 450, 300
    batch = np.repeat([0, 1, 2], 150)
    age = rng.uniform(20, 80, n)
    a = (age - 50) / 15
    z = 0.6 * a + 0.8 * rng.normal(size=n)
    Y = ((0.8 * a ** 2 + 0.5 * a)[:, None] + 2.0 * (z - z.mean())[:, None]
         + rng.normal(size=(n, p)) + rng.normal(size=(3, p))[batch]).T

    def noise_var(**opts):
        Yh = comcat(Y, batch, z[:, None], age[:, None], gam_df=6,
                    preserve_df='same', **opts)[0]
        return np.median(residuals_around_age(Yh, age).var(1))

    v_old, v_new = noise_var(), noise_var(residual_delta=True)
    # in-sample fit removes ~k/n of the noise variance (k ~ 15 parameters)
    check("residual_delta: residual variance ~ true noise (1)", 0.9 < v_new < 1.02,
          f"{v_new:.3f}")
    # default: shrunk by ~ 1 / (1 + var(nuisance effect) / var(noise)) ~ 1/5
    check("default: residual variance shrunk", v_old < 0.3, f"{v_old:.3f}")


def test_nuisance_scale_removes_heteroscedasticity():
    print("nuisance_scale: IQM-dependent variance removed, age-dependent kept")
    Y, batch, z, age, *_ = simulate(z_scale=0.35)
    zc = z[:, 0]
    kw = dict(gam_df=6, preserve_df='same')

    def log_sd_slopes(Yh):
        R = residuals_around_age(Yh, age[:, 0])
        # site-wise standardization to remove residual site scale differences
        for s in np.unique(batch):
            R[:, batch == s] /= R[:, batch == s].std(1, keepdims=True)
        L = 0.5 * np.log(R ** 2 + 1e-12)
        D = np.column_stack([np.ones_like(zc), (zc - zc.mean()) / zc.std(),
                             (age[:, 0] - 50) / 15])
        return np.median(np.linalg.lstsq(D, L.T, rcond=None)[0][1:], axis=1)

    b_off = log_sd_slopes(comcat(Y, batch, z, age, **kw)[0])
    b_on = log_sd_slopes(comcat(Y, batch, z, age, nuisance_scale=True, **kw)[0])
    check("IQM effect on log SD present without option", b_off[0] > 0.25,
          f"slope {b_off[0]:.3f}")
    check("IQM effect on log SD removed with option", abs(b_on[0]) < 0.05,
          f"slope {b_on[0]:.3f}")
    check("age effect on log SD preserved", abs(b_on[1] - b_off[1]) < 0.05 and b_on[1] > 0.1,
          f"age slope {b_off[1]:.3f} -> {b_on[1]:.3f}")


def test_scale_lr_test():
    print("nuisance_scale: likelihood-ratio test of IQM variance effects")
    for z_scale, label in ((0.0, "null"), (0.35, "effect")):
        Y, batch, z, age, *_ = simulate(z_scale=z_scale, seed=3)
        est = comcat(Y, batch, z, age, gam_df=6, preserve_df='same',
                     nuisance_scale=True, return_estimates=True)[4]
        frac = np.mean(est['scale_p'] < 0.05)
        if label == "null":
            check("null: ~5% of features p < 0.05", 0.01 < frac < 0.10,
                  f"{100 * frac:.1f}%  mean LR {np.mean(est['scale_lr']):.2f} "
                  f"(df {est['scale_lr_df']})")
        else:
            check("effect: most features p < 0.05", frac > 0.9, f"{100 * frac:.1f}%")


def test_from_training_reproduces_in_sample():
    print("comcat_from_training reproduces the in-sample result with all options")
    Y, batch, z, age, *_ = simulate()
    bounds = [(z.min(), z.max())]
    pbounds = [(age.min(), age.max())]
    for opts in (dict(preserve_df='same'), dict(residual_delta=True),
                 dict(nuisance_scale=True),
                 dict(preserve_df='same', nuisance_scale=True)):
        Yh, *_, est = comcat(Y, batch, z, age, gam_df=6, return_estimates=True,
                             smooth_term_bounds=bounds, preserve_bounds=pbounds, **opts)
        Yft = comcat_from_training(Y, batch, z, age, estimates=est)
        err = np.max(np.abs(Yh - Yft))
        check(f"from_training == comcat  {opts}", err < 1e-8, f"max_abs={err:.1e}")


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} test groups passed.")

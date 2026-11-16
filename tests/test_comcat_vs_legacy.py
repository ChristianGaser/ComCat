"""
The current comcat.py against the frozen earlier version (evaluation/comcat_legacy.py).

The simplification of 2026-11-16 removed options but must not change the model that was
kept: comcat() has to equal the earlier version with ComBat's site variances estimated
from the residuals (combat_compatible=True, residual_delta=True), for all settings that
still exist: with and without sites, nuisance covariates and preserved covariates,
location only, linear nuisance terms, a reference site, splines for preserved covariates,
subject terms, clipping, new data (comcat_from_training) and cross-fitting.

Also checked: without nuisance covariates the result is ComBat (combat_family.combat_eb),
removed options are accepted when they describe this version and refused otherwise.

Run:  python tests/test_comcat_vs_legacy.py
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'evaluation'))

from comcat import comcat, comcat_from_training, comcat_crossfit  # noqa: E402
import comcat_legacy as L  # noqa: E402
from combat_family import combat_eb  # noqa: E402

warnings.simplefilter('ignore')
TOL = 1e-10
OLD = dict(combat_compatible=True, residual_delta=True)     # the model that was kept


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:62s} {detail}")
    assert ok, name


def same(name, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    nan_equal = np.array_equal(np.isnan(a), np.isnan(b))
    d = float(np.nanmax(np.abs(a - b))) if a.size and np.any(np.isfinite(a)) else 0.0
    check(name, nan_equal and d <= TOL, f"max_abs={d:.1e}")


def data(seed=0, n_sites=5, nonneg=True):
    """Sites of unequal size (one with three subjects), three nuisance covariates with
    site differences, age and group preserved, site effects in mean and variance."""
    rng = np.random.default_rng(seed)
    sizes = np.array([40, 25, 12, 6, 3][:n_sites])
    batch = np.repeat(np.arange(n_sites), sizes)
    n, p = batch.size, 150
    age = rng.uniform(20, 80, n)
    group = rng.integers(0, 2, n).astype(float)
    Z = rng.normal(size=(n, 3)) + 0.5 * rng.normal(size=(n_sites, 3))[batch]
    Z[:, 2] = np.exp(Z[:, 2])                                  # skewed
    Y = (rng.normal(size=(p, 1)) * 0.02 * age + 0.3 * rng.normal(size=(p, 1)) * group
         + 0.4 * rng.normal(size=(p, n_sites))[:, batch]
         + 0.3 * rng.normal(size=(p, 1)) * Z[:, 0] + 0.2 * rng.normal(size=(p, 1)) * Z[:, 1] ** 2
         + rng.normal(size=(p, n)) * (1 + 0.3 * rng.uniform(size=n_sites))[batch])
    if nonneg:
        Y = Y - Y.min() + 0.1
        Y[:5] *= 0.02                    # features near 0: clipping matters
    return Y, batch, Z, np.column_stack([age, group])


def test_same_as_legacy():
    Y, batch, Z, P = data()
    cases = {
        'sites + nuisance + preserve': dict(),
        'location only': dict(mean_only=True),
        'linear nuisance terms': dict(nuisance_model='linear'),
        'reference site': dict(ref_batch=1),
        'reference site, location only': dict(ref_batch=1, mean_only=True),
        'no clipping': dict(clip=False),
        'preserve_df': dict(preserve_df=5),
    }
    for name, o in cases.items():
        new = comcat(Y, batch, Z, P, **o)
        old = L.comcat(Y, batch, Z, P, **OLD, **o)
        for k, part in enumerate(('data', 'beta', 'gamma', 'delta')):
            same(f"{name}: {part}", new[k], old[k])
    same('sites only', comcat(Y, batch, None, P)[0], L.comcat(Y, batch, None, P, **OLD)[0])
    same('nuisance only (one site)', comcat(Y, None, Z, P)[0],
         L.comcat(Y, None, Z, P, **OLD)[0])
    same('nuisance only, nothing preserved', comcat(Y, None, Z)[0],
         L.comcat(Y, None, Z, **OLD)[0])
    Ys = Y - Y.mean()                                          # signed data: never clipped
    same('signed data', comcat(Ys, batch, Z, P)[0], L.comcat(Ys, batch, Z, P, **OLD)[0])
    # constant and missing features, a site with a single subject
    Yb = Y.copy()
    Yb[3] = 1.0
    Yb[7, 4] = np.nan
    b1 = batch.copy()
    b1[-1] = 9
    same('constant and missing features, single-subject site', comcat(Yb, b1, Z, P)[0],
         L.comcat(Yb, b1, Z, P, **OLD)[0])


def test_combat_without_nuisance():
    Y, batch, Z, P = data(nonneg=False)
    same('no nuisance covariates: ComBat (combat_family.combat_eb)',
         comcat(Y, batch, None, P)[0], combat_eb(Y, batch, P)[0])


def test_subject_terms_same_as_legacy():
    rng = np.random.default_rng(3)
    n_sub, n_site = 12, 4
    subject = np.repeat(np.arange(n_sub), n_site)
    batch = np.tile(np.arange(n_site), n_sub)
    n, p = subject.size, 80
    Z = rng.normal(size=(n, 2))
    age = np.repeat(rng.uniform(20, 60, n_sub), n_site)
    Y = (rng.normal(size=(p, n_sub))[:, subject] + 0.5 * rng.normal(size=(p, n_site))[:, batch]
         + 0.3 * rng.normal(size=(p, 1)) * Z[:, 0] + 0.3 * rng.normal(size=(p, n)))
    same('subject terms', comcat(Y, batch, Z, age, subject=subject)[0],
         L.comcat(Y, batch, Z, age, subject=subject, **OLD)[0])
    same('subject terms, sites only', comcat(Y, batch, None, age, subject=subject)[0],
         L.comcat(Y, batch, None, age, subject=subject, **OLD)[0])
    for st in (False, True):
        same(f"cross-fit over subjects, subject_terms={st}",
             comcat_crossfit(Y, batch, Z, age, folds=4, subject=subject, subject_terms=st),
             L.comcat_crossfit(Y, batch, Z, age, folds=4, subject=subject, subject_terms=st,
                               **OLD))


def test_new_data_and_crossfit_same_as_legacy():
    Y, batch, Z, P = data(seed=1)
    tr = np.arange(Y.shape[1]) % 3 != 0
    te = ~tr
    bte = batch[te].copy()
    bte[:2] = 77                                               # a site not in the training
    for name, o in {'default': dict(), 'location only': dict(mean_only=True),
                    'linear': dict(nuisance_model='linear'), 'reference site': dict(ref_batch=0),
                    'preserve_df': dict(preserve_df=5, preserve_bounds=[(20, 80), None])}.items():
        en = comcat(Y[:, tr], batch[tr], Z[tr], P[tr], return_estimates=True, **o)[4]
        eo = L.comcat(Y[:, tr], batch[tr], Z[tr], P[tr], return_estimates=True, **OLD, **o)[4]
        a = comcat_from_training(Y[:, te], bte, Z[te], P[te], estimates=en, unseen_batch='mean')
        b = L.comcat_from_training(Y[:, te], bte, Z[te], P[te], estimates=eo,
                                   unseen_batch='mean')
        same(f"new data, {name}", a, b)
    ctrl = P[:, 1] == 0
    same('cross-fit', comcat_crossfit(Y, batch, Z, P, folds=4),
         L.comcat_crossfit(Y, batch, Z, P, folds=4, **OLD))
    same('cross-fit, fitted on controls',
         comcat_crossfit(Y, batch, Z, P[:, :1], folds=4, fit=ctrl),
         L.comcat_crossfit(Y, batch, Z, P[:, :1], folds=4, fit=ctrl, **OLD))
    same('cross-fit, one site', comcat_crossfit(Y, None, Z, P, folds=4),
         L.comcat_crossfit(Y, None, Z, P, folds=4, **OLD))
    same('cross-fit, preserve_df', comcat_crossfit(Y, batch, Z, P, folds=3, preserve_df=5),
         L.comcat_crossfit(Y, batch, Z, P, folds=3, preserve_df=5, **OLD))


def test_removed_options():
    Y, batch, Z, P = data(seed=2)
    ref = comcat(Y, batch, Z, P)[0]
    # names of earlier versions that describe this version are accepted
    ok = dict(eb=True, combat_compatible=True, residual_delta=True, min_scale_n=2,
              single_site_scale=False, nuisance_scale=False, nuisance_penalty=None, gam_df=10,
              smooth_term_bounds=None, nuisance_model='pspline')
    same('compatible options of earlier versions accepted', comcat(Y, batch, Z, P, **ok)[0],
         ref)
    same('... also by comcat_crossfit',
         comcat_crossfit(Y, batch, Z, P, folds=3, eb=True, min_scale_n=2),
         comcat_crossfit(Y, batch, Z, P, folds=3))
    for o in (dict(eb=False), dict(combat_compatible=False), dict(residual_delta=False),
              dict(min_scale_n=4), dict(single_site_scale=True), dict(nuisance_scale=True),
              dict(nuisance_penalty='gcv'), dict(gam_df=6), dict(smooth_term_bounds=(0, 1)),
              dict(nuisance_model='spline')):
        try:
            comcat(Y, batch, Z, P, **o)
            raised = False
        except ValueError as e:
            raised = 'comcat_legacy' in str(e)
        check(f"refused with a pointer to the earlier version: {o}", raised)
    try:
        comcat(Y, batch, Z, P, no_such_option=1)
        raised = False
    except TypeError:
        raised = True
    check('unknown option: TypeError', raised)
    # estimates of the removed unpenalized splines cannot be applied
    eo = L.comcat(Y, batch, Z, P, nuisance_model='spline', return_estimates=True)[4]
    try:
        comcat_from_training(Y, batch, Z, P, estimates=eo)
        raised = False
    except ValueError:
        raised = True
    check('estimates of the unpenalized splines refused', raised)


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} test groups passed.")

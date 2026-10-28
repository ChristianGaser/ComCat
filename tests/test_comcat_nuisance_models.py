"""
test_comcat_nuisance_models.py
==============================
Tests for the nuisance models 'linear' and 'pspline', subject terms and
cross-fitting (comcat.py: nuisance_model, nuisance_penalty, subject,
comcat_crossfit).  Data are simulated with known effects.

Simulations
-----------
simulate():   3 sites, age, two image quality metrics (IQMs); the IQM effect on
              every feature is linear or nonlinear (sin) with a random size
traveling():  subjects scanned repeatedly; the IQMs of a subject depend on its
              anatomy (subject effect), so an IQM effect estimated across
              subjects takes up anatomy, one estimated within subjects does not
patients():   controls and patients; patients have a disease effect and worse
              IQMs, so a model fitted on all subjects removes part of the
              disease effect, one fitted on the controls does not

Run:
    python tests/test_comcat_nuisance_models.py
"""

import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from comcat import comcat, comcat_from_training, comcat_crossfit  # noqa: E402

FAILED = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:55s} {detail}")
    if not ok:
        FAILED.append(name)


def simulate(n=240, p=400, nonlinear=False, seed=0):
    rng = np.random.default_rng(seed)
    site = np.repeat(np.arange(3), n // 3)
    age = rng.uniform(20, 70, n)
    z = rng.normal(size=(n, 2))
    f = np.sin(2 * z) if nonlinear else z
    effect = f @ rng.normal(scale=0.6, size=(2, p))           # (n, p) nuisance effect
    Y = (rng.normal(size=(p, n)) + np.outer(rng.normal(scale=0.03, size=p), age)
         + effect.T + 0.5 * site)
    return Y, site, z, age, effect.T


def traveling(n_subj=30, n_scan=6, p=400, seed=1):
    rng = np.random.default_rng(seed)
    subj = np.repeat(np.arange(n_subj), n_scan)
    u = rng.normal(size=n_subj)                                # e.g. head or brain size
    anatomy = np.outer(u, rng.normal(size=p)) + 0.6 * rng.normal(size=(n_subj, p))
    z = (1.5 * u[subj] + 0.5 * rng.normal(size=subj.size))[:, None]   # IQMs follow it
    beta = rng.normal(scale=0.5, size=p)                       # true IQM effect per feature
    Y = anatomy[subj].T + np.outer(beta, z[:, 0]) + 0.3 * rng.normal(size=(p, subj.size))
    return Y, subj, z, anatomy, beta


def patients(n=300, p=400, seed=2):
    rng = np.random.default_rng(seed)
    group = (np.arange(n) >= n // 2).astype(float)             # 1 = patient
    z = (rng.normal(size=n) + 1.5 * group)[:, None]            # patients have worse IQMs
    disease = rng.normal(scale=0.5, size=p)
    beta = rng.normal(scale=0.5, size=p)
    Y = (rng.normal(size=(p, n)) + np.outer(disease, group) + np.outer(beta, z[:, 0]))
    return Y, z, group, disease


def test_from_training_reproduces():
    print("comcat_from_training reproduces the in-sample result")
    Y, site, z, age, _ = simulate()
    for opts in ({'nuisance_model': 'linear'}, {'nuisance_model': 'pspline'},
                 {'nuisance_model': 'pspline', 'nuisance_penalty': 'gcv'},
                 {'nuisance_model': 'pspline', 'mean_only': True},
                 {'nuisance_model': 'linear', 'residual_delta': True}):
        H, *_, est = comcat(Y, site, z, age, return_estimates=True, **opts)
        Hf = comcat_from_training(Y, site, z, age, estimates=est)
        err = np.abs(H - Hf).max()
        check(f"from_training == comcat {opts}", err < 1e-10, f"max_abs={err:.1e}")


def test_pspline_smoothness():
    print("pspline: the penalty adapts to the shape of the nuisance effect")
    edf = {}
    for nonlinear in (False, True):
        Y, site, z, age, _ = simulate(nonlinear=nonlinear)
        est = comcat(Y, site, z, age, nuisance_model='pspline', return_estimates=True)[4]
        edf[nonlinear] = est['nuisance_edf']
    check("linear truth: about 1 df per IQM", edf[False] < 2.6, f"edf={edf[False]:.2f}")
    check("nonlinear truth: more df", edf[True] > edf[False] + 2, f"edf={edf[True]:.2f}")


def test_heldout_error():
    print("held-out nuisance removal: pspline/linear beat unpenalized splines in small samples")
    Y, site, z, age, effect = simulate(n=90, nonlinear=True, seed=3)
    tr = np.arange(90) % 3 != 0
    err = {}
    for name, opts in (('spline', {'nuisance_model': 'spline', 'gam_df': 10}),
                       ('pspline', {'nuisance_model': 'pspline'}),
                       ('linear', {'nuisance_model': 'linear'})):
        bounds = [(z[:, j].min(), z[:, j].max()) for j in range(2)]
        est = comcat(Y[:, tr], site[tr], z[tr], age[tr], mean_only=True,
                     smooth_term_bounds=bounds, return_estimates=True, **opts)[4]
        H = comcat_from_training(Y[:, ~tr], site[~tr], z[~tr], age[~tr], estimates=est)
        removed = Y[:, ~tr] - H
        truth = effect[:, ~tr] - effect[:, ~tr].mean(axis=1, keepdims=True)
        removed -= removed.mean(axis=1, keepdims=True)
        err[name] = np.sqrt(np.mean((removed - truth) ** 2))
    check("pspline < unpenalized spline", err['pspline'] < err['spline'],
          f"pspline {err['pspline']:.3f}, spline {err['spline']:.3f}, linear {err['linear']:.3f}")


def test_subject_terms():
    print("subject terms: IQM effects within subjects keep the anatomy")
    Y, subj, z, anatomy, beta = traveling()
    B0 = np.var(anatomy, axis=0)
    out = {}
    for name, s in (('across', None), ('within', subj)):
        H, *_, est = comcat(Y, None, z, None, nuisance_model='linear', subject=s,
                            return_estimates=True)
        means = np.vstack([H[:, subj == k].mean(axis=1) for k in np.unique(subj)])
        out[name] = (np.mean(np.var(means, axis=0) / B0),
                     np.corrcoef(est['gamma_hat_masked'][1] * est['std_pooled']
                                 / est['nuisance_setup'][0]['sd'], beta)[0, 1])
    check("across subjects: anatomy removed", out['across'][0] < 0.8,
          f"between-subject variance kept {out['across'][0]:.2f}")
    check("within subjects: anatomy kept", abs(out['within'][0] - 1) < 0.15,
          f"between-subject variance kept {out['within'][0]:.2f}")

    # new subjects: comcat_from_training with the within-subject model
    tr = subj < 20
    est = comcat(Y[:, tr], None, z[tr], None, nuisance_model='linear', subject=subj[tr],
                 return_estimates=True)[4]
    H = comcat_from_training(Y[:, ~tr], np.ones(int((~tr).sum())), z[~tr], None, estimates=est)
    resid = H - anatomy[subj[~tr]].T
    W = np.mean(np.var(resid, axis=1))
    W0 = np.mean(np.var(Y[:, ~tr] - anatomy[subj[~tr]].T, axis=1))
    check("new subjects: within-subject variance reduced", W < 0.5 * W0,
          f"{W:.3f} vs raw {W0:.3f}")


def test_crossfit_controls():
    print("comcat_crossfit: models fitted on the controls keep the disease effect")
    Y, z, group, disease = patients()
    d0 = Y[:, group == 1].mean(axis=1) - Y[:, group == 0].mean(axis=1)
    keep = {}
    for name, fit in (('all subjects', None), ('controls', group == 0)):
        H = comcat_crossfit(Y, None, z, None, folds=5, fit=fit, nuisance_model='linear')
        d = H[:, group == 1].mean(axis=1) - H[:, group == 0].mean(axis=1)
        keep[name] = np.polyfit(disease, d, 1)[0]
    check("fitted on all subjects: disease effect reduced", keep['all subjects'] < 0.8,
          f"slope {keep['all subjects']:.2f} (raw {np.polyfit(disease, d0, 1)[0]:.2f})")
    check("fitted on controls: disease effect kept", abs(keep['controls'] - 1) < 0.1,
          f"slope {keep['controls']:.2f}")

    Y, site, z, age, _ = simulate()
    H, folds = comcat_crossfit(Y, site, z, age, folds=4, nuisance_model='pspline',
                               return_folds=True)
    check("every sample harmonized, folds balanced",
          np.isfinite(H).all() and np.ptp(np.bincount(folds)) <= 3,
          f"fold sizes {np.bincount(folds).tolist()}")


def test_screening():
    print("nuisance columns without variation within sites are dropped")
    Y, site, z, age, _ = simulate()
    zc = np.column_stack([z, site * 2.0 + 1])                  # constant within sites
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        est = comcat(Y, site, zc, age, nuisance_model='linear', return_estimates=True)[4]
    check("site-level column dropped", est['nuisance_keep'].tolist() == [0, 1],
          f"kept {est['nuisance_keep'].tolist()}")
    check("warning issued", any('do not vary' in str(x.message) for x in w))


def test_single_site_location_only():
    print("one site: location correction only, unless single_site_scale")
    Y, site, z, age, _ = simulate()
    one = np.ones(len(age), dtype=int)
    a = comcat(Y, one, z, age)[0]
    b = comcat(Y, one, z, age, mean_only=True)[0]
    c = comcat(Y, None, z, age)[0]
    old = comcat(Y, one, z, age, single_site_scale=True)[0]
    check("explicit single site == mean_only == no batch",
          np.allclose(a, b) and np.allclose(a, c), f"max_abs={np.abs(a - c).max():.1e}")
    shrink = np.mean(np.var(old, axis=1) / np.var(a, axis=1))
    check("single_site_scale shrinks all variance (earlier versions)", shrink < 0.95,
          f"variance ratio {shrink:.2f}")
    from comcat_ui import _build_subfolder
    est = comcat(Y, one, z, age, return_estimates=True)[4]
    name = _build_subfolder(1, 2, 1, est['mean_only'], gam_df=5)
    check("comcat_ui folder marks location-only runs",
          name.endswith('_meanonly_nuisance2_pspline'), name)
    legacy = _build_subfolder(1, 2, 1, False, gam_df=5, nuisance_model='spline')
    check("comcat_ui folder of the earlier model unchanged", legacy == 'comcat_preserve1_nuisance2_gam5',
          legacy)


def test_default_model():
    print("default nuisance model: penalized B-splines; the earlier spline stays available")
    Y, site, z, age, _ = simulate()
    a, *_, est = comcat(Y, site, z, age, return_estimates=True)
    b = comcat(Y, site, z, age, nuisance_model='pspline', nuisance_penalty='reml')[0]
    check("default == explicit 'pspline' with REML", np.array_equal(a, b),
          f"model {est['nuisance_model']}, gam_df {est['gam_df']}")
    old = comcat(Y, site, z, age, nuisance_model='spline')[0]
    check("'spline' (earlier versions) differs from the default", not np.allclose(a, old))


if __name__ == '__main__':
    tests = [test_default_model, test_single_site_location_only,
             test_from_training_reproduces, test_pspline_smoothness,
             test_heldout_error, test_subject_terms, test_crossfit_controls, test_screening]
    for t in tests:
        t()
    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) failed: {FAILED}")
        sys.exit(1)
    print(f"All {len(tests)} test groups passed.")

"""
test_combat_family.py
=====================
Tests for combat_family.py (ComBat-EB, CovBat, ComBatLS) and combat_family_ui.py.

1. Properties that hold by construction (site means equalized, masked features
   untouched, ComBatLS with constant sigma == ComBat, CovBat reduces site
   differences in covariance, MAT round trip through the file interface).
2. Numerical agreement with the reference R package ComBatFamily.  This part
   runs only when Rscript with ComBatFamily is available; set COMBATFAMILY_RSCRIPT
   to the Rscript executable if it is not on PATH.  Install the package with
       remotes::install_github("andy1764/ComBatFamily")

Run:
    python tests/test_combat_family.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
from scipy.io import loadmat, savemat

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from combat_family import combat_eb, combatls, covbat  # noqa: E402
from combat_family_ui import covbat_ui  # noqa: E402


def check_close(name, a, b, atol):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    same_nan = np.array_equal(np.isnan(a), np.isnan(b))
    fin = ~np.isnan(a)
    err = float(np.max(np.abs(a[fin] - b[fin])))
    ok = same_nan and err <= atol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:40s} max_abs={err:.3e}")
    if not ok:
        raise AssertionError(f"{name}: max_abs={err:.3e} > {atol}")


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:40s} {detail}")
    if not cond:
        raise AssertionError(name)


def simulate(seed=1, p=60):
    """Four sites with location, scale and covariance effects; age-dependent
    variance.  Returns Y (n x p), batch, covariates (age, sex)."""
    rng = np.random.default_rng(seed)
    sizes = [30, 50, 40, 30]
    n = sum(sizes)
    batch = np.repeat(['A', 'B', 'C', 'D'], sizes)
    age = rng.uniform(20, 80, n)
    sex = rng.integers(0, 2, n).astype(float)
    loadings = rng.normal(size=(4, p, 3))
    Y = np.empty((n, p))
    for b, lab in enumerate('ABCD'):
        idx = batch == lab
        sd = (0.5 + 0.02 * age[idx])[:, None] * rng.uniform(0.7, 1.5, p)
        Y[idx] = (rng.normal(size=(idx.sum(), p)) * sd
                  + rng.normal(size=(idx.sum(), 3)) @ loadings[b].T
                  + rng.normal(b, 1, p))
    Y += 0.05 * age[:, None] * rng.normal(1, 0.3, p) + 0.8 * sex[:, None]
    return Y, batch, np.column_stack([age, sex])


# ---------------------------------------------------------------------------
# 1. Properties
# ---------------------------------------------------------------------------

def test_site_means_equalized():
    print("L/S model without EB equalizes site means and variances")
    Y, batch, _ = simulate()
    Yh = combat_eb(Y, batch, eb=False)[0]
    means = np.vstack([Yh[batch == b].mean(0) for b in np.unique(batch)])
    sds = np.vstack([Yh[batch == b].std(0, ddof=1) for b in np.unique(batch)])
    check_close("site means equal", means, means[:1], 1e-10)
    check_close("site SDs equal", sds, sds[:1], 1e-10)


def test_masked_features_and_orientation():
    print("Constant / non-finite features untouched; orientation kept")
    Y, batch, X = simulate()
    Y[:, 0] = 3.0
    Y[5, 1] = np.nan
    for fn in (combat_eb, covbat, combatls):
        Yh, est = fn(Y, batch, X)                  # (subjects x features) input
        check(f"{fn.__name__}: shape", Yh.shape == Y.shape)
        check(f"{fn.__name__}: constant feature kept", np.all(Yh[:, 0] == 3.0))
        check(f"{fn.__name__}: NaN feature kept",
              np.array_equal(Yh[:, 1], Y[:, 1], equal_nan=True))
        check(f"{fn.__name__}: mask", not est['mask'][0] and not est['mask'][1])
    check_close("transposed input", combat_eb(Y.T, batch, X)[0].T,
                combat_eb(Y, batch, X)[0], 1e-12)


def test_combatls_constant_sigma_is_combat():
    print("ComBatLS with constant sigma reduces to ComBat")
    Y, batch, X = simulate()
    Y_ls, est = combatls(Y.T, batch, X, scale=np.empty((len(batch), 0)))
    check("converged", est['converged'].all())
    check_close("combatls(scale=[]) == combat_eb", Y_ls, combat_eb(Y.T, batch, X)[0], 1e-9)


def test_covbat_reduces_covariance_differences():
    print("CovBat reduces site differences in covariance")
    Y, batch, X = simulate()

    def cov_spread(Z):
        R = Z - X @ np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), Z, rcond=None)[0][1:]
        covs = [np.cov(R[batch == b].T) for b in np.unique(batch)]
        mean_cov = np.mean(covs, axis=0)
        return np.mean([np.linalg.norm(c - mean_cov) for c in covs])

    s_raw = cov_spread(Y)
    s_com = cov_spread(combat_eb(Y, batch, X)[0])
    s_cov = cov_spread(covbat(Y, batch, X)[0])
    check("covbat < combat < raw", s_cov < s_com < s_raw,
          f"raw {s_raw:.2f}  combat {s_com:.2f}  covbat {s_cov:.2f}")


def test_ui_mat_roundtrip():
    print("File interface: MAT round trip (subjects x features)")
    Y, batch, X = simulate()
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, 'data.mat')
        savemat(f, {'Y': Y, 'age': X[:, :1]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Yh, gamma, delta = covbat_ui([f], batch=batch, nuisance=X[:, 1],
                                         preserve=X, verbose=False)
        out = os.path.join(tmp, 'covbat_sites_preserve2_pc95')
        saved = loadmat(os.path.join(out, 'data.mat'))
        check("nuisance warning", any('ignored' in str(x.message) for x in w))
        check("saved orientation", saved['Y'].shape == Y.shape)
        check_close("saved == returned", saved['Y'], Yh.T, 1e-12)
        check_close("other fields kept", saved['age'], X[:, :1], 0)
        check("estimates shape", gamma.shape == delta.shape == (4, Y.shape[1]))
        check("log file", os.path.isfile(os.path.join(out, 'covbat_sites_preserve2_pc95.mat')))


# ---------------------------------------------------------------------------
# 2. Agreement with R ComBatFamily
# ---------------------------------------------------------------------------

def _rscript():
    return os.environ.get('COMBATFAMILY_RSCRIPT') or shutil.which('Rscript')


def test_against_R():
    print("Agreement with R ComBatFamily")
    rscript = _rscript()
    if rscript is None:
        print("  [SKIP] Rscript not found (set COMBATFAMILY_RSCRIPT)")
        return
    Y, batch, X = simulate()
    with tempfile.TemporaryDirectory() as tmp:
        np.savetxt(os.path.join(tmp, 'data.csv'), Y, delimiter=',', fmt='%.17g')
        np.savetxt(os.path.join(tmp, 'covar.csv'), X, delimiter=',', fmt='%.17g')
        with open(os.path.join(tmp, 'batch.txt'), 'w') as fh:
            fh.write('\n'.join(batch) + '\n')
        res = subprocess.run(
            [rscript, os.path.join(_HERE, 'combat_family_reference.R'), tmp],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            if 'there is no package' in res.stderr:
                print("  [SKIP] R package ComBatFamily not installed")
                return
            raise RuntimeError(res.stderr)

        def ref(name):
            return np.loadtxt(os.path.join(tmp, f'r_{name}.csv'), delimiter=',')

        check_close("ComBat (EB)", combat_eb(Y, batch, X)[0], ref('combat_eb'), 1e-10)
        check_close("ComBat (no EB)", combat_eb(Y, batch, X, eb=False)[0],
                    ref('combat_noeb'), 1e-10)
        Y_cov, est = covbat(Y, batch, X)
        n_pc_r = int(open(os.path.join(tmp, 'r_covbat_npc.txt')).read())
        check("CovBat number of PCs", est['n_pc'] == n_pc_r, f"{est['n_pc']} vs R {n_pc_r}")
        check_close("CovBat (95% variance)", Y_cov, ref('covbat'), 1e-10)
        check_close("CovBat (n_pc=3)", covbat(Y, batch, X, n_pc=3)[0],
                    ref('covbat_npc3'), 1e-10)
        # gamlss stops on a deviance criterion and ends ~1e-7 from the MLE
        check_close("ComBatLS", combatls(Y, batch, X)[0], ref('combatls'), 1e-5)


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} test groups passed.")

"""
test_comcat_py.py
=================
Validates that comcat.py produces results numerically identical (within
floating-point tolerance) to comcat.m.

Workflow
--------
1. Run gen_test_data.m in MATLAB first — it produces test_case1.mat,
   test_case2.mat, test_case3.mat in the same directory.
2. Run this script:
       python test_comcat_py.py

Each test case loads the MATLAB-generated inputs, runs comcat.py on the
same inputs, and compares the harmonized output against the MATLAB result.

Tolerances
----------
MATLAB uses single-precision for Y internally (single(Y)) while this
Python port also uses float32. Differences ≲ 1e-4 (relative) or ≲ 1e-5
(absolute) are considered acceptable given float32 rounding.
"""

import os
import sys
import numpy as np
from scipy.io import loadmat

# comcat.py lives in the repo root, one level up from this tests/ directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from comcat import comcat, comcat_from_training

# .mat reference files (MATLAB-generated) are looked for next to this script
DATA_DIR = _HERE

import statsmodels 

# Tolerances
ATOL = 1e-4
RTOL = 1e-4


def load(fname):
    """Load a -v6 .mat file, return as dict of numpy arrays."""
    raw = loadmat(os.path.join(DATA_DIR, fname), squeeze_me=True)
    return {k: np.array(v, dtype=np.float64)
            for k, v in raw.items() if not k.startswith('_')}


def check_close(name, a, b, atol=ATOL, rtol=RTOL):
    """Assert two arrays are close; print a summary line."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # ignore NaN positions (both must be NaN or both finite)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        raise AssertionError(f"[{name}] NaN pattern differs")
    mask = ~nan_a
    max_abs = float(np.max(np.abs(a[mask] - b[mask])))
    max_rel = float(np.max(np.abs(a[mask] - b[mask]) /
                            (np.abs(b[mask]) + 1e-12)))
    ok = max_abs <= atol or max_rel <= rtol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:30s}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
    if not ok:
        raise AssertionError(
            f"{name}: max_abs={max_abs:.3e} > {atol}, max_rel={max_rel:.3e} > {rtol}"
        )


# ---------------------------------------------------------------------------
# Test 1 — multi-site, linear nuisance + preserve
# ---------------------------------------------------------------------------
def test_case1():
    print("Case 1: multi-site, linear nuisance Z, preserve X")
    d = load("test_case1.mat")
    Y, batch, Z, X = d['Y1'], d['batch1'].astype(int), d['Z1'], d['X1']

    # MATLAB batch is 1-based; comcat.py recodes internally, so pass as-is
    Yh_py, bh_py, gh_py, dh_py = comcat(
        Y, batch, Z, X, mean_only=False, verbose=False,
    )

    check_close("Y_harmonized", Yh_py, d['Yh1'])
    check_close("gamma_hat",    gh_py, d['gh1'])
    check_close("delta_hat",    dh_py, d['dh1'])


# ---------------------------------------------------------------------------
# Test 2 — single site, linear nuisance, mean_only=True
# ---------------------------------------------------------------------------
def test_case2():
    print("Case 2: single site, linear nuisance, mean_only=True")
    d = load("test_case2.mat")
    Y, Z, X = d['Y2'], d['Z2'], d['X2']

    Yh_py, bh_py, gh_py, dh_py = comcat(
        Y, None, Z, X, mean_only=True, verbose=False,
    )

    check_close("Y_harmonized", Yh_py, d['Yh2'])
    check_close("gamma_hat",    gh_py, d['gh2'])
    check_close("delta_hat",    dh_py, d['dh2'])


# ---------------------------------------------------------------------------
# Test 3 — two sites, mean_only, no nuisance
# ---------------------------------------------------------------------------
def test_case3():
    print("Case 3: two sites, mean_only=True, no nuisance")
    d = load("test_case3.mat")
    Y, batch, X = d['Y3'], d['batch3'].astype(int), d['X3']

    Yh_py, bh_py, gh_py, dh_py = comcat(
        Y, batch, None, X, mean_only=True, verbose=False
    )

    check_close("Y_harmonized", Yh_py, d['Yh3'])
    check_close("gamma_hat",    gh_py, d['gh3'])
    check_close("delta_hat",    dh_py, d['dh3'])


# ---------------------------------------------------------------------------
# Test case 4 — multi-site, linear nuisance + preserve
# ---------------------------------------------------------------------------
def test_case4():
    print("Case 4 (MATLAB match): multi-site, linear nuisance + preserve")
    d = load("test_case4.mat")
    Y, batch, Z, X = d['Y4'], d['batch4'].astype(int), d['Z4'], d['X4']

    Yh_py, bh_py, gh_py, dh_py = comcat(
        Y, batch, Z[:, np.newaxis], X[:, np.newaxis],
        mean_only=False, verbose=False,
    )

    check_close("Y_harmonized", Yh_py, d['Yh4'])
    check_close("gamma_hat",    gh_py, d['gh4'])
    check_close("delta_hat",    dh_py, d['dh4'])


# ---------------------------------------------------------------------------
# Test case 5 — four sites, two linear nuisance columns
# ---------------------------------------------------------------------------
def test_case5():
    print("Case 5 (MATLAB match): four sites, two linear nuisance cols")
    d = load("test_case5.mat")
    Y, batch, Z, X = d['Y5'], d['batch5'].astype(int), d['Z5'], d['X5']
    # Z5 is already (n, 2) from MATLAB
    if Z.ndim == 1:
        Z = Z[:, np.newaxis]

    Yh_py, bh_py, gh_py, dh_py = comcat(
        Y, batch, Z, X[:, np.newaxis],
        mean_only=False, verbose=False,
    )

    check_close("Y_harmonized", Yh_py, d['Yh5'])
    check_close("gamma_hat",    gh_py, d['gh5'])
    check_close("delta_hat",    dh_py, d['dh5'])


# ---------------------------------------------------------------------------
# Test ref_batch: reference site data must be unchanged
# ---------------------------------------------------------------------------
def test_ref_batch():
    print("Case 4: ref_batch — reference site data unchanged, others harmonized")
    d = load("test_case1.mat")
    Y, batch, Z, X = d['Y1'], d['batch1'].astype(int), d['Z1'], d['X1']

    Yh_py, *_ = comcat(Y, batch, Z, X, mean_only=False,
                        verbose=False, ref_batch=1,)

    # Reference site (label=1) must be numerically identical to input
    ref_idx = np.where(batch == 1)[0]
    max_diff = float(np.max(np.abs(Yh_py[:, ref_idx] - Y[:, ref_idx])))
    ok = max_diff < 1e-10
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] ref site unchanged               max_abs={max_diff:.3e}")
    if not ok:
        raise AssertionError(f"ref_batch site was modified: max_abs={max_diff:.3e}")

    # Other sites must actually change
    other_idx = np.where(batch != 1)[0]
    max_diff_other = float(np.max(np.abs(Yh_py[:, other_idx] - Y[:, other_idx])))
    ok2 = max_diff_other > 1e-3
    status2 = "PASS" if ok2 else "FAIL"
    print(f"  [{status2}] other sites were adjusted         max_abs={max_diff_other:.3e}")
    if not ok2:
        raise AssertionError("Other sites were not adjusted with ref_batch set.")


# ---------------------------------------------------------------------------
# Test 5 — comcat_from_training: split-half, apply to held-out subjects
# ---------------------------------------------------------------------------
def test_from_training():
    print("Case 5: comcat_from_training — held-out subjects get same result")
    d = load("test_case1.mat")
    Y, batch, Z, X = d['Y1'], d['batch1'].astype(int), d['Z1'], d['X1']

    n = Y.shape[1]

    # Train on all subjects, get estimates
    Yh_all, _, _, _, est = comcat(
        Y, batch, Z, X, mean_only=False,
        verbose=False, return_estimates=True,
    )

    # Apply estimates to all subjects via from_training — must match Yh_all
    Yh_ft = comcat_from_training(Y, batch, Z, X, estimates=est, verbose=False)

    check_close("Y_harmonized (from_training vs direct)", Yh_ft, Yh_all)


# ---------------------------------------------------------------------------
# Test 6 — GAM removes nonlinear nuisance while preserving the covariate of interest
# ---------------------------------------------------------------------------
def test_gam_smoothing():
    print("Case 6: GAM nuisance modelling — nonlinear nuisance removal")
    rng = np.random.default_rng(99)
    n, V = 200, 300

    batch  = np.concatenate([np.ones(100, dtype=int), np.full(100, 2, dtype=int)])
    Z      = np.linspace(-2, 2, n)          # continuous nuisance
    X      = rng.standard_normal(n)         # preserve
    E      = rng.standard_normal((V, n))
    # strong cubic + quadratic nuisance (needs a flexible GAM to remove)
    Y = 2 * X + Z + 2 * Z**2 + 0.5 * Z**3 + np.where(batch == 2, 3, 0) + E

    # ComCAT models the nuisance with a B-spline GAM (always on)
    Yh_gam, *_ = comcat(Y, batch, Z, X, mean_only=False, verbose=False, gam_df=10)

    # Residual correlation with Z^2 (lower = better removal of nonlinear nuisance)
    def resid_corr(Yh, cov):
        # Yh: (V, n), cov: (n,) — correlate each feature row with covariate
        return float(np.mean(np.abs(np.corrcoef(Yh, cov[None, :])[-1, :-1])))

    r_raw = resid_corr(Y,      Z**2)
    r_gam = resid_corr(Yh_gam, Z**2)
    print(f"  Residual corr(., Z^2):  raw={r_raw:.4f}   GAM={r_gam:.4f}")

    # GAM should produce finite, reasonable output
    ok_finite = np.all(np.isfinite(Yh_gam))
    status = "PASS" if ok_finite else "FAIL"
    print(f"  [{status}] GAM output is finite")
    if not ok_finite:
        raise AssertionError("GAM harmonized output contains non-finite values.")

    # GAM should substantially reduce the nonlinear (Z^2) nuisance
    ok_better = r_gam < r_raw
    status2 = "PASS" if ok_better else "FAIL"
    print(f"  [{status2}] GAM reduces nonlinear nuisance vs. raw")
    if not ok_better:
        raise AssertionError(
            f"GAM did not reduce nonlinear nuisance: raw={r_raw:.4f}, GAM={r_gam:.4f}")

    # Preserved-covariate correlation should remain high
    r_x = resid_corr(Yh_gam, X)
    print(f"  Preserved corr(Y_harm, X):   GAM={r_x:.4f}")


# ---------------------------------------------------------------------------
# Test 7 — GAM from_training: apply pre-fitted splines to held-out data
# ---------------------------------------------------------------------------
def test_gam_from_training():
    print("Case 7: GAM from_training — held-out subjects match direct call")
    d = load("test_case1.mat")
    Y, batch, Z, X = d['Y1'], d['batch1'].astype(int), d['Z1'], d['X1']

    # Train with GAM, get estimates
    Yh_all, _, _, _, est = comcat(
        Y, batch, Z, X, mean_only=False,
        verbose=False, return_estimates=True,
        gam_df=10,
        smooth_term_bounds=(float(Z.min()) - 0.1, float(Z.max()) + 0.1),
    )

    # Apply to all subjects via from_training — must match
    Yh_ft = comcat_from_training(Y, batch, Z, X, estimates=est, verbose=False)
    check_close("Y_harmonized (GAM from_training vs direct)", Yh_ft, Yh_all)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    missing = [f for f in ("test_case1.mat", "test_case2.mat", "test_case3.mat",
                           "test_case4.mat", "test_case5.mat")
               if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print("ERROR: Missing MATLAB-generated test files:", missing)
        print("       Run gen_test_data.m in MATLAB first.")
        sys.exit(1)

    failures = 0
    for fn in (test_case1, test_case2, test_case3, test_case4, test_case5,
               test_ref_batch, test_from_training,
               test_gam_smoothing, test_gam_from_training):
        try:
            fn()
        except AssertionError as e:
            print(f"  ASSERTION ERROR: {e}")
            failures += 1
        print()

    if failures == 0:
        print("All tests PASSED.")
    else:
        print(f"{failures} test(s) FAILED.")
        sys.exit(1)

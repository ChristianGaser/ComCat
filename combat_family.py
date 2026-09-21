"""
combat_family.py — ComBat (empirical Bayes), CovBat and ComBatLS harmonization.

NumPy port of the reference R implementation in the ComBatFamily package
(A. A. Chen et al., https://github.com/andy1764/ComBatFamily, v0.2.2):

    combat_eb()  <->  comfam(data, bat, covar, lm, y ~ covariates)
                      ComBat with empirical Bayes (Johnson et al., 2007;
                      Fortin et al., 2017, 2018)
    covbat()     <->  covfam(data, bat, covar, lm, y ~ covariates)
                      CovBat (Chen et al., 2022, Hum Brain Mapp 43:1179)
    combatls()   <->  combatls(data, bat, covar, y ~ covariates,
                               sigma.formula = ~ covariates)
                      i.e. comfam(model = gamlss, family = NO(), ...)
                      ComBatLS (Gardner et al., 2025, Hum Brain Mapp 46:e70197)

The R package fits one regression model per feature.  Here all features are
fitted together with matrix operations, so voxel- or vertex-wise data with
10^4-10^5 features are harmonized in seconds to minutes instead of hours
(one gamlss fit per feature for ComBatLS).  tests/test_combat_family.py
compares the results against the R package when R is available.

Conventions follow comcat.py
----------------------------
Y         : (n_features, n_subjects)   (transposed automatically if needed)
batch     : (n_subjects,) site/scanner labels, at least two sites
preserve  : (n_subjects, n_preserve) covariates whose effects are kept,
            entered linearly (as ComCAT enters its preserve covariates)

None of these methods removes covariate effects: every covariate in the model
is preserved.  They have no counterpart to ComCAT's nuisance (IQM) removal.

Returns
-------
Every public function returns (Y_harmonized, estimates):
Y_harmonized : (n_features, n_subjects), same orientation as the input
estimates    : dict with the batch parameters mapped to the full feature space
               ('gamma_hat', 'delta_hat', 'gamma_star', 'delta_star', each of
               shape (n_batch, n_features)), 'batch_levels', 'mask' and
               method-specific entries (see each function).

Features that are constant or contain non-finite values are left unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np

# R: while (change > 10e-5)  — note 10e-5 == 1e-4
_EB_CONV = 1e-4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def combat_eb(
    Y: np.ndarray,
    batch: np.ndarray,
    preserve: np.ndarray | None = None,
    eb: bool = True,
    mean_only: bool = False,
    verbose: bool = False,
):
    """ComBat with empirical Bayes — port of ComBatFamily::comfam(model = lm).

    Parameters
    ----------
    Y         : (n_features, n_subjects) data
    batch     : (n_subjects,) site labels
    preserve  : (n_subjects, n_preserve) covariates to preserve, or None
    eb        : use empirical Bayes shrinkage of the batch parameters
    mean_only : adjust batch means only (batch variances are left unchanged)
    verbose   : print progress
    """
    prep = _prepare(Y, batch, preserve, 'ComBat', verbose)
    Ym = prep.valid()

    stand_mean, stand_sd, var_pooled = _standardize_lm(Ym, prep.batch_mod, prep.covar)
    Y_adj, est = _adjust_batches(Ym, stand_mean, stand_sd, prep.batches,
                                 eb, mean_only, 'ComBat', verbose)
    est['var_pooled'] = var_pooled
    return prep.finish(Y_adj, est)


def covbat(
    Y: np.ndarray,
    batch: np.ndarray,
    preserve: np.ndarray | None = None,
    eb: bool = True,
    mean_only: bool = False,
    percent_var: float = 0.95,
    n_pc: int | None = None,
    std_var: bool = True,
    score_eb: bool = False,
    verbose: bool = False,
):
    """CovBat — port of ComBatFamily::covfam(model = lm).

    Step 1 runs ComBat (combat_eb).  Step 2 runs a PCA of the ComBat residuals
    (covariate effects removed) and harmonizes the mean and variance of the
    leading principal-component scores across sites (ComBat without covariates),
    which removes site differences in the covariance between features.
    Finally the scores are projected back and covariate effects are added.

    Parameters
    ----------
    percent_var : number of harmonized PCs is the smallest number explaining
                  more than this proportion of the variance (R default 0.95)
    n_pc        : harmonize exactly this many PCs (overrides percent_var)
    std_var     : scale features to unit variance before the PCA (R default)
    score_eb    : use empirical Bayes when harmonizing the scores (R default False)
    Remaining parameters as in combat_eb(); eb and mean_only apply to step 1.

    Additional estimates
    --------------------
    'n_pc'          : number of harmonized PCs
    'pc_var'        : proportion of variance explained by each PC
    'score_gamma_hat', 'score_delta_hat' : (n_batch, n_pc) site effects on the scores
    """
    prep = _prepare(Y, batch, preserve, 'CovBat', verbose)
    Ym = prep.valid()
    n = Ym.shape[1]

    # ------------------------------------------------ step 1: ComBat
    stand_mean, stand_sd, var_pooled = _standardize_lm(Ym, prep.batch_mod, prep.covar)
    Y_com, est = _adjust_batches(Ym, stand_mean, stand_sd, prep.batches,
                                 eb, mean_only, 'CovBat', verbose)
    del Ym

    # ------------------------------------------------ step 2: PCA of residuals
    # R: com_res <- dat.combat - stand.mean;  prcomp(com_res, center=TRUE, scale.=std.var)
    R = (Y_com - stand_mean).T                     # (n_subjects, n_valid)
    del Y_com
    center = R.mean(axis=0)
    R -= center
    if std_var:
        scale = R.std(axis=0, ddof=1)
        scale[scale == 0] = 1.0
        R /= scale
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    del R
    scores = U * S                                 # prcomp()$x
    pc_var = S ** 2 / np.sum(S ** 2)

    if n_pc is not None:
        npc = int(n_pc)
        if not 1 <= npc <= len(S):
            raise ValueError(f"n_pc={n_pc} must be between 1 and {len(S)}.")
    else:
        # R: which(cumsum(sdev^2 / sum(sdev^2)) > percent.var)[1]
        above = np.cumsum(pc_var) > percent_var
        npc = int(np.argmax(above)) + 1 if above.any() else len(S)
    if verbose:
        print(f"[CovBat] Harmonizing {npc} of {len(S)} PC scores "
              f"({100 * np.sum(pc_var[:npc]):.1f}% of variance)")

    # ComBat without covariates on the scores (R: comfam(scores, bat, eb = FALSE))
    sc = scores[:, :npc].T                         # (npc, n_subjects)
    sc_mean, sc_sd, _ = _standardize_lm(sc, prep.batch_mod, np.empty((n, 0)))
    sc_adj, sc_est = _adjust_batches(sc, sc_mean, sc_sd, prep.batches,
                                     score_eb and npc > 1, False, 'CovBat', False)
    scores[:, :npc] = sc_adj.T

    # ------------------------------------------------ project back
    X_rec = scores @ Vt
    del scores, Vt
    if std_var:
        X_rec *= scale
    X_rec += center
    Y_adj = X_rec.T + stand_mean

    est['var_pooled'] = var_pooled
    Y_harm, est_full = prep.finish(Y_adj, est)
    est_full['n_pc'] = npc
    est_full['pc_var'] = pc_var
    est_full['score_gamma_hat'] = sc_est['gamma_hat']
    est_full['score_delta_hat'] = sc_est['delta_hat']
    return Y_harm, est_full


def combatls(
    Y: np.ndarray,
    batch: np.ndarray,
    preserve: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    eb: bool = True,
    mean_only: bool = False,
    max_iter: int = 2000,
    tol: float = 1e-8,
    verbose: bool = False,
):
    """ComBatLS — port of ComBatFamily::combatls() (comfam with gamlss, NO family).

    Each feature is modelled as a normal location-scale model
        y ~ N(mu, sigma^2),  mu = batch + preserve,  log(sigma) = 1 + scale
    fitted by maximum likelihood (RS algorithm as in gamlss).  Data are
    standardized with the covariate-dependent sigma, site effects in location
    and scale are removed as in ComBat, and the covariate effects on both the
    mean and the variance are reintroduced.  Batch is not part of the sigma
    model, as in the ComBatLS paper (sigma.formula = ~ age + sex).

    Parameters
    ----------
    scale    : (n_subjects, n_scale) covariates for log(sigma).  None (default)
               uses the preserve covariates, as in the ComBatLS paper.  An empty
               array gives a constant sigma, which reduces ComBatLS to ComBat.
    max_iter : maximum number of RS iterations per feature
    tol      : convergence tolerance on the change of log(sigma) and of the
               fitted mean relative to sigma (gamlss uses a looser deviance
               criterion, c.crit = 0.001)
    Remaining parameters as in combat_eb().

    Additional estimates
    --------------------
    'theta'     : (1 + n_scale, n_features) coefficients of log(sigma)
    'converged' : (n_features,) bool, RS algorithm converged
    """
    prep = _prepare(Y, batch, preserve, 'ComBatLS', verbose)
    n = prep.n_subjects
    if scale is None:
        scale = prep.covar
    else:
        scale = _to_col_matrix(scale, n)
    W = np.hstack([np.ones((n, 1)), scale])
    if np.linalg.matrix_rank(W) < W.shape[1]:
        raise ValueError("Scale covariates are collinear (or constant).")

    Ym = prep.valid()
    X = np.hstack([prep.batch_mod, prep.covar])
    if verbose:
        print(f"[ComBatLS] Fitting location-scale model for {Ym.shape[0]} features "
              f"({prep.covar.shape[1]} location, {scale.shape[1]} scale covariate(s))")
    beta, theta, converged = _fit_location_scale(Ym, X, W, max_iter, tol)
    if not np.all(converged):
        warnings.warn(
            f"ComBatLS location-scale model did not converge for "
            f"{int(np.sum(~converged))} feature(s). Results may be unreliable.",
            RuntimeWarning, stacklevel=2,
        )

    stand_mean = _stand_mean(beta, prep.batch_mod, prep.covar)
    stand_sd = np.exp(W @ theta).T                 # (n_valid, n_subjects)
    Y_adj, est = _adjust_batches(Ym, stand_mean, stand_sd, prep.batches,
                                 eb, mean_only, 'ComBatLS', verbose)

    Y_harm, est_full = prep.finish(Y_adj, est)
    est_full['theta'] = prep.to_full(theta)
    conv_full = np.ones(prep.mask.shape[0], dtype=bool)
    conv_full[prep.mask] = converged
    est_full['converged'] = conv_full
    return Y_harm, est_full


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

class _Prepared:
    """Validated inputs plus the bookkeeping needed to rebuild full outputs."""

    def __init__(self, Y, transposed, mask, levels, codes, covar):
        self.Y = Y                                 # (n_features, n_subjects)
        self.transposed = transposed
        self.mask = mask                           # features that are harmonized
        self.levels = levels
        self.n_subjects = len(codes)
        n_batch = len(levels)
        self.batches = [np.flatnonzero(codes == b) for b in range(n_batch)]
        self.batch_mod = (codes[:, None] == np.arange(n_batch)[None, :]).astype(float)
        self.covar = covar

    def to_full(self, arr):
        """Map (k, n_valid) parameters to (k, n_features), zeros elsewhere."""
        full = np.zeros((arr.shape[0], self.mask.shape[0]), dtype=np.float64)
        full[:, self.mask] = arr
        return full

    def valid(self):
        """Data of the harmonized features (a view when all features are valid)."""
        return self.Y if self.mask.all() else self.Y[self.mask]

    def finish(self, Y_adj, est):
        Y_harm = self.Y                            # private float64 copy; filled in place
        bad = ~np.all(np.isfinite(Y_adj), axis=1)
        if np.any(bad):
            warnings.warn(
                f"{int(bad.sum())} feature(s) gave non-finite harmonized values "
                "and were left unchanged.", RuntimeWarning, stacklevel=3,
            )
            Y_adj[bad] = self.valid()[bad]
        Y_harm[self.mask] = Y_adj

        est_full = {k: self.to_full(v) for k, v in est.items() if v.ndim == 2}
        if 'var_pooled' in est:
            est_full['var_pooled'] = self.to_full(est['var_pooled'][None, :])[0]
        est_full['batch_levels'] = self.levels
        est_full['mask'] = self.mask
        if self.transposed:
            Y_harm = Y_harm.T
        return Y_harm, est_full


def _prepare(Y, batch, preserve, tag, verbose) -> _Prepared:
    if batch is None or np.asarray(batch).size == 0:
        raise ValueError(f"{tag} needs site labels (batch); got none.")
    batch = np.asarray(batch).ravel()
    n = batch.size

    Y = np.array(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D, got shape {Y.shape}.")
    transposed = False
    if Y.shape[1] != n:
        if Y.shape[0] == n:
            Y = Y.T
            transposed = True
        else:
            raise ValueError(f"Shape mismatch: Y {Y.shape}, n_subjects={n}")

    levels, codes = np.unique(batch, return_inverse=True)
    counts = np.bincount(codes)
    if len(levels) < 2:
        raise ValueError(f"{tag} needs at least two sites; all subjects share "
                         f"site {levels[0]!r}.")
    if np.any(counts < 2):
        raise ValueError(f"Site(s) {levels[counts < 2].tolist()} contain a single "
                         "subject; their variance cannot be estimated. Merge or "
                         "drop them first.")

    covar = _to_col_matrix(preserve, n)
    X = np.hstack([(codes[:, None] == np.arange(len(levels))).astype(float), covar])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("Preserve covariates are confounded with site (design "
                         "matrix is rank-deficient). Remove the confounded "
                         "covariate(s).")

    finite = np.all(np.isfinite(Y), axis=1)
    sd = np.zeros(Y.shape[0])
    sd[finite] = np.std(Y[finite], axis=1, ddof=1)
    mask = finite & (sd > 0)
    if not np.any(mask):
        raise ValueError("No feature with finite, non-constant values.")

    if verbose:
        print(f"[{tag}] {n} subjects, {len(levels)} sites, "
              f"{covar.shape[1]} preserved covariate(s), "
              f"{int(mask.sum())} of {mask.size} features harmonized")
    return _Prepared(Y, transposed, mask, levels, codes, covar)


def _to_col_matrix(arr, n: int) -> np.ndarray:
    """Return an (n, k) float matrix; None or empty gives (n, 0)."""
    if arr is None or np.asarray(arr).size == 0:
        return np.empty((n, 0), dtype=np.float64)
    arr = np.array(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != n:
        arr = arr.T
    if arr.shape[0] != n:
        raise ValueError(f"Covariate shape {arr.shape} does not match "
                         f"n_subjects={n}.")
    return arr


# ---------------------------------------------------------------------------
# Standardization models
# ---------------------------------------------------------------------------

def _stand_mean(beta, batch_mod, covar):
    """Covariate-adjusted grand mean, R: predict(fit, newdata = pmod).

    pmod replaces each subject's batch indicators by the batch proportions
    n_b / n, so the intercept is the size-weighted mean of the batch means.
    """
    n_batch = batch_mod.shape[1]
    props = batch_mod.mean(axis=0)
    grand_mean = props @ beta[:n_batch]            # (n_valid,)
    stand_mean = np.repeat(grand_mean[:, None], batch_mod.shape[0], axis=1)
    if covar.shape[1] > 0:
        stand_mean += (covar @ beta[n_batch:]).T
    return stand_mean


def _standardize_lm(Ym, batch_mod, covar):
    """ComBat standardization (model = lm): OLS on [batch one-hot, covariates]."""
    X = np.hstack([batch_mod, covar])
    beta = np.linalg.lstsq(X, Ym.T, rcond=None)[0]   # (k, n_valid)
    resid = Ym - (X @ beta).T
    # R: apply(data - resid_mean, 2, var) * (n - 1)/n
    var_pooled = np.var(resid, axis=1)
    del resid
    return _stand_mean(beta, batch_mod, covar), np.sqrt(var_pooled)[:, None], var_pooled


def _deviance(r, eta):
    """-2 log-likelihood of N(mu, exp(eta)^2) given residuals r = y - mu."""
    return np.sum(np.log(2 * np.pi) + 2 * eta + r ** 2 * np.exp(-2 * eta), axis=0)


def _fit_location_scale(Ym, X, W, max_iter, tol, max_elements=4_000_000):
    """ML fit of y_j ~ N(X beta_j, exp(W theta_j)^2) for every feature j.

    RS algorithm as in gamlss (family NO, identity link for mu, log link for
    sigma): alternate a weighted least-squares update of beta (weights
    1/sigma^2) with a Fisher-scoring update of theta (working weights 2),
    halving the theta step if the deviance increases.  Features are processed
    in chunks so memory stays bounded for large data.

    Returns beta (kx, p), theta (kw, p), converged (p,)
    """
    p, n = Ym.shape
    kx, kw = X.shape[1], W.shape[1]
    XX = (X[:, :, None] * X[:, None, :]).reshape(n, kx * kx)
    X_pinv = np.linalg.pinv(X)
    W_pinv = np.linalg.pinv(W)

    beta = np.empty((kx, p))
    theta = np.empty((kw, p))
    converged = np.zeros(p, dtype=bool)

    step = max(1, max_elements // n)
    for start in range(0, p, step):
        y_all = Ym[start:start + step].T           # (n, pc)

        # start: OLS mean, constant sigma
        b_all = X_pinv @ y_all
        r = y_all - X @ b_all
        t_all = np.zeros((kw, y_all.shape[1]))
        t_all[0] = 0.5 * np.log(np.mean(r ** 2, axis=0))

        # iterate only features that have not converged yet; a few features
        # with strong scale effects can need hundreds of RS iterations
        active = np.arange(y_all.shape[1])
        for _ in range(max_iter):
            y, b, t = y_all[:, active], b_all[:, active], t_all[:, active]
            eta = W @ t

            # mu step: weighted least squares with weights 1/sigma^2
            w = np.exp(-2 * eta)
            A = (XX.T @ w).T.reshape(-1, kx, kx)
            rhs = (X.T @ (w * y)).T[..., None]
            b_new = np.linalg.solve(A, rhs)[..., 0].T
            d_mu = np.max(np.abs(X @ (b_new - b)) * np.exp(-eta), axis=0)
            r = y - X @ b_new

            # sigma step: Fisher scoring on log(sigma)
            dev_old = _deviance(r, eta)
            d_t = W_pinv @ (0.5 * (r ** 2 * w - 1))
            t_new = t + d_t
            eta_new = W @ t_new
            dev_new = _deviance(r, eta_new)
            for _ in range(20):                    # step halving (gamlss autostep)
                worse = dev_new > dev_old
                if not np.any(worse):
                    break
                d_t[:, worse] *= 0.5
                t_new[:, worse] = t[:, worse] + d_t[:, worse]
                eta_new[:, worse] = W @ t_new[:, worse]
                dev_new[worse] = _deviance(r[:, worse], eta_new[:, worse])
            d_theta = np.max(np.abs(t_new - t), axis=0)

            b_all[:, active] = b_new
            t_all[:, active] = t_new
            done = (d_mu < tol) & (d_theta < tol)
            converged[start + active[done]] = True
            active = active[~done]
            if active.size == 0:
                break

        beta[:, start:start + step] = b_all
        theta[:, start:start + step] = t_all

    return beta, theta, converged


# ---------------------------------------------------------------------------
# Batch effect estimation and removal
# ---------------------------------------------------------------------------

def _adjust_batches(Ym, stand_mean, stand_sd, batches, eb, mean_only, tag, verbose):
    """Estimate and remove batch effects in location and scale.

    Port of the '#### Obtain location and scale adjustments ####' and
    '#### Harmonize the data ####' sections of ComBatFamily::comfam.
    """
    data = np.subtract(Ym, stand_mean)             # data_stand
    data /= stand_sd

    gamma_hat = np.vstack([data[:, idx].mean(axis=1) for idx in batches])
    delta_hat = np.vstack([data[:, idx].var(axis=1, ddof=1) for idx in batches])

    if eb and data.shape[0] > 1:
        if verbose:
            print(f"[{tag}] Empirical Bayes estimation of site effects")
        gamma_star, delta_star = _eb_estimates(data, batches, gamma_hat, delta_hat)
    else:
        gamma_star, delta_star = gamma_hat.copy(), delta_hat.copy()

    if mean_only:
        delta_star = np.ones_like(delta_star)

    if verbose:
        print(f"[{tag}] Adjusting the data")
    for i, idx in enumerate(batches):
        data[:, idx] = ((data[:, idx] - gamma_star[i][:, None])
                        / np.sqrt(delta_star[i])[:, None])

    data *= stand_sd
    data += stand_mean
    est = dict(gamma_hat=gamma_hat, delta_hat=delta_hat,
               gamma_star=gamma_star, delta_star=delta_star)
    return data, est


def _eb_estimates(data, batches, gamma_hat, delta_hat):
    """Parametric empirical Bayes, literal port of the comfam() loop."""
    gamma_star = np.empty_like(gamma_hat)
    delta_star = np.empty_like(delta_hat)

    for i, idx in enumerate(batches):
        n_b = len(idx)

        # method of moments estimates of the priors
        g_bar = np.mean(gamma_hat[i])
        g_var = np.var(gamma_hat[i], ddof=1)
        d_bar = np.mean(delta_hat[i])
        d_var = np.var(delta_hat[i], ddof=1)
        d_a = (2 * d_var + d_bar ** 2) / d_var
        d_b = (d_bar * d_var + d_bar ** 3) / d_var

        bdat = data[:, idx]
        g_orig = gamma_hat[i]
        g_old = gamma_hat[i]
        d_old = delta_hat[i]

        change_old = 1.0
        change = 1.0
        count = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            while change > _EB_CONV:
                g_new = (n_b * g_var * g_orig + d_old * g_bar) / (n_b * g_var + d_old)
                sum2 = np.sum((bdat - g_new[:, None]) ** 2, axis=1)
                d_new = (sum2 / 2 + d_b) / (n_b / 2 + d_a - 1)

                # R: max(abs(g_new - g_old)/g_old, abs(d_new - d_old)/d_old)
                change = np.nanmax(np.concatenate([np.abs(g_new - g_old) / g_old,
                                                   np.abs(d_new - d_old) / d_old]))
                if count > 30 and change > change_old:
                    warnings.warn(
                        "Empirical Bayes step failed to converge after 30 "
                        "iterations, using estimate before change between "
                        "iterations increases.", RuntimeWarning, stacklevel=4,
                    )
                    break
                g_old, d_old = g_new, d_new
                change_old = change
                count += 1

        gamma_star[i] = g_new
        delta_star[i] = d_new

    return gamma_star, delta_star

# Decentralized ComCAT — GAM Basis Equivalence (Mode A)

This documents how ComCAT's **GAM B-spline nuisance basis** is reproduced in a
decentralized setting so the result matches a centralized `comcat()` run without
pooling raw imaging data. It is implemented in
[`decentralized_comcat.py`](../decentralized_comcat.py) and builds on the
decentralized fit POC ([`poc_decentralized_comcat.py`](poc_decentralized_comcat.py)),
which already proved the regression / standardization / L/S steps reproduce
centralized output to machine precision.

**Scope decisions (as implemented):**

- **Mode A only** — knots are reconstructed from the *unique values* of the
  smooth covariate, shared across sites. This is bitwise-exact and matches the
  current centralized quantile-knot default.
- **GAM B-splines only** — ComCAT uses B-splines; the polynomial nuisance path is
  not decentralized (it existed only for B-spline-vs-polynomial comparison).
- **`comcat.py` is not modified** — equivalence is achieved by constructing
  `statsmodels` BSplines with *injected* knots and handing them to the existing
  `spline_constructors` hook that `comcat_from_training` already honours.

---

## 1. Problem statement

In `comcat()`, each smooth nuisance column is expanded into a B-spline basis
before entering the design matrix (`_build_nuisance_basis`, [comcat.py](../comcat.py)):
`statsmodels` `BSplines` with `df`, `degree=3`.

The only **data-dependent** quantity is the inner knot vector — the empirical
quantiles of the covariate (boundary knots = min/max unless bounds are given).
In a decentralized setting the covariate is split across sites, so the knot fit
cannot be computed naively without pooling.

Everything *downstream* of the basis (the design-matrix regression, std, γ, δ,
adjustment) is already solved by the POC. **So GAM-equivalence reduces to one
question:**

> How do we determine the B-spline knot vector from cross-site information,
> identically to centralized `comcat()`?

Once the knots are known, basis evaluation is purely local
(`BSplines.transform(local_x)`) — the same mechanism the existing apply path
(`comcat_from_training`) already uses via `spline_constructors`.

---

## 2. Key empirical findings (validated against statsmodels 0.14.6)

1. **Knots can be injected.** `BSplines(x, df, degree, knot_kwds=[{'knots':
   inner, 'lower_bound':lo, 'upper_bound':hi}])` builds a basis with explicit
   inner knots, bypassing quantile fitting. The resulting `.basis` and
   `.transform(x_new)` are **bitwise identical** to a quantile-fit `BSplines`
   that landed on the same knots.

2. **The aggregator can reproduce statsmodels' exact knots.** statsmodels places
   inner knots from `unique(x within bounds)` via an R-compatible quantile rule.
   Recomputing on the pooled set of unique covariate values reproduces the inner
   knots **bitwise**; boundary knots are the global min/max. Because the
   constructor takes `unique()` internally, passing the pooled unique values is
   equivalent to passing the pooled raw column.

3. **`bs.basis == bs.transform(same x)` bitwise**, and `transform()` depends only
   on the stored knots. Centralized `comcat()` builds the basis from `bs.basis`
   while decentralized sites build it from `bs.transform()` — these match.

Consequence: **B-spline equivalence is bitwise**, not merely machine-precision.

---

## 3. Design: separate "knot fit" (global, once) from "basis eval" (local)

The basis-defining parameters are captured in a small, serializable **spec** per
smooth column:

```python
spec = {degree, df, lower_bound, upper_bound, inner_knots}
```

Pipeline roles (mirroring Bostami et al.):

```text
  site  ── unique covariate values ──▶  aggregator ── spec(s) ──▶  site
                                        (reconstruct knots)        (build constructor,
                                                                    evaluate basis locally)
```

The knot fit is a single extra round at the front of the decentralized protocol
(before the design-matrix regression), because the expanded nuisance columns are
part of the design matrix `XZ`.

Helpers in [`decentralized_comcat.py`](../decentralized_comcat.py):
`bspline_spec_from_pooled_unique`, `build_constructor`, `expand_nuisance`.

---

## 4. Mode A — shared unique covariate values

Each site sends, **for smooth columns only**, its sorted unique covariate values
(one short 1-D vector per column; no subject linkage, no imaging data). The
aggregator forms `unique(∪ site uniques)`, then reconstructs the exact knot spec
(Finding 2). Imaging data `Y` never leaves a site.

| Property | Value |
|----------|-------|
| Equivalence | **bitwise** (vs. centralized quantile-knot default) |
| Covariate leakage | sorted unique values of smooth columns only |
| Imaging data leakage | none |

Rationale: covariates such as age/sex/TIV are routinely shareable in federated
harmonization; the sensitive imaging data never moves, and de-duplicating to
unique values weakens any linkage.

`gam_df` is computed centrally from the pooled `n` (same heuristic as
[comcat.py:156](../comcat.py#L156)) so every site uses an identical value.

---

## 5. Protocol

Round B (knot fit) is added before the regression rounds from the POC:

```text
Round 0   sites→agg : n_i, ΣY, ΣY²                         → common mask, n
Round B   sites→agg : unique smooth-covariate values        → knot spec(s)   (NEW)
          agg→sites : spec(s)  (each site builds an identical constructor)
Round 1   sites→agg : XᵢᵀXᵢ, XᵢᵀDᵢ   (design incl. expanded basis) → β
Round 2   sites→agg : resid SS, grand-mean parts            → std_pooled, grand_mean
Round 3   sites→agg : Xnᵢᵀ·, per-batch var, Σz, Σz²         → γ, δ
Apply     local      (reuses comcat_from_training)
```

Rounds 0 and B are both upstream covariate/`Y`-summary exchanges and can be
merged into one message. Assumes the Bostami et al. topology: **each site holds
exactly one batch**, so per-batch variance (δ batch rows) is computed locally;
the nuisance δ rows are pooled.

---

## 6. Why `comcat.py` stays untouched

`comcat_from_training` ([comcat.py:512](../comcat.py#L512)) already reads
`estimates['spline_constructors']` and applies basis columns via
`bs.transform()` ([comcat.py:478](../comcat.py#L478)). The decentralized module
assembles an `estimates` dict whose `spline_constructors` are BSplines built with
injected knots — so both the fit rounds and the apply step evaluate an identical
basis with no library changes.

---

## 7. Equivalence achieved (verified)

`decentralized_comcat.py` self-test (3 sites, GAM on `age`, `preserve` covariate):

- **GAM basis: bitwise-identical** to centralized.
- **Overall harmonized output: ~1e-13** vs. centralized — the residual comes from
  the regression rounds using the normal-equations form
  `pinv(ΣXᵢᵀXᵢ)·ΣXᵢᵀYᵢ` (identity `pinv(X)=pinv(XᵀX)Xᵀ`), the same machine-precision
  behaviour as the POC and the order reported in Bostami et al. (3e-15).

---

## 8. Not implemented (deliberately)

- **Polynomial nuisance decentralization** — ComCAT uses GAM; the polynomial path
  was only for comparison.
- **Moment-only / explicit-knot knot sourcing** — Mode A covers the needed cases;
  other modes would only reduce covariate-value disclosure further at the cost of
  exactness or extra user burden.
- **Real network transport** — the module simulates message-passing in-process.
  Wiring to a transport (e.g. COINSTAC) means serializing the per-round dicts; the
  knot **spec** is already serialization-friendly (plain floats + a short array)
  rather than live `BSplines` objects.
```

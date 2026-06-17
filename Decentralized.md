# Decentralized ComCAT — Usage

Harmonize data that is split across **sites which never share raw imaging data**.
Each site keeps its own `Y`; only aggregate statistics are exchanged, and the
result matches a centralized [`comcat()`](comcat.py) run to machine precision
(the GAM B-spline basis is reproduced bitwise).

> **Status: experimental.** This is a standalone module
> ([`decentralized_comcat.py`](decentralized_comcat.py)). It is not wired into
> `comcat_ui` / the CLI and is not part of the packaged `comcat*` modules. The
> message-passing is simulated in-process (see [Real deployments](#real-deployments)).
>
> For the algorithm/design and equivalence proofs, see
> [tests/DECENTRALIZED-GAM-DESIGN.md](tests/DECENTRALIZED-GAM-DESIGN.md); for a
> minimal worked example without GAM, see
> [tests/poc_decentralized_comcat.py](tests/poc_decentralized_comcat.py).

---

## When to use it

Use the decentralized path when datasets cannot be pooled (data-use agreements,
confidentiality, storage), but you still want a single harmonization equivalent
to running centralized ComCAT on the combined data.

If you *can* pool the data, use [`comcat()`](comcat.py) directly — it is simpler
and is the reference this module reproduces.

---

## Assumptions & scope

- **One batch per site.** Each site holds exactly one batch/scanner (the topology
  from Bostami et al. 2022). Per-batch variance is therefore computed locally.
- **GAM B-splines only** (Mode A). Every nuisance column is modelled with a
  B-spline GAM, matching centralized ComCAT (which always uses GAM).
- **`ref_batch` is not supported** in this module (all sites are harmonized to the
  pooled grand mean). Everything else — `preserve` covariates, `mean_only`,
  the common feature mask — works as in `comcat()`.
- Requires `statsmodels` (for the B-spline basis).

---

## What leaves a site (privacy)

Raw imaging data `Y` **never** leaves a site. Across the protocol rounds, each
site sends only aggregate statistics:

| Round | Sent to aggregator | Purpose |
|-------|--------------------|---------|
| 0 | feature counts, `ΣY`, `ΣY²` | common feature mask |
| B | **unique values** of each smooth covariate | reconstruct exact B-spline knots |
| 1 | design Gram blocks `XᵢᵀXᵢ`, `XᵢᵀYᵢ` | global regression (β) |
| 2 | residual sums, grand-mean parts | pooled std, grand mean |
| 3 | nuisance Gram blocks, per-batch variances | L/S model (γ, δ) |

The only covariate-level disclosure is the **sorted unique values** of the smooth
columns (e.g. age) — not subject linkage, not imaging data. (This is "Mode A" in
the design doc; it is what makes the knots — and therefore the basis — bitwise
identical to centralized.)

---

## Quick start

The inputs are **per-site lists** (one entry per site), all in the same site
order:

```python
import numpy as np
from decentralized_comcat import decentralized_harmonize

# --- data held at each site (never pooled) -------------------------------
# site_Y[k]:        (n_features, n_subjects_at_site_k)
# site_nuisance[k]: covariate(s) to remove, e.g. age   (or None)
# site_preserve[k]: covariate(s) to keep,   e.g. score (or None)
site_Y        = [Y_siteA, Y_siteB, Y_siteC]
site_nuisance = [age_A,   age_B,   age_C]
site_preserve = [score_A, score_B, score_C]

# each site holds one batch; label them and give each site its batch index
batch_levels   = ["siteA", "siteB", "siteC"]   # the batch labels
site_batch_idx = [0, 1, 2]                      # index into batch_levels per site

harmonized, estimates = decentralized_harmonize(
    site_Y, site_batch_idx, site_nuisance, site_preserve, batch_levels,
)
# harmonized[k] is the harmonized (n_features, n_subjects) array for site k
```

`harmonized[k]` is bit-for-bit the columns that centralized
`comcat(np.hstack(site_Y), batch, nuisance=..., preserve=...)` would produce for
site *k* (up to ~1e-13 from the regression; the GAM basis is exact).

---

## Two-step API: fit once, apply locally

`decentralized_harmonize` is a convenience wrapper. The two underlying steps
mirror a real federated run:

```python
from decentralized_comcat import decentralized_fit
from comcat import comcat_from_training

# 1) FIT — run the aggregation rounds, produce the global `estimates` dict.
#    (In a real deployment this is the aggregator's job; see below.)
estimates = decentralized_fit(
    site_Y, site_batch_idx, site_nuisance, site_preserve, batch_levels,
    smooth_cols=None,   # default: GAM on every nuisance column
    gam_df=None,        # default: min(10, max(5, n_total // 30))
    degree=3,
    mean_only=False,
)

# 2) APPLY — each site harmonizes its own data locally with the shared estimates.
labels_k = np.full(site_Y[k].shape[1], batch_levels[site_batch_idx[k]])
Y_harm_k = comcat_from_training(
    site_Y[k], labels_k,
    nuisance=site_nuisance[k], preserve=site_preserve[k],
    estimates=estimates,
)
```

The `estimates` dict is the same structure produced by
`comcat(..., return_estimates=True)`, so the apply step reuses the existing,
unmodified [`comcat_from_training()`](comcat.py).

### Key parameters

| Parameter | Meaning |
|-----------|---------|
| `site_Y` | list of `(n_features, n_subjects)` arrays, one per site |
| `site_batch_idx` | per-site batch index into `batch_levels` (usually `[0,1,…]`) |
| `site_nuisance` | per-site nuisance covariate(s) to remove, or `None` |
| `site_preserve` | per-site covariate(s) to preserve, or `None` |
| `batch_levels` | the batch labels (length = number of sites/batches) |
| `smooth_cols` | nuisance column indices to model with GAM (default: all) |
| `gam_df` | B-spline basis dimension (default: sample-size heuristic on pooled `n`) |
| `mean_only` | adjust mean only, no variance scaling |

---

## Equivalence guarantees

Verified by the module's self-test (`python decentralized_comcat.py`):

- **GAM B-spline basis: bitwise-identical** to centralized — the aggregator
  reconstructs statsmodels' exact knot vector from the pooled unique covariate
  values.
- **Overall harmonized output: ~1e-13** vs. centralized. The small residual comes
  from decentralized regression using the normal-equations form
  `pinv(ΣXᵢᵀXᵢ)·ΣXᵢᵀYᵢ` (the same machine-precision behaviour reported in Bostami
  et al., 3e-15).

---

## Real deployments

This module **simulates** message-passing in-process: the "sites" are just entries
in the input lists, and the aggregation functions are called directly. To run it
across machines you wire the per-round payloads through your transport (e.g.
COINSTAC, sockets, message queue):

- The site-side functions (`site_round0/1/2/3`, `site_basis_stats`) produce the
  payloads to send up.
- The aggregator-side functions (`aggregate_mask`, `aggregate_basis_specs`,
  `aggregate_beta`, `aggregate_std_grandmean`, `aggregate_gamma_delta`) combine
  them and produce values to broadcast down.
- Payloads are plain NumPy arrays / dicts. The B-spline **knot spec**
  (`bspline_spec_from_pooled_unique` → `build_constructor`) is serialization-
  friendly (floats + a short array), so you broadcast the spec rather than a live
  `statsmodels` object.

See [tests/DECENTRALIZED-GAM-DESIGN.md](tests/DECENTRALIZED-GAM-DESIGN.md) §5 for
the round-by-round protocol.

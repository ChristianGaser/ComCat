> [!WARNING]
> This project is **currently under construction** and might contain bugs. **If you experience any issues, please [let me know](https://github.com/ChristianGaser/ComCat/issues)!**

> [!CAUTION]
> **Current scope is limited to BrainAGE and normative modelling workflows.**  
> For general GLM-based group analyses, harmonizing data with ComCAT and then running a standard GLM inflates degrees of freedom and leads to inflated false-positive rates. The two-step correction proposed by Li et al. that addresses this problem is **not yet implemented** in the production pipeline (`comcat_ui.py`). It is only available inside the simulation framework (`simulate_comcat.py`) for evaluation purposes. Until the Li et al. correction is integrated, avoid using ComCAT-harmonized data as input to group-comparison GLMs.

# ComCAT — Combating CovariATe Effects

ComCAT is a Python toolkit for **harmonizing multi-site neuroimaging data**. It removes scanner/site batch effects and unwanted nuisance covariate effects from high-dimensional data (e.g., voxel- or vertex-wise brain maps) while preserving biological signals of interest such as age or group membership. ComCAT is a heavily extended Python port of the original MATLAB-based ComBat/ComCAT implementation.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [File Overview](#file-overview)
- [Core API — `comcat.py`](#core-api--comcatpy)
  - [`comcat()` — harmonize data](#comcat--harmonize-data)
  - [`comcat_from_training()` — apply to new data](#comcat_from_training--apply-to-new-data)
  - [Nuisance modelling: B-spline GAM](#nuisance-modelling-b-spline-gam)
- [File I/O Interface — `comcat_ui.py`](#file-io-interface--comcat_uipy)
  - [Python usage](#python-usage)
  - [Command-line usage](#command-line-usage)
  - [Supported file formats](#supported-file-formats)
  - [Output naming](#output-naming)
- [Batch Processing — `run_comcat_from_files.py`](#batch-processing--run_comcat_from_filespy)
- [Simulation Tools](#simulation-tools)
  - [`simulate_comcat.py`](#simulate_comcatpy)
  - [`simulate_comcat_ui.py`](#simulate_comcat_uipy)
- [Testing — `test_comcat_py.py`](#testing--test_comcat_pypy)
- [Common Workflows](#common-workflows)
- [Parameter Reference](#parameter-reference)

---

## Overview

ComCAT implements a harmonization model that:

1. **Removes additive and multiplicative batch effects** (scanner / site differences).
2. **Regresses out nuisance covariates** (e.g., image quality metrics, total intracranial volume) using B-spline GAM smoothing.
3. **Preserves biological covariates of interest** (e.g., age, group) so they are not inadvertently removed.

The method is equivalent to ComBat when only batch correction is requested (except that no emprical Bayes is implemented), and extends it with flexible nuisance modelling when covariates are provided.

If you are interested in ComCat's theoretical background, you can read about it [here](ComCat-Theory.md).

---

## Installation

ComCAT has no mandatory installation step — clone the repository and install dependencies. Requires **Python ≥ 3.8**.

**Install required dependencies only:**
```bash
pip install -r requirements.txt
```

| Dependency | Required | Purpose |
|---|---|---|
| `numpy>=1.22` | Yes | Array operations |
| `scipy>=1.8` | Yes | `.mat` file I/O (v5–v7.2), stats |
| `nibabel>=4.0` | Yes | NIfTI / GIFTI file I/O |
| `statsmodels>=0.13` | Yes | B-spline GAM for nuisance modelling |
| `h5py>=3.0` | Yes | MATLAB v7.3 `.mat` files |
| `matplotlib>=3.5` | Yes | Plots in simulation tools |

Add the repository directory to your Python path or pass it via `sys.path.insert` (see `run_comcat_from_files.py` for an example).

---

## File Overview

| File | Purpose |
|---|---|
| `comcat.py` | Core harmonization algorithm (low-level API) |
| `comcat_ui.py` | High-level file I/O interface + CLI entry point |
| `run_comcat_from_files.py` | Example/template script for processing multiple datasets |
| `simulate_comcat.py` | Monte-Carlo simulation comparing ComCAT vs GLM AnCova |
| `simulate_comcat_ui.py` | Parameter sweep over simulation conditions |
| `decentralized_comcat.py` | Experimental decentralized/federated harmonization (sites never pool raw data) — see **[Decentralized.md](Decentralized.md)** |
| `tests/test_comcat_py.py` | Numerical validation against MATLAB reference output |

> **Decentralized / federated use:** to harmonize data split across sites that
> cannot share raw data, see **[Decentralized.md](Decentralized.md)**. The result
> matches centralized `comcat()` to machine precision.

---

## Core API — `comcat.py`

### `comcat()` — harmonize data

```python
from comcat import comcat

Y_harmonized, beta_hat, gamma_hat, delta_hat = comcat(
    Y,           # (n_features, n_subjects) data matrix
    batch,       # (n_subjects,) site/scanner labels
    nuisance,    # (n_subjects, n_nuisance) variables to remove  — or None
    preserve,    # (n_subjects, n_preserve) variables to keep    — or None
    mean_only=False,
    verbose=True,
)
```

**Returns**

| Value | Shape | Description |
|---|---|---|
| `Y_harmonized` | `(n_features, n_subjects)` | Harmonized data |
| `beta_hat` | `(n_params, n_features)` | Full design-matrix coefficients |
| `gamma_hat` | `(n_batch+n_Z, n_features)` | Additive batch/nuisance effects |
| `delta_hat` | `(n_batch, n_features)` | Multiplicative batch effects |

**Key parameters**

| Parameter | Default | Description |
|---|---|---|
| `mean_only` | `False` | Adjust mean only; skip variance scaling |
| `ref_batch` | `None` | Site label to treat as reference (its data is left untouched) |
| `return_estimates` | `False` | Return a 5th element (dict) for use with `comcat_from_training()` |
| `smooth_term_bounds` | `None` | Explicit boundary knots for the nuisance splines |
| `gam_df` | `None` | B-spline basis dimension per nuisance column (auto-selected if `None`) |
| `verbose` | `False` | Print progress messages |

**Single-site / nuisance-only mode**

Passing `batch=None` (or an empty array) treats all subjects as one site.  
ComCAT then only regresses out nuisance covariates while optionally preserving others — no scanner correction is applied.

### `comcat_from_training()` — apply to new data

```python
from comcat import comcat, comcat_from_training

# Step 1: fit on training data and save estimates
Y_train_harm, _, _, _, estimates = comcat(
    Y_train, batch_train, nuisance_train, preserve_train,
    return_estimates=True,
    smooth_term_bounds=(lo, hi),  # fix bounds for consistent knot placement
)

# Step 2: apply to new (test) data
Y_test_harm = comcat_from_training(
    Y_test, batch_test, nuisance_test, preserve_test,
    estimates=estimates,
)
```

> [!IMPORTANT]
> When using train/test workflows, always specify `smooth_term_bounds` so knot positions are identical between training and new data.

### Nuisance modelling: B-spline GAM

Every nuisance column is always modelled with a B-spline GAM (a flexible
non-linear fit; `statsmodels` is required). There is no linear or polynomial
option. The GAM is configured with `gam_df` (basis dimension) and, for
train/test workflows, `smooth_term_bounds` (explicit knot bounds).

**Recommended `gam_df` values by covariate type**

| Covariate | Recommended `gam_df` |
|---|---|
| Age (20–90 yr) | 6 – 8 |
| TIV / ICV (cm³) | 5 – 7 |
| Continuous score | 5 – 6 |
| General rule | `max(5, n // 30)`, capped at 10 |

When `gam_df=None` (default), the value is auto-selected as `min(10, max(5, n // 30))`.

---

## File I/O Interface — `comcat_ui.py`

`comcat_ui.py` wraps `comcat()` with file loading/saving logic, so you can work directly with neuroimaging files or data matrices stored on disk.

### Python usage

```python
from comcat_ui import comcat_ui
import numpy as np

batch    = np.loadtxt("batch.txt").astype(int)
nuisance = np.loadtxt("nuisance.txt")   # (n_subjects, n_nuisance)
preserve = np.loadtxt("preserve.txt")   # (n_subjects, n_preserve)

Y_harmonized, gamma_hat, delta_hat = comcat_ui(
    files          = ["subj_001.nii.gz", "subj_002.nii.gz", ...],
    batch          = batch,
    nuisance       = nuisance,
    preserve       = preserve,
    mean_only      = False,
    save_estimates = True,
    verbose        = True,
)
```

### Command-line usage

```bash
python comcat_ui.py --help
```

### Supported file formats

| Format | Extension(s) | Notes |
|---|---|---|
| NIfTI | `.nii`, `.nii.gz` | One file per subject; includes CIFTI |
| GIFTI | `.func.gii`, `.shape.gii`, `.gii` | Surface data |
| MATLAB | `.mat` | Single file containing variable `Y` of shape `(n_features × n_subjects)`, i.e. created by [BA_data2mat.m](https://github.com/ChristianGaser/BrainAGE); supports v5–v7.3 (requires `h5py` for v7.3) |
| Plain text / CSV | `.txt`, `.csv` | Single file; shape auto-detected |

### Output naming

Harmonized files are saved into an auto-generated subfolder next to the input files. The folder name encodes the harmonization settings, for example:

```
comcat_sites_preserve1_nuisance1_gam6/
combat_sites/
comcat_nuisance2_linear/
```

Override the folder name with the `subfolder` parameter. For MAT/TXT files, the output file is placed inside the subfolder with the same file name.

---

## Batch Processing — `run_comcat_from_files.py`

`run_comcat_from_files.py` is a ready-to-use template script for processing multiple datasets. Edit the `SAMPLES` dictionary and configuration section at the top of the file, then run:

```bash
python run_comcat_from_files.py
```

**Input file formats expected**

`batch.txt` — one site label per line (integer or string):
```
1
1
2
2
```

`nuisance.txt` — space- or tab-delimited, one row per subject, no header:
```
25.3  1450.2  1
31.7  1380.5  0
```

`preserve.txt` — same format as `nuisance.txt`.

`data.mat` — MATLAB file containing variable `Y` of shape `(n_features × n_subjects)`, i.e created by [BA_data2mat.m](https://github.com/ChristianGaser/BrainAGE).

**Setting the ComCAT directory**

The script resolves `comcat_ui.py` via the `COMCAT_DIR` variable. Either edit it directly or set the environment variable before running:

```bash
export COMCAT_DIR=/path/to/ComCat
python run_comcat_from_files.py
```

---

## Simulation Tools

### `simulate_comcat.py`

Monte-Carlo simulation that compares the statistical performance of **ComCAT** against a standard **GLM AnCova** approach. Use it to understand how nuisance amplitude and covariate–effect-of-interest correlation affect false-positive rates and effect-size recovery.

```python
from simulate_comcat import simulate_comcat

avgD, FPR = simulate_comcat(
    a=[1.0, 0.2, 0.0, 0.5],  # [EoI_amp, nuisance_amp, mult_amp, nuisance–EoI covariance]
    no_preserving=False,
    n=1000,
    n_sim=500,
    n_nuisance=1,
    mean_only=True,
    gam_df=6,       # B-spline basis dimension per nuisance term (GAM always on)
)
# avgD / FPR shape: (2,) → [AnCova, ComCAT (GAM)]
```

The simulation applies an optional **Zhao et al. two-step correction** (`apply_2step_correction=True` by default) to pre-whiten harmonized data and correct for degrees-of-freedom inflation introduced by ComCAT.

### `simulate_comcat_ui.py`

Runs a full parameter sweep over nuisance amplitude, nuisance–effect-of-interest covariance, and number of nuisance covariates. Results are saved to a `.mat` file for further analysis.

```bash
python simulate_comcat_ui.py                              # default sweep
python simulate_comcat_ui.py --n 1000 --n-sim 5000 --no-fig
python simulate_comcat_ui.py --mean-only --output my_results.mat
```

**Sweep dimensions**

| Dimension | Default values |
|---|---|
| Nuisance amplitude `a2` | 0, 0.05, …, 0.30 |
| Nuisance–EoI covariance `a4` | 0, 0.05, …, 0.50 |
| Number of nuisance covariates | 1, 2, 5, 10 |

Results are stored as arrays of shape `(n_a2, n_a4, n_nuisance, n_methods)` for both mean Cohen's D and false-positive rate.

---

## Testing — `test_comcat_py.py`

Validates that `comcat.py` produces results numerically identical to the MATLAB reference implementation within float32 tolerances (`atol=1e-4`, `rtol=1e-4`).

**Step 1** — Generate test data in MATLAB and place the `test_case*.mat` files in the `tests/` directory (the test script looks for them next to itself):
```matlab
gen_test_data   % produces test_case1.mat, test_case2.mat, test_case3.mat
```

**Step 2** — Run the Python tests:
```bash
python tests/test_comcat_py.py
```

Three test cases are covered:

| Case | Configuration |
|---|---|
| Case 1 | Multi-site, linear nuisance + preserve |
| Case 2 | Single site, linear nuisance, `mean_only=True` |
| Case 3 | Multi-site, B-spline GAM nuisance (requires `statsmodels`) |

---

## Common Workflows

### 1. Multi-site harmonization with age preservation

```python
from comcat import comcat
import numpy as np

# Y: (n_voxels, n_subjects), batch: site IDs, age: variable to preserve
Y_harm, *_ = comcat(Y, batch, nuisance=iqm, preserve=age[:, None])
```

### 2. Nuisance regression without site correction

```python
# Pass batch=None for single-site nuisance removal
Y_harm, *_ = comcat(Y, batch=None, nuisance=iqm, preserve=age[:, None])
```

### 3. Processing NIfTI files via the UI

```python
from comcat_ui import comcat_ui
import numpy as np

comcat_ui(
    files    = sorted(glob("data/*.nii.gz")),
    batch    = np.loadtxt("scanner.txt").astype(int),
    nuisance = np.loadtxt("iqm.txt")[:, None],
    preserve = np.loadtxt("age.txt")[:, None],
    save_estimates=True,
)
```

### 4. Train on one dataset, apply to new subjects

```python
from comcat import comcat, comcat_from_training

# Training
_, _, _, _, estimates = comcat(
    Y_train, batch_train, nuisance_train, preserve_train,
    return_estimates=True,
    smooth_term_bounds=(18.0, 90.0),  # age range covering both datasets
)

# New data
Y_new_harm = comcat_from_training(
    Y_new, batch_new, nuisance_new, preserve_new,
    estimates=estimates,
)
```

---

## Parameter Reference

### `comcat()` full signature

```python
comcat(
    Y,                          # (n_features, n_subjects)
    batch,                      # (n_subjects,) — site labels; None for single site
    nuisance       = None,      # (n_subjects, n_nuisance)
    preserve       = None,      # (n_subjects, n_preserve)
    mean_only      = False,     # True → mean correction only
    verbose        = False,
    ref_batch      = None,      # reference site label (left untouched)
    return_estimates = False,   # return fitted parameter dict
    smooth_term_bounds = None,  # (lo, hi) or [(lo0,hi0), ...] or None
    gam_df         = None,      # B-spline df per nuisance column; None = auto
)
```

### `comcat_ui()` full signature

```python
comcat_ui(
    files,                      # list of file paths
    batch          = None,      # (n_subjects,)
    nuisance       = None,      # (n_subjects, n_nuisance)
    preserve       = None,      # (n_subjects, n_preserve)
    mean_only      = False,
    subfolder      = None,      # override auto-generated output folder name
    save_estimates = False,     # save gamma/delta estimates alongside data
    verbose        = True,
    smooth_term_bounds = None,
    gam_df         = None,
)
```

---

## Citation / Origin

ComCAT is based on the ComBat algorithm originally developed by W. Evan Johnson and Cheng Li, and extended in the neuroimaging context by Jean-Philippe Fortin. The MATLAB implementation and extensions are by Christian Gaser (University of Jena).

For issues or feature requests, please use the [GitHub issue tracker](https://github.com/ChristianGaser/ComCat/issues).

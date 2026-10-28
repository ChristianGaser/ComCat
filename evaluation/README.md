# Real-data evaluations

These scripts produce the numbers in [HeldOut-Evaluation.md](../HeldOut-Evaluation.md),
and the empirical Bayes results in the README, CHANGELOG and ComCat-Theory.md. They need
the BrainAGE mat-files (`Y`, s6rp1/s4rp1 at 8 mm) and the site/IQM/age tables. Set the
paths in `data.py` or with environment variables:

| Variable | Default | Content |
|---|---|---|
| `COMCAT_TABLES` | `~/Library/CloudStorage/Dropbox/Harmonization-Oxford` | `<sample>/tables/*.txt` |
| `COMCAT_DATA` | `/Users/gaser/Dropbox/BrainAGE` | ON-Harmony, Buchert, ABIDE, MR-ART, Tohoku |
| `COMCAT_DATA_EXT` | `/Volumes/UltraMax/BrainAGE` | NormSample, ADNI |

| Script | What it does |
|---|---|
| `heldout_run.py [samples]` | Location model variants (current, truncated SVD, ridge, IQM subsets, linear, PCs, within-subject), in-sample vs held-out |
| `heldout_delta.py [samples]` | The same with the variance step (mean only, default δ, `residual_delta`) |
| `heldout_ui_single.py` | Single-site samples as `comcat_ui` runs them (one coded site, variance step) |
| `transfer.py` | IQM effects estimated within subjects on MR-ART, applied to other samples |
| `eb_splithalf.py` | Empirical Bayes: split-half error of the site parameters |
| `eb_traveling.py` | Empirical Bayes: held-out traveling subjects; alternative location targets (`eb_variants.py`) |
| `eb_impact.py` | Empirical Bayes: in-sample change of the harmonized data by site size |

The model variants are not ComCAT options. `heldout.py` and `eb_variants.py` inject them
by patching comcat internals (`pinv`, the nuisance basis, the EB step), so use these
modules only in evaluation scripts. Each script runs in seconds to a few minutes. On a
busy machine, limit the BLAS threads, e.g.
`VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 python evaluation/heldout_run.py`.

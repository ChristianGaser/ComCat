"""
run_combat_family_from_files.py — run CovBat, ComBatLS and ComBat (EB) on the
same text/MAT inputs as run_comcat_from_files.py.

For every sample and MAT file, each method in METHODS writes its result to its
own subfolder next to the MAT file (combateb_*, covbat_*, combatls_*), next to
ComCAT's comcat_* / combat_* folders.

Only samples with site labels are listed: these methods remove site effects
and have no nuisance-removal step, so single-site samples (Tohoku, MR-ART)
have nothing to harmonize.  Nuisance (IQM) files are therefore not used.

Usage
-----
    cd /path/to/Harmonization-Oxford     # folder containing the */tables dirs
    python /path/to/ComCat/run_combat_family_from_files.py

    or set TABLES_DIR (environment variable or below) to that folder.
"""

import os
import sys

COMCAT_DIR = os.environ.get(
    "COMCAT_DIR",
    os.path.dirname(os.path.abspath(__file__)),  # default: same folder as this script
)
sys.path.insert(0, os.path.abspath(COMCAT_DIR))

import numpy as np
from combat_family_ui import combat_eb_ui, combatls_ui, covbat_ui

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Folder that contains the <sample>/tables/ directories
TABLES_DIR = os.environ.get("TABLES_DIR", os.getcwd())

SAMPLES = {
    1: dict(
        mat_files=[
            "/Users/gaser/Dropbox/BrainAGE/s4rp1_4mm_ON-Harmony80_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s4rp1_8mm_ON-Harmony80_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s8rp1_4mm_ON-Harmony80_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s8rp1_8mm_ON-Harmony80_CAT12.9.mat",
        ],
        batch_file    = "ON-Harmony/tables/scanner80.txt",
        preserve_file = "ON-Harmony/tables/age80.txt",
    ),
    3: dict(
        mat_files=[
            "/Users/gaser/Dropbox/BrainAGE/s4rp1_4mm_Buchert531_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s4rp1_8mm_Buchert531_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s8rp1_4mm_Buchert531_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s8rp1_8mm_Buchert531_CAT12.9.mat",
        ],
        batch_file    = "Buchert/tables/scannerID531.txt",
        preserve_file = "Buchert/tables/age531.txt",
    ),
    4: dict(
        mat_files=[
            "/Users/gaser/Dropbox/BrainAGE/s4rp1_4mm_ABIDE437_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s4rp1_8mm_ABIDE437_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s8rp1_4mm_ABIDE437_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s8rp1_8mm_ABIDE437_CAT12.9.mat",
        ],
        batch_file    = "ABIDE/tables/Scanner437.txt",
        preserve_file = "ABIDE/tables/age437.txt",
    ),
    5: dict(
        mat_files=[
            "/Users/gaser/Dropbox/BrainAGE/s4rp1_4mm_NormSample2870_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s4rp1_8mm_NormSample2870_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s8rp1_4mm_NormSample2870_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s8rp1_8mm_NormSample2870_CAT12.9.mat",
        ],
        batch_file    = "normativeSample/tables/site2870.txt",
        preserve_file = "normativeSample/tables/age2870.txt",
    ),
}

SAMPLE  = 1                                   # key of SAMPLES
METHODS = ["covbat", "combatls", "combateb"]  # any subset

# Options (defaults follow the R ComBatFamily package)
MEAN_ONLY      = False  # True → adjust site means only, skip variance scaling
EB             = True   # empirical Bayes estimation of site effects
PERCENT_VAR    = 0.95   # CovBat: harmonize PCs explaining this proportion of variance
N_PC           = None   # CovBat: fixed number of PCs (overrides PERCENT_VAR)
SAVE_ESTIMATES = False  # save additive (gamma) and multiplicative (delta) site effects

# ---------------------------------------------------------------------------
# Load covariates
# ---------------------------------------------------------------------------

cfg = SAMPLES[SAMPLE]
batch = np.loadtxt(os.path.join(TABLES_DIR, cfg["batch_file"]), dtype=str)
preserve = None
if cfg["preserve_file"] is not None:
    preserve = np.loadtxt(os.path.join(TABLES_DIR, cfg["preserve_file"]))
    if preserve.ndim == 1:
        preserve = preserve[:, np.newaxis]

print(f"Subjects  : {batch.size}")
print(f"Sites     : {len(np.unique(batch))}")
if preserve is not None:
    print(f"Preserve  : {preserve.shape[1]} column(s)  — {cfg['preserve_file']}")

# ---------------------------------------------------------------------------
# Run all methods on all MAT files
# ---------------------------------------------------------------------------

common = dict(batch=batch, preserve=preserve, mean_only=MEAN_ONLY, eb=EB,
              save_estimates=SAVE_ESTIMATES, verbose=True)

for mat_file in cfg["mat_files"]:
    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"{method}: {mat_file}")
        print(f"{'='*60}")
        if method == "covbat":
            Y_harmonized, _, _ = covbat_ui([mat_file], percent_var=PERCENT_VAR,
                                           n_pc=N_PC, **common)
        elif method == "combatls":
            Y_harmonized, _, _ = combatls_ui([mat_file], **common)
        else:
            Y_harmonized, _, _ = combat_eb_ui([mat_file], **common)
        print(f"Harmonized data shape: {Y_harmonized.shape}")

"""
run_comcat_from_files.py — Example script for running ComCAT from text/MAT files.

Reads all covariates from plain text files and the data matrix from a MAT file,
then calls comcat_ui to harmonize and save the result.

Expected file formats
---------------------
batch.txt
    Single column, one label per row (integer or string site/scanner IDs).
    Example:
        1
        1
        2
        2
        3

nuisance.txt
    Multiple columns (n_subjects × n_nuisance), space- or tab-delimited,
    one row per subject.  Column header lines must be absent (pure numbers).
    Example (age, TIV, sex):
        25.3  1450.2  1
        31.7  1380.5  0
        28.1  1510.8  1

preserve.txt
    One or more columns (n_subjects × n_preserve), same format as nuisance.txt.
    Example (diagnosis score, IQ):
        0  112
        1  98
        0  105

data.mat
    MATLAB file (.mat, v5–v7.3) containing a variable named 'Y' with shape
    (n_features × n_subjects).  This is the format produced by ComCAT's MATLAB
    tools (cat_stat_comcat.m).

Usage
-----
    python run_comcat_from_files.py

    Or adjust the file paths / options at the top of the script and run directly.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path to the ComCat directory — edit this when moving the script
# ---------------------------------------------------------------------------
# Option A (recommended): set COMCAT_DIR to the folder that contains comcat_ui.py.
#   Use an absolute path or a path relative to this script file.
#
#   Absolute example:
#     COMCAT_DIR = "/Users/me/projects/ComCat"
#
#   Relative example (ComCat is one level up from this script):
#     COMCAT_DIR = os.path.join(os.path.dirname(__file__), "..", "ComCat")
#
# Option B: set the COMCAT_DIR environment variable before running the script:
#     export COMCAT_DIR=/Users/me/projects/ComCat
#     python run_comcat_from_files.py

COMCAT_DIR = os.environ.get(
    "COMCAT_DIR",
    os.path.dirname(os.path.abspath(__file__)),  # default: same folder as this script
)
sys.path.insert(0, os.path.abspath(COMCAT_DIR))

import numpy as np
from comcat_ui import comcat_ui

# ---------------------------------------------------------------------------
# Configuration — edit these paths and options to match your data
# ---------------------------------------------------------------------------

# One entry per sample: (mat_files, batch_file, nuisance_file, preserve_file)
# Set batch_file to None for single-site samples (no site correction).
SAMPLES = {
    1: dict(
        mat_files=[
            "/Users/gaser/Dropbox/BrainAGE/s4rp1_4mm_ON-Harmony80_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s4rp1_8mm_ON-Harmony80_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s8rp1_4mm_ON-Harmony80_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s8rp1_8mm_ON-Harmony80_CAT12.9.mat",
        ],
        batch_file    = "ON-Harmony/tables/scanner80.txt",
        nuisance_file = "ON-Harmony/tables/IQMrall80.txt",
        preserve_file = "ON-Harmony/tables/age80.txt",
    ),
    2: dict(
        mat_files=[
            "/Users/gaser/Dropbox/BrainAGE/s4rp1_4mm_Tohoku121_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s4rp1_8mm_Tohoku121_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s8rp1_4mm_Tohoku121_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s8rp1_8mm_Tohoku121_CAT12.9.mat",
        ],
        batch_file    = None,
        nuisance_file = "ADNI/tables/IQMrall121.txt",
        preserve_file = "ADNI/tables/age121.txt",
    ),
    3: dict(
        mat_files=[
            "/Users/gaser/Dropbox/BrainAGE/s4rp1_4mm_Buchert531_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s4rp1_8mm_Buchert531_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s8rp1_4mm_Buchert531_CAT12.9.mat",
            "/Users/gaser/Dropbox/BrainAGE/s8rp1_8mm_Buchert531_CAT12.9.mat",
        ],
        batch_file    = "Buchert/tables/scannerID531.txt",
        nuisance_file = "Buchert/tables/IQMrall531.txt",
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
        nuisance_file = "ABIDE/tables/IQMrall437.txt",
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
        nuisance_file = "normativeSample/tables/IQMrall2870.txt",
        preserve_file = "normativeSample/tables/age2870.txt",
    ),
}

# Select the sample to process (1–5).
SAMPLE = 1
MAT_FILES     = SAMPLES[SAMPLE]["mat_files"]
BATCH_FILE    = SAMPLES[SAMPLE]["batch_file"]
NUISANCE_FILE = SAMPLES[SAMPLE]["nuisance_file"]
PRESERVE_FILE = SAMPLES[SAMPLE]["preserve_file"]

# ComCAT options
MEAN_ONLY    = False   # True → adjust mean only, skip variance scaling
POLY_DEGREE  = 2       # polynomial degree (used only when SMOOTH_TERMS=None)
SMOOTH_TERMS = 'all'   # 'all' = B-spline GAM for all nuisance columns;
                       # None  = polynomial expansion; [0, 2] = GAM for cols 0 & 2
GAM_DF       = None    # None = auto: min(15, max(5, n//30)); set an int to override
SAVE_ESTIMATES = False # save additive (gamma) and multiplicative (delta) estimates

# ---------------------------------------------------------------------------
# Load covariates from text files
# ---------------------------------------------------------------------------

# Batch labels: None → single site (no scanner correction)
if BATCH_FILE is not None:
    batch_raw = np.loadtxt(BATCH_FILE, dtype=str)        # shape: (n_subjects,)
    _, batch = np.unique(batch_raw, return_inverse=True) # integer-coded labels
else:
    batch = None

# Nuisance covariates (multiple columns are handled automatically by np.loadtxt)
nuisance = np.loadtxt(NUISANCE_FILE)   # shape: (n_subjects,) or (n_subjects, n_cols)
if nuisance.ndim == 1:
    nuisance = nuisance[:, np.newaxis] # ensure 2-D

# Preserve covariates (same format)
preserve = np.loadtxt(PRESERVE_FILE)   # shape: (n_subjects,) or (n_subjects, n_cols)
if preserve.ndim == 1:
    preserve = preserve[:, np.newaxis]

n_subjects = nuisance.shape[0]
print(f"Subjects  : {n_subjects}")
print(f"Sites     : {len(np.unique(batch)) if batch is not None else 1}")
print(f"Nuisance  : {nuisance.shape[1]} column(s)  — {NUISANCE_FILE}")
print(f"Preserve  : {preserve.shape[1]} column(s)  — {PRESERVE_FILE}")

# ---------------------------------------------------------------------------
# Run ComCAT — iterate over all MAT files
# ---------------------------------------------------------------------------

for mat_file in MAT_FILES:
    print(f"\n{'='*60}")
    print(f"Processing: {mat_file}")
    print(f"{'='*60}")

    Y_harmonized, gamma_hat, delta_hat = comcat_ui(
        files          = [mat_file],
        batch          = batch,
        nuisance       = nuisance,
        preserve       = preserve,
        mean_only      = MEAN_ONLY,
        poly_degree    = POLY_DEGREE,
        smooth_terms   = SMOOTH_TERMS,
        gam_df         = GAM_DF,
        save_estimates = SAVE_ESTIMATES,
        verbose        = True,
    )

    print(f"Harmonized data shape: {Y_harmonized.shape}")

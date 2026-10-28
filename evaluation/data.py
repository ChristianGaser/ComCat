"""Data loaders for the real-data evaluations (read only).

Paths can be set with environment variables:
  COMCAT_TABLES     site/IQM/age tables  (<sample>/tables/*.txt)
  COMCAT_DATA       BrainAGE mat-files (s6rp1_8mm_*.mat)
  COMCAT_DATA_EXT   second location (NormSample, ADNI)
"""
import os
import re
import numpy as np
from scipy.io import loadmat

T = os.environ.get('COMCAT_TABLES', '/Users/gaser/Library/CloudStorage/Dropbox/Harmonization-Oxford')
B = os.environ.get('COMCAT_DATA', '/Users/gaser/Dropbox/BrainAGE')
U = os.environ.get('COMCAT_DATA_EXT', '/Volumes/UltraMax/BrainAGE')

SAMPLES = {
    'ON-Harmony': (f'{B}/s6rp1_8mm_ON-Harmony80_CAT12.9.mat', 'ON-Harmony/tables/scanner80.txt',
                   'ON-Harmony/tables/IQMrall80.txt', 'ON-Harmony/tables/age80.txt'),
    'Buchert':    (f'{B}/s6rp1_8mm_Buchert531_CAT12.9.mat', 'Buchert/tables/scannerID531.txt',
                   'Buchert/tables/IQMrall531.txt', 'Buchert/tables/age531.txt'),
    'ABIDE':      (f'{B}/s6rp1_8mm_ABIDE437_CAT12.9.mat', 'ABIDE/tables/Scanner437.txt',
                   'ABIDE/tables/IQMrall437.txt', 'ABIDE/tables/age437.txt'),
    'NormSample': (f'{U}/s6rp1_8mm_NormSample2870_CAT12.9.mat', 'normativeSample/tables/site2870.txt',
                   'normativeSample/tables/IQMrall2870.txt', 'normativeSample/tables/age2870.txt'),
    'ADNI':       (f'{U}/s4rp1_8mm_200AD_404MCI_231Normal_CAT12.9.mat', 'ADNI/tables/site835.txt',
                   'ADNI/tables/IQMrall835.txt', 'ADNI/tables/age835.txt'),
    'MR-ART':     (f'{B}/s6rp1_8mm_MRART436_CAT12.9.mat', None,
                   'MR-ART/tables/IQMrall436.txt', 'MR-ART/tables/age436.txt'),
    'Tohoku':     (f'{B}/s6rp1_8mm_Tohoku121_CAT12.9.mat', None,
                   'Tohoku/tables/IQMrall121.txt', 'Tohoku/tables/age121.txt'),
}


def load(name):
    mat, bf, nf, pf = SAMPLES[name]
    try:
        d = loadmat(mat, squeeze_me=True, struct_as_record=False)
        Y = np.asarray(d['Y'], dtype=np.float64)             # (n, p)
    except NotImplementedError:                               # v7.3 (HDF5)
        import h5py
        with h5py.File(mat, 'r') as f:
            Y = np.array(f['Y'], dtype=np.float64)
            age_m = np.array(f['age']).ravel() if 'age' in f else None
        if Y.shape[0] != len(np.loadtxt(f'{T}/{nf}')):
            Y = Y.T
        d = {'Y': Y} if age_m is None else {'Y': Y, 'age': age_m}
    iqm = np.loadtxt(f'{T}/{nf}')
    site = np.loadtxt(f'{T}/{bf}', dtype=str) if bf else np.full(len(iqm), '1')
    age = np.loadtxt(f'{T}/{pf}')
    if age.ndim > 1:
        age = age[:, 0]
    files = [str(f) for f in np.ravel(d['sample'].data)] if 'sample' in d else None
    assert Y.shape[0] == len(site) == len(iqm) == len(age), (Y.shape, len(site), len(iqm), len(age))
    # features with variance (as comcat's mask) and mean >= 0.05 (gray matter)
    keep = (Y.std(0) > 0) & (Y.mean(0) >= 0.05)
    return dict(Y=Y[:, keep].T, site=site, iqm=iqm, age=age, files=files,
                mat_age=np.ravel(d.get('age', np.full(len(site), np.nan))))


def subject_ids(files):
    return np.array([re.search(r'sub-([^_/]+)', f).group(1) for f in files])

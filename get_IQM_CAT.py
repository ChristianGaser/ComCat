#!/usr/bin/env python3
"""get_IQM_CAT.py — read image quality ratings (IQM) from CAT12 xml files.

Python port of get_IQM_CAT.m. For each CAT12 ``cat*.xml`` report file it reads
the seven quality ratings below from the ``<qualityratings>`` block and saves
them to text files (progress bar omitted).

    ICR, NCR, res_ECR, res_RMS, contrastr, SIQR, IQR

Outputs (n = number of input files):
    IQM{n}.txt      one row per subject, the 7 values space-separated

Usage
-----
    # explicit files
    python get_IQM_CAT.py /path/report/cat_sub-01.xml /path/report/cat_sub-02.xml

    # a directory or glob (only names matching ^cat.*\\.xml$ are kept)
    python get_IQM_CAT.py /path/report

    # no argument: searches the current directory for ^cat.*\\.xml$
    python get_IQM_CAT.py

Or as a library:
    from get_IQM_CAT import get_iqm_cat
    iqm = get_iqm_cat(list_of_xml_files)   # ndarray (n_subjects, 7)
"""

from __future__ import annotations

import os
import re
import sys
import glob
import xml.etree.ElementTree as ET

import numpy as np

NAME = "IQM"
LIST_IQM = ["ICR", "NCR", "res_ECR", "res_RMS", "contrastr", "SIQR", "IQR"]

# spm_select default filter in the original: '^cat.*\.xml$'
_CAT_XML_RE = re.compile(r"^cat.*\.xml$")


def read_qualityratings(xml_path: str) -> list[float]:
    """Return the LIST_IQM values (in order) from a CAT xml file.

    Raises ValueError if the file has no <qualityratings> block or is missing
    one of the requested ratings (mirrors the MATLAB error behaviour).
    """
    root = ET.parse(xml_path).getroot()
    qr = root.find(".//qualityratings")
    if qr is None and root.tag == "qualityratings":
        qr = root
    if qr is None:
        raise ValueError(f"No quality measures found in {xml_path}")

    values = []
    for key in LIST_IQM:
        el = qr.find(key)
        if el is None or el.text is None or el.text.strip() == "":
            raise ValueError(f"Missing quality rating '{key}' in {xml_path}")
        try:
            values.append(float(el.text.strip()))
        except ValueError as exc:
            raise ValueError(
                f"Could not parse rating '{key}'={el.text.strip()!r} in {xml_path}"
            ) from exc
    return values


def _write_rows(path: str, rows) -> None:
    """Write each row as space-separated '%g ' values + newline (matches MATLAB)."""
    with open(path, "w") as fid:
        for row in rows:
            fid.write("".join("%g " % v for v in row))
            fid.write("\n")
    print(f"\nValues saved in {path}.")


def get_iqm_cat(files, name: str = NAME) -> np.ndarray:
    """Read IQM ratings from `files` and save IQM{n}.txt.

    Returns an (n_subjects, 7) array of the ratings (column order = LIST_IQM).
    """
    files = list(files)
    n = len(files)
    if n == 0:
        raise ValueError("No xml files given.")

    iqm = np.array([read_qualityratings(f) for f in files], dtype=float)  # (n, 7)

    # IQM{n}.txt — one subject per row
    _write_rows(f"{name}{n}.txt", iqm)

    return iqm


def _collect_files(args) -> list[str]:
    """Resolve CLI args to a sorted file list, keeping only ^cat.*\\.xml$ names."""
    if not args:
        args = [os.getcwd()]

    candidates: list[str] = []
    for a in args:
        if os.path.isdir(a):
            candidates.extend(glob.glob(os.path.join(a, "*.xml")))
        elif any(ch in a for ch in "*?["):
            candidates.extend(glob.glob(a))
        else:
            candidates.append(a)

    # apply the ^cat.*\.xml$ filter only when it would not drop explicit choices,
    # i.e. always filter directory/glob expansions; keep explicit files as given.
    filtered = [f for f in candidates if _CAT_XML_RE.match(os.path.basename(f))]
    return sorted(filtered)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    files = _collect_files(argv)
    if not files:
        print("No files matching '^cat.*\\.xml$' found.", file=sys.stderr)
        return 1
    get_iqm_cat(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

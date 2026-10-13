#!/usr/bin/env python3
"""get_TIV_CAT.py — read total intracranial volume (TIV) from CAT12 xml files.

For each CAT12 ``cat*.xml`` report file it reads ``vol_TIV`` (in cm³) from the
top-level ``<subjectmeasures>`` block and saves the values to a text file. File
selection and output format are the same as in get_IQM_CAT.py.

Outputs (n = number of input files):
    TIV{n}.txt      one row per subject, the TIV value

Usage
-----
    # explicit files
    python get_TIV_CAT.py /path/report/cat_sub-01.xml /path/report/cat_sub-02.xml

    # a directory or glob (only names matching ^cat.*\\.xml$ are kept)
    python get_TIV_CAT.py /path/report

    # no argument: searches the current directory for ^cat.*\\.xml$
    python get_TIV_CAT.py

Or as a library:
    from get_TIV_CAT import get_tiv_cat
    tiv = get_tiv_cat(list_of_xml_files)   # ndarray (n_subjects,)
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import numpy as np

from get_IQM_CAT import _collect_files, _write_rows

NAME = "TIV"
FIELD = "vol_TIV"


def read_tiv(xml_path: str) -> float:
    """Return subjectmeasures.vol_TIV from a CAT xml file.

    Only the top-level <subjectmeasures> block is read: <ratings_help> contains
    an earlier <subjectmeasures> block with help texts, and <subjectratings>
    contains a vol_TIV rating. Raises ValueError if vol_TIV is missing or not a
    number.
    """
    root = ET.parse(xml_path).getroot()
    sm = root if root.tag == "subjectmeasures" else root.find("subjectmeasures")
    el = None if sm is None else sm.find(FIELD)
    if el is None or el.text is None or el.text.strip() == "":
        raise ValueError(f"No subjectmeasures.{FIELD} found in {xml_path}")
    try:
        return float(el.text.strip())
    except ValueError as exc:
        raise ValueError(
            f"Could not parse {FIELD}={el.text.strip()!r} in {xml_path}"
        ) from exc


def get_tiv_cat(files, name: str = NAME) -> np.ndarray:
    """Read TIV from `files` and save TIV{n}.txt.

    Returns an (n_subjects,) array of TIV values.
    """
    files = list(files)
    n = len(files)
    if n == 0:
        raise ValueError("No xml files given.")

    tiv = np.array([read_tiv(f) for f in files], dtype=float)  # (n,)

    # TIV{n}.txt — one subject per row
    _write_rows(f"{name}{n}.txt", tiv[:, None])

    return tiv


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    files = _collect_files(argv)
    print(f"subjectmeasures.{FIELD}")
    if not files:
        print("No files matching '^cat.*\\.xml$' found.", file=sys.stderr)
        return 1
    get_tiv_cat(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

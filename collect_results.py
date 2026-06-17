#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Bundle result artifacts into a single zip for transfer between machines.
import os
import sys
import glob
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))

# Result files to collect: per-domain metric JSONs, P1/P3 outputs, and run logs.
PATTERNS = [
    "run_code/results_3seed/*.json",
    "run_code/results_p3_real/*.json",
    "run_code/results_p3_rewired/*.json",
    "run_code/results_p3_removed/*.json",
    "run_code/results_baselines/*.json",
    "run_code/results_scarce/*.json",
    "run_code/results_A_anchors/*.json",
    "run_code/run_tomorrow.log",
    "run_code/p3_local.log",
    "RESULTS.txt",
    "P1_cosine.json",
    "P1_cosine_implicit.json",
]


def main():
    # Expand every glob, keep only files, and de-duplicate.
    files = []
    for pat in PATTERNS:
        files += glob.glob(os.path.join(ROOT, *pat.split("/")))
    files = sorted({f for f in files if os.path.isfile(f)})
    if not files:
        print("no result files found; run run_4070.py first.")
        sys.exit(1)
    # Store each file under its path relative to the project root so it unzips back in place.
    out = os.path.join(ROOT, "ontospan_results.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            arc = os.path.relpath(f, ROOT)
            z.write(f, arc)
            print(f"  + {arc}")
    size_kb = os.path.getsize(out) / 1024
    print(f"\npacked {len(files)} files -> {out}  ({size_kb:.0f} KB)")
    print("unzip ontospan_results.zip at the project root to restore files.")


if __name__ == "__main__":
    main()

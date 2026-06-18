#!/usr/bin/env python3
# Runner: uncapped (full-data) legal NER, none vs. explicit.
#
# Repeats the legal main-table comparison on the full CAIL data (no 600/100/300
# cap), same protocol otherwise (depth-2, fp32, five epochs, three seeds). Tests
# whether the type-level mutex gain holds beyond the low-resource budget.
import os
import sys
import json
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(ROOT, "run_code", "run_grid_safe.py")
PY = sys.executable
SAVE = os.path.join(ROOT, "run_code", "results_law_full")


def _mean_f1(results, mode):
    runs = results.get(f"1.0|{mode}") or results.get(f"1|{mode}") or []
    return sum(r["F1"] for r in runs) / len(runs) if runs else float("nan")


def main():
    try:
        import torch
        print("CUDA:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU/MPS, slow)")
    except ImportError:
        print("torch not installed"); sys.exit(1)

    # cap 0 = use all sentences (uncapped); everything else matches the main table.
    cmd = [PY, "-u", GRID, "--domains", "law", "--fracs", "1.0", "--modes", "none,explicit",
           "--seeds", "42,1,2", "--train-cap", "0", "--dev-cap", "0", "--test-cap", "0",
           "--gnn-depth", "2", "--epochs", "5", "--encoder", "hfl/chinese-macbert-base",
           "--save-dir", SAVE]
    print("\n" + "=" * 72 + "\nLaw uncapped (full CAIL, none vs explicit)\n" + "=" * 72, flush=True)
    subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"))

    path = os.path.join(SAVE, "law_paper.json")
    if not os.path.exists(path):
        print("no law_paper.json produced; see errors above."); return
    res = json.load(open(path, encoding="utf-8")).get("results", {})
    none, expl = _mean_f1(res, "none"), _mean_f1(res, "explicit")
    print("\n" + "=" * 72 + "\nLaw uncapped (full data, mean test-F1 over seeds 42/1/2)\n" + "=" * 72)
    print(f"  none     : {none:.4f}")
    print(f"  explicit : {expl:.4f}   (delta vs none: {expl - none:+.4f})")
    print("\nIf the delta holds (or grows) at full data, the type-level gain is not a")
    print("low-resource artifact -- the 'greenhouse' concern is answered.")
    print("\nSend back run_code/results_law_full/law_paper.json.")


if __name__ == "__main__":
    main()

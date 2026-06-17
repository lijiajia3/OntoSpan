#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Unified experiment entry point.
#
# Dispatches to the grid driver, the P1 geometry check, and the baseline runner
# via subprocess. Stages: law-small / law-full / medical-full / all.
import os, sys, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(ROOT, "run_code", "run_grid_safe.py")
P1 = os.path.join(ROOT, "verify_p1_cosine.py")
BASELINES = os.path.join(ROOT, "run_baselines_4070.py")
ENC = "hfl/chinese-macbert-base"
PY = sys.executable
ENV = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
UNCAPPED = "100000"


def sh(cmd, label):
    print(f"\n{'='*72}\n▶ {label}\n  {' '.join(cmd)}\n{'='*72}", flush=True)
    r = subprocess.run(cmd, env=ENV, cwd=ROOT)
    if r.returncode != 0:
        print(f"  step exited non-zero (rc={r.returncode}); continuing.", flush=True)
    return r.returncode


def need_law_data():
    d = os.path.join(ROOT, "data", "law")
    miss = [sp for sp in ("train", "dev", "test") if not os.path.exists(os.path.join(d, f"{sp}.jsonl"))]
    if miss:
        print("missing legal data data/law/{train,dev,test}.jsonl (" + ",".join(miss) + ").")
        print("  CAIL information-extraction subset: download into data/raw/cail/, then")
        print("    python run_code/download_datasets.py --dataset cail")
        print("  (3 type-level mutex pairs for law, e.g. plaintiff/defendant, are in build_datasets.py).")
        return False
    return True


# Build the command line for one main-table grid run (subprocess-isolated driver).
def grid(domain, fracs, seeds, cap, save_sub, modes="none,explicit"):
    return [PY, "-u", GRID, "--domains", domain, "--fracs", fracs, "--modes", modes,
            "--seeds", seeds, "--train-cap", cap, "--dev-cap", cap, "--test-cap", cap,
            "--gnn-depth", "2", "--encoder", ENC,
            "--save-dir", os.path.join(ROOT, "run_code", save_sub)]


# P1 geometry check; the decoupled anchor learning rate is required for the geometry to form.
def p1(domain, frac, seed="42"):
    return [PY, "-u", P1, "--pairs", f"{domain}:{frac}", "--seed", seed,
            "--epochs", "5", "--mode", "implicit", "--anchor-lr", "0.05"]


def law_small():
    if not need_law_data(): return
    sh(grid("law", "0.1,0.3,1.0", "42", "600", "results_law_small"), "law small main table (cap600, seed42)")
    sh(p1("law", "1.0"), "law P1 geometry (type-level plaintiff/defendant)")
    print("\n-> explicit-none > 0 means type-level repulsion helps NER; P1 checks whether mutex cosine < 0.")


def law_full():
    if not need_law_data(): return
    sh(grid("law", "0.1,0.3,1.0", "42,1,2", UNCAPPED, "results_law_full"), "law full main table (uncapped, 3 seeds)")
    sh(p1("law", "1.0"), "law P1 geometry")
    sh([PY, "-u", BASELINES, "--domains", "law", "--fracs", "0.1,0.3,1.0", "--seeds", "42,1,2"],
       "law baselines (biaffine/GP/CRF)")


def medical_full():
    sh(grid("medical", "1.0", "42,1,2", UNCAPPED, "results_medical_full"),
       "medical full main table (uncapped frac1.0, none/explicit, 3 seeds)")
    print("\n→ Expectation: explicit−none approaches 0 at full data, consistent with the prior becoming redundant once labels are abundant.")


def all_stages():
    print("[all] main table + P3 + P1 + baselines already exist; this adds two full runs + law:")
    medical_full()
    sh(grid("tcm", "1.0", "42,1,2", UNCAPPED, "results_tcm_full"), "TCM full main table (uncapped, 3 seeds)")
    if need_law_data():
        law_full()
    print("\n done; use collect_results.py to bundle results.")


STAGES = {"law-small": law_small, "law-full": law_full,
          "medical-full": medical_full, "all": all_stages}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print(__doc__); print("available stages:", ", ".join(STAGES)); sys.exit(1)
    try:
        import torch
        print("✓ CUDA:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (will use CPU/MPS, slow)")
    except ImportError:
        print("torch not installed"); sys.exit(1)
    STAGES[sys.argv[1]]()


if __name__ == "__main__":
    main()

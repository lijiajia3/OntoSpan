#!/usr/bin/env python3
# Runner: P3 perturbation control on the legal type-level mutex.
#
# Runs the legal NER explicit model at the full (capped) fraction under three
# mutex variants -- real / rewired / removed -- so the causal role of the
# authoritative suspect/victim edge can be tested. If the gain over the no-prior
# baseline collapses under rewired/removed, the specific edge identity matters.
import os
import sys
import json
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(ROOT, "run_code", "run_grid_safe.py")
PY = sys.executable

# Same backbone, budget, and protocol as the legal main table.
COMMON = ["--domains", "law", "--fracs", "1.0", "--modes", "explicit", "--seeds", "42,1,2",
          "--train-cap", "600", "--dev-cap", "100", "--test-cap", "300",
          "--gnn-depth", "2", "--encoder", "hfl/chinese-macbert-base"]


def _mean_f1(results, mode):
    # The fraction may be keyed as "1.0|<mode>" or "1|<mode>".
    runs = results.get(f"1.0|{mode}") or results.get(f"1|{mode}") or []
    return sum(r["F1"] for r in runs) / len(runs) if runs else float("nan")


def run_variant(kind):
    save = os.path.join(ROOT, "run_code", f"results_p3_{kind}")
    cmd = [PY, "-u", GRID, *COMMON, "--mutex-perturb", kind, "--save-dir", save]
    print("\n" + "=" * 72 + f"\nP3 law | {kind}\n" + "=" * 72, flush=True)
    subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"))
    path = os.path.join(save, "law_paper.json")
    if os.path.exists(path):
        return _mean_f1(json.load(open(path, encoding="utf-8")).get("results", {}), "explicit")
    return float("nan")


def main():
    try:
        import torch
        print("CUDA:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU/MPS, slow)")
    except ImportError:
        print("torch not installed"); sys.exit(1)

    res = {k: run_variant(k) for k in ("real", "rewired", "removed")}

    # No-prior baseline from the legal main table, if present.
    none = float("nan")
    lp = os.path.join(ROOT, "run_code", "results_law", "law_paper.json")
    if os.path.exists(lp):
        none = _mean_f1(json.load(open(lp, encoding="utf-8")).get("results", {}), "none")

    print("\n" + "=" * 72 + "\nP3 law (frac 1.0, mean test-F1 over seeds 42/1/2)\n" + "=" * 72)
    print(f"  none     : {none:.4f}")
    for k in ("real", "rewired", "removed"):
        print(f"  {k:<8} : {res[k]:.4f}   (delta vs none: {res[k] - none:+.4f})")
    print("\nP3 is supported if real > rewired and real > removed: the specific")
    print("authoritative mutex edge, not just graph topology, drives the gain.")
    print("\nSend back run_code/results_p3_{real,rewired,removed}/law_paper.json.")


if __name__ == "__main__":
    main()

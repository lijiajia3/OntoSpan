#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Runner: legal NER (type-level mutex).
#
# Legal NER with a type-level mutex (suspect vs. victim) plus the P1 geometry
# check; optionally also runs the external baselines.
import os, sys, json, subprocess, argparse, statistics as st

ROOT = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(ROOT, "run_code", "run_grid_safe.py")
P1 = os.path.join(ROOT, "verify_p1_cosine.py")
BASE = os.path.join(ROOT, "run_baselines_4070.py")
ENC = "hfl/chinese-macbert-base"
PY = sys.executable
ENV = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
SAVE = os.path.join(ROOT, "run_code", "results_law")


def check_data():
    d = os.path.join(ROOT, "data", "law")
    miss = [sp for sp in ("train", "dev", "test") if not os.path.exists(os.path.join(d, f"{sp}.jsonl"))]
    if not miss:
        return True
    print("=" * 72)
    print("missing legal data data/law/{train,dev,test}.jsonl --", ",".join(miss))
    print("=" * 72)
    print("CAIL2021-IE (theft-case IE) acquisition (competition licence; manual):")
    print("  1) register/download train.json (with entity span annotations) from the CAIL IE track")
    print("  2) place it under data/raw/cail/")
    print("  3) python run_code/download_datasets.py --dataset cail --raw-dir data/raw/cail")
    print("     (built-in parser: NHCS->suspect / NHVI->victim ..., normalized to char offsets -> data/law/)")
    print("  type-level mutex (suspect/victim, etc.) is in build_datasets.py MUTEX_TYPE['law'].")
    print("\n  no CAIL but want to test the pipeline? use any span-NER jsonl (per line {\"text\":..,\"entities\":[[s,e,type],..]})")
    print("  as data/law/{train,dev,test}.jsonl; mutex applies only when type names include suspect/victim, etc.")
    return False


def run_main(seeds, cap):
    cmd = [PY, "-u", GRID, "--domains", "law", "--fracs", "0.1,0.3,1.0",
           "--modes", "none,explicit", "--seeds", seeds,
           "--train-cap", str(cap), "--dev-cap", str(max(50, cap // 6)),
           "--test-cap", str(max(50, cap // 2)), "--gnn-depth", "2",
           "--lr", "2e-5", "--encoder", ENC, "--save-dir", SAVE]
    print("\n> law main table (none/explicit, fp32, depth2):", " ".join(cmd[3:]), flush=True)
    subprocess.run(cmd, env=ENV, cwd=ROOT)


def show_table():
    path = os.path.join(SAVE, "law_paper.json")
    if not os.path.exists(path):
        print("law_paper.json not found; the main table may have failed (see errors above)."); return
    res = json.load(open(path, encoding="utf-8")).get("results", {})
    def mean(k): v = [r["F1"] for r in res.get(k, [])]; return st.mean(v) if v else float("nan")
    print("\n" + "=" * 60 + "\nlaw NER -- test micro-F1 + delta-F1 (explicit-none)\n" + "=" * 60)
    print(f"{'frac':>6} | {'none':>9} | {'explicit':>9} | {'ΔF1':>9}")
    for fr in ["0.1", "0.3", "1.0"]:
        n, e = mean(f"{fr}|none"), mean(f"{fr}|explicit")
        print(f"{fr:>6} | {n:>9.4f} | {e:>9.4f} | {e-n:>+9.4f}")
    print("\nreading: explicit-none clearly > 0 means type-level mutex helps NER")
    print("  (positive evidence that repulsion helps; decision-axis aligned).")
    print("  <=0 -> type-level is also absorbed by supervision.")


def run_p1():
    print("\n> law P1 geometry (cosine of type-mutex suspect/victim)", flush=True)
    subprocess.run([PY, "-u", P1, "--pairs", "law:1.0", "--seed", "42",
                    "--epochs", "5", "--mode", "implicit", "--anchor-lr", "0.05"], env=ENV, cwd=ROOT)
    print("  (note: law has a single type-mutex pair, suspect/victim, in the theft-case data, so P1 is based on one pair--"
          "negative cosine supports the geometry, but with a small sample; the main-table delta-F1 is primary.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="quick: 1 seed")
    ap.add_argument("--baselines", action="store_true", help="also run biaffine/GP/CRF")
    ap.add_argument("--train-cap", type=int, default=600)
    ap.add_argument("--no-p1", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    try:
        import torch
        dev = a.device or ("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
        print("✓ CUDA:", torch.cuda.get_device_name(0)) if dev == "cuda" else print("non-CUDA dev=" + dev)
    except ImportError:
        print("torch not installed"); sys.exit(1)

    if not check_data():
        sys.exit(1)
    os.makedirs(SAVE, exist_ok=True)
    seeds = "42" if a.quick else "42,1,2"
    print(f"\n{'='*72}\nlaw NER experiment | seeds={seeds} | cap{a.train_cap} | fp32 | type-level mutex\n{'='*72}")

    run_main(seeds, a.train_cap)
    show_table()
    if not a.no_p1:
        run_p1()
    if a.baselines:
        print("\n> law baselines (biaffine/GP/CRF)", flush=True)
        subprocess.run([PY, "-u", BASE, "--domains", "law", "--fracs", "0.1,0.3,1.0",
                        "--seeds", seeds], env=ENV, cwd=ROOT)

    print("\n done; results in run_code/results_law/; bundle with collect_results.py.")
    print("  the delta-F1 rows indicate whether type-level mutex helps.")


if __name__ == "__main__":
    main()

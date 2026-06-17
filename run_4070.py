#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Runner: main table (multi-seed) + perturbation (P3) + geometry (P1).
#
# One-command runner (named for the RTX 4070 it was run on; CUDA-agnostic):
# main table, P3 perturbation, and the P1 geometry check, then bundles results.
import os
import sys
import time
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_GRID = os.path.join(ROOT, "run_code", "run_grid_safe.py")
ENCODER = "hfl/chinese-macbert-base"

STAGE = "all"
_VALID = ("all", "core", "full", "everything", "p3", "main", "p1")
if len(sys.argv) > 1 and sys.argv[1] in _VALID:
    STAGE = sys.argv[1]

COMMON = ["--epochs", "5", "--batch-size", "4", "--accum", "2", "--gnn-depth", "2",
          "--train-cap", "600", "--dev-cap", "100", "--test-cap", "300", "--amp"]


def banner(msg):
    print("\n" + "=" * 72 + "\n" + msg + "\n" + "=" * 72, flush=True)


def check_torch_cuda():
    try:
        import torch
    except ImportError:
        print("torch not installed; install the CUDA build, e.g.:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}  (torch {torch.__version__})")
        return
    print("CUDA unavailable -- running on CPU; P3 may take hours.")
    print("    likely a CPU-only torch; reinstall the CUDA build:")
    print("    pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu121")
    try:
        input("    continue on CPU anyway? Enter to continue, Ctrl+C to quit... ")
    except KeyboardInterrupt:
        sys.exit(0)


def ensure_macbert():
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("transformers not installed: pip install transformers")
        sys.exit(1)
    try:
        AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
        AutoModel.from_pretrained(ENCODER, local_files_only=True)
        print("macbert already cached; skipping download.")
        return
    except Exception:
        print("no local macbert cache; downloading once (~400MB)...")
    def _dl():
        AutoTokenizer.from_pretrained(ENCODER)
        AutoModel.from_pretrained(ENCODER)

    try:
        _dl()
        print("macbert downloaded and cached.")
        return
    except Exception as e1:
        print(f"direct huggingface.co failed ({type(e1).__name__}); retrying via hf-mirror.com...")
    try:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        _dl()
        print("macbert downloaded via hf-mirror.com and cached.")
    except Exception as e2:
        print(f"macbert download failed both ways: {e2}")
        print("  three options:")
        print("   1) set a mirror and rerun: set HF_ENDPOINT=https://hf-mirror.com then python run_4070.py")
        print("   2) copy ~/.cache/huggingface/hub/models--hfl--chinese-macbert-base")
        print("      to C:\\Users\\<user>\\.cache\\huggingface\\hub\\ (no download, most reliable)")
        print("   3) connect via VPN and rerun.")
        sys.exit(1)


def _run(extra_args, save_dir):
    cmd = [sys.executable, "-u", RUN_GRID, *extra_args, "--save-dir", save_dir]
    print("\n▶ " + " ".join(extra_args + ["--save-dir", os.path.relpath(save_dir, ROOT)]), flush=True)
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0:
        print(f"cell failed (rc={rc}); stopping.")
        sys.exit(1)


def run_grid(args, save_dir):
    _run(args + COMMON, save_dir)


def stage_p3(grad=False):
    rew = os.path.join(ROOT, "run_code", "results_p3_rewired")
    rem = os.path.join(ROOT, "run_code", "results_p3_removed")
    rea = os.path.join(ROOT, "run_code", "results_p3_real")
    med = "0.1,0.3,1.0" if grad else "1.0"
    banner(f"P3 control multi-seed (medical[{med}] + removed-tcm)" + ("  (with medical gradient)" if grad else ""))
    run_grid(["--domains", "tcm", "--fracs", "0.1", "--modes", "explicit",
              "--seeds", "42,1,2", "--mutex-perturb", "removed"], rem)
    targets = [("rewired", rew), ("removed", rem)] + ([("real", rea)] if grad else [])
    for pert, sd in targets:
        run_grid(["--domains", "medical", "--fracs", med, "--modes", "explicit",
                  "--seeds", "42,1,2", "--mutex-perturb", pert], sd)


def stage_main():
    banner("main table 3-seed (none, explicit x 2 domains x 3 fracs x seeds 42,1,2)")
    out = os.path.join(ROOT, "run_code", "results_3seed")
    run_grid(["--domains", "tcm,medical", "--modes", "none,explicit",
              "--fracs", "0.1,0.3,1.0", "--seeds", "42,1,2", "--mutex-perturb", "real"], out)


def stage_p1():
    banner("P1 cosine test (mutex-pair representation should be significantly negative)")
    script = os.path.join(ROOT, "verify_p1_cosine.py")
    if not os.path.exists(script):
        print("verify_p1_cosine.py not found; skipping P1.")
        return
    rc = subprocess.run([sys.executable, "-u", script, "--epochs", "5"], cwd=ROOT).returncode
    if rc != 0:
        print("P1 script exited non-zero (does not affect main results; rerun python verify_p1_cosine.py).")


def stage_full():
    banner("uncapped full run (epochs=8, full train/test, dev500) -- multi-day!")
    print("data: medical 18.4k + tcm 5.6k train sentences x 3 fracs x {none,explicit} x 3 seeds x 8ep.")
    print("expect several days on a 4070 (1-2 days even on an A100).")
    print("uncapped numbers supersede the capped ones for the final version.")
    try:
        ans = input("start this multi-day job now? type yes to continue, anything else to quit: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    if ans != "yes":
        print("skipped the full run; use `python run_4070.py full` when needed.")
        return
    out = os.path.join(ROOT, "run_code", "results_full")
    full = ["--epochs", "8", "--batch-size", "8", "--accum", "2", "--gnn-depth", "2",
            "--train-cap", "0", "--dev-cap", "500", "--test-cap", "0", "--amp"]
    _run(["--domains", "tcm,medical", "--modes", "none,explicit",
          "--fracs", "0.1,0.3,1.0", "--seeds", "42,1,2", "--mutex-perturb", "real"] + full, out)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _read_results(path):
    import json
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for k, runs in d.get("results", {}).items():
        out[k] = [r["F1"] for r in runs]
    return out


def summarize():
    lines = []

    def w(s=""):
        print(s, flush=True)
        lines.append(s)

    rc = os.path.join(ROOT, "run_code")
    w("\n" + "#" * 72)
    w("# results summary (read back from the generated JSON)")
    w("#" * 72)

    did = {"p3": STAGE in ("p3", "all", "core", "everything"),
           "main": STAGE in ("main", "all", "core", "everything"),
           "p1": STAGE in ("p1", "all", "core", "everything"),
           "full": STAGE in ("full", "everything")}

    if did["p3"]:
        w("\n[P3 polarity control] each cell = mean test-F1 over seeds")
        w(f"{'domain/frac':<14}{'real':>9}{'rewired':>9}{'removed':>9}"
          f"{'sem-d':>9}{'dens-d':>9}   (sem=real-rewired, dens=rewired-removed)")
        anchors = [("tcm", "0.1")]
        med = _read_results(os.path.join(rc, "results_p3_real", "medical_paper.json")) or {}
        med_fracs = sorted({k.split("|")[0] for k in med}) or ["1.0"]
        anchors += [("medical", f) for f in med_fracs]
        for dom, fr in anchors:
            vals = {}
            for pert in ("real", "rewired", "removed"):
                r = _read_results(os.path.join(rc, f"results_p3_{pert}", f"{dom}_paper.json"))
                vals[pert] = _mean(r.get(f"{float(fr)}|explicit", [])) if r else float("nan")
            sem = vals["real"] - vals["rewired"]
            den = vals["rewired"] - vals["removed"]
            w(f"{dom+' '+fr:<14}{vals['real']:>9.4f}{vals['rewired']:>9.4f}"
              f"{vals['removed']:>9.4f}{sem:>+9.4f}{den:>+9.4f}")
        w("-> removed lowest = repulsion (polarity) matters; real>rewired = authentic semantics help.")

    for tag, sub in (("main 3-seed (capped)", "results_3seed"),
                     ("uncapped full", "results_full")):
        if (sub == "results_3seed" and not did["main"]) or \
           (sub == "results_full" and not did["full"]):
            continue
        w(f"\n[{tag}] delta-F1 from none to explicit")
        w(f"{'domain':<10}{'frac':>6}{'none':>9}{'explicit':>10}{'Δ':>9}")
        for dom in ("tcm", "medical"):
            r = _read_results(os.path.join(rc, sub, f"{dom}_paper.json"))
            if not r:
                w(f"{dom:<10}  (no result file)")
                continue
            for fr in (0.1, 0.3, 1.0):
                nF = _mean(r.get(f"{fr}|none", []))
                eF = _mean(r.get(f"{fr}|explicit", []))
                w(f"{dom:<10}{fr:>6}{nF:>9.4f}{eF:>10.4f}{eF-nF:>+9.4f}")

    if did["p1"]:
        import json
        p1p = os.path.join(ROOT, "P1_cosine.json")
        if os.path.exists(p1p):
            w("\n[P1 cosine] mutex-pair cosine (expect mutex<0 and < random/attract)")
            w(f"{'domain@frac':<14}{'mutex':>9}{'random':>9}{'attract':>9}{'p(m<r)':>9}{'  P1?':>6}")
            for r in json.load(open(p1p, encoding="utf-8")):
                w(f"{r['domain']+'@'+str(r['frac']):<14}{r['mutex']['mean']:>9.4f}"
                  f"{r['random']['mean']:>9.4f}{r['attract']['mean']:>9.4f}"
                  f"{r['p_mutex_lt_random']:>9.4f}{'  ✓' if r['P1_supported'] else '  ✗':>6}")
        else:
            w("\n[P1 cosine] no P1_cosine.json (P1 step may have failed; run verify_p1_cosine.py)")

    out_txt = os.path.join(ROOT, "RESULTS.txt")
    open(out_txt, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    w(f"\nsaved the same content to: {out_txt}")


def main():
    banner(f"OntoSpan runner | STAGE = {STAGE}")
    if not os.path.exists(RUN_GRID):
        print(f"{RUN_GRID} not found\n  place run_4070.py at the project root (parent of run_code).")
        sys.exit(1)
    check_torch_cuda()
    ensure_macbert()
    t0 = time.time()
    if STAGE in ("p3", "all", "core", "everything"):
        stage_p3(grad=STAGE in ("all", "everything"))
    if STAGE in ("main", "all", "core", "everything"):
        stage_main()
    if STAGE in ("p1", "all", "core", "everything"):
        stage_p1()
    if STAGE in ("full", "everything"):
        stage_full()
    banner(f"all done | elapsed {(time.time() - t0) / 60:.1f} min")
    summarize()
    collector = os.path.join(ROOT, "collect_results.py")
    if os.path.exists(collector):
        print("\n[bundle results]")
        try:
            subprocess.run([sys.executable, collector], cwd=ROOT)
        except Exception as e:
            print(f"(auto-bundle skipped: {e}; run python collect_results.py)")
    print("\n-> bundle ontospan_results.zip and unzip at the project root.")
    print("  results map to the paper tables (Section 6.3 Table 5 / Tables 3-4).")
    try:
        input("\npress Enter to close... ")
    except Exception:
        pass


if __name__ == "__main__":
    main()

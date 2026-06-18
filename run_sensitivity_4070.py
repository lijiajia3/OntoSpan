#!/usr/bin/env python3
# Runner: hyper-parameter sensitivity for the rebuttal/appendix figure.
#
# Two sweeps, both on the capped low-resource budget and the same protocol as the
# main table (depth-2 default, fp32, five epochs, three seeds), written to
# run_code/results_sens_*/ as <domain>_paper.json.
#
#   (a) GNN depth, EXPLICIT realization -- the deployed method. depth in {1,2,3,4}
#       on medicine frac 0.1 (cleanest low-resource gain). Tests whether depth-2 is a
#       good operating point and whether deeper message passing over-smooths.
#   (b) lambda, IMPLICIT realization -- lambda only feeds the implicit signed-Laplacian
#       regularizer (the explicit realization ignores it). lambda in {0,0.05,0.1,0.5,1}.
#       On medicine 0.1 (instance-level) the curve is expected flat == none (quantifying
#       the "implicit is inert" claim); on law 1.0 (type-level) lambda regularizes the
#       scored type prototypes E directly, so a non-flat curve would be a zero-overhead
#       type-level signal worth reporting.
#
# Send back the whole run_code/results_sens_*/ tree; plot locally with
# allpaper/make_sensitivity_fig.py.
import os
import sys
import json
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(ROOT, "run_code", "run_grid_safe.py")
PY = sys.executable
ENC = "hfl/chinese-macbert-base"
SEEDS = "42,1,2"
CAP = ["--train-cap", "600", "--dev-cap", "100", "--test-cap", "300"]

LAMBDAS = [0.0, 0.05, 0.1, 0.5, 1.0]
DEPTHS = [1, 2, 3, 4]


def _run(tag, domain, frac, mode, save, extra):
    cmd = [PY, "-u", GRID, "--domains", domain, "--fracs", frac, "--modes", mode,
           "--seeds", SEEDS, *CAP, "--epochs", "5", "--encoder", ENC,
           "--save-dir", save, *extra]
    print("\n" + "=" * 72 + f"\n{tag}\n" + "=" * 72, flush=True)
    subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"))


def _mean_f1(save, domain, frac, mode):
    p = os.path.join(save, f"{domain}_paper.json")
    if not os.path.exists(p):
        return float("nan")
    res = json.load(open(p, encoding="utf-8")).get("results", {})
    runs = res.get(f"{frac}|{mode}") or (res.get("1|" + mode) if frac == "1.0" else None) or []
    return sum(r["F1"] for r in runs) / len(runs) if runs else float("nan")


def main():
    try:
        import torch
        print("CUDA:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU/MPS, slow)")
    except ImportError:
        print("torch not installed"); sys.exit(1)

    # ---- (a) GNN-depth sweep, explicit, medicine 0.1 ----
    for d in DEPTHS:
        save = os.path.join(ROOT, "run_code", f"results_sens_depth{d}")
        _run(f"depth={d} | explicit | medical 0.1", "medical", "0.1", "explicit", save,
             ["--gnn-depth", str(d)])
    # none reference (depth-independent)
    save_n_med = os.path.join(ROOT, "run_code", "results_sens_none_med01")
    _run("none ref | medical 0.1", "medical", "0.1", "none", save_n_med, ["--gnn-depth", "2"])

    # ---- (b) lambda sweep, implicit; medicine 0.1 (instance) + law 1.0 (type) ----
    for lam in LAMBDAS:
        tag = str(lam).replace(".", "p")
        for dom, fr in (("medical", "0.1"), ("law", "1.0")):
            save = os.path.join(ROOT, "run_code", f"results_sens_lam{tag}_{dom}")
            _run(f"lambda={lam} | implicit | {dom} {fr}", dom, fr, "implicit", save,
                 ["--gnn-depth", "2", "--lam", str(lam)])
    save_n_law = os.path.join(ROOT, "run_code", "results_sens_none_law10")
    _run("none ref | law 1.0", "law", "1.0", "none", save_n_law, ["--gnn-depth", "2"])

    # ---- summary ----
    print("\n" + "=" * 72 + "\nSENSITIVITY SUMMARY (mean test-F1 over seeds 42/1/2)\n" + "=" * 72)
    none_med = _mean_f1(save_n_med, "medical", "0.1", "none")
    print(f"[depth] medicine 0.1, explicit   (none ref = {none_med:.4f})")
    for d in DEPTHS:
        f1 = _mean_f1(os.path.join(ROOT, "run_code", f"results_sens_depth{d}"), "medical", "0.1", "explicit")
        print(f"   depth={d}: {f1:.4f}   (delta vs none: {f1 - none_med:+.4f})")

    none_law = _mean_f1(save_n_law, "law", "1.0", "none")
    print(f"\n[lambda] implicit")
    for dom, fr, ref in (("medical", "0.1", none_med), ("law", "1.0", none_law)):
        print(f"  {dom} {fr}  (none ref = {ref:.4f})")
        for lam in LAMBDAS:
            tag = str(lam).replace(".", "p")
            f1 = _mean_f1(os.path.join(ROOT, "run_code", f"results_sens_lam{tag}_{dom}"), dom, fr, "implicit")
            print(f"     lambda={lam}: {f1:.4f}   (delta vs none: {f1 - ref:+.4f})")
    print("\nReads: a flat lambda curve on medicine == implicit is inert (quantified);")
    print("a depth curve peaking near 2 supports the depth-2 choice; a non-flat lambda")
    print("curve on law would be a zero-overhead type-level signal.")
    print("\nSend back the whole run_code/results_sens_*/ tree.")


if __name__ == "__main__":
    main()

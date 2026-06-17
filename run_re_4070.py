#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Runner: relation-extraction transfer (CMeIE).
#
# Relation-extraction transfer grid on CMeIE, used as a negative-transfer control;
# one subprocess per config.
import os, sys, json, time, argparse, subprocess
from itertools import product

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(ROOT, "run_code", "ontospan_re_train.py")
ENC = "hfl/chinese-macbert-base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="none,implicit,explicit")
    ap.add_argument("--fracs", default="0.1,0.3,1.0")
    ap.add_argument("--seeds", default="42,1,2")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--train-cap", type=int, default=800)
    ap.add_argument("--dev-cap", type=int, default=400)
    ap.add_argument("--gnn-depth", type=int, default=2)
    ap.add_argument("--mutex-only", action="store_true",
                    help="keep only mutex edges (drop attractive); test the pure repulsion mechanism")
    ap.add_argument("--onto-lr", type=float, default=0.0,
                    help="separate lr for relation embeddings + GAT (decoupled); 0.01-0.03 suggested")
    ap.add_argument("--amp", action="store_true", help="fp16 (default fp32; not recommended)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    try:
        import torch
        dev = a.device or ("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"{'CUDA: '+torch.cuda.get_device_name(0) if dev=='cuda' else 'non-CUDA dev='+dev}")
    except ImportError:
        print("torch not installed"); sys.exit(1)

    modes = a.modes.split(","); fracs = a.fracs.split(","); seeds = a.seeds.split(",")
    save = os.path.join(ROOT, "run_code", "results_re"); tmp = os.path.join(save, ".tmp")
    os.makedirs(tmp, exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    jobs = [(fr, sd, m) for fr in fracs for sd in seeds for m in modes]
    print("=" * 70 + f"\nRE grid | {len(jobs)} runs | cap{a.train_cap} depth{a.gnn_depth} "
          f"{'fp16' if a.amp else 'fp32'}\n" + "=" * 70)

    res = {}; t0 = time.time()
    for i, (fr, sd, m) in enumerate(jobs, 1):
        outp = os.path.join(tmp, f"{fr}_{m}_{sd}.json")
        cmd = [sys.executable, "-u", TRAIN, "--mode", m, "--frac", fr, "--seed", sd,
               "--epochs", str(a.epochs), "--train-cap", str(a.train_cap),
               "--dev-cap", str(a.dev_cap), "--gnn-depth", str(a.gnn_depth),
               "--encoder", ENC, "--out", outp]
        if a.mutex_only: cmd += ["--mutex-only"]
        if a.onto_lr > 0: cmd += ["--onto-lr", str(a.onto_lr)]
        if a.amp: cmd += ["--amp"]
        if a.device: cmd += ["--device", a.device]
        print(f"  [{i}/{len(jobs)}] frac={fr} seed={sd} {m} …", flush=True)
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0 or not os.path.exists(outp):
            print(f"    ✗ FAILED rc={r.returncode}", flush=True); continue
        f1 = json.load(open(outp, encoding="utf-8"))["F1"]
        res.setdefault((fr, m), []).append(f1)
        print(f"    ✓ F1={f1:.4f}  ({(time.time()-t0)/60:.0f}min)", flush=True)

    meta = {"kind": "re_grid", "config": {"modes": a.modes, "fracs": a.fracs, "seeds": a.seeds,
            "train_cap": a.train_cap, "epochs": a.epochs, "gnn_depth": a.gnn_depth, "amp": a.amp},
            "results": {f"{fr}|{m}": v for (fr, m), v in res.items()}}
    os.makedirs(save, exist_ok=True)
    json.dump(meta, open(os.path.join(save, "cmeie_re.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    _table(res, fracs, modes)
    print(f"\nsaved run_code/results_re/cmeie_re.json | elapsed {(time.time()-t0)/60:.1f}min")


def _mean(xs): return sum(xs) / len(xs) if xs else float("nan")


def _table(res, fracs, modes):
    print("\n" + "=" * 70 + "\nRE -- mean dev micro-F1 + delta-F1 vs none\n" + "=" * 70)
    print(f"{'frac':>6} | " + " | ".join(f"{m:>10}" for m in modes) + " ||   Δimplicit   Δexplicit")
    for fr in fracs:
        n = _mean(res.get((fr, "none"), []))
        row = f"{fr:>6} | " + " | ".join(f"{_mean(res.get((fr,m),[])):>10.4f}" for m in modes)
        di = _mean(res.get((fr, "implicit"), [])) - n
        de = _mean(res.get((fr, "explicit"), [])) - n
        print(row + f" ||  {di:>+9.4f}  {de:>+9.4f}")
    print("\nreading: explicit-none > 0 at low frac (0.1) indicates the signed repulsion prior helps low-resource RE")
    print("      (the mechanism applied to a relational task).")


if __name__ == "__main__":
    main()

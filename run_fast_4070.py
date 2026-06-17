#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Parallel grid runner (one config per subprocess).
#
# Runs the full grid (main table + P3) with several parallel workers to saturate
# one GPU; the P1 geometry check is run separately at the end.
import os, sys, json, time, argparse, subprocess
from itertools import product

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_GRID = os.path.join(ROOT, "run_code", "run_grid_safe.py")
ENCODER = "hfl/chinese-macbert-base"
_CAP = os.environ.get("FAST_CAP", "600"); _EP = os.environ.get("FAST_EP", "5")
WCOMMON = ["--epochs", _EP, "--batch-size", "4", "--accum", "2", "--train-cap", _CAP,
           "--dev-cap", str(max(50, int(_CAP) // 6)), "--test-cap", str(max(50, int(_CAP) // 2)),
           "--lr", "2e-5", "--lmax", "10", "--lam", "0.1", "--gamma", "1.0",
           "--gnn-depth", "2", "--rho", "5", "--aug-mask", "0.0", "--encoder", ENCODER]


def banner(m): print("\n" + "=" * 72 + "\n" + m + "\n" + "=" * 72, flush=True)


def ensure_macbert():
    os.environ.pop("HF_HUB_OFFLINE", None); os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from transformers import AutoTokenizer, AutoModel
    try:
        AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
        AutoModel.from_pretrained(ENCODER, local_files_only=True); print("macbert local cache ready."); return
    except Exception:
        print("downloading macbert (mirror fallback on failure)...")
    for ep in (None, "https://hf-mirror.com"):
        try:
            if ep: os.environ["HF_ENDPOINT"] = ep
            AutoTokenizer.from_pretrained(ENCODER); AutoModel.from_pretrained(ENCODER); print("✓ macbert ok"); return
        except Exception:
            pass
    print("macbert download failed; copy the cache from another machine"); sys.exit(1)


def build_jobs():
    jobs = []
    seeds = ["42", "1", "2"]
    for dom, fr, sd, mode in product(["tcm", "medical"], ["0.1", "0.3", "1.0"], seeds, ["none", "explicit"]):
        jobs.append(("results_3seed", dom, fr, mode, sd, "real"))
    p3cells = [("tcm", "0.1"), ("medical", "0.1"), ("medical", "0.3"), ("medical", "1.0")]
    for pert, (dom, fr), sd in product(["real", "rewired", "removed"], p3cells, seeds):
        jobs.append((f"results_p3_{pert}", dom, fr, "explicit", sd, pert))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=3,
                    help="number of parallel workers (~1.5-2G each);"
                         "lower it if GPU memory overflows")
    ap.add_argument("--amp", action="store_true", help="use fp16 (faster but may distort TCM; default fp32)")
    ap.add_argument("--onto-lr", type=float, default=0.0, help="separate lr for ontology embeddings (P1/TCM only; hurts medical)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    try:
        import torch
        dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"{'CUDA: '+torch.cuda.get_device_name(0) if dev=='cuda' else 'non-CUDA dev='+dev+' (suggest parallel=2)'}")
    except ImportError:
        print("torch not installed"); sys.exit(1)
    ensure_macbert()

    jobs = build_jobs()
    _n = int(os.environ.get("FAST_NJOBS", "0"))
    if _n > 0: jobs = jobs[:_n]
    tmp = os.path.join(ROOT, "run_code", ".fast_tmp"); os.makedirs(tmp, exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    banner(f"OntoSpan parallel grid | {len(jobs)} runs | parallel={a.parallel} | "
           f"{'fp16(amp)' if a.amp else 'fp32(no-amp)'} | batch4")
    print("(main 36 + P3 36; batch4 each; P1 run separately at the end)")

    collected = {}
    t0 = time.time(); running = []; qi = 0; done = 0

    for f in os.listdir(tmp):
        try: os.remove(os.path.join(tmp, f))
        except Exception: pass
    total = len(jobs)

    def launch(job):
        save_sub, dom, fr, mode, sd, pert = job
        outp = os.path.join(tmp, f"{save_sub}_{dom}_{fr}_{mode}_{sd}.json")
        cmd = [sys.executable, "-u", RUN_GRID, "--worker", "--domain", dom, "--frac", fr,
               "--mode", mode, "--seed", sd, "--out", outp, "--mutex-perturb", pert, *WCOMMON]
        if a.amp: cmd += ["--amp"]
        if a.onto_lr > 0: cmd += ["--onto-lr", str(a.onto_lr), "--onto-scope", "both"]
        if a.device: cmd += ["--device", a.device]
        log = open(outp + ".log", "w")
        running.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), job, outp, log))

    while qi < len(jobs) or running:
        while qi < len(jobs) and len(running) < a.parallel:
            launch(jobs[qi]); qi += 1
            print(f"  {qi}/{len(jobs)} launched (running {len(running)})", flush=True)
        time.sleep(2)
        for tup in running[:]:
            p, job, outp, log = tup
            if p.poll() is None: continue
            running.remove(tup); log.close()
            save_sub, dom, fr, mode, sd, pert = job
            if p.returncode != 0 or not os.path.exists(outp):
                print(f"  FAIL {job} rc={p.returncode} (see {outp}.log)", flush=True); continue
            r = json.load(open(outp, encoding="utf-8"))
            collected.setdefault((save_sub, dom), {}).setdefault(f"{float(fr)}|{mode}", []).append(r)
            done += 1
            pct = 100 * done / total
            el = (time.time() - t0) / 60
            eta = el / done * (total - done)
            print(f"  ✓ [{done}/{total} = {pct:.0f}%] {save_sub} {dom} {fr} {mode} s{sd} "
                  f"F1={r['F1']:.4f}  | elapsed {el:.0f}min, eta {eta:.0f}min", flush=True)

    for (save_sub, dom), results in collected.items():
        d = os.path.join(ROOT, "run_code", save_sub); os.makedirs(d, exist_ok=True)
        fracs = sorted({k.split("|")[0] for k in results})
        meta = {"domain": dom, "kind": save_sub,
                "config": {"fracs": ",".join(fracs), "seeds": "42,1,2",
                           "modes": "none,explicit", "epochs": 5, "batch_size": 4, "accum": 2,
                           "train_cap": 600, "dev_cap": 100, "test_cap": 300,
                           "amp": bool(a.amp), "onto_lr": a.onto_lr},
                "results": results}
        json.dump(meta, open(os.path.join(d, f"{dom}_paper.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  ✓ saved {save_sub}/{dom}_paper.json", flush=True)

    banner(f"grid done | elapsed {(time.time()-t0)/60:.1f} min")

    banner("P1 cosine (implicit + anchor-lr)")
    vp = os.path.join(ROOT, "verify_p1_cosine.py")
    if os.path.exists(vp):
        subprocess.run([sys.executable, "-u", vp, "--epochs", "5", "--mode", "implicit",
                        "--anchor-lr", "0.05"], cwd=ROOT, env=env)

    coll = os.path.join(ROOT, "collect_results.py")
    if os.path.exists(coll):
        banner("bundle"); subprocess.run([sys.executable, coll], cwd=ROOT)
    print("\n bundle ontospan_results.zip (contains fp32 main/P3 + P1).")
    try: input("\npress Enter to close... ")
    except Exception: pass


if __name__ == "__main__":
    main()

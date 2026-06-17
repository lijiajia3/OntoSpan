#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Runner: external baselines (biaffine / global-pointer / BERT-CRF / W2NER).
#
# Same backbone and budget as OntoSpan; one subprocess per config, merge-writing
# the per-domain JSON so methods can be added incrementally.
import os
import sys
import json
import time
import argparse
import subprocess
from itertools import product

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, "run_code", "baselines.py")
ENCODER = "hfl/chinese-macbert-base"


def banner(m): print("\n" + "=" * 72 + "\n" + m + "\n" + "=" * 72, flush=True)


def ensure_macbert():
    os.environ.pop("HF_HUB_OFFLINE", None); os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("transformers not installed: pip install transformers"); sys.exit(1)
    try:
        AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
        AutoModel.from_pretrained(ENCODER, local_files_only=True)
        print("macbert already cached."); return
    except Exception:
        print("no local macbert cache; downloading once (~400MB)...")
    for ep in (None, "https://hf-mirror.com"):
        try:
            if ep: os.environ["HF_ENDPOINT"] = ep
            AutoTokenizer.from_pretrained(ENCODER); AutoModel.from_pretrained(ENCODER)
            print(f"macbert downloaded{' (mirror)' if ep else ''}."); return
        except Exception as e:
            print(f"  download failed ({'mirror' if ep else 'direct'}): {type(e).__name__}")
    print("macbert download failed. Copy ~/.cache/huggingface/hub/models--hfl--chinese-macbert-base")
    print("  to C:\\Users\\<user>\\.cache\\huggingface\\hub\\ and rerun."); sys.exit(1)


def check_torchcrf(methods):
    if "bert_crf" not in methods:
        return
    try:
        import torchcrf
    except ImportError:
        print("bert_crf baseline needs pytorch-crf: pip install pytorch-crf"); sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=1,
                    help="number of parallel workers; one run already saturates an 8G GPU, so 1 is recommended")
    ap.add_argument("--methods", default="biaffine,global_pointer,bert_crf")
    ap.add_argument("--domains", default="tcm,medical")
    ap.add_argument("--fracs", default="0.1,0.3,1.0")
    ap.add_argument("--seeds", default="42,1,2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--train-cap", type=int, default=600)
    ap.add_argument("--dev-cap", type=int, default=100)
    ap.add_argument("--test-cap", type=int, default=300)
    ap.add_argument("--amp", action="store_true", help="use fp16 (default fp32, matching the main table; not recommended)")
    a = ap.parse_args()

    COMMON = ["--epochs", str(a.epochs), "--batch-size", "4", "--accum", "2",
              "--train-cap", str(a.train_cap), "--dev-cap", str(a.dev_cap),
              "--test-cap", str(a.test_cap)] + (["--amp"] if a.amp else [])
    methods = a.methods.split(","); domains = a.domains.split(",")
    fracs = a.fracs.split(","); seeds = a.seeds.split(",")
    save_dir = os.path.join(ROOT, "run_code", "results_baselines")
    tmp_dir = os.path.join(save_dir, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    try:
        import torch
        dev = a.device or ("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
        if dev == "cuda":
            print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print(f"non-CUDA (dev={dev}); use --parallel 1-2.")
    except ImportError:
        print("torch not installed"); sys.exit(1)
    check_torchcrf(methods)
    ensure_macbert()

    jobs = [(fr, sd, dom, m) for fr in fracs for sd in seeds for dom in domains for m in methods]
    banner(f"Baselines grid | {len(jobs)} runs | parallel={a.parallel} | batch4 (matching OntoSpan)")

    collected = {dom: {} for dom in domains}
    t0 = time.time()
    running = []
    qi = 0; done = 0

    def launch(job):
        fr, sd, dom, m = job
        outp = os.path.join(tmp_dir, f"{m}_{dom}_{fr}_{sd}.json")
        cmd = [sys.executable, "-u", BASE, "--worker", "--method", m, "--domain", dom,
               "--frac", fr, "--seed", sd, "--out", outp, "--encoder", ENCODER, *COMMON]
        if a.device: cmd += ["--device", a.device]
        log = open(outp + ".log", "w")
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        running.append((p, job, outp, log))
        print(f"  launch [{m} {dom} frac={fr} seed={sd}]  (running {len(running)})", flush=True)

    while qi < len(jobs) or running:
        while qi < len(jobs) and len(running) < a.parallel:
            launch(jobs[qi]); qi += 1
        time.sleep(2)
        for tup in running[:]:
            p, job, outp, log = tup
            if p.poll() is None:
                continue
            running.remove(tup); log.close()
            fr, sd, dom, m = job
            if p.returncode != 0 or not os.path.exists(outp):
                print(f"  FAILED [{m} {dom} {fr} {sd}] rc={p.returncode} (see {outp}.log)", flush=True)
                continue
            r = json.load(open(outp, encoding="utf-8"))
            collected[dom].setdefault(f"{fr}|{m}", []).append(r)
            done += 1
            print(f"  ✓ done [{m} {dom} frac={fr} seed={sd}]  F1={r['F1']:.4f}  "
                  f"({done}/{len(jobs)})", flush=True)

    for dom in domains:
        path = os.path.join(save_dir, f"{dom}_baselines.json")
        merged = {}
        if os.path.exists(path):
            try:
                merged = json.load(open(path, encoding="utf-8")).get("results", {})
            except Exception:
                merged = {}
        n_old = len(merged)
        merged.update(collected[dom])
        meta = {"domain": dom, "kind": "baselines",
                "config": {"methods": a.methods, "fracs": a.fracs, "seeds": a.seeds,
                           "epochs": 5, "batch_size": 4, "accum": 2,
                           "train_cap": a.train_cap, "dev_cap": 100, "test_cap": 300},
                "results": merged}
        json.dump(meta, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  saved {path}  (old {n_old} keys + new {len(collected[dom])} -> {len(merged)} total)", flush=True)

    banner(f"all done | elapsed {(time.time()-t0)/60:.1f} min")
    summarize(collected, methods, fracs, domains)
    collector = os.path.join(ROOT, "collect_results.py")
    if os.path.exists(collector):
        try:
            subprocess.run([sys.executable, collector], cwd=ROOT)
        except Exception as e:
            print(f"(bundle skipped: {e})")
    print("\n-> bundle ontospan_results.zip; baselines are under results_baselines/.")
    try:
        input("\npress Enter to close... ")
    except Exception:
        pass


def _mean(xs):
    xs = [x for x in xs]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(collected, methods, fracs, domains):
    print("\n[Baselines] mean test-F1 over seeds (batch4, same setup as OntoSpan)")
    head = f"{'domain/frac':<14}" + "".join(f"{m:>16}" for m in methods)
    print(head)
    for dom in domains:
        for fr in fracs:
            row = f"{dom+' '+fr:<14}"
            for m in methods:
                runs = collected.get(dom, {}).get(f"{fr}|{m}", [])
                v = _mean([r["F1"] for r in runs])
                row += f"{v:>16.4f}"
            print(row)
    print("-> compare against the explicit column in run_code/results_3seed.")


if __name__ == "__main__":
    main()

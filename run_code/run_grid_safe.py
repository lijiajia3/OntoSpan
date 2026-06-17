# -*- coding: utf-8 -*-
# Subprocess-isolated grid driver.
#
# Runs each (domain, mode, fraction, seed) configuration in its own subprocess so
# that GPU memory is fully released between runs (avoids accumulation/OOM). Also
# supports injecting a type-level mutex (--type-mutex) for the type-mutex study.
import os, sys, json, time, argparse, subprocess, random
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

DOMAIN_DATASET = {"tcm": "tcmner", "medical": "cmeee"}


def _subsample(data, cap, seed=42):
    if not cap or cap <= 0 or len(data) <= cap:
        return data
    r = random.Random(seed); idx = sorted(r.sample(range(len(data)), cap))
    return [data[i] for i in idx]


def _load_domain(domain, caps):
    from ontospan_experiment import read_jsonl, build_types, load_ontology
    ddir = os.path.join(_ROOT, "data", domain)
    paths = {sp: os.path.join(ddir, f"{sp}.jsonl") for sp in ("train", "dev", "test")}
    tr = _subsample(read_jsonl(paths["train"]), caps[0])
    dv = _subsample(read_jsonl(paths["dev"]),   caps[1])
    te = _subsample(read_jsonl(paths["test"]),  caps[2])
    onto = load_ontology(domain); types = build_types(tr + dv, onto)
    return tr, dv, te, onto, types


def worker(a):
    import torch
    from ontospan_experiment import HFEncoder
    from run_ablation import train_one
    tr, dv, te, onto, types = _load_domain(a.domain, (a.train_cap, a.dev_cap, a.test_cap))
    # Optionally inject a type-level mutex: all type pairs, flip attract->mutex, or explicit pairs.
    tmx = getattr(a, "type_mutex", None)
    if tmx:
        rt = [t for t in types if t != "∅"]
        onto = dict(onto)
        if tmx == "all":
            onto["tmutex"] = list(onto.get("tmutex", [])) + \
                [(rt[i], rt[j]) for i in range(len(rt)) for j in range(i + 1, len(rt))]
        elif tmx == "flip":
            onto["tmutex"] = list(onto.get("tmutex", [])) + [(a_, b_) for a_, b_, _w in onto.get("attract", [])]
            onto["attract"] = []
        else:
            onto["tmutex"] = list(onto.get("tmutex", [])) + \
                [tuple(p.split(":")) for p in tmx.split(",") if ":" in p]
    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

    class Cfg: pass
    cfg = Cfg()
    for k in ("lmax", "lr", "lam", "gamma", "gnn_depth", "rho", "batch_size", "epochs", "accum", "aug_mask", "mutex_perturb", "onto_lr", "onto_scope", "score_anchors"):
        setattr(cfg, k, getattr(a, k, None))
    cfg.amp = bool(getattr(a, "amp", False)); cfg.ckpt_dir = None
    r = train_one(lambda: HFEncoder(a.encoder), types, onto, a.mode, a.frac, a.seed,
                  tr, dv, te, cfg, dev, tag=a.domain)
    json.dump(r, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)


def driver(a):
    domains = a.domains.split(","); fracs = [float(x) for x in a.fracs.split(",")]
    modes = a.modes.split(","); seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(a.save_dir, exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    print(f"=== run_grid_safe: {domains} × {modes} × fracs{fracs} × seeds{seeds} | "
          f"epochs={a.epochs} batch={a.batch_size} accum={a.accum} cap={a.train_cap}/{a.dev_cap}/{a.test_cap} ===")
    for domain in domains:
        tr, dv, te, onto, types = _load_domain(domain, (a.train_cap, a.dev_cap, a.test_cap))
        print(f"\n### {domain}: train/dev/test={len(tr)}/{len(dv)}/{len(te)} | types({len(types)})={types} | imutex={len(onto['imutex'])}")
        results = {}; t0 = time.time()
        for frac in fracs:
            for seed in seeds:
                for mode in modes:
                    outp = os.path.join(a.save_dir, f".tmp_{domain}_{frac}_{mode}_{seed}.json")
                    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--worker",
                           "--domain", domain, "--frac", str(frac), "--mode", mode, "--seed", str(seed), "--out", outp,
                           "--epochs", str(a.epochs), "--batch-size", str(a.batch_size), "--accum", str(a.accum),
                           "--train-cap", str(a.train_cap), "--dev-cap", str(a.dev_cap), "--test-cap", str(a.test_cap),
                           "--lr", str(a.lr), "--lmax", str(a.lmax), "--lam", str(a.lam), "--gamma", str(a.gamma),
                           "--gnn-depth", str(a.gnn_depth), "--rho", str(a.rho), "--aug-mask", str(a.aug_mask),
                           "--encoder", a.encoder]
                    cmd += ["--mutex-perturb", a.mutex_perturb]
                    if getattr(a, "onto_lr", 0): cmd += ["--onto-lr", str(a.onto_lr), "--onto-scope", a.onto_scope]
                    if getattr(a, "score_anchors", False): cmd += ["--score-anchors"]
                    if getattr(a, "type_mutex", None): cmd += ["--type-mutex", a.type_mutex]
                    if a.device: cmd += ["--device", a.device]
                    if a.amp: cmd += ["--amp"]
                    print(f"  → [{domain} frac={frac} {mode} seed={seed}] subprocess …", flush=True)
                    p = subprocess.run(cmd, env=env)
                    if p.returncode != 0 or not os.path.exists(outp):
                        print(f"  ✗ FAILED {domain} {frac} {mode} {seed} (rc={p.returncode})"); sys.exit(1)
                    r = json.load(open(outp, encoding="utf-8")); os.remove(outp)
                    results.setdefault(f"{frac}|{mode}", []).append(r)
                    print(f"     ✓ test-F1={r['F1']:.4f}  dev_best={r['dev_best_F1']:.4f} @ep{r.get('best_epoch')}", flush=True)
        meta = {"domain": domain, "dataset": DOMAIN_DATASET.get(domain, domain),
                "n_train": len(tr), "n_dev": len(dv), "n_test": len(te),
                "types": types, "tmutex": onto["tmutex"], "n_imutex": len(onto["imutex"]),
                "config": {"fracs": a.fracs, "seeds": a.seeds, "modes": a.modes, "epochs": a.epochs,
                           "lr": a.lr, "lam": a.lam, "gamma": a.gamma, "gnn_depth": a.gnn_depth, "rho": a.rho,
                           "lmax": a.lmax, "train_cap": a.train_cap, "dev_cap": a.dev_cap, "test_cap": a.test_cap,
                           "encoder": a.encoder, "accum": a.accum, "aug_mask": a.aug_mask, "batch_size": a.batch_size,
                           "mutex_perturb": a.mutex_perturb},
                "elapsed_min": round((time.time() - t0) / 60, 2), "results": results}
        path = os.path.join(a.save_dir, f"{domain}_paper.json")
        json.dump(meta, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  ✓ saved {path}  ({meta['elapsed_min']} min)", flush=True)
    print("\n=== grid done (subprocess-isolated, no OOM) ===")


def main():
    ap = argparse.ArgumentParser(description="OntoSpan grid with subprocess isolation (avoids MPS memory buildup)")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--domain"); ap.add_argument("--frac", type=float); ap.add_argument("--mode")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out")
    ap.add_argument("--domains", default="tcm,medical"); ap.add_argument("--fracs", default="0.1,0.3,1.0")
    ap.add_argument("--modes", default="none,implicit,explicit"); ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--train-cap", type=int, default=600); ap.add_argument("--dev-cap", type=int, default=100)
    ap.add_argument("--test-cap", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--lmax", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.1); ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--gnn-depth", type=int, default=2); ap.add_argument("--rho", type=int, default=5)
    ap.add_argument("--aug-mask", type=float, default=0.0)
    ap.add_argument("--onto-lr", type=float, default=0.0, help="separate learning rate for ontology embeddings (0 = same as encoder)")
    ap.add_argument("--onto-scope", default="both", choices=["both", "anchor"], help="scope of onto-lr")
    ap.add_argument("--score-anchors", action="store_true",
                    help="route instance anchors into scoring (parent score = max(type, best anchor)); off by default")
    ap.add_argument("--type-mutex", default=None,
                    help="inject type-level repulsion. all=all type pairs; flip=flip attract-adjacent pairs to repulsion;"
                         "or '中药:方剂,症状:证候' for specific pairs. default = original ontology.")
    ap.add_argument("--encoder", default="hfl/chinese-macbert-base"); ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action="store_true", help="CUDA AMP (fp16); ignored on non-CUDA devices")
    ap.add_argument("--mutex-perturb", choices=["real", "rewired", "removed"], default="real",
                    help="P3 control: real / rewired (random-rewire mutex edges) / removed")
    ap.add_argument("--save-dir", default=os.path.join(_HERE, "results_scratch"),
                    help="output dir for standalone driver mode; default results_scratch (temporary)")
    a = ap.parse_args()
    worker(a) if a.worker else driver(a)


if __name__ == "__main__":
    main()

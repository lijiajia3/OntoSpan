# -*- coding: utf-8 -*-
# Ablation and baseline-comparison driver.
#
# Runs none/implicit/explicit side by side, printing per-epoch P/R/F1, a comparison
# table with delta-F1 vs. the no-prior baseline, and per-type F1.
import os, sys, json, argparse, random
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from ontospan_experiment import (HFEncoder, StubEncoder, make_model, run_epoch,
                                 evaluate, read_jsonl, build_types, load_ontology,
                                 set_seed, mask_augment, free_memory, perturb_ontology, W_RELATED)

MODE_TAG = {"none": "none (base)", "implicit": "implicit (3.5)",
            "explicit": "explicit (3.6)", "sgcn": "SGCN baseline",
            "sigat": "SiGAT baseline"}


def train_one(enc_factory, types, onto, mode, frac, seed,
              train, dev_d, test, args, device, tag=""):
    set_seed(seed)
    free_memory(device)
    onto = perturb_ontology(onto, getattr(args, "mutex_perturb", "real"), seed)
    n = max(1, int(len(train) * frac))
    idx = list(range(len(train))); random.shuffle(idx)
    sub = [train[i] for i in idx[:n]]
    model = make_model(enc_factory, types, onto, mode, args.lmax, device, gnn_depth=args.gnn_depth,
                       score_anchors=getattr(args, "score_anchors", False),
                       me_weight=getattr(args, "me_weight", 1.0))
    onto_lr = float(getattr(args, "onto_lr", 0) or 0)
    onto_scope = getattr(args, "onto_scope", "both")
    if onto_lr > 0:
        onto_ps = [model.anchor] if getattr(model, "anchor", None) is not None else []
        if onto_scope != "anchor":
            onto_ps = onto_ps + list(model.type_emb.parameters())
        oids = {id(p) for p in onto_ps}
        base_ps = [p for p in model.parameters() if id(p) not in oids]
        optim = torch.optim.AdamW([{"params": base_ps, "lr": args.lr},
                                   {"params": onto_ps, "lr": onto_lr}])
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = getattr(args, "amp", False) and str(device) == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    accum = max(1, int(getattr(args, "accum", 1)))
    aug_p = float(getattr(args, "aug_mask", 0.0))
    extra = ("  [AMP fp16]" if use_amp else "") + (f"  [accum×{accum}]" if accum > 1 else "") \
            + (f"  [mask {aug_p}]" if aug_p > 0 else "")
    print(f"\n──[{tag} frac={frac} seed={seed} | {MODE_TAG.get(mode, mode)}]  training {len(sub)} sentences" + extra + "──")
    best_devF, best_state, best_ep = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        sub_ep = mask_augment(sub, aug_p, seed=seed * 1000 + ep) if aug_p > 0 else sub
        tl = run_epoch(model, sub_ep, optim, device, rho=args.rho, lam=args.lam,
                       gamma=args.gamma, bs=args.batch_size, scaler=scaler, accum=accum)
        P, R, F = evaluate(model, dev_d, device, bs=args.batch_size, use_amp=use_amp)["micro"]
        star = ""
        if F >= best_devF:
            best_devF, best_ep = F, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = "  *best"
        print(f"   epoch {ep:2d}/{args.epochs}  train_loss={tl:8.2f}  dev:  P={P:.4f}  R={R:.4f}  F1={F:.4f}{star}")
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        ck_dir = getattr(args, "ckpt_dir", None)
        if ck_dir:
            os.makedirs(ck_dir, exist_ok=True)
            ck = os.path.join(ck_dir, f"best_{tag}_{mode}_frac{frac}_seed{seed}.pt")
            torch.save({"state_dict": best_state, "dev_f1": best_devF, "epoch": best_ep,
                        "types": types, "mode": mode}, ck)
            print(f"   saved best weights {ck} (dev-F1={best_devF:.4f} @ep{best_ep})")
    tev = evaluate(model, test, device, bs=args.batch_size, use_amp=use_amp,
                   return_details=bool(getattr(args, "save_predictions", False)))
    tP, tR, tF = tev["micro"]
    print(f"   └─ test@best-dev(ep{best_ep}): P={tP:.4f}  R={tR:.4f}  F1={tF:.4f}")
    out = {"seed": int(seed), "P": tP, "R": tR, "F1": tF,
           "dev_best_F1": best_devF, "best_epoch": best_ep,
           "per_type": {l: list(v) for l, v in tev["per_type"].items()}}
    if "predictions" in tev:
        out["predictions"] = tev["predictions"]
    return out


def summarize(res, modes, fracs, seeds):
    print("\n" + "=" * 86)
    print(f"  ablation + baseline comparison (test span-NER, mean+/-std over {len(seeds)} seeds)")
    print("=" * 86)
    hdr = f"{'frac':>6} | {'method':<12} | {'P':>8} | {'R':>8} | {'F1 (mean±std)':>16} | {'dF1 vs base':>12}"
    print(hdr); print("-" * len(hdr))
    for frac in fracs:
        base = np.mean([d["F1"] for d in res[(frac, "none")]]) if (frac, "none") in res else None
        for mode in modes:
            arr = res[(frac, mode)]
            P = np.mean([d["P"] for d in arr]); R = np.mean([d["R"] for d in arr])
            F = np.mean([d["F1"] for d in arr]); Fs = np.std([d["F1"] for d in arr])
            delta = "—" if (base is None or mode == "none") else f"{F - base:+.4f}"
            print(f"{frac:>6} | {mode:<12} | {P:>8.4f} | {R:>8.4f} | {F:>8.4f}±{Fs:.4f} | {delta:>12}")
    big = max(fracs)
    labels = sorted({l for mode in modes for l in res[(big, mode)][0]["per_type"]})
    if labels:
        print(f"\n  per-type F1 (frac={big}, seed={seeds[0]}):")
        print(f"    {'type':<10} " + " ".join(f"{m:>10}" for m in modes))
        for lab in labels:
            row = []
            for mode in modes:
                pt = res[(big, mode)][0]["per_type"].get(lab)
                row.append(f"{pt[2]:.3f}" if pt else "  -  ")
            print(f"    {lab:<10} " + " ".join(f"{c:>10}" for c in row))


def smoke():
    print("=== SMOKE: synthetic data + stub encoder (per-epoch P/R/F1 + 3-method table) ===")
    raw = [
        ("方中附子与半夏同用属配伍禁忌", [(2, 4, "中药"), (5, 7, "中药")]),
        ("桂枝汤主治外感风寒", [(0, 3, "方剂"), (0, 2, "中药")]),
        ("患者头痛服用甘草", [(2, 4, "症状"), (6, 8, "中药")]),
        ("乌头反贝母临床慎用", [(0, 2, "中药"), (3, 5, "中药")]),
    ] * 8
    onto = {"attract": [("中药", "方剂", W_RELATED)], "tmutex": [],
            "imutex": [("附子", "半夏"), ("乌头", "贝母")], "parent": "中药"}
    types = build_types(raw, onto)

    class A: epochs = 5; lr = 5e-3; lmax = 4; lam = 0.2; gamma = 1.0; rho = 6; batch_size = 8; gnn_depth = 2
    modes = ["none", "implicit", "explicit"]; fracs = [1.0]; seeds = [0]
    res = {}
    for mode in modes:
        r = train_one(lambda: StubEncoder(48), types, onto, mode, 1.0, 0,
                      raw, raw, raw, A, "cpu", tag="smoke")
        res[(1.0, mode)] = [r]
    summarize(res, modes, fracs, seeds)
    print("\n  per-epoch P/R/F1 + 3-method table + per-type F1 all passed.")


def main():
    ap = argparse.ArgumentParser(description="OntoSpan Section-3 ablation + baseline comparison")
    ap.add_argument("--domain", choices=["law", "tcm", "medical"])
    ap.add_argument("--data-root", default=os.path.join(_ROOT, "data"))
    ap.add_argument("--encoder", default="hfl/chinese-macbert-base")
    ap.add_argument("--modes", default="none,implicit,explicit", help="comma-separated; all three modes by default")
    ap.add_argument("--fracs", default="1.0", help="low-resource fractions, comma-separated, e.g. 0.1,0.2,1.0")
    ap.add_argument("--seeds", default="42", help="comma-separated")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lmax", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.1, help="implicit regularization strength (lambda)")
    ap.add_argument("--gamma", type=float, default=1.0, help="instance-layer weight (gamma)")
    ap.add_argument("--gnn-depth", type=int, default=2, help="explicit SignedGAT layers (multi-hop)")
    ap.add_argument("--rho", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action="store_true",
                    help="CUDA AMP (fp16); ignored on non-CUDA devices")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation steps; effective batch = batch_size * accum")
    ap.add_argument("--aug-mask", type=float, default=0.0,
                    help="char random-masking probability (training set only; 0 = off)")
    ap.add_argument("--ckpt-dir", default=None, help="if set, save best-dev .pt weights to this directory")
    ap.add_argument("--mutex-perturb", choices=["real", "rewired", "removed", "random_sign"], default="real",
                    help="P3 control: real / rewired / removed")
    ap.add_argument("--me-weight", type=float, default=1.0,
                    help="negative-message/implicit-ME weight")
    ap.add_argument("--save-predictions", action="store_true",
                    help="include per-sample gold/predicted spans in result JSON")
    ap.add_argument("--save-json", default=None)
    ap.add_argument("--score-anchors", action="store_true",
                    help="route instance anchors into span scoring (parent score = max(type, best anchor));"
                         "off by default.")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        smoke(); return
    if not a.domain:
        ap.error("specify --domain {law,tcm,medical} (or --smoke)")

    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    ddir = os.path.join(a.data_root, a.domain)
    need = {sp: os.path.join(ddir, f"{sp}.jsonl") for sp in ("train", "dev", "test")}
    miss = [p for p in need.values() if not os.path.exists(p)]
    if miss:
        print(f"missing data: {miss}\n  run run_code/download_datasets.py --dataset <name>."); return
    train, dev_d, test = (read_jsonl(need[s]) for s in ("train", "dev", "test"))
    onto = load_ontology(a.domain)
    types = build_types(train + dev_d, onto)
    modes = a.modes.split(","); fracs = [float(x) for x in a.fracs.split(",")]; seeds = [int(s) for s in a.seeds.split(",")]
    print(f"domain={a.domain} device={dev} | types({len(types)})={types}")
    print(f"train/dev/test={len(train)}/{len(dev_d)}/{len(test)} | type-mutex={onto['tmutex']} instance-mutex pairs={len(onto['imutex'])}")
    print(f"modes={modes} fracs={fracs} seeds={seeds} epochs={a.epochs} lam={a.lam} gamma={a.gamma} gnn_depth={a.gnn_depth}")

    enc_factory = lambda: HFEncoder(a.encoder)
    res = {}
    for frac in fracs:
        for seed in seeds:
            for mode in modes:
                r = train_one(enc_factory, types, onto, mode, frac, seed,
                              train, dev_d, test, a, dev, tag=a.domain)
                res.setdefault((frac, mode), []).append(r)
    summarize(res, modes, fracs, seeds)
    if a.save_json:
        os.makedirs(os.path.dirname(a.save_json) or ".", exist_ok=True)
        dump = {f"{f}|{m}": v for (f, m), v in res.items()}
        json.dump({"domain": a.domain, "config": vars(a), "results": dump},
                  open(a.save_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\nsaved -> {a.save_json}")
    print("\nreading: delta-F1 vs base -- implicit/explicit are largest at low fractions;"
          "\n      explicit can exceed implicit when the mutex graph is dense/multi-hop.")


if __name__ == "__main__":
    main()

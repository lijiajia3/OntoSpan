#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Verify P1: mutex-pair representation-cosine geometry.
#
# Trains a model, then compares the cosine of mutex pairs against attractive and
# random pairs. With a decoupled ontology learning rate, the predicted antipodal
# geometry (mutex cosine significantly negative) forms.
import os
import sys
import json
import random
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_CODE = os.path.join(ROOT, "run_code")
for p in (RUN_CODE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
from run_code.ontospan_experiment import (HFEncoder, make_model, read_jsonl,
                                  build_types, load_ontology, perturb_ontology,
                                  run_epoch, evaluate, set_seed, free_memory)
from run_code.run_ablation import train_one

LR_OVERRIDE = None
ANCHOR_LR = None


class Cfg:
    lmax = 10
    lr = 2e-5
    lam = 0.1
    gamma = 1.0
    gnn_depth = 2
    rho = 5
    batch_size = 4
    epochs = 5
    accum = 2
    aug_mask = 0.0
    mutex_perturb = "real"
    amp = False
    ckpt_dir = None


def _cos(H, i, j):
    return F.cosine_similarity(H[i].unsqueeze(0), H[j].unsqueeze(0)).item()


def _stats(xs):
    if not xs:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "frac_neg": float("nan")}
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return {"n": len(xs), "mean": m, "std": var ** 0.5,
            "frac_neg": sum(1 for x in xs if x < 0) / len(xs)}


def _perm_test(a, b, n_perm=10000, seed=0):
    if not a or not b:
        return float("nan")
    obs = sum(a) / len(a) - sum(b) / len(b)
    pool = a + b
    na = len(a)
    rng = random.Random(seed)
    hit = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        d = sum(pool[:na]) / na - sum(pool[na:]) / (len(pool) - na)
        if d <= obs:
            hit += 1
    return (hit + 1) / (n_perm + 1)


def all_node_reps(model):
    model.eval()
    with torch.no_grad():
        if model.mode == "explicit" and model.anchor is not None:
            x = torch.cat([model.type_emb.weight, model.anchor], 0)
            H = model.gnn(x, model.pos_nbr, model.neg_nbr)
        elif model.mode == "explicit":
            H = model.gnn(model.type_emb.weight, model.pos_nbr, model.neg_nbr)
        else:
            H = (torch.cat([model.type_emb.weight, model.anchor], 0)
                 if model.anchor is not None else model.type_emb.weight)
    return H.detach().float().cpu()


def verify_one(domain, frac, epochs, seed, device, encoder, tmp, mode="explicit", lam=0.1):
    ddir = os.path.join(ROOT, "data", domain)
    tr = read_jsonl(os.path.join(ddir, "train.jsonl"))[:600]
    dv = read_jsonl(os.path.join(ddir, "dev.jsonl"))[:100]
    te = read_jsonl(os.path.join(ddir, "test.jsonl"))[:300]
    onto = load_ontology(domain)
    types = build_types(tr + dv, onto)

    print(f"\n=== P1 train: {domain} frac={frac} {mode} {epochs}ep seed={seed}"
          + (f" | onto-lr={ANCHOR_LR}" if ANCHOR_LR else "") + " ===")

    if ANCHOR_LR:
        set_seed(seed); free_memory(device)
        onto_real = perturb_ontology(onto, "real", seed)
        n = max(1, int(len(tr) * frac)); idx = list(range(len(tr))); random.shuffle(idx)
        sub = [tr[i] for i in idx[:n]]
        model = make_model(lambda: HFEncoder(encoder), types, onto_real, mode, 10, device, gnn_depth=2)
        onto_ps = ([model.anchor] if model.anchor is not None else []) + list(model.type_emb.parameters())
        oids = {id(p) for p in onto_ps}
        base_ps = [p for p in model.parameters() if id(p) not in oids]
        optim = torch.optim.AdamW([{"params": base_ps, "lr": 2e-5},
                                   {"params": onto_ps, "lr": ANCHOR_LR}])
        best, best_state = -1.0, None
        for ep in range(1, epochs + 1):
            run_epoch(model, sub, optim, device, rho=5, lam=lam, gamma=1.0, bs=4, scaler=None, accum=2)
            f1 = evaluate(model, dv, device, bs=4)["micro"][2]
            if f1 >= best:
                best, best_state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"   test F1={evaluate(model, te, device, bs=4)['micro'][2]:.4f} (dev best={best:.4f})")
    else:
        cfg = Cfg(); cfg.epochs = epochs; cfg.lam = lam
        cfg.lr = LR_OVERRIDE if LR_OVERRIDE else cfg.lr; cfg.ckpt_dir = tmp
        train_one(lambda: HFEncoder(encoder), types, onto, mode, frac, seed,
                  tr, dv, te, cfg, device, tag=domain)
        onto_real = perturb_ontology(onto, "real", seed)
        model = make_model(lambda: HFEncoder(encoder), types, onto_real, mode, 10, device, gnn_depth=2)
        ck = os.path.join(tmp, f"best_{domain}_{mode}_frac{frac}_seed{seed}.pt")
        model.load_state_dict(torch.load(ck, map_location=device)["state_dict"])

    H = all_node_reps(model)
    K = model.K

    mutex = [(i, j) for i, j in model.type_mutex] + \
            [(K + a, K + b) for a, b in model.inst_mutex]
    attract = [(i, j) for i, j, _w in model.type_attract]
    if model.anchor is not None and model.parent_idx is not None:
        attract += [(K + m, model.parent_idx) for m in range(model.M)]

    edge = set()
    for i, j in mutex + attract:
        edge.add((min(i, j), max(i, j)))
    N = H.size(0)
    nodes = [i for i in range(N) if i != 0]
    rng = random.Random(seed)
    target = max(len(mutex) * 5, 200)
    rand, tries = [], 0
    while len(rand) < target and tries < target * 80:
        tries += 1
        a, b = rng.sample(nodes, 2)
        key = (min(a, b), max(a, b))
        if key in edge:
            continue
        rand.append((a, b))

    cm = [_cos(H, i, j) for i, j in mutex]
    ca = [_cos(H, i, j) for i, j in attract]
    cr = [_cos(H, i, j) for i, j in rand]
    sm, sa, sr = _stats(cm), _stats(ca), _stats(cr)
    p_mr = _perm_test(cm, cr, seed=seed)
    p_ma = _perm_test(cm, ca, seed=seed)

    passed = (sm["mean"] < 0) and (sm["mean"] < sr["mean"]) and (sm["mean"] < sa["mean"]) \
        and (p_mr < 0.05)
    return {"domain": domain, "frac": frac, "epochs": epochs, "seed": seed,
            "mutex": sm, "attract": sa, "random": sr,
            "p_mutex_lt_random": p_mr, "p_mutex_lt_attract": p_ma,
            "P1_supported": bool(passed)}


def main():
    ap = argparse.ArgumentParser(description="Verify P1: mutex-pair representation cosine is significantly negative")
    ap.add_argument("--pairs", default="tcm:0.1,medical:1.0",
                    help="comma-separated domain:frac")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--mode", default="explicit", choices=["explicit", "implicit"],
                    help="which realization's geometry to measure; Property 1 concerns the implicit regularizer")
    ap.add_argument("--lam", type=float, default=0.1, help="ontology-regularization strength")
    ap.add_argument("--lr-override", type=float, default=0.0, help=">0 overrides the global learning rate")
    ap.add_argument("--anchor-lr", type=float, default=0.0, help="separate learning rate for ontology embeddings (encoder stays 2e-5)")
    ap.add_argument("--encoder", default="hfl/chinese-macbert-base")
    a = ap.parse_args()
    global LR_OVERRIDE, ANCHOR_LR
    LR_OVERRIDE = a.lr_override if a.lr_override and a.lr_override > 0 else None
    ANCHOR_LR = a.anchor_lr if a.anchor_lr and a.anchor_lr > 0 else None

    device = a.device or ("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"P1 cosine test | device={device} | mode={a.mode}")
    tmp = os.path.join(RUN_CODE, ".p1_ckpt")
    os.makedirs(tmp, exist_ok=True)

    out = []
    for tok in a.pairs.split(","):
        dom, fr = tok.split(":")
        out.append(verify_one(dom.strip(), float(fr), a.epochs, a.seed, device,
                              a.encoder, tmp, mode=a.mode, lam=a.lam))

    print("\n" + "#" * 78)
    print("# P1: mutex-pair representation cosine (expect mutex < 0, and < random, < attract)")
    print("#" * 78)
    hdr = f"{'domain@frac':<14}{'mutex(mean)':>13}{'%neg':>7}{'random':>10}{'attract':>10}" \
          f"{'p(m<r)':>9}{'p(m<a)':>9}{'  P1?':>6}"
    print(hdr)
    for r in out:
        print(f"{r['domain']+'@'+str(r['frac']):<14}"
              f"{r['mutex']['mean']:>13.4f}{r['mutex']['frac_neg']*100:>6.0f}%"
              f"{r['random']['mean']:>10.4f}{r['attract']['mean']:>10.4f}"
              f"{r['p_mutex_lt_random']:>9.4f}{r['p_mutex_lt_attract']:>9.4f}"
              f"{'  ✓' if r['P1_supported'] else '  ✗':>6}")
    print("-> check = mutex cosine significantly negative and lowest; P1 supported.")

    path = os.path.join(ROOT, f"P1_cosine_{a.mode}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nsaved machine-readable result: {path}")


if __name__ == "__main__":
    main()

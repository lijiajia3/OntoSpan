#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Relation-extraction training loop.
#
# Training and evaluation loop for the relation-extraction transfer experiment.
import os, sys, json, time, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from ontospan_experiment import HFEncoder, StubEncoder, SignedGAT, set_seed, free_memory
from ontospan_re import load_re, RE_DIR, NA

W_ATTRACT, W_MUTEX = 1.0, 1.0


class OntoRE(nn.Module):
    def __init__(self, encoder, relations, attract, mutex, mode="explicit", gnn_depth=1):
        super().__init__()
        self.encoder = encoder
        self.relations = relations
        self.R = len(relations)
        self.mode = mode
        d = encoder.hidden
        self.rel_emb = nn.Embedding(self.R, d)
        self.pair_proj = nn.Sequential(nn.Linear(4 * d, d), nn.LayerNorm(d), nn.GELU())
        self.Wbi = nn.Parameter(torch.empty(d, d)); nn.init.xavier_uniform_(self.Wbi)
        self.rel_bias = nn.Parameter(torch.zeros(self.R))
        self.attract, self.mutex = attract, mutex
        self.pos_nbr = [[] for _ in range(self.R)]
        self.neg_nbr = [[] for _ in range(self.R)]
        def add(adj, i, j): adj[i].append(j); adj[j].append(i)
        for i, j in attract: add(self.pos_nbr, i, j)
        for i, j in mutex:   add(self.neg_nbr, i, j)
        self.gnn = SignedGAT(d, depth=gnn_depth) if mode == "explicit" else None

    def relation_reps(self):
        if self.mode == "explicit":
            return self.gnn(self.rel_emb.weight, self.pos_nbr, self.neg_nbr)
        return self.rel_emb.weight

    @staticmethod
    def _pool(reps, offs, cs, ce):
        idx = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce]
        if not idx: idx = [min(range(len(offs)), key=lambda k: abs(offs[k][0] - cs))]
        return reps[idx].mean(0)

    def pair_feats(self, reps, offs, pairs, ents):
        feats = []
        for si, oi in pairs:
            es = self._pool(reps, offs, ents[si][0], ents[si][1])
            eo = self._pool(reps, offs, ents[oi][0], ents[oi][1])
            feats.append(torch.cat([es, eo, es - eo, es * eo]))
        return self.pair_proj(torch.stack(feats)) if feats else \
            torch.zeros(0, self.encoder.hidden, device=self.rel_emb.weight.device)

    def score(self, pair_rep, rreps):
        return (pair_rep @ self.Wbi) @ rreps.t() + self.rel_bias

    def ontology_loss(self, gamma=1.0):
        if self.mode != "implicit":
            return torch.zeros((), device=self.rel_emb.weight.device)
        E = F.normalize(self.rel_emb.weight, dim=1)
        reg = torch.zeros((), device=E.device)
        for a, b in self.attract: reg = reg + W_ATTRACT * (E[a] - E[b]).pow(2).sum()
        for a, b in self.mutex:   reg = reg + W_MUTEX * (E[a] + E[b]).pow(2).sum()
        return gamma * reg


def make_pairs(d, r2i, rho=5, training=True, rng=random):
    ents, triples = d["entities"], d["triples"]
    gold = {(si, oi): r2i[p] for si, oi, p in triples if p in r2i}
    pairs, labels = [], []
    for (si, oi), lab in gold.items():
        pairs.append((si, oi)); labels.append(lab)
    negs = [(si, oi) for si in range(len(ents)) for oi in range(len(ents))
            if si != oi and (si, oi) not in gold]
    if training:
        rng.shuffle(negs)
        negs = negs[:max(1, rho * max(1, len(gold)))]
    for si, oi in negs:
        pairs.append((si, oi)); labels.append(0)
    return pairs, labels


def run_epoch(model, data, optim, device, r2i, rho=5, bs=4, accum=2, scaler=None):
    model.train(); use_amp = scaler is not None
    order = list(range(len(data))); random.shuffle(order)
    cew = torch.ones(model.R, device=device); cew[0] = 0.1
    tot = 0.0; optim.zero_grad(); pending = 0
    nb = (len(data) + bs - 1) // bs
    for bi, k in enumerate(range(0, len(data), bs)):
        batch = [data[i] for i in order[k:k + bs]]
        with torch.autocast(device_type="cuda", enabled=use_amp):
            reps_list, offs_list = model.encoder([d["text"] for d in batch])
            rreps = model.relation_reps()
            logits_all, y_all = [], []
            for d, reps, offs in zip(batch, reps_list, offs_list):
                if not d["entities"]: continue
                pairs, labels = make_pairs(d, r2i, rho, training=True)
                if not pairs: continue
                pf = model.pair_feats(reps, offs, pairs, d["entities"])
                logits_all.append(model.score(pf, rreps))
                y_all.append(torch.tensor(labels, device=device))
            loss = None
            if logits_all:
                loss = F.cross_entropy(torch.cat(logits_all), torch.cat(y_all), weight=cew)
                if model.mode == "implicit":
                    loss = loss + 0.1 * model.ontology_loss()
        if loss is not None:
            tot += float(loss.detach())
            (scaler.scale(loss / accum) if scaler else loss / accum).backward(); pending += 1
        if ((bi + 1) % accum == 0 or bi + 1 == nb) and pending:
            if scaler: scaler.step(optim); scaler.update()
            else: optim.step()
            optim.zero_grad(); pending = 0
        if (bi & 63) == 63: free_memory(device)
    return tot


@torch.no_grad()
def evaluate(model, data, device, r2i, bs=4):
    model.eval(); tp = fp = fn = 0
    for k in range(0, len(data), bs):
        batch = data[k:k + bs]
        reps_list, offs_list = model.encoder([d["text"] for d in batch])
        rreps = model.relation_reps()
        for d, reps, offs in zip(batch, reps_list, offs_list):
            ents = d["entities"]
            gold = {(si, oi, r2i[p]) for si, oi, p in d["triples"] if p in r2i}
            allpairs = [(si, oi) for si in range(len(ents)) for oi in range(len(ents)) if si != oi]
            pred = set()
            if allpairs:
                pf = model.pair_feats(reps, offs, allpairs, ents)
                yp = model.score(pf, rreps).argmax(1).tolist()
                for (si, oi), p in zip(allpairs, yp):
                    if p != 0: pred.add((si, oi, p))
            tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"P": P, "R": R, "F1": (2 * P * R / (P + R) if P + R else 0.0), "tp": tp, "fp": fp, "fn": fn}


def train_eval_once(enc_factory, relations, attract, mutex, mode, frac, seed,
                    train, dev, device, epochs=5, lr=2e-5, bs=4, accum=2, rho=5,
                    gnn_depth=2, amp=False, onto_lr=0.0):
    set_seed(seed); free_memory(device)
    r2i = {r: i for i, r in enumerate(relations)}
    n = max(1, int(len(train) * frac)); idx = list(range(len(train))); random.shuffle(idx)
    sub = [train[i] for i in idx[:n]]
    enc = enc_factory()
    if hasattr(enc, "model"): enc = enc.to(device)
    model = OntoRE(enc, relations, attract, mutex, mode=mode, gnn_depth=gnn_depth).to(device)
    onto_lr = float(onto_lr or 0)
    if onto_lr > 0:
        onto_ps = list(model.rel_emb.parameters()) + (list(model.gnn.parameters()) if model.gnn else [])
        oids = {id(p) for p in onto_ps}
        base_ps = [p for p in model.parameters() if id(p) not in oids]
        optim = torch.optim.AdamW([{"params": base_ps, "lr": lr}, {"params": onto_ps, "lr": onto_lr}])
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if (amp and str(device) == "cuda") else None
    best = {"F1": -1}; print(f"──[RE {mode} frac={frac} seed={seed}] training {len(sub)} sentences--", flush=True)
    for ep in range(1, epochs + 1):
        loss = run_epoch(model, sub, optim, device, r2i, rho=rho, bs=bs, accum=accum, scaler=scaler)
        ev = evaluate(model, dev, device, r2i, bs=bs)
        tag = " *best" if ev["F1"] > best["F1"] else ""
        if ev["F1"] > best["F1"]: best = dict(ev, epoch=ep)
        print(f"   ep {ep}/{epochs} loss={loss:.1f} dev P={ev['P']:.3f} R={ev['R']:.3f} F1={ev['F1']:.4f}{tag}", flush=True)
    return best


def smoke():
    print("== RE smoke (StubEncoder, synthetic data) ==")
    relations = [NA, "病因", "临床表现", "实验室检查", "影像学检查"]
    mutex = [(3, 4)]; attract = [(1, 2)]
    r2i = {r: i for i, r in enumerate(relations)}
    def mk(text, trips):
        ents, ei = [], {}
        for name, ty in set([(s, "x") for s, _, _ in trips] + [(o, "x") for _, _, o in trips]):
            cs = text.find(name); ents.append((cs, cs + len(name), ty, name)); ei[name] = len(ents) - 1
        return {"text": text, "entities": ents, "triples": [(ei[s], ei[o], p) for s, p, o in trips]}
    data = [mk("感冒由病毒引起常伴发热血常规异常", [("感冒", "病因", "病毒"), ("感冒", "临床表现", "发热"),
                                                  ("感冒", "实验室检查", "血常规")]) for _ in range(12)]
    dev = data[:4]
    device = "cpu"
    for mode in ["none", "implicit", "explicit"]:
        best = train_eval_once(lambda: StubEncoder(hidden=48), relations, attract, mutex,
                               mode, 1.0, 42, data, dev, device, epochs=3, bs=4, accum=1, gnn_depth=2)
        print(f"  [{mode}] dev best F1={best['F1']:.4f}")
    print("smoke passed: all three modes train + evaluate.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mode", default="explicit", choices=["none", "implicit", "explicit"])
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--train-cap", type=int, default=800)
    ap.add_argument("--dev-cap", type=int, default=400)
    ap.add_argument("--encoder", default="hfl/chinese-macbert-base")
    ap.add_argument("--gnn-depth", type=int, default=2)
    ap.add_argument("--mutex-only", action="store_true",
                    help="keep only mutex edges, drop attractive ones; test the pure repulsion mechanism")
    ap.add_argument("--onto-lr", type=float, default=0.0,
                    help="separate learning rate for relation embeddings + SignedGAT (0 = same for all)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        smoke(); return
    device = a.device or ("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    train, dev, relations, attract, mutex, stats = load_re(caps=(a.train_cap, a.dev_cap, None))
    if a.mutex_only:
        attract = []
    print(f"device={device} | relations={len(relations)} mutex={len(mutex)} attract={len(attract)}"
          f"{' (mutex-only)' if a.mutex_only else ''} | "
          f"train={stats['train']['sentences']} dev={stats['dev']['sentences']}")
    enc_factory = lambda: HFEncoder(a.encoder)
    best = train_eval_once(enc_factory, relations, attract, mutex, a.mode, a.frac, a.seed,
                           train, dev, device, epochs=a.epochs, gnn_depth=a.gnn_depth, amp=a.amp,
                           onto_lr=a.onto_lr)
    print(f"\n[RE {a.mode} frac={a.frac} seed={a.seed}] test(dev)-F1={best['F1']:.4f}")
    if a.out:
        json.dump(best, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()

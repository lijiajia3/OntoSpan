#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# External NER baselines (biaffine / global-pointer / BERT-CRF / W2NER).
#
# Standard span/sequence NER baselines on a shared MacBERT backbone and budget;
# the dense word-pair models use negative sampling to avoid collapse.
import os, sys, json, math, time, random, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from ontospan_experiment import (HFEncoder, read_jsonl, build_types, load_ontology,
                                  gold_token_spans, set_seed, free_memory,
                                  run_epoch, evaluate)

DOMAIN_DATASET = {"tcm": "tcmner", "medical": "cmeee"}


class BiaffineNER(nn.Module):
    def __init__(self, encoder, types, lmax=10, proj=128):
        super().__init__()
        self.encoder = encoder
        self.types = types
        self.t2i = {t: i for i, t in enumerate(types)}
        self.K = len(types)
        self.lmax = lmax
        self.mode = "biaffine"
        d = encoder.hidden
        self.start = nn.Sequential(nn.Linear(d, proj), nn.GELU())
        self.end = nn.Sequential(nn.Linear(d, proj), nn.GELU())
        self.U = nn.Parameter(torch.empty(self.K, proj + 1, proj + 1))
        nn.init.xavier_uniform_(self.U)
        self._d = d

    def type_representations(self):
        return torch.zeros(1, device=self.U.device)

    def span_reps(self, reps, offs):
        T = reps.size(0)
        feats, cspans = [], []
        for i in range(T):
            for j in range(i, min(i + self.lmax, T)):
                feats.append(torch.cat([reps[i], reps[j]]))
                cspans.append((offs[i][0], offs[j][1]))
        if not feats:
            return torch.zeros(0, 2 * self._d, device=reps.device), []
        return torch.stack(feats), cspans

    def score_with(self, sr, treps):
        d = self._d
        s = self.start(sr[:, :d]); e = self.end(sr[:, d:])
        one = torch.ones(s.size(0), 1, device=s.device)
        s1 = torch.cat([s, one], 1); e1 = torch.cat([e, one], 1)
        return torch.einsum('np,kpq,nq->nk', s1, self.U, e1)


def _rope(x, pos):
    h = x.size(-1); half = h // 2
    freqs = 1.0 / (10000 ** (torch.arange(0, half, device=x.device).float() / half))
    ang = pos.float().unsqueeze(1) * freqs.unsqueeze(0)
    cos = ang.cos().unsqueeze(1); sin = ang.sin().unsqueeze(1)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class GlobalPointerNER(nn.Module):
    def __init__(self, encoder, types, lmax=10, head=64, rope=True):
        super().__init__()
        self.encoder = encoder
        self.types = types
        self.t2i = {t: i for i, t in enumerate(types)}
        self.K = len(types)
        self.lmax = lmax
        self.head = head
        self.rope = rope
        self.mode = "global_pointer"
        self.dense = nn.Linear(encoder.hidden, self.K * head * 2)

    def type_representations(self):
        return torch.zeros(1, device=self.dense.weight.device)

    def span_reps(self, reps, offs):
        T = reps.size(0)
        if T == 0:
            return torch.zeros(0, 2 * self.K * self.head, device=reps.device), []
        qk = self.dense(reps).view(T, self.K, 2, self.head)
        q, k = qk[:, :, 0, :], qk[:, :, 1, :]
        if self.rope:
            pos = torch.arange(T, device=reps.device)
            q, k = _rope(q, pos), _rope(k, pos)
        q = q.reshape(T, -1); k = k.reshape(T, -1)
        feats, cspans = [], []
        for i in range(T):
            for j in range(i, min(i + self.lmax, T)):
                feats.append(torch.cat([q[i], k[j]]))
                cspans.append((offs[i][0], offs[j][1]))
        if not feats:
            return torch.zeros(0, 2 * self.K * self.head, device=reps.device), []
        return torch.stack(feats), cspans

    def score_with(self, sr, treps):
        Kh = self.K * self.head
        q = sr[:, :Kh].view(-1, self.K, self.head)
        k = sr[:, Kh:].view(-1, self.K, self.head)
        return (q * k).sum(-1) / math.sqrt(self.head)


class BertCRF(nn.Module):
    def __init__(self, encoder, types):
        super().__init__()
        from torchcrf import CRF
        self.encoder = encoder
        self.types = types
        self.ent_types = types[1:]
        self.tags = ["O"] + [f"{p}-{t}" for t in self.ent_types for p in ("B", "I")]
        self.tag2i = {t: i for i, t in enumerate(self.tags)}
        self.num_tags = len(self.tags)
        self.cls = nn.Linear(encoder.hidden, self.num_tags)
        self.crf = CRF(self.num_tags, batch_first=True)

    def _emit(self, texts):
        reps_list, offs_list = self.encoder(texts)
        lens = [r.size(0) for r in reps_list]
        Tmax = max(lens) if lens else 1
        d = self.encoder.hidden
        dev = self.cls.weight.device
        padded = torch.zeros(len(texts), Tmax, d, device=dev)
        mask = torch.zeros(len(texts), Tmax, dtype=torch.bool, device=dev)
        for b, r in enumerate(reps_list):
            if r.size(0):
                padded[b, :r.size(0)] = r; mask[b, :r.size(0)] = True
        emissions = self.cls(padded)
        return emissions, mask, offs_list, lens

    def bio_labels(self, offs, gold):
        n = len(offs); tags = [0] * n
        for cs, ce, lab in sorted(gold, key=lambda g: g[1] - g[0]):
            bt = self.tag2i.get(f"B-{lab}"); it = self.tag2i.get(f"I-{lab}")
            if bt is None: continue
            toks = [i for i, (ts, te) in enumerate(offs) if ts >= cs and te <= ce]
            if not toks: continue
            tags[toks[0]] = bt
            for i in toks[1:]: tags[i] = it
        return tags

    def decode_spans(self, tag_ids, offs):
        spans = set(); i = 0; n = len(tag_ids)
        while i < n:
            t = self.tags[tag_ids[i]]
            if t.startswith("B-"):
                lab = t[2:]; j = i + 1
                while j < n and self.tags[tag_ids[j]] == f"I-{lab}": j += 1
                spans.add((offs[i][0], offs[j - 1][1], lab)); i = j
            else:
                i += 1
        return spans


def _crf_batch_tags(model, batch, mask, Tmax, dev):
    tags = torch.zeros(len(batch), Tmax, dtype=torch.long, device=dev)
    reps_offs = model.encoder([t for t, _ in batch])[1]
    for b, ((text, gold), offs) in enumerate(zip(batch, reps_offs)):
        lab = model.bio_labels(offs, gold)
        for i, v in enumerate(lab): tags[b, i] = v
    return tags


def train_crf(model, data, optim, device, bs=4, accum=1):
    model.train(); accum = max(1, int(accum)); tot = 0.0
    order = list(range(len(data))); random.shuffle(order)
    n_batches = (len(data) + bs - 1) // bs
    optim.zero_grad(); pending = 0
    for bi, k in enumerate(range(0, len(data), bs)):
        batch = [data[i] for i in order[k:k + bs]]
        emissions, mask, offs_list, lens = model._emit([t for t, _ in batch])
        if emissions.size(0) == 0 or not mask.any(): continue
        Tmax = emissions.size(1)
        tags = torch.zeros(len(batch), Tmax, dtype=torch.long, device=device)
        for b, ((text, gold), offs) in enumerate(zip(batch, offs_list)):
            for i, v in enumerate(model.bio_labels(offs, gold)): tags[b, i] = v
        mask[:, 0] = True
        loss = -model.crf(emissions, tags, mask=mask, reduction="mean")
        tot += float(loss.detach())
        (loss / accum).backward(); pending += 1
        if ((bi + 1) % accum == 0 or bi + 1 == n_batches) and pending > 0:
            optim.step(); optim.zero_grad(); pending = 0
        if (bi & 63) == 63: free_memory(device)
    return tot


@torch.no_grad()
def eval_crf(model, data, device, bs=4):
    model.eval(); tp = fp = fn = 0; per = {}
    for k in range(0, len(data), bs):
        batch = data[k:k + bs]
        emissions, mask, offs_list, lens = model._emit([t for t, _ in batch])
        if emissions.size(0) == 0: continue
        mask[:, 0] = True
        paths = model.crf.decode(emissions, mask=mask)
        for (text, gold), offs, path in zip(batch, offs_list, paths):
            gold_set = {(cs, ce, lab) for cs, ce, lab in gold}
            pred_set = model.decode_spans(path[:len(offs)], offs)
            for key in pred_set:
                per.setdefault(key[2], [0, 0, 0])
                if key in gold_set: tp += 1; per[key[2]][0] += 1
                else: fp += 1; per[key[2]][1] += 1
            for key in gold_set:
                per.setdefault(key[2], [0, 0, 0])
                if key not in pred_set: fn += 1; per[key[2]][2] += 1
    def prf(t, p_, n_):
        p = t / (t + p_) if t + p_ else 0.0; r = t / (t + n_) if t + n_ else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)
    P, R, Fm = prf(tp, fp, fn)
    return {"micro": (P, R, Fm), "per_type": {l: prf(*v) for l, v in sorted(per.items())}}


_DIST_BOUNDS = torch.tensor([1, 2, 3, 4, 6, 10, 18, 34])


class W2NER(nn.Module):
    def __init__(self, encoder, types, conv_dim=64, dilations=(1, 2, 3)):
        super().__init__()
        self.encoder = encoder
        self.types = types
        self.ent_types = types[1:]
        self.num_labels = 2 + len(self.ent_types)
        self.mode = "w2ner"
        d = encoder.hidden; c = conv_dim
        self.head_mlp = nn.Linear(d, c)
        self.tail_mlp = nn.Linear(d, c)
        self.dist_emb = nn.Embedding(len(_DIST_BOUNDS) + 1, c)
        self.region_emb = nn.Embedding(3, c)
        self.grid_proj = nn.Linear(4 * c, c)
        self.convs = nn.ModuleList([nn.Conv2d(c, c, 3, padding=dl, dilation=dl) for dl in dilations])
        self.U = nn.Parameter(torch.empty(self.num_labels, c + 1, c + 1))
        nn.init.xavier_uniform_(self.U)
        self.cls = nn.Linear(c * len(dilations), self.num_labels)
        self._c = c

    def _grid_logits(self, H):
        T = H.size(0); c = self._c; dev = H.device
        head = self.head_mlp(H); tail = self.tail_mlp(H)
        idx = torch.arange(T, device=dev)
        delta = idx.unsqueeze(0) - idx.unsqueeze(1)
        dist_idx = torch.bucketize(delta.abs(), _DIST_BOUNDS.to(dev))
        region_idx = (delta > 0).long() * 2 + (delta == 0).long()
        hi = head.unsqueeze(1).expand(T, T, c)
        tj = tail.unsqueeze(0).expand(T, T, c)
        g = torch.cat([hi, tj, self.dist_emb(dist_idx), self.region_emb(region_idx)], -1)
        g = self.grid_proj(g).permute(2, 0, 1).unsqueeze(0)
        x = torch.cat([F.gelu(conv(g)) for conv in self.convs], 1).squeeze(0).permute(1, 2, 0)
        one = torch.ones(T, 1, device=dev)
        h1 = torch.cat([head, one], 1); t1 = torch.cat([tail, one], 1)
        biaff = torch.einsum('ip,lpq,jq->ijl', h1, self.U, t1)
        return self.cls(x) + biaff

    def gold_grid(self, offs, gold, dev):
        T = len(offs); Y = torch.zeros(T, T, dtype=torch.long, device=dev)
        for cs, ce, lab in gold:
            if lab not in self.ent_types:
                continue
            toks = [i for i, (ts, te) in enumerate(offs) if ts >= cs and te <= ce]
            if not toks:
                continue
            h, t = toks[0], toks[-1]
            Y[t, h] = 2 + self.ent_types.index(lab)
            for k in range(h, t):
                Y[k, k + 1] = 1
        return Y

    def decode_spans(self, grid_lab, offs):
        spans = set(); T = len(offs)
        for i in range(T):
            for j in range(i + 1):
                lab = int(grid_lab[i, j])
                if lab >= 2:
                    spans.add((offs[j][0], offs[i][1], self.ent_types[lab - 2]))
        return spans


def train_w2ner(model, data, optim, device, bs=4, accum=1, rho=5):
    model.train(); accum = max(1, int(accum)); tot = 0.0
    order = list(range(len(data))); random.shuffle(order)
    n_batches = (len(data) + bs - 1) // bs
    optim.zero_grad(); pending = 0
    for bi, k in enumerate(range(0, len(data), bs)):
        batch = [data[i] for i in order[k:k + bs]]
        reps_list, offs_list = model.encoder([t for t, _ in batch])
        loss = 0.0; nb = 0
        for (text, gold), reps, offs in zip(batch, reps_list, offs_list):
            if reps.size(0) == 0:
                continue
            logits = model._grid_logits(reps).reshape(-1, model.num_labels)
            Y = model.gold_grid(offs, gold, reps.device).reshape(-1)
            pos = (Y > 0).nonzero(as_tuple=True)[0]
            neg = (Y == 0).nonzero(as_tuple=True)[0]
            n_neg = min(neg.numel(), max(20, rho * pos.numel()))
            if neg.numel() > n_neg:
                neg = neg[torch.randperm(neg.numel(), device=Y.device)[:n_neg]]
            sel = torch.cat([pos, neg]) if pos.numel() else neg
            loss = loss + F.cross_entropy(logits[sel], Y[sel])
            nb += 1
        if nb == 0:
            continue
        loss = loss / nb; tot += float(loss.detach())
        (loss / accum).backward(); pending += 1
        if ((bi + 1) % accum == 0 or bi + 1 == n_batches) and pending > 0:
            optim.step(); optim.zero_grad(); pending = 0
        if (bi & 63) == 63:
            free_memory(device)
    return tot


@torch.no_grad()
def eval_w2ner(model, data, device, bs=4):
    model.eval(); tp = fp = fn = 0; per = {}
    for k in range(0, len(data), bs):
        batch = data[k:k + bs]
        reps_list, offs_list = model.encoder([t for t, _ in batch])
        for (text, gold), reps, offs in zip(batch, reps_list, offs_list):
            pred_set = (model.decode_spans(model._grid_logits(reps).argmax(-1), offs)
                        if reps.size(0) else set())
            gold_set = {(cs, ce, lab) for cs, ce, lab in gold}
            for key in pred_set:
                per.setdefault(key[2], [0, 0, 0])
                if key in gold_set: tp += 1; per[key[2]][0] += 1
                else: fp += 1; per[key[2]][1] += 1
            for key in gold_set:
                per.setdefault(key[2], [0, 0, 0])
                if key not in pred_set: fn += 1; per[key[2]][2] += 1
    def prf(t, p_, n_):
        p = t / (t + p_) if t + p_ else 0.0; r = t / (t + n_) if t + n_ else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)
    P, R, Fm = prf(tp, fp, fn)
    return {"micro": (P, R, Fm), "per_type": {l: prf(*v) for l, v in sorted(per.items())}}


def build_model(method, encoder, types, lmax):
    if method == "biaffine":       return BiaffineNER(encoder, types, lmax=lmax)
    if method == "global_pointer": return GlobalPointerNER(encoder, types, lmax=lmax)
    if method == "bert_crf":       return BertCRF(encoder, types)
    if method == "w2ner":          return W2NER(encoder, types)
    raise ValueError(f"unknown baseline: {method}")


def train_baseline_once(method, enc_factory, types, frac, seed, train, dev_d, test,
                        a, device, verbose=False):
    set_seed(seed); free_memory(device)
    n = max(1, int(len(train) * frac))
    idx = list(range(len(train))); random.shuffle(idx)
    sub = [train[i] for i in idx[:n]]
    enc = enc_factory()
    if hasattr(enc, "model"): enc = enc.to(device)
    model = build_model(method, enc, types, a.lmax).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=a.lr)
    use_amp = bool(getattr(a, "amp", False)) and str(device) == "cuda" and method not in ("bert_crf", "w2ner")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    accum = max(1, int(getattr(a, "accum", 1)))
    print(f"\n──[{method} {DOMAIN_DATASET.get(a.domain,a.domain)} frac={frac} seed={seed}] "
          f"training {len(sub)} sentences  bs={a.batch_size} accum={accum}"
          + ("  [AMP fp16]" if use_amp else ""), flush=True)
    best_f1, best_ep = -1.0, 0; best_state = None
    for ep in range(1, a.epochs + 1):
        if method == "bert_crf":
            tl = train_crf(model, sub, optim, device, bs=a.batch_size, accum=accum)
            dev_f1 = eval_crf(model, dev_d, device, bs=a.batch_size)["micro"][2]
        elif method == "w2ner":
            tl = train_w2ner(model, sub, optim, device, bs=a.batch_size, accum=accum, rho=a.rho)
            dev_f1 = eval_w2ner(model, dev_d, device, bs=a.batch_size)["micro"][2]
        else:
            tl = run_epoch(model, sub, optim, device, rho=a.rho, lam=0.0, gamma=0.0,
                           bs=a.batch_size, scaler=scaler, accum=accum)
            dev_f1 = evaluate(model, dev_d, device, bs=a.batch_size, use_amp=use_amp)["micro"][2]
        star = ""
        if dev_f1 >= best_f1:
            best_f1, best_ep = dev_f1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = "  *best"
        print(f"   epoch {ep}/{a.epochs}  loss={tl:.2f}  dev-F1={dev_f1:.4f}{star}", flush=True)
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    if method == "bert_crf":
        tev = eval_crf(model, test, device, bs=a.batch_size)
    elif method == "w2ner":
        tev = eval_w2ner(model, test, device, bs=a.batch_size)
    else:
        tev = evaluate(model, test, device, bs=a.batch_size, use_amp=use_amp)
    tP, tR, tF = tev["micro"]
    print(f"   └─ test@best-dev(ep{best_ep}): P={tP:.4f} R={tR:.4f} F1={tF:.4f}", flush=True)
    return {"method": method, "P": tP, "R": tR, "F1": tF, "dev_best_F1": best_f1,
            "best_epoch": best_ep, "per_type": {l: list(v) for l, v in tev["per_type"].items()}}


def _load_domain(domain, caps):
    ddir = os.path.join(_ROOT, "data", domain)
    paths = {sp: os.path.join(ddir, f"{sp}.jsonl") for sp in ("train", "dev", "test")}

    def sub(data, cap, seed=42):
        if not cap or cap <= 0 or len(data) <= cap: return data
        r = random.Random(seed); return [data[i] for i in sorted(r.sample(range(len(data)), cap))]
    tr = sub(read_jsonl(paths["train"]), caps[0])
    dv = sub(read_jsonl(paths["dev"]), caps[1])
    te = sub(read_jsonl(paths["test"]), caps[2])
    onto = load_ontology(domain)
    types = build_types(tr + dv, onto)
    return tr, dv, te, types


def main():
    ap = argparse.ArgumentParser(description="OntoSpan baseline (biaffine / global_pointer / bert_crf / w2ner)")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--method", required=True, choices=["biaffine", "global_pointer", "bert_crf", "w2ner"])
    ap.add_argument("--domain", required=True); ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out")
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lmax", type=int, default=10); ap.add_argument("--rho", type=int, default=5)
    ap.add_argument("--train-cap", type=int, default=600); ap.add_argument("--dev-cap", type=int, default=100)
    ap.add_argument("--test-cap", type=int, default=300)
    ap.add_argument("--encoder", default="hfl/chinese-macbert-base"); ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action="store_true")
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    tr, dv, te, types = _load_domain(a.domain, (a.train_cap, a.dev_cap, a.test_cap))
    r = train_baseline_once(a.method, lambda: HFEncoder(a.encoder), types, a.frac, a.seed,
                            tr, dv, te, a, dev, verbose=True)
    if a.out:
        json.dump(r, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()

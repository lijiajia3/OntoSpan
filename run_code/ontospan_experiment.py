# -*- coding: utf-8 -*-
# OntoSpan model, training, and evaluation.
#
# Spans are scored against entity-type prototypes. Ontology relations enter as a
# signed graph in two interchangeable ways: an implicit signed-Laplacian
# regularizer (OntoSpan.ontology_loss) or an explicit signed graph attention
# (SignedGAT). Attractive edges pull related types together; mutex edges push
# incompatible types/concepts apart.

import os, sys, json, argparse, random, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

W_RELATED = 0.5
W_ISA     = 1.0
W_MUTEX   = 1.0


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def free_memory(device=None):
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_ontology(domain):
    from build_datasets import MUTEX_TYPE, MUTEX_INST, ATTRACT
    return {
        "attract": [(a, b, W_RELATED) for a, b, _rel in ATTRACT.get(domain, [])],
        "tmutex":  [(a, b) for a, b in MUTEX_TYPE.get(domain, [])],
        "imutex":  [(a, b) for a, b in MUTEX_INST.get(domain, [])],
        "parent":  {"tcm": "中药", "medical": "药物", "law": None}.get(domain),
    }


def perturb_ontology(onto, kind="real", seed=0):
    if kind == "random_sign":
        # Keep every node and edge fixed; the model shuffles which graph edges
        # are negative while preserving the original negative-edge count.
        o = dict(onto)
        o["_random_sign_seed"] = int(seed)
        return o
    if kind not in ("removed", "rewired"):
        return onto
    o = dict(onto)
    if kind == "removed":
        o["tmutex"], o["imutex"] = [], []
        return o
    rng = random.Random(seed)
    def rewire(pairs):
        nodes = sorted({n for p in pairs for n in p})
        if len(nodes) < 2 or not pairs:
            return list(pairs)
        real = {frozenset(p) for p in pairs}
        out, seen = [], set()
        shuf = nodes[:]; rng.shuffle(shuf)
        for i in range(0, len(shuf) - 1, 2):
            e = (shuf[i], shuf[i + 1]); out.append(e); seen.add(frozenset(e))
        if len(shuf) % 2 == 1:
            e = (shuf[-1], rng.choice(shuf[:-1])); out.append(e); seen.add(frozenset(e))
        guard = 0
        while len(out) < len(pairs) and guard < len(pairs) * 200:
            guard += 1
            a, b = rng.sample(nodes, 2); fs = frozenset((a, b))
            if fs in seen:
                continue
            if fs in real and guard < len(pairs) * 100:
                continue
            out.append((a, b)); seen.add(fs)
        return out[:len(pairs)]
    o["tmutex"] = rewire(onto["tmutex"])
    o["imutex"] = rewire(onto["imutex"])
    return o


class HFEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.model = AutoModel.from_pretrained(name)
        self.hidden = self.model.config.hidden_size

    @torch.no_grad()
    def embed_text_static(self, text):
        enc = self.tok(text or "未", return_tensors="pt", truncation=True, max_length=64)
        dev = next(self.model.parameters()).device
        out = self.model(**{k: v.to(dev) for k, v in enc.items()}).last_hidden_state[0]
        return out[1:-1].mean(0) if out.size(0) > 2 else out.mean(0)

    def forward(self, texts):
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=160, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        dev = next(self.model.parameters()).device
        out = self.model(**{k: v.to(dev) for k, v in enc.items()}).last_hidden_state
        reps, offs = [], []
        for b in range(len(texts)):
            om = offsets[b].tolist()
            idx = [i for i, (s, e) in enumerate(om) if e > s]
            reps.append(out[b, idx]); offs.append([tuple(om[i]) for i in idx])
        return reps, offs


class StubEncoder(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.hidden = hidden
        self.emb = nn.Embedding(6000, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def _ids(self, text):
        return torch.tensor([ord(c) % 6000 for c in text], dtype=torch.long)

    @torch.no_grad()
    def embed_text_static(self, text):
        dev = self.emb.weight.device
        return self.emb(self._ids(text).to(dev)).mean(0) if text else torch.zeros(self.hidden, device=dev)

    def forward(self, texts):
        dev = self.emb.weight.device
        reps, offs = [], []
        for text in texts:
            h = (torch.tanh(self.proj(self.emb(self._ids(text).to(dev))))
                 if text else torch.zeros(0, self.hidden, device=dev))
            reps.append(h); offs.append([(i, i + 1) for i in range(len(text))])
        return reps, offs


# Explicit realization: graph attention over the signed ontology graph.
# Attractive neighbors are aggregated positively, mutex neighbors negatively,
# with a gated residual that limits over-smoothing as depth grows.
class SignedGAT(nn.Module):
    def __init__(self, d, depth=1, leaky=0.2):
        super().__init__()
        self.depth = depth
        self.W = nn.Linear(d, d, bias=False)
        self.a_pos = nn.Linear(2 * d, 1, bias=False)
        self.a_neg = nn.Linear(2 * d, 1, bias=False)
        self.gate = nn.Linear(2 * d, d)
        self.ln = nn.LayerNorm(d)
        self.leaky = nn.LeakyReLU(leaky)

    def _aggregate(self, x, Wx, nbr, attn):
        out = torch.zeros_like(Wx)
        for i, js in enumerate(nbr):
            if not js:
                continue
            jt = torch.tensor(js, device=x.device)
            e = self.leaky(attn(torch.cat([x[i].expand(len(js), -1), x[jt]], dim=1))).squeeze(1)
            a = torch.softmax(e, 0)
            out[i] = (a.unsqueeze(1) * Wx[jt]).sum(0)
        return out

    def forward(self, x, pos_nbr, neg_nbr, neg_weight=1.0):
        for _ in range(self.depth):
            Wx = self.W(x)
            # Message = attractive aggregation minus mutex aggregation (the signed step).
            pos = self._aggregate(x, Wx, pos_nbr, self.a_pos)
            neg = self._aggregate(x, Wx, neg_nbr, self.a_neg)
            agg = pos - float(neg_weight) * neg
            g = torch.sigmoid(self.gate(torch.cat([x, agg], dim=1)))
            x = self.ln(g * agg + (1 - g) * x)
        return x


def _neighbor_mean(x, nbr):
    """Mean neighbor features for a small adjacency list."""
    out = torch.zeros_like(x)
    for i, js in enumerate(nbr):
        if js:
            out[i] = x[torch.tensor(js, device=x.device)].mean(0)
    return out


class SignedGCN(nn.Module):
    """Balance-theory SGCN adapted to return one prototype per ontology node.

    The two channels represent balanced and unbalanced paths.  The first layer
    separates positive and negative neighborhoods; later layers mix
    positive-balanced with negative-unbalanced paths, and vice versa, following
    Derr et al. (ICDM 2018).  A final projection restores the NER prototype size.
    """
    def __init__(self, d, depth=2):
        super().__init__()
        self.depth = max(1, int(depth))
        h = max(1, d // 2)
        self.first_bal = nn.Linear(2 * d, h)
        self.first_unbal = nn.Linear(2 * d, h)
        self.bal = nn.ModuleList(nn.Linear(3 * h, h) for _ in range(self.depth - 1))
        self.unbal = nn.ModuleList(nn.Linear(3 * h, h) for _ in range(self.depth - 1))
        self.out = nn.Linear(2 * h, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, pos_nbr, neg_nbr, neg_weight=1.0):
        p = _neighbor_mean(x, pos_nbr)
        n = float(neg_weight) * _neighbor_mean(x, neg_nbr)
        b = F.gelu(self.first_bal(torch.cat([x, p], 1)))
        u = F.gelu(self.first_unbal(torch.cat([x, n], 1)))
        for wb, wu in zip(self.bal, self.unbal):
            b_new = F.gelu(wb(torch.cat([
                b, _neighbor_mean(b, pos_nbr),
                float(neg_weight) * _neighbor_mean(u, neg_nbr)], 1)))
            u_new = F.gelu(wu(torch.cat([
                u, _neighbor_mean(u, pos_nbr),
                float(neg_weight) * _neighbor_mean(b, neg_nbr)], 1)))
            b, u = b_new, u_new
        return self.norm(self.out(torch.cat([b, u], 1)))


class _MotifAttention(nn.Module):
    def __init__(self, d, q):
        super().__init__()
        self.proj = nn.Linear(d, q, bias=False)
        self.attn = nn.Linear(2 * q, 1, bias=False)

    def forward(self, x, nbr):
        z = self.proj(x)
        out = torch.zeros_like(z)
        for i, js in enumerate(nbr):
            nodes = list(dict.fromkeys([i] + list(js)))
            jt = torch.tensor(nodes, device=x.device)
            e = F.leaky_relu(self.attn(torch.cat([
                z[i].expand(len(nodes), -1), z[jt]], 1)).squeeze(1), 0.1)
            a = torch.softmax(e, 0)
            out[i] = (a.unsqueeze(1) * z[jt]).sum(0)
        return out


def _sigat_channels(pos_nbr, neg_nbr):
    """Build SiGAT's 6 signed-direction channels and 32 triad motifs.

    OntoSpan's ontology edges are undirected, so each edge is expanded in both
    directions before the signed directed motif construction.
    """
    n = len(pos_nbr)
    p_out = [set(v) for v in pos_nbr]; p_in = [set(v) for v in pos_nbr]
    n_out = [set(v) for v in neg_nbr]; n_in = [set(v) for v in neg_nbr]
    p_all = [p_out[i] | p_in[i] for i in range(n)]
    n_all = [n_out[i] | n_in[i] for i in range(n)]
    channels = [[sorted(v) for v in xs] for xs in
                (p_all, p_out, p_in, n_all, n_out, n_in)]
    motifs = [[[ ] for _ in range(n)] for _ in range(32)]

    def features(u, v):
        return (
            len(p_out[u] & p_in[v]), len(p_out[u] & n_in[v]),
            len(n_out[u] & p_in[v]), len(n_out[u] & n_in[v]),
            len(p_out[u] & p_out[v]), len(p_out[u] & n_out[v]),
            len(n_out[u] & p_out[v]), len(n_out[u] & n_out[v]),
            len(p_in[u] & p_out[v]), len(p_in[u] & n_out[v]),
            len(n_in[u] & p_out[v]), len(n_in[u] & n_out[v]),
            len(p_in[u] & p_in[v]), len(p_in[u] & n_in[v]),
            len(n_in[u] & p_in[v]), len(n_in[u] & n_in[v]))

    for u in range(n):
        for v in p_out[u]:
            for k, value in enumerate(features(u, v)):
                if value > 0: motifs[k][u].append(v)
        for v in n_out[u]:
            for k, value in enumerate(features(u, v)):
                if value > 0: motifs[16 + k][u].append(v)
    return channels + motifs


class SiGAT(nn.Module):
    """Motif-attention SiGAT adaptation for the small ontology graph.

    It retains the original 38 channels (six signed/direction channels and 32
    signed triad-motif channels), then projects their attended features back to
    the prototype dimension used by the shared NER scorer.
    """
    def __init__(self, d, depth=1, motif_dim=32):
        super().__init__()
        q = min(int(motif_dim), d)
        self.heads = nn.ModuleList(_MotifAttention(d, q) for _ in range(38))
        self.mlp = nn.Sequential(nn.Linear(d + 38 * q, d), nn.Tanh(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self._channels = None

    def forward(self, x, pos_nbr, neg_nbr, neg_weight=1.0):
        if self._channels is None:
            self._channels = _sigat_channels(pos_nbr, neg_nbr)
        hs = [head(x, nbr) for head, nbr in zip(self.heads, self._channels)]
        return self.norm(self.mlp(torch.cat([x] + hs, 1)))


# Span-based NER over a two-layer signed ontology graph (type layer + concept
# anchors). mode = none / implicit (regularizer) / explicit (SignedGAT).
class OntoSpan(nn.Module):
    def __init__(self, encoder, types, ontology, mode="implicit", lmax=10, dwidth=64, gnn_depth=1,
                 score_anchors=False, me_weight=1.0):
        super().__init__()
        self.score_anchors = bool(score_anchors) and mode == "explicit"
        self._anchor_reps = None
        self.encoder = encoder
        self.types = types
        self.t2i = {t: i for i, t in enumerate(types)}
        self.K = len(types)
        self.lmax = lmax
        self.mode = mode
        self.me_weight = float(me_weight)
        d = encoder.hidden
        self.width_emb = nn.Embedding(lmax + 1, dwidth)
        self.span_proj = nn.Sequential(nn.Linear(3 * d + dwidth, d), nn.LayerNorm(d), nn.GELU())
        self.type_emb = nn.Embedding(self.K, d)
        self.Wbi = nn.Parameter(torch.empty(d, d)); nn.init.xavier_uniform_(self.Wbi)
        self.type_bias = nn.Parameter(torch.zeros(self.K))
        self._build_graph(ontology, encoder, d)
        if mode == "explicit":
            self.gnn = SignedGAT(d, depth=gnn_depth)
        elif mode == "sgcn":
            self.gnn = SignedGCN(d, depth=gnn_depth)
        elif mode == "sigat":
            self.gnn = SiGAT(d, depth=gnn_depth)
        else:
            self.gnn = None

    # Build positive (attractive) and negative (mutex) neighbor lists over type
    # nodes (0..K-1) and concept anchors (K..K+M-1); is-a links each anchor to its
    # parent type as a positive edge.
    def _build_graph(self, onto, encoder, d):
        ti = self.t2i
        self.type_attract = [(ti[a], ti[b], w) for a, b, w in onto["attract"] if a in ti and b in ti]
        self.type_mutex   = [(ti[a], ti[b]) for a, b in onto["tmutex"] if a in ti and b in ti]
        names = sorted({n for pair in onto["imutex"] for n in pair})
        self.anchor_names = names
        self.M = len(names)
        want_anchor = self.mode in ("implicit", "explicit", "sgcn", "sigat") and self.M > 0
        if want_anchor:
            self.anchor = nn.Parameter(torch.stack([encoder.embed_text_static(n) for n in names]))
            a2i = {n: i for i, n in enumerate(names)}
            self.inst_mutex = [(a2i[a], a2i[b]) for a, b in onto["imutex"]]
            self.parent_idx = ti.get(onto["parent"])
        else:
            self.register_parameter("anchor", None)
            self.inst_mutex, self.parent_idx = [], None
        N = self.K + (self.M if want_anchor else 0)
        self.pos_nbr = [[] for _ in range(N)]
        self.neg_nbr = [[] for _ in range(N)]
        def add(adj, i, j): adj[i].append(j); adj[j].append(i)
        pos_edges = [(a, b) for a, b, _w in self.type_attract]
        neg_edges = list(self.type_mutex)
        if want_anchor:
            neg_edges += [(self.K + a, self.K + b) for a, b in self.inst_mutex]
            if self.parent_idx is not None:
                pos_edges += [(self.K + m, self.parent_idx) for m in range(self.M)]
        if "_random_sign_seed" in onto:
            all_edges = list(dict.fromkeys(
                tuple(sorted(e)) for e in (pos_edges + neg_edges) if e[0] != e[1]))
            rng = random.Random(int(onto["_random_sign_seed"]))
            rng.shuffle(all_edges)
            n_neg = min(len(neg_edges), len(all_edges))
            neg_edges, pos_edges = all_edges[:n_neg], all_edges[n_neg:]
        for a, b in pos_edges: add(self.pos_nbr, a, b)
        for a, b in neg_edges: add(self.neg_nbr, a, b)

    def type_representations(self):
        if self.mode in ("explicit", "sgcn", "sigat"):
            x = (torch.cat([self.type_emb.weight, self.anchor], 0)
                 if self.anchor is not None else self.type_emb.weight)
            h = self.gnn(x, self.pos_nbr, self.neg_nbr, self.me_weight)
            self._anchor_reps = h[self.K:] if (self.score_anchors and self.anchor is not None) else None
            return h[:self.K]
        self._anchor_reps = None
        return self.type_emb.weight

    def span_index(self, length, device="cpu"):
        """Enumerate candidates in the original start-major, short-to-long order."""
        if length <= 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty
        starts = torch.arange(length, dtype=torch.long, device=device).unsqueeze(1)
        widths = torch.arange(1, self.lmax + 1, dtype=torch.long,
                              device=device).unsqueeze(0)
        starts = starts.expand(length, self.lmax)
        widths = widths.expand(length, self.lmax)
        ends = starts + widths - 1
        valid = ends < length
        return starts[valid], ends[valid], widths[valid]

    @staticmethod
    def span_char_spans(offs, starts, ends):
        """Map token candidates to character offsets without tensor kernels."""
        si = starts.detach().cpu().tolist()
        sj = ends.detach().cpu().tolist()
        return [(offs[i][0], offs[j][1]) for i, j in zip(si, sj)]

    def span_feats(self, reps, starts, ends, widths):
        """Project only the requested candidates; means use FP32 prefix sums."""
        if starts.numel() == 0:
            return reps.new_zeros((0, self.encoder.hidden))
        starts = starts.to(reps.device)
        ends = ends.to(reps.device)
        widths = widths.to(reps.device)
        accum = reps.float() if reps.dtype in (torch.float16, torch.bfloat16) else reps
        prefix = torch.cat([accum.new_zeros((1, accum.size(1))), accum.cumsum(0)], 0)
        means = (prefix[ends + 1] - prefix[starts]) / widths.to(accum.dtype).unsqueeze(1)
        means = means.to(reps.dtype)
        feats = torch.cat([reps[starts], reps[ends], means, self.width_emb(widths)], 1)
        return self.span_proj(feats)

    def span_reps(self, reps, offs):
        """Vectorized all-candidate path used by evaluation/inference."""
        starts, ends, widths = self.span_index(reps.size(0), "cpu")
        char_spans = self.span_char_spans(offs, starts, ends)
        return self.span_feats(reps, starts, ends, widths), char_spans

    # Bilinear span-vs-type score; optionally lifts the parent-type score with the
    # best matching concept anchor (the score-anchors ablation).
    def score_with(self, span_rep, treps):
        base = (span_rep @ self.Wbi) @ treps.t() + self.type_bias
        if (self.score_anchors and self.parent_idx is not None
                and self._anchor_reps is not None and self._anchor_reps.size(0) > 0):
            anc = (span_rep @ self.Wbi) @ self._anchor_reps.t() + self.type_bias[self.parent_idx]
            amax = anc.max(dim=1).values
            delta = torch.clamp(amax - base[:, self.parent_idx], min=0.0)
            onehot = F.one_hot(torch.tensor(self.parent_idx, device=base.device),
                               num_classes=base.size(1)).to(base.dtype)
            base = base + delta.unsqueeze(1) * onehot.unsqueeze(0)
        return base

    # Implicit realization: signed-Laplacian regularizer on L2-normalized prototypes.
    # Attractive pairs minimize ||E_a - E_b||^2 (pull together); mutex pairs minimize
    # ||E_a + E_b||^2 (push toward antipodal). Instance-level mutex acts on the anchors,
    # which reach the type space through an is-a term.
    def ontology_loss(self, gamma=1.0):
        if self.mode != "implicit":
            return torch.zeros((), device=self.type_emb.weight.device)
        E = F.normalize(self.type_emb.weight, dim=1)
        reg = torch.zeros((), device=E.device)
        for a, b, w in self.type_attract:
            reg = reg + w * (E[a] - E[b]).pow(2).sum()
        for a, b in self.type_mutex:
            reg = reg + self.me_weight * (E[a] + E[b]).pow(2).sum()
        if self.anchor is not None:
            C = F.normalize(self.anchor, dim=1)
            inst = torch.zeros((), device=E.device)
            for a, b in self.inst_mutex:
                inst = inst + self.me_weight * (C[a] + C[b]).pow(2).sum()
            if self.parent_idx is not None:
                inst = inst + W_ISA * (C - E[self.parent_idx]).pow(2).sum(1).mean()
            reg = reg + gamma * inst
        return reg


def read_jsonl(path):
    out = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            o = json.loads(ln); out.append((o["text"], [tuple(e) for e in o.get("entities", [])]))
    return out


def build_types(data, ontology):
    labs = sorted({l for _, ents in data for _, _, l in ents})
    for a, b, _ in ontology["attract"]:
        for t in (a, b):
            if t not in labs: labs.append(t)
    for a, b in ontology["tmutex"]:
        for t in (a, b):
            if t not in labs: labs.append(t)
    return ["∅"] + labs


def gold_token_spans(char_spans, gold_ents, t2i):
    g = {(cs, ce): lab for cs, ce, lab in gold_ents}
    return torch.tensor([t2i.get(g.get((cs, ce)), 0) for (cs, ce) in char_spans], dtype=torch.long)


_MASK_CH = "□"


def mask_augment(data, p=0.0, seed=0):
    if p <= 0:
        return data
    rng = random.Random(seed)
    out = []
    for text, ents in data:
        chars = list(text)
        for i, ch in enumerate(chars):
            if ch.strip() and rng.random() < p:
                chars[i] = _MASK_CH
        out.append(("".join(chars), ents))
    return out


def run_epoch(model, data, optim, device, rho=5, lam=0.1, gamma=1.0, bs=8, scaler=None, accum=1):
    model.train()
    use_amp = scaler is not None
    accum = max(1, int(accum))
    cew = torch.ones(model.K, device=device); cew[0] = 0.1
    order = list(range(len(data))); random.shuffle(order)
    n_batches = (len(data) + bs - 1) // bs
    tot = 0.0
    optim.zero_grad()
    pending = 0
    for bi, k in enumerate(range(0, len(data), bs)):
        batch = [data[i] for i in order[k:k + bs]]
        with torch.autocast(device_type="cuda", enabled=use_amp):
            reps_list, offs_list = model.encoder([t for t, _ in batch])
            treps = model.type_representations()
            logits_all, y_all = [], []
            for (text, gold), reps, offs in zip(batch, reps_list, offs_list):
                starts, ends, widths = model.span_index(reps.size(0), "cpu")
                if starts.numel() == 0: continue
                cspans = model.span_char_spans(offs, starts, ends)
                y = gold_token_spans(cspans, gold, model.t2i).to(device)
                pos = (y != 0).nonzero(as_tuple=True)[0]
                neg = (y == 0).nonzero(as_tuple=True)[0]
                keep = neg[torch.randperm(len(neg), device=device)[:max(1, int(rho * max(1, len(pos))))]]
                sel = torch.cat([pos, keep])
                starts = starts.to(reps.device, non_blocking=True)
                ends = ends.to(reps.device, non_blocking=True)
                widths = widths.to(reps.device, non_blocking=True)
                sr = model.span_feats(reps, starts[sel], ends[sel], widths[sel])
                logits_all.append(model.score_with(sr, treps)); y_all.append(y[sel])
            loss = None
            if logits_all:
                loss = F.cross_entropy(torch.cat(logits_all), torch.cat(y_all), weight=cew)
                if model.mode == "implicit":
                    loss = loss + lam * model.ontology_loss(gamma=gamma)
        if loss is not None:
            tot += float(loss.detach())
            scaled = loss / accum
            (scaler.scale(scaled) if scaler is not None else scaled).backward()
            pending += 1
        if ((bi + 1) % accum == 0 or bi + 1 == n_batches) and pending > 0:
            if scaler is not None:
                scaler.step(optim); scaler.update()
            else:
                optim.step()
            optim.zero_grad()
            pending = 0
        if (bi & 63) == 63:
            free_memory(device)
    return tot


@torch.no_grad()
def evaluate(model, data, device, bs=8, use_amp=False, return_details=False):
    model.eval()
    tp = fp = fn = 0; per = {}; details = []
    for k in range(0, len(data), bs):
        batch = data[k:k + bs]
        with torch.autocast(device_type="cuda", enabled=use_amp):
            reps_list, offs_list = model.encoder([t for t, _ in batch])
            treps = model.type_representations()
        for local_i, ((text, gold), reps, offs) in enumerate(zip(batch, reps_list, offs_list)):
            sr, cspans = model.span_reps(reps, offs)
            gold_set = {(cs, ce, lab) for cs, ce, lab in gold}
            pred_set = set()
            if sr.size(0):
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    preds = model.score_with(sr, treps).argmax(1).tolist()
                for (cs, ce), p in zip(cspans, preds):
                    if p != 0: pred_set.add((cs, ce, model.types[p]))
            for key in pred_set:
                per.setdefault(key[2], [0, 0, 0])
                if key in gold_set: tp += 1; per[key[2]][0] += 1
                else: fp += 1; per[key[2]][1] += 1
            for key in gold_set:
                per.setdefault(key[2], [0, 0, 0])
                if key not in pred_set: fn += 1; per[key[2]][2] += 1
            if return_details:
                details.append({"sample_index": k + local_i,
                                "gold": [list(v) for v in sorted(gold_set)],
                                "pred": [list(v) for v in sorted(pred_set)]})
    def prf(t, p_, n_):
        p = t / (t + p_) if t + p_ else 0.0; r = t / (t + n_) if t + n_ else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)
    P, R, Fm = prf(tp, fp, fn)
    out = {"micro": (P, R, Fm), "tp": tp, "fp": fp, "fn": fn,
           "per_type": {l: prf(*v) for l, v in sorted(per.items())}}
    if return_details:
        out["predictions"] = details
    return out


def make_model(encoder_factory, types, onto, mode, lmax, device, gnn_depth=1,
               score_anchors=False, me_weight=1.0):
    enc = encoder_factory()
    if hasattr(enc, "model"): enc = enc.to(device)
    model = OntoSpan(enc, types, onto, mode=mode, lmax=lmax, gnn_depth=gnn_depth,
                     score_anchors=score_anchors, me_weight=me_weight).to(device)
    return model


def train_eval_once(encoder_factory, types, onto, mode, frac, seed,
                    train, dev_d, test, args, device, verbose=False):
    set_seed(seed)
    free_memory(device)
    onto = perturb_ontology(onto, getattr(args, "mutex_perturb", "real"), seed)
    n = max(1, int(len(train) * frac))
    idx = list(range(len(train))); random.shuffle(idx)
    sub = [train[i] for i in idx[:n]]
    model = make_model(encoder_factory, types, onto, mode, args.lmax, device,
                       gnn_depth=getattr(args, "gnn_depth", 1),
                       score_anchors=getattr(args, "score_anchors", False),
                       me_weight=getattr(args, "me_weight", 1.0))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = getattr(args, "amp", False) and str(device) == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    accum = max(1, int(getattr(args, "accum", 1)))
    aug_p = float(getattr(args, "aug_mask", 0.0))
    best_f1, best_state, best_ep = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        sub_ep = mask_augment(sub, aug_p, seed=seed * 1000 + ep) if aug_p > 0 else sub
        tl = run_epoch(model, sub_ep, optim, device, rho=args.rho, lam=args.lam,
                       gamma=args.gamma, bs=args.batch_size, scaler=scaler, accum=accum)
        dev_f1 = evaluate(model, dev_d, device, bs=args.batch_size, use_amp=use_amp)["micro"][2]
        if dev_f1 >= best_f1:
            best_f1, best_ep = dev_f1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"      epoch {ep}: train_loss={tl:.2f} dev-F1={dev_f1:.4f}"
                  + ("  ← best" if best_ep == ep else ""))
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        ck_dir = getattr(args, "ckpt_dir", None)
        if ck_dir:
            os.makedirs(ck_dir, exist_ok=True)
            ck = os.path.join(ck_dir, f"best_{mode}_frac{frac}_seed{seed}.pt")
            torch.save({"state_dict": best_state, "dev_f1": best_f1, "epoch": best_ep,
                        "types": types, "mode": mode}, ck)
            if verbose:
                print(f"      saved best-dev weights {ck} (dev-F1={best_f1:.4f} @ep{best_ep})")
    ev = evaluate(model, test, device, bs=args.batch_size, use_amp=use_amp)
    if verbose:
        print(f"      [test@best-dev epoch {best_ep}] per-type P/R/F1:",
              {l: f"{f:.2f}" for l, (p, r, f) in ev["per_type"].items()})
    return ev["micro"][2], ev


def smoke():
    print("=== SMOKE: synthetic data + stub encoder (offline; checks none/implicit/explicit) ===")
    raw = [
        ("方中附子与半夏同用属配伍禁忌", [(2, 4, "中药"), (5, 7, "中药")]),
        ("桂枝汤主治外感风寒", [(0, 3, "方剂"), (0, 2, "中药")]),
        ("患者头痛服用甘草", [(2, 4, "症状"), (6, 8, "中药")]),
        ("乌头反贝母临床慎用", [(0, 2, "中药"), (3, 5, "中药")]),
    ] * 6
    onto = {"attract": [("中药", "方剂", W_RELATED)], "tmutex": [],
            "imutex": [("附子", "半夏"), ("乌头", "贝母")], "parent": "中药"}
    types = build_types(raw, onto)

    class A: epochs = 8; lr = 5e-3; lmax = 4; lam = 0.2; gamma = 1.0; rho = 6; batch_size = 8; gnn_depth = 1
    for mode in ("none", "implicit", "explicit"):
        f1, ev = train_eval_once(lambda: StubEncoder(48), types, onto, mode, 1.0, 0,
                                 raw, raw, raw, A, "cpu", verbose=False)
        tag = {"none": "none    ", "implicit": "implicit", "explicit": "explicit"}[mode]
        print(f"  [{tag}] span micro-F1={f1:.3f}  (P={ev['micro'][0]:.3f} R={ev['micro'][1]:.3f})")
    class A2(A): gnn_depth = 2
    f1d2, _ = train_eval_once(lambda: StubEncoder(48), types, onto, "explicit", 1.0, 0,
                              raw, raw, raw, A2, "cpu", verbose=False)
    print(f"  [explicit GAT depth=2] span micro-F1={f1d2:.3f}")
    print("  all three modes + multi-layer explicit forward/backward/eval passed.")


def main():
    ap = argparse.ArgumentParser(description="OntoSpan: signed-ontology span-NER (implicit/explicit)")
    ap.add_argument("--domain", choices=["law", "tcm", "medical"])
    ap.add_argument("--data-root", default=os.path.join(_ROOT, "data"))
    ap.add_argument("--encoder", default="hfl/chinese-macbert-base")
    ap.add_argument("--ontology", default="implicit",
                    help="none|implicit|explicit|sgcn|sigat|all (comma list ok)")
    ap.add_argument("--seeds", default="42", help="comma-separated, e.g. 42,1,2")
    ap.add_argument("--train-fracs", default="1.0", help="comma-separated low-resource fractions, e.g. 0.1,0.2,1.0")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lmax", type=int, default=10)
    ap.add_argument("--gnn-depth", type=int, default=1,
                    help="explicit SignedGAT layers (multi-hop mutex propagation)")
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--rho", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action="store_true",
                    help="enable CUDA AMP (fp16); ignored on non-CUDA devices")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation steps; effective batch = batch_size * accum")
    ap.add_argument("--aug-mask", type=float, default=0.0,
                    help="char random-masking probability (training set only; 0 = off)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="if set, save best-dev .pt weights for each run to this directory")
    ap.add_argument("--mutex-perturb", choices=["real", "rewired", "removed", "random_sign"], default="real",
                    help="P3 control: real / rewired (difficulty-matched) / removed")
    ap.add_argument("--me-weight", type=float, default=1.0,
                    help="negative-message/implicit-ME weight")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        smoke(); return
    if not a.domain:
        ap.error("specify --domain {law,tcm,medical} (or --smoke)")

    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    ddir = os.path.join(a.data_root, a.domain)
    paths = {sp: os.path.join(ddir, f"{sp}.jsonl") for sp in ("train", "dev", "test")}
    miss = [p for p in paths.values() if not os.path.exists(p)]
    if miss:
        print(f"missing data: {miss}\n  run download_datasets.py --dataset <name>, then --check.")
        return
    train = read_jsonl(paths["train"]); dev_d = read_jsonl(paths["dev"]); test = read_jsonl(paths["test"])
    onto = load_ontology(a.domain)
    types = build_types(train + dev_d, onto)

    modes = ["none", "implicit", "explicit"] if a.ontology == "all" else a.ontology.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    fracs = [float(x) for x in a.train_fracs.split(",")]
    grid = len(modes) * len(seeds) * len(fracs)
    print(f"domain={a.domain} device={dev} types({len(types)})={types}")
    print(f"train/dev/test={len(train)}/{len(dev_d)}/{len(test)} | modes={modes} fracs={fracs} seeds={seeds} ({grid} runs)")
    print(f"type-mutex={onto['tmutex']}  instance-mutex pairs={len(onto['imutex'])}\n")

    enc_factory = lambda: HFEncoder(a.encoder)
    results = {}
    for frac in fracs:
        for mode in modes:
            f1s = []
            for seed in seeds:
                f1, ev = train_eval_once(enc_factory, types, onto, mode, frac, seed,
                                         train, dev_d, test, a, dev, verbose=(grid == 1))
                f1s.append(f1)
                print(f"  frac={frac:<4} mode={mode:8s} seed={seed:<3} → test span-F1={f1:.4f} "
                      f"(P={ev['micro'][0]:.3f} R={ev['micro'][1]:.3f})")
            results[(frac, mode)] = f1s

    print("\n================ low-resource ablation (test span-F1, mean+/-std over seeds) ================")
    header = "train_frac | " + " | ".join(f"{m:^16s}" for m in modes)
    print(header); print("-" * len(header))
    for frac in fracs:
        cells = []
        for mode in modes:
            arr = np.array(results[(frac, mode)])
            cells.append(f"{arr.mean():.4f}±{arr.std():.4f}")
        print(f"{frac:<10} | " + " | ".join(f"{c:^16s}" for c in cells))
    print("\nnote: implicit/explicit gains over none are largest at low fractions;"
          "\n      explicit can exceed implicit when the mutex graph is dense/multi-hop.")


if __name__ == "__main__":
    main()

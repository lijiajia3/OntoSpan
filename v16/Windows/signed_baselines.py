"""Task-adapted SGCN/SiGAT; see METHOD_SOURCES.md. Not original paper modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F

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

"""Pre-registered 10-seed paired grid (fixed before any result is seen).

cells : law@600, medical@180, medical@600, tcm@600   (frac of the 600-sentence cap)
modes : none | explicit(real) | explicit+random_sign | explicit+removed(ME dropped) | sgcn
seeds : 42 1 2 3 4 5 6 7 8 9
Unmodified v16 worker.train(); one machine, one torch build for every arm.
"""
import os, sys, time
from pathlib import Path
import torch
import adapter, worker
from protocol import ENCODER, settings

LR_TYPE = float(os.environ['BIG_LR_TYPE']) if os.environ.get('BIG_LR_TYPE') else None
CUR = {}
_orig_make = adapter.make_model
_orig_adamw = torch.optim.AdamW


def _make(j, types, onto, encoder):
    m = _orig_make(j, types, onto, encoder)
    CUR['model'] = m
    return m


def _adamw(params, lr, **kw):
    if LR_TYPE is None:
        return _orig_adamw(params, lr=lr, **kw)
    m = CUR['model']
    special = [m.type_emb.weight] + ([m.anchor] if getattr(m, 'anchor', None) is not None else [])
    ids = {id(p) for p in special}
    rest = [p for p in m.parameters() if id(p) not in ids]
    return _orig_adamw([{'params': rest, 'lr': lr}, {'params': special, 'lr': LR_TYPE}], lr=lr, **kw)


adapter.make_model = _make
torch.optim.AdamW = _adamw

REV = 'a986e004d2a7f2a1c2f5a3edef4e20604a974ed1'
OUT = Path.home() / 'onto' / ('big_grid' if LR_TYPE is None else 'big_grid_typelr%g' % LR_TYPE)
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
CELLS = (('law', 1.0), ('medical', 0.3), ('medical', 1.0), ('tcm', 1.0))
ARMS = (('none', 'real', 'none'), ('explicit', 'real', 'explicit'),
        ('explicit', 'random_sign', 'randsign'), ('explicit', 'removed', 'removed'),
        ('sgcn', 'real', 'sgcn'))


def job(domain, frac, mode, perturb, tag, seed):
    s = settings()
    j = dict(family='big', domain=domain, frac=frac, mode=mode, seed=seed, encoder=ENCODER,
             encoder_revision=REV, epochs=5, lr=2e-5, lmax=10, gnn_depth=2, rho=5, lam=.1, gamma=1.,
             me_weight=1., perturb=perturb, train_cap=600, dev_cap=100, test_cap=300,
             batch_size=s['batch_size'], accum=s['accum'], memory_limit_gib=s['memory_limit_gib'],
             fast_spans=s['fast_spans'], type_mutex=False, precision='fp32',
             data_policy='legacy_original_splits_v1', lr_type=LR_TYPE)
    j['id'] = f'big_{domain}_f{frac:g}_{tag}_s{seed}'.replace('.', 'p')
    return j


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = [job(d, f, m, p, t, sd) for sd in SEEDS for d, f in CELLS for m, p, t in ARMS]
    print('planned', len(plan), flush=True)
    for j in plan:
        out = OUT / j['id']
        if (out / 'COMPLETE.json').exists():
            continue
        t = time.time()
        try:
            r = worker.train(j, out, device='cuda')
        except Exception as e:
            print('FAIL', j['id'], repr(e)[:200], flush=True)
            continue
        m = r['metrics']
        print('DONE', j['id'], 'F1=%.5f tp=%d fp=%d fn=%d dev=%.5f ep=%d %.0fs' % (
            m['F1'], m['tp'], m['fp'], m['fn'], r['dev_best'], r['best_epoch'], time.time() - t), flush=True)


if __name__ == '__main__':
    main()

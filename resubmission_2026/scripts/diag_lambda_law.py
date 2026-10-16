"""R3.4 diagnostic: law, implicit regularizer, lambda sweep with E-change logging.

Runs the unmodified v16 worker.train() and only wraps (a) geometry() to also record
how far the type embeddings moved, and (b) AdamW to optionally give type embeddings
their own learning rate. No repository file is edited.
"""
import json, sys, time
from pathlib import Path
import torch
import adapter, worker
from protocol import ENCODER, settings

REV = 'a986e004d2a7f2a1c2f5a3edef4e20604a974ed1'
OUT = Path.home() / 'onto' / 'diag_law_lambda'
CUR = {}
_orig_make = adapter.make_model
_orig_geo = worker.geometry
_orig_adamw = torch.optim.AdamW


def make_model(j, types, onto, encoder):
    m = _orig_make(j, types, onto, encoder)
    CUR['model'] = m
    CUR.pop('E0', None)
    return m


def geometry(model):
    g = _orig_geo(model)
    E = model.type_emb.weight.detach().float().cpu()
    if 'E0' not in CUR:
        CUR['E0'] = E.clone()
    d = (E - CUR['E0'])
    g['E_row_norm'] = E.norm(dim=1).tolist()
    g['dE_row_l2'] = d.norm(dim=1).tolist()
    g['dE_rel_fro'] = float(d.norm() / CUR['E0'].norm())
    g['types'] = list(model.types)
    return g


def adamw(params, lr, **kw):
    lt = CUR.get('lr_type')
    if lt is None:
        return _orig_adamw(params, lr=lr, **kw)
    m = CUR['model']
    special = [m.type_emb.weight] + ([m.anchor] if getattr(m, 'anchor', None) is not None else [])
    ids = {id(p) for p in special}
    rest = [p for p in m.parameters() if id(p) not in ids]
    return _orig_adamw([{'params': rest, 'lr': lr}, {'params': special, 'lr': lt}], lr=lr, **kw)


adapter.make_model = make_model
worker.geometry = geometry
torch.optim.AdamW = adamw


def job(seed, lam, lr_type):
    s = settings()
    tag = 'base' if lr_type is None else f'typelr{lr_type:g}'
    j = dict(family='diag_lambda', domain='law', frac=1.0, mode='implicit', seed=seed,
             encoder=ENCODER, encoder_revision=REV, epochs=5, lr=2e-5, lmax=10, gnn_depth=2, rho=5,
             lam=lam, gamma=1.0, me_weight=1.0, perturb='real', train_cap=600, dev_cap=100, test_cap=300,
             batch_size=s['batch_size'], accum=s['accum'], memory_limit_gib=s['memory_limit_gib'],
             fast_spans=s['fast_spans'], type_mutex=False, precision='fp32',
             data_policy='legacy_original_splits_v1', lr_type=lr_type)
    j['id'] = f'law_s{seed}_lam{lam:g}_{tag}'.replace('.', 'p')
    return j


def main():
    plan = [(sd, lam, lt) for lt in (None, 1e-3) for lam in (0.0, 0.1, 1.0) for sd in (42, 1, 2)]
    if len(sys.argv) > 1:
        plan = plan[:int(sys.argv[1])]
    OUT.mkdir(parents=True, exist_ok=True)
    for sd, lam, lt in plan:
        j = job(sd, lam, lt)
        out = OUT / j['id']
        if (out / 'COMPLETE.json').exists():
            print('skip', j['id'], flush=True); continue
        CUR['lr_type'] = lt
        t = time.time()
        r = worker.train(j, out, device='cuda')
        m = r['metrics']
        print('DONE', j['id'], 'F1=%.5f tp=%d fp=%d fn=%d' % (m['F1'], m['tp'], m['fp'], m['fn']),
              'best_ep', r['best_epoch'], 'dE_rel=%.5f' % r['geometry']['best']['dE_rel_fro'],
              'reg %.4f->%.4f' % (r['geometry']['initial']['implicit_regularizer'],
                                  r['geometry']['best']['implicit_regularizer']),
              '%.0fs' % (time.time() - t), flush=True)


def repro():
    """Re-run recorded law/none and law/explicit runs on this hardware; compare to paper."""
    from protocol import jobs
    out_root = Path.home() / 'onto' / 'diag_repro'
    for j in jobs():
        if j['family'] == 'main' and j['domain'] == 'law' and j['frac'] == 1.0 and j['mode'] in ('none', 'explicit'):
            j = dict(j, encoder_revision=REV, lr_type=None)
            out = out_root / j['id']
            if (out / 'COMPLETE.json').exists():
                continue
            CUR['lr_type'] = None
            r = worker.train(j, out, device='cuda')
            m = r['metrics']
            print('REPRO', j['id'], 'F1=%.5f tp=%d fp=%d fn=%d' % (m['F1'], m['tp'], m['fp'], m['fn']), flush=True)



def det():
    """Same job, same machine, three times: is training bitwise repeatable?"""
    from protocol import jobs
    base = [j for j in jobs() if j['id'] == 'main_law_f1_none_s1'][0]
    for tag in ('a', 'b', 'c'):
        j = dict(base, encoder_revision=REV, lr_type=None)
        CUR['lr_type'] = None
        r = worker.train(j, Path.home() / 'onto' / 'diag_det' / tag, device='cuda')
        m = r['metrics']
        print('DET', tag, 'F1=%.5f tp=%d fp=%d fn=%d' % (m['F1'], m['tp'], m['fp'], m['fn']),
              'dev=%.5f' % r['dev_best'], 'ep', r['best_epoch'], flush=True)


if len(sys.argv) > 1 and sys.argv[1] == 'det':
    det()
    sys.exit(0)


if len(sys.argv) > 1 and sys.argv[1] == 'repro':
    repro()
    sys.exit(0)


if __name__ == '__main__':
    main()

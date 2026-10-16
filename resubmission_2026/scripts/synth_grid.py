"""Controlled synthetic check of the mutual-exclusion (ME) mechanism (R4 / R3).

Question: when two types are genuinely confusable, have few training examples and
the ontology's ME edge sits exactly on that pair, does an ME-aware model beat
(a) no ontology, (b) the same graph with permuted signs, (c) the graph without ME edges?

Unmodified v16 worker.train() is used; only the data tuple and the ontology are injected.
Each replicate = an independent synthetic dataset (data seed = training seed).
See PREREG_synth.md: protocol, metric and decision rule were fixed before any arm comparison.
"""
import os, random, sys, time, json
from pathlib import Path
import torch
import adapter, worker
from adapter import legacy
from protocol import ENCODER, settings

LR_TYPE = float(os.environ['SYN_LR_TYPE']) if os.environ.get('SYN_LR_TYPE') else None
K_RARE = int(os.environ.get('SYN_K', '10'))          # train sentences per ME-pair type
ONLY_NONE = os.environ.get('SYN_ONLY_NONE') == '1'   # pilot: difficulty calibration only
P_SHARED_PREFIX = float(os.environ.get('SYN_S', '0.6'))   # prefix cue uninformative
P_TAIL = float(os.environ.get('SYN_Q', '0.7'))            # tail cue informative

CUR = {}
_orig_make, _orig_adamw = adapter.make_model, torch.optim.AdamW


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

# ---- synthetic ontology: two real ME pairs, one related pair, two easy types ----
ME1, ME2, REL, EASY = ('甲类', '乙类'), ('丙类', '丁类'), ('戊类', '己类'), ('庚类', '辛类')
ME_TYPES = ME1 + ME2
SYN_ONTO = lambda: {'attract': [(REL[0], REL[1], legacy.W_RELATED)],
                    'tmutex': [ME1, ME2], 'imutex': [], 'parent': None}
legacy.load_ontology = lambda domain: SYN_ONTO()

PREFIX = {
    '甲类': ['嫌疑人', '被告人', '犯罪分子'], '乙类': ['被害人', '受害者', '伤者'],
    '丙类': ['承包方', '出让人', '委托人'], '丁类': ['受让方', '购买人', '受托人'],
    '戊类': ['甲方', '申请人', '原告'], '己类': ['乙方', '被申请人', '被诉方'],
    '庚类': ['地点位于', '案发地为', '现场在'], '辛类': ['涉案金额', '价值约为', '共计人民币']}
SHARED = {ME1: ['当事人', '该男子', '一名男子'], ME2: ['相关方', '该单位', '一名负责人']}
TAIL = {
    '甲类': ['随后逃离现场', '被民警抓获', '供述了犯罪事实'], '乙类': ['随后被送往医院', '经鉴定为轻伤', '要求赔偿损失'],
    '丙类': ['按期交付了标的物', '提出解除合同', '收取了定金'], '丁类': ['拒绝支付尾款', '要求继续履行', '主张违约责任'],
    '戊类': ['提交了全部材料', '请求确认权利'], '己类': ['提交了答辩意见', '请求驳回请求'],
    '庚类': ['已被警方封控', '附近设有监控'], '辛类': ['已全部追缴', '尚未能够退还']}
NEUTRAL = ['事件发生在当晚', '相关情况已通报', '有关部门已介入', '具体经过仍在调查']
POOL = list('张王李赵陈刘杨黄周吴徐孙马朱胡郭何林高梁郑罗宋谢唐韩曹许邓萧冯曾程蔡彭潘袁')
FILL = ['在城东某处', '于上周三', '在一家商铺内', '在公司门口', '于昨日下午', '在小区附近']


def name(r):
    return ''.join(r.choice(POOL) for _ in range(r.choice((2, 3))))


def sentence(t, r):
    pair = ME1 if t in ME1 else ME2 if t in ME2 else None
    if pair and r.random() < P_SHARED_PREFIX:
        pre = r.choice(SHARED[pair])
    else:
        pre = r.choice(PREFIX[t])
    ent = name(r)
    tail = r.choice(TAIL[t]) if (not pair or r.random() < P_TAIL) else r.choice(NEUTRAL)
    text = f'据了解，{pre}{ent}{r.choice(FILL)}，{tail}。'
    s = text.index(ent, 4 + len(pre))
    return text, [(s, s + len(ent), t)]


def build(seed, k):
    r = random.Random(10_000 + seed)
    counts_train = {**{t: k for t in ME_TYPES}, **{t: 40 for t in REL}, **{t: 60 for t in EASY}}
    counts_dev = {**{t: 15 for t in ME_TYPES}, **{t: 10 for t in REL}, **{t: 10 for t in EASY}}
    counts_test = {**{t: 40 for t in ME_TYPES}, **{t: 20 for t in REL}, **{t: 30 for t in EASY}}
    out = []
    for counts in (counts_train, counts_dev, counts_test):
        rows = [sentence(t, r) for t, n in counts.items() for _ in range(n)]
        r.shuffle(rows)
        out.append(rows)
    return out


def job(seed, mode, perturb, tag):
    s = settings()
    j = dict(family='synth', domain='synth', frac=1.0, mode=mode, seed=seed, encoder=ENCODER,
             encoder_revision='a986e004d2a7f2a1c2f5a3edef4e20604a974ed1', epochs=5, lr=2e-5, lmax=10,
             gnn_depth=2, rho=5, lam=.1, gamma=1., me_weight=1., perturb=perturb, train_cap=10 ** 6,
             dev_cap=100, test_cap=300, batch_size=s['batch_size'], accum=s['accum'],
             memory_limit_gib=s['memory_limit_gib'], fast_spans=s['fast_spans'], type_mutex=False,
             precision='fp32', data_policy='synthetic_v1', lr_type=LR_TYPE, k_rare=K_RARE)
    j['id'] = f'synth_k{K_RARE}_{tag}_s{seed}'
    return j


ARMS = (('none', 'real', 'none'), ('implicit', 'real', 'implicit'), ('explicit', 'real', 'explicit'),
        ('explicit', 'random_sign', 'randsign'), ('explicit', 'removed', 'removed'))
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
OUT = Path.home() / 'onto' / ('synth_pilot' if ONLY_NONE else
                              f'synth_k{K_RARE}' + ('' if LR_TYPE is None else f'_typelr{LR_TYPE:g}'))


def main():
    arms = ARMS[:1] if ONLY_NONE else ARMS
    seeds = SEEDS[:2] if ONLY_NONE else SEEDS
    OUT.mkdir(parents=True, exist_ok=True)
    for sd in seeds:
        tr, dv, te = build(sd, K_RARE)
        audit = dict(train_count=len(tr), capped_count=len(tr), dev=dict(n=len(dv)), evaluation=dict(n=len(te)),
                     train_hash='synthetic', dev_hash='synthetic', evaluation_hash='synthetic',
                     selection='synthetic v1; see PREREG_synth.md', evaluation_name='synthetic test')
        for mode, pert, tag in arms:
            j = job(sd, mode, pert, tag)
            out = OUT / j['id']
            if (out / 'COMPLETE.json').exists():
                continue
            t = time.time()
            try:
                r = worker.train(j, out, device='cuda', data=(tr, dv, te, tr + dv, audit))
            except Exception as e:
                print('FAIL', j['id'], repr(e)[:200], flush=True)
                continue
            m = r['metrics']
            pt = r['per_type']
            tp = sum(pt.get(x, {}).get('tp', 0) for x in ME_TYPES)
            fp = sum(pt.get(x, {}).get('fp', 0) for x in ME_TYPES)
            fn = sum(pt.get(x, {}).get('fn', 0) for x in ME_TYPES)
            print('DONE', j['id'], 'F1=%.4f ME-F1=%.4f dev=%.4f ep=%d %.0fs' % (
                m['F1'], 2 * tp / max(1, 2 * tp + fp + fn), r['dev_best'], r['best_epoch'], time.time() - t), flush=True)


if __name__ == '__main__':
    main()

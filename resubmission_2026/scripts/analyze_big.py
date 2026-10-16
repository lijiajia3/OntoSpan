"""Paired analysis of the 10-seed grid. Pure python; exact sign-flip test; Holm over all comparisons."""
import itertools, json, random, statistics as st, sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'onto' / 'big_grid'
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
res = {}
for d in root.glob('big_*'):
    p = d / 'result.json'
    if not p.exists():
        continue
    r = json.loads(p.read_text(encoding='utf-8'))
    j = r['job']
    tag = d.name.split('_f')[1].split('_', 1)[1].rsplit('_s', 1)[0]
    cell = f"{j['domain']}@{r['splits']['train_count']}"
    res[(cell, tag, j['seed'])] = dict(F1=r['metrics']['F1'], dev=r['dev_best'], tp=r['metrics']['tp'],
                                       fp=r['metrics']['fp'], fn=r['metrics']['fn'])

cells = sorted({k[0] for k in res})
tags = ['none', 'explicit', 'randsign', 'removed', 'sgcn']


def signflip_p(diffs):
    n = len(diffs)
    obs = abs(sum(diffs))
    cnt = 0
    for signs in itertools.product((1, -1), repeat=n):
        if abs(sum(s * d for s, d in zip(signs, diffs))) >= obs - 1e-12:
            cnt += 1
    return cnt / 2 ** n


def boot_ci(diffs, B=5000, seed=0):
    rng = random.Random(seed)
    m = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(B))
    return m[int(.025 * B)], m[int(.975 * B)]


print('== mean test F1 (std) over seeds; n per cell')
for c in cells:
    row = []
    for t in tags:
        xs = [res[(c, t, s)]['F1'] for s in SEEDS if (c, t, s) in res]
        row.append(f"{t}={st.mean(xs):.4f}({st.stdev(xs) if len(xs) > 1 else 0:.3f},n={len(xs)})" if xs else f"{t}=NA")
    print(f"{c:12s}", '  '.join(row))

pairs = [('explicit', 'none'), ('explicit', 'randsign'), ('explicit', 'removed'), ('sgcn', 'explicit')]
rows = []
for c in cells:
    for a, b in pairs:
        ss = [s for s in SEEDS if (c, a, s) in res and (c, b, s) in res]
        if len(ss) < 5:
            continue
        diffs = [res[(c, a, s)]['F1'] - res[(c, b, s)]['F1'] for s in ss]
        lo, hi = boot_ci(diffs)
        rows.append(dict(cell=c, a=a, b=b, n=len(ss), mean=st.mean(diffs) * 100, lo=lo * 100, hi=hi * 100,
                         p=signflip_p(diffs), wins=sum(d > 0 for d in diffs)))
order = sorted(range(len(rows)), key=lambda i: rows[i]['p'])
m = len(rows)
run_max = 0.0
for rank, i in enumerate(order):
    adj = min(1.0, rows[i]['p'] * (m - rank))
    run_max = max(run_max, adj)
    rows[i]['holm'] = run_max

print('\n== paired differences in F1 points (a - b), seed-level; 95% bootstrap CI over seeds; exact sign-flip p; Holm')
print(f"{'cell':12s} {'a - b':22s} n  mean    [95% CI]          wins  p      Holm")
for r in rows:
    print(f"{r['cell']:12s} {r['a'] + ' - ' + r['b']:22s} {r['n']:2d} {r['mean']:+6.2f}  [{r['lo']:+6.2f},{r['hi']:+6.2f}]  {r['wins']:2d}/{r['n']}  {r['p']:.3f}  {r['holm']:.3f}")

print('\n== between-seed std of F1 for none (training-seed noise floor, F1 points)')
for c in cells:
    xs = [res[(c, 'none', s)]['F1'] * 100 for s in SEEDS if (c, 'none', s) in res]
    if len(xs) > 1:
        print(f"{c:12s} sd={st.stdev(xs):.2f}  range={min(xs):.1f}-{max(xs):.1f}")

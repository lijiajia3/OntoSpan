"""Pre-registered analysis of the synthetic ME check (see PREREG_synth.md).
Primary metric: micro-F1 over the four ME types. Secondary: overall micro-F1.
16 comparisons: {explicit-none, explicit-randsign, explicit-removed, implicit-none} x k{10,40} x protocol{default,typelr}.
Paired by seed, exact sign-flip test, 95% bootstrap CI over seeds, Holm over the 16."""
import itertools, json, random, statistics as st
from pathlib import Path

HOME = Path.home() / 'onto'
ME = ('甲类', '乙类', '丙类', '丁类')
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
ARMS = ('none', 'implicit', 'explicit', 'randsign', 'removed')


def load(root):
    out = {}
    for d in Path(root).glob('synth_k*/synth_k*'):
        p = d / 'result.json'
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding='utf-8'))
        j = r['job']
        tag = d.name.split('_', 2)[2].rsplit('_s', 1)[0]
        pt = r['per_type']
        tp = sum(pt.get(t, {}).get('tp', 0) for t in ME)
        fp = sum(pt.get(t, {}).get('fp', 0) for t in ME)
        fn = sum(pt.get(t, {}).get('fn', 0) for t in ME)
        out[(j['k_rare'], j['lr_type'], tag, j['seed'])] = dict(
            me=100 * 2 * tp / max(1, 2 * tp + fp + fn), all=100 * r['metrics']['F1'])
    return out


def flip_p(x):
    obs = abs(sum(x))
    c = sum(abs(sum(s * v for s, v in zip(sg, x))) >= obs - 1e-9
            for sg in itertools.product((1, -1), repeat=len(x)))
    return c / 2 ** len(x)


def ci(x, B=5000):
    r = random.Random(0)
    m = sorted(st.mean(r.choices(x, k=len(x))) for _ in range(B))
    return m[int(.025 * B)], m[int(.975 * B)]


def main():
    res = load(HOME)
    cells = sorted({(k, lt) for k, lt, _, _ in res}, key=lambda c: (c[0], str(c[1])))
    print('== mean ME-type F1 (sd) / overall F1, n seeds')
    for k, lt in cells:
        row = []
        for a in ARMS:
            v = [res[(k, lt, a, s)] for s in SEEDS if (k, lt, a, s) in res]
            if v:
                row.append('%s=%.1f(%.1f)/%.1f n=%d' % (a, st.mean(x['me'] for x in v), st.stdev(x['me'] for x in v) if len(v) > 1 else 0,
                                                       st.mean(x['all'] for x in v), len(v)))
        print(f'k={k:2d} lr_type={lt}:', ' | '.join(row))
    pairs = (('explicit', 'none'), ('explicit', 'randsign'), ('explicit', 'removed'), ('implicit', 'none'))
    rows = []
    for k, lt in cells:
        for a, b in pairs:
            ss = [s for s in SEEDS if (k, lt, a, s) in res and (k, lt, b, s) in res]
            if len(ss) < 5:
                continue
            for metric in ('me', 'all'):
                x = [res[(k, lt, a, s)][metric] - res[(k, lt, b, s)][metric] for s in ss]
                lo, hi = ci(x)
                rows.append(dict(k=k, lt=lt, a=a, b=b, metric=metric, n=len(ss), mean=st.mean(x), lo=lo, hi=hi,
                                 p=flip_p(x), wins=sum(v > 0 for v in x)))
    prim = [r for r in rows if r['metric'] == 'me']
    order = sorted(range(len(prim)), key=lambda i: prim[i]['p'])
    m, run = len(prim), 0.
    for rank, i in enumerate(order):
        run = max(run, min(1., prim[i]['p'] * (m - rank)))
        prim[i]['holm'] = run
    print(f"\n== PRIMARY (ME-type micro-F1), paired difference in points, Holm over {m} comparisons")
    for r in sorted(prim, key=lambda r: (r['k'], str(r['lt']), r['a'], r['b'])):
        print(f"k={r['k']:2d} lr_type={str(r['lt']):6s} {r['a']:8s}-{r['b']:8s} n={r['n']:2d} {r['mean']:+6.2f} "
              f"[{r['lo']:+6.2f},{r['hi']:+6.2f}] wins {r['wins']}/{r['n']} p={r['p']:.3f} Holm={r['holm']:.3f}")
    print('\n== SECONDARY (overall micro-F1), unadjusted')
    for r in sorted([r for r in rows if r['metric'] == 'all'], key=lambda r: (r['k'], str(r['lt']), r['a'], r['b'])):
        print(f"k={r['k']:2d} lr_type={str(r['lt']):6s} {r['a']:8s}-{r['b']:8s} {r['mean']:+6.2f} "
              f"[{r['lo']:+6.2f},{r['hi']:+6.2f}] p={r['p']:.3f}")
    ok = [r for r in prim if r['a'] == 'explicit' and r['b'] == 'none']
    print('\n== decision rule: explicit beats none AND randsign AND removed, Holm<0.05, same cell')
    for k, lt in cells:
        cs = [r for r in prim if r['k'] == k and r['lt'] == lt and r['a'] == 'explicit' and r['b'] in ('none', 'randsign', 'removed')]
        passed = len(cs) == 3 and all(r['mean'] > 0 and r['holm'] < .05 for r in cs)
        print(f'k={k} lr_type={lt}: {"PASS" if passed else "not detected"}')


if __name__ == '__main__':
    main()

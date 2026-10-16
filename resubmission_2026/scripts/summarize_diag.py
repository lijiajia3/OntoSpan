import json, sys, statistics as st
from pathlib import Path

root = Path.home() / 'onto' / 'diag_law_lambda'
runs = {}
for d in sorted(root.glob('law_s*')):
    p = d / 'result.json'
    if not p.exists():
        continue
    r = json.loads(p.read_text(encoding='utf-8'))
    j = r['job']
    ev = [json.loads(l) for l in (d / 'evaluation.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    key = (j['lr_type'], j['lam'], j['seed'])
    types = r['geometry']['best'].get('types', [])
    dE = r['geometry']['best']['dE_row_l2']
    live = {t: round(dE[i], 5) for i, t in enumerate(types) if t in ('受害人', '犯罪嫌疑人')}
    pairs = r['geometry']['best']['type_pairs']
    live_pair = [p_ for p_ in pairs if set(p_['pair']) == {'受害人', '犯罪嫌疑人'}]
    runs[key] = dict(F1=r['metrics']['F1'], tp=r['metrics']['tp'], fp=r['metrics']['fp'], fn=r['metrics']['fn'],
                     dev=r['dev_best'], ep=r['best_epoch'],
                     reg0=r['geometry']['initial']['implicit_regularizer'],
                     reg1=r['geometry']['best']['implicit_regularizer'],
                     dE_rel=r['geometry']['best']['dE_rel_fro'], live_dE=live,
                     cos0=[p_['raw_cos'] for p_ in r['geometry']['initial']['type_pairs']
                           if set(p_['pair']) == {'受害人', '犯罪嫌疑人'}],
                     cos1=[p_['raw_cos'] for p_ in live_pair],
                     pred=[json.dumps(x['predicted']) for x in ev])

print('lr_type  lam   seed |   F1     tp   fp   fn | dev    ep | reg0 -> reg1   | dE_rel   | live dE | cos(受害人,嫌疑人) 0->1 | diff-vs-lam0')
for (lt, lam, sd), v in sorted(runs.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
    base = runs.get((lt, 0.0, sd))
    diff = ''
    if base is not None and lam != 0.0:
        n = sum(a != b for a, b in zip(v['pred'], base['pred']))
        diff = f"{n}/{len(v['pred'])} sents differ"
    print(f"{str(lt):7s} {lam:5g} {sd:4d} | {v['F1']:.4f} {v['tp']:5d} {v['fp']:4d} {v['fn']:4d} | {v['dev']:.4f} {v['ep']} |"
          f" {v['reg0']:.3f}->{v['reg1']:.3f} | {v['dE_rel']:.5f} | {v['live_dE']} | {v['cos0']}->{v['cos1']} | {diff}")

print('\nmean F1 over seeds')
for lt in sorted({k[0] for k in runs}, key=str):
    for lam in sorted({k[1] for k in runs}):
        xs = [runs[(lt, lam, s)]['F1'] for s in (42, 1, 2) if (lt, lam, s) in runs]
        ds = [runs[(lt, lam, s)]['dev'] for s in (42, 1, 2) if (lt, lam, s) in runs]
        if xs:
            print(f'  lr_type={lt} lam={lam:g}: test F1 {st.mean(xs):.4f} (n={len(xs)})  dev {st.mean(ds):.4f}')

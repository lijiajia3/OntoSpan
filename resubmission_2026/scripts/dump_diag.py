import json
from pathlib import Path
root = Path.home() / 'onto' / 'diag_law_lambda'
out = []
base = {}
for d in sorted(root.glob('law_s*')):
    p = d / 'result.json'
    if not p.exists():
        continue
    r = json.loads(p.read_text(encoding='utf-8')); j = r['job']
    ev = [json.dumps(json.loads(l)['predicted']) for l in (d / 'evaluation.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    g = r['geometry']
    pair = lambda blk: [x['raw_cos'] for x in g[blk]['type_pairs'] if set(x['pair']) == {'受害人', '犯罪嫌疑人'}]
    base[(j['lr_type'], j['lam'], j['seed'])] = ev
    out.append(dict(run=d.name, protocol='default' if j['lr_type'] is None else 'typelr%g' % j['lr_type'], lam=j['lam'], seed=j['seed'],
                    F1=r['metrics']['F1'], tp=r['metrics']['tp'], fp=r['metrics']['fp'], fn=r['metrics']['fn'],
                    dev_best=r['dev_best'], best_epoch=r['best_epoch'],
                    reg_initial=g['initial']['implicit_regularizer'], reg_best=g['best']['implicit_regularizer'],
                    dE_rel_fro=g['best']['dE_rel_fro'], cos_initial=pair('initial'), cos_best=pair('best'), _ev=ev))
for o in out:
    b = base.get((None if o['protocol'] == 'default' else float(o['protocol'][6:]), 0.0, o['seed']))
    o['sentences_differing_from_lam0'] = None if (b is None or o['lam'] == 0) else sum(x != y for x, y in zip(o['_ev'], b))
    del o['_ev']
json.dump(out, open(Path.home() / 'onto' / 'diag_lambda_runs.json', 'w'), ensure_ascii=False, indent=1)
print(len(out))

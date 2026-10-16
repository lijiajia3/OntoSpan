"""Implementation check from stored geometry: effective cosine of the suspect/victim prototypes after the graph module."""
import json, glob, statistics as st
from pathlib import Path
POS = {1, 4, 5, 7, 8, 9}   # seeds whose random-sign draw made the live edge positive
out = {}
for root, proto in (('big_grid', 'MacBERT, primary'), ('big_grid_typelr0.001', 'MacBERT, type emb. 1e-3'), ('big_grid_roberta', 'RoBERTa, primary')):
    for p in glob.glob(str(Path.home() / 'onto' / root / 'big_law_f1_*' / 'result.json')):
        r = json.load(open(p)); j = r['job']; g = r.get('geometry', {})
        tag = Path(p).parent.name.split('_f1_')[1].rsplit('_s', 1)[0]
        if tag == 'randsign':
            tag = 'randsign(+)' if j['seed'] in POS else 'randsign(-)'
        for when in ('initial', 'best'):
            pr = [x for x in g.get(when, {}).get('type_pairs', []) if set(x['pair']) == {'受害人', '犯罪嫌疑人'}]
            if pr:
                out.setdefault((proto, tag, when), []).append(pr[0]['effective_cos'])
rows = []
for (proto, tag, when), v in sorted(out.items()):
    rows.append(dict(protocol=proto, config=tag, when=when, n=len(v), mean=st.mean(v), sd=st.stdev(v) if len(v) > 1 else 0, values=v))
    print(f"{proto:26s} {tag:12s} {when:8s} n={len(v):2d} cos={st.mean(v):+.3f} sd={st.stdev(v) if len(v)>1 else 0:.3f}")
json.dump(rows, open(Path.home() / 'onto' / 'me_geometry.json', 'w'), ensure_ascii=False, indent=1)

import json, glob, sys
from pathlib import Path
out=[]
for root,proto in (("big_grid","default"),("big_grid_typelr0.001","typelr1e-3"),("big_grid_roberta","roberta-default")):
    for p in glob.glob(str(Path.home()/'onto'/root/'big_*'/'result.json')):
        r=json.load(open(p)); j=r['job']; m=r['metrics']
        tag=Path(p).parent.name.split('_f')[1].split('_',1)[1].rsplit('_s',1)[0]
        pt=r['per_type']
        out.append(dict(kind='main' if not proto.startswith('roberta') else 'roberta',protocol=proto,domain=j['domain'],frac=j['frac'],train=r['splits']['train_count'],arm=tag,seed=j['seed'],
                        F1=m['F1'],P=m['P'],R=m['R'],tp=m['tp'],fp=m['fp'],fn=m['fn'],dev=r['dev_best'],ep=r['best_epoch'],
                        per_type={k:[v['tp'],v['fp'],v['fn']] for k,v in pt.items()},
                        secs=r['timing']['total_seconds']))
for p in glob.glob(str(Path.home()/'onto'/'synth_k*'/'synth_k*'/'result.json')):
    r=json.load(open(p)); j=r['job']; m=r['metrics']
    tag=Path(p).parent.name.split('_',2)[2].rsplit('_s',1)[0]
    out.append(dict(kind='synth',protocol='default' if j['lr_type'] is None else 'typelr1e-3',k=j['k_rare'],arm=tag,seed=j['seed'],
                    F1=m['F1'],tp=m['tp'],fp=m['fp'],fn=m['fn'],dev=r['dev_best'],ep=r['best_epoch'],
                    per_type={k:[v['tp'],v['fp'],v['fn']] for k,v in r['per_type'].items()}))
json.dump(out,open(Path.home()/'onto'/'all_runs.json','w'),ensure_ascii=False)
print(len(out))

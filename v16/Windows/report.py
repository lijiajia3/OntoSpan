"""Evidence, not an automatic publication-acceptance verdict."""
import json
import statistics
import zipfile
from pathlib import Path
from protocol import SEEDS,ROOT,settings
from integrity import validate,records
from metrics import paired,holm
from worker import dump


def comparisons(js):
    specs=[]
    def group(**kw):return [next(j for j in js if j['seed']==s and all(j[k]==v for k,v in kw.items())) for s in SEEDS]
    def add(name,left,right):specs.append((name,group(**left),group(**right)))
    def main(d,f,m='explicit'):return dict(family='main',domain=d,frac=f,mode=m)
    for d in ['medical','tcm','law']:
        for f in [.1,.3,1.]:add(f'main/{d}/{f}',main(d,f,'none'),main(d,f))
    for d,f in [('medical',.1),('medical',.3),('medical',1.),('tcm',.1),('law',1.)]:
        for p in ['removed','rewired']:
            add(f'p3/real-minus-{p}/{d}/{f}',dict(family='p3',domain=d,frac=f,perturb=p),main(d,f))
    for d,f in [('medical',.1),('law',1.)]:
        add(f'random/real-minus-random/{d}',dict(family='random_sign',domain=d),main(d,f))
        add(f'backbone/{d}',dict(family='backbone',domain=d,mode='none'),dict(family='backbone',domain=d,mode='explicit'))
        for m in ['sgcn','sigat']:
            add(f'signed/explicit-minus-{m}/{d}',dict(family='signed_baseline',domain=d,mode=m),main(d,f))
    for w in [0.,.25,.5,2.]:
        add(f'me_weight/{w}-minus-default',main('law',1.),dict(family='me_weight',me_weight=w))
    for l in [.05,.1,.5,1.]:add(f'lambda/{l}-minus-zero',dict(family='lambda',lam=0.),dict(family='lambda',lam=l))
    for d in ['medical','tcm']:
        add(f'uncapped/{d}',dict(family='uncapped',domain=d,mode='none'),dict(family='uncapped',domain=d,mode='explicit'))
    for f in [.1,.3,1.]:
        for m in ['none','explicit']:
            add(f'onaxis/minus-{m}/{f}',main('tcm',f,m),dict(family='tcm_onaxis',frac=f))
    for m in ['biaffine','global_pointer','bert_crf','w2ner']:
        add(f'traditional/explicit-minus-{m}',dict(family='traditional',mode=m),main('medical',.1))
    return specs


def refresh(manifest,folder,with_stats=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    good={};missing=[];invalid={}
    for j in manifest['jobs']:
        p=folder/'jobs'/j['id']
        if not (p/'COMPLETE.json').exists():missing.append(j['id']);continue
        r,error=validate(p,j)
        if error:invalid[j['id']]=error
        else:good[j['id']]=r
    stats=[];pending=[]
    specs=comparisons(manifest['jobs'])
    if with_stats:
        for name,a,b in specs:
            if not all(j['id'] in good for j in a+b):pending.append(name);continue
            try:
                row=paired([records(folder/'jobs'/j['id']/'evaluation.jsonl') for j in a],
                           [records(folder/'jobs'/j['id']/'evaluation.jsonl') for j in b],manifest['analysis']['bootstrap_replicates'])
                row['comparison']=name;stats.append(row)
            except ValueError as e:pending.append(name+': '+str(e))
        holm(stats,len(specs));dump(folder/'STATISTICS.json',dict(comparisons=stats,pending=pending,planned_comparisons=len(specs)))
    else:
        # Previous statistics cannot silently survive changed/missing artifacts.
        for name in ['STATISTICS.json']:
            p=folder/name
            if p.exists():p.unlink()
    groups={}
    for r in good.values():
        j=r['job'];k=(j['family'],j['domain'],j['frac'],j['mode'],j['encoder'],j['perturb'],j['lam'],j['me_weight'])
        groups.setdefault(k,[]).append(r)
    lines=['# v16 审稿实验汇总','',f'完成并校验 {len(good)}/174；缺项 {len(missing)}；无效 {len(invalid)}。',
           '', '技术完成不等于科学结论成立，更不等于保证录用。正、零、负增益均保留。',
           '旧方法与原train/dev/test抽样；默认batch8×1，不是旧batch4×2逐位复现。',
           '', '| 分组 | 种子数 | F1均值±样本SD（%） | 训练秒均值 |', '|---|---:|---:|---:|']
    for k,rs in groups.items():
        fs=[r['metrics']['F1']*100 for r in rs]
        sd=statistics.stdev(fs) if len(fs)>1 else None
        lines.append(f'| {k} | {len(rs)}/3 | {statistics.mean(fs):.3f} ± {sd:.3f} | {statistics.mean(r["timing"]["train_seconds"] for r in rs):.1f} |' if sd is not None
                     else f'| {k} | 1/3 | {fs[0]:.3f} ± 未完成 | {rs[0]["timing"]["train_seconds"]:.1f} |')
    lines+=['','## 配对统计','']
    if with_stats:
        lines+=['固定模型、归一化文本簇配对bootstrap CI；p为双侧配对随机置换。所有47项预设比较统一Holm校正。不能解释为新训练seed的泛化保证。',
                '| 比较（右减左） | ΔF1 pp | 95% CI pp | p | p Holm |','|---|---:|---|---:|---:|']
        for r in stats:
            mark='<sup>†</sup>' if r['p_holm']<.05 else ('<sup>‡</sup>' if r['p_holm']<.1 else '')
            lines.append(f'| {r["comparison"]} | {100*r["delta"]:+.3f}{mark} | {100*r["ci95"][0]:+.3f}, {100*r["ci95"][1]:+.3f} | {r["p_randomization"]:.5f} | {r["p_holm"]:.5f} |')
        lines.append('† Holm调整p<0.05；‡ Holm调整p<0.10。标记表示差异，正负方向由ΔF1决定；没有标记不等于证明无效。')
        lines.append(f'待完成或不能配对：{len(pending)} 项。')
    else:lines.append('尚未重新计算统计；运行07_SUMMARIZE.py。缺少作业时不能标为全部通过。')
    lines+=['','## 解释边界','',
            '- R1相关证据包括适用边界、implicit负结果、随机符号边、SGCN/SiGAT、敏感性、uncapped和TCM类型边控制。',
            '- R2相关证据包括P3逐种子方差、传统基线跨域差异、lambda扫描、理论条件和固定gold评估划分。',
            '- R3相关证据包括第二backbone、配对训练成本，以及implicit几何probe与explicit任务模型的分离。',
            '- 配对检验描述固定模型在评估文本上的不确定性，不能替代大量独立训练seed。',
            '- 统计不显著或结果为负时仍完整保留，不选择性删除。',
            '', '## 缺项 / 无效', '']
    lines += [f'- {j}' for j in missing]+[f'- INVALID {j}: {e}' for j,e in invalid.items()]
    (folder/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(folder/'SUMMARY.json',dict(completed=len(good),total=174,missing=missing,invalid=invalid,
                                  results=list(good.values()),statistics_computed=with_stats))
    with zipfile.ZipFile(folder/'FULL_SUMMARY.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['SUMMARY.md','SUMMARY.json','MANIFEST.json','STATISTICS.json']:
            p=folder/name
            if p.exists():z.write(p,name)
        for j in good:
            for name in ['result.json','config.json','history.json','evaluation.jsonl','dev.jsonl','COMPLETE.json']:
                p=folder/'jobs'/j/name;z.write(p,p.relative_to(folder))
        z.write(ROOT/'REVIEWER_CHECKLIST.md','REVIEWER_CHECKLIST.md')
        for p in sorted((folder/'logs').glob('*.log')):
            if p.stat().st_size<20*1024*1024:z.write(p,p.relative_to(folder))
    print(f'完成并校验 {len(good)}/174；无效 {len(invalid)}。摘要：{folder / "FULL_SUMMARY.zip"}',flush=True)
    return good,missing,invalid

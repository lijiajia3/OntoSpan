"""Immutable reviewer grid, independent of all v7-v12 outputs."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SEEDS=(42,1,2)
ENCODER='hfl/chinese-macbert-base'
SECOND='hfl/chinese-roberta-wwm-ext'


def settings():
    s=json.loads((ROOT/'settings.json').read_text(encoding='utf-8'))
    if not isinstance(s['batch_size'],int) or not isinstance(s['accum'],int) or min(s['batch_size'],s['accum'])<1:
        raise ValueError('batch_size/accum must be positive integers')
    if not 0<float(s['memory_limit_gib'])<=8: raise ValueError('memory_limit_gib must be in (0,8]')
    if not isinstance(s['fast_spans'],bool): raise ValueError('fast_spans must be boolean')
    if not isinstance(s['bootstrap_replicates'],int) or s['bootstrap_replicates']<100: raise ValueError('bootstrap_replicates >=100 required')
    return s


def jobs():
    s=settings();out=[]
    def add(family,domain,frac,mode,seed,**extra):
        j=dict(family=family,domain=domain,frac=frac,mode=mode,seed=seed,
               encoder=ENCODER,epochs=5,lr=2e-5,lmax=10,gnn_depth=2,rho=5,
               lam=.1,gamma=1.,me_weight=1.,perturb='real',train_cap=600,
               dev_cap=100,test_cap=300,batch_size=s['batch_size'],accum=s['accum'],
               memory_limit_gib=s['memory_limit_gib'],fast_spans=s['fast_spans'],
               type_mutex=False,precision='fp32',data_policy='legacy_original_splits_v1')
        j.update(extra)
        j['id']=f'{family}_{domain}_f{frac:g}_{mode}_s{seed}'+(''.join(f'_{k}{v}' for k,v in sorted(extra.items())))
        j['id']=j['id'].replace('/','-').replace('.','p')
        out.append(j)
    for d in ['medical','tcm','law']:
        for f in [.1,.3,1.]:
            for seed in SEEDS:
                for m in ['none','explicit']: add('main',d,f,m,seed)
    for d,f in [('medical',.1),('medical',.3),('medical',1.),('tcm',.1),('law',1.)]:
        for p in ['removed','rewired']:
            for seed in SEEDS: add('p3',d,f,'explicit',seed,perturb=p)
    for d,f in [('medical',.1),('law',1.)]:
        for seed in SEEDS:
            add('random_sign',d,f,'explicit',seed,perturb='random_sign')
            for m in ['none','explicit']: add('backbone',d,f,m,seed,encoder=SECOND)
            add('signed_baseline',d,f,'sgcn',seed)
            add('signed_baseline',d,f,'sigat',seed,gnn_depth=1)
    for l in [0.,.05,.1,.5,1.]:
        for seed in SEEDS: add('lambda','medical',.1,'implicit',seed,lam=l)
    for w in [0.,.25,.5,2.]:
        for seed in SEEDS: add('me_weight','law',1.,'explicit',seed,me_weight=w)
    for d in ['medical','tcm']:
        for seed in SEEDS:
            for m in ['none','explicit']: add('uncapped',d,1.,m,seed,train_cap=0,dev_cap=0,test_cap=0)
    for f in [.1,.3,1.]:
        for seed in SEEDS: add('tcm_onaxis','tcm',f,'explicit',seed,type_mutex=True)
    for m in ['biaffine','global_pointer','bert_crf','w2ner']:
        for seed in SEEDS: add('traditional','medical',.1,m,seed)
    assert len(out)==len({j['id'] for j in out})==174
    key_ids={j['id'] for j in select(out,'key')}
    return [j for j in out if j['id'] in key_ids]+[j for j in out if j['id'] not in key_ids]


def select(js,stage):
    if stage=='all': return list(js)
    if stage=='main': return [j for j in js if j['family']=='main']
    if stage=='review': return [j for j in js if j['family']!='main']
    if stage=='key':
        return [j for j in js if (j['family']=='main' and (j['domain'],j['frac']) in [('medical',.1),('law',1.)])
                or (j['family']=='p3' and j['domain']=='law' and j['perturb']=='removed')]
    raise ValueError('Unknown stage '+stage)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()


def file_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def fingerprints():
    paths=list(ROOT.glob('*.py'))+list((ROOT/'vendor').glob('*.py'))+list((ROOT/'data').glob('*/*.jsonl'))
    paths += [ROOT/n for n in ['settings.json','DESIGN.md','requirements_windows.txt','SOURCE_PROVENANCE.json']]
    return {p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted(paths)}


def manifest():
    import torch
    from huggingface_hub import snapshot_download
    revisions={}
    for name in [ENCODER,SECOND]:
        try:revisions[name]=Path(snapshot_download(name,local_files_only=True)).name
        except Exception:revisions[name]=None
    versions={}
    for name in ['torch','transformers','tokenizers','numpy','safetensors','huggingface-hub','pytorch-crf']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    runtime=dict(python=platform.python_version(),platform=platform.system(),packages=versions,
                 cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 model_revisions=revisions)
    js=jobs()
    for j in js:j['encoder_revision']=revisions[j['encoder']]
    m=dict(version='v16-legacy-b8',jobs=js,fingerprints=fingerprints(),runtime=runtime,
           analysis=dict(bootstrap_replicates=settings()['bootstrap_replicates'],seed=20260911,planned_comparisons=47))
    m['namespace']=digest(m)[:16]
    return m

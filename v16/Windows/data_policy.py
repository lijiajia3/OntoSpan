"""Reproduce the historical split/cap protocol without filtering any rows."""
import json
import random
import unicodedata
from protocol import ROOT,digest


def key(text):
    return ''.join(unicodedata.normalize('NFKC',text).casefold().split())


def read(path):
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            o=json.loads(line);text=o['text'];gold=[tuple(e) for e in o.get('entities',[])]
            if not key(text):raise ValueError('Empty normalized text')
            if any(not isinstance(a,int) or not isinstance(b,int) or not 0<=a<b<=len(text) or not isinstance(t,str) for a,b,t in gold):
                raise ValueError('Invalid gold offsets in '+str(path))
            rows.append((text,gold))
    return rows


def subsample(rows,cap):
    if not cap or len(rows)<=cap:return list(rows)
    return [rows[i] for i in sorted(random.Random(42).sample(range(len(rows)),cap))]


def unchanged_audit(rows):
    return dict(source_rows=len(rows),kept=len(rows),overlap_removed=0,
                conflicting_removed=0,duplicates_removed=0)


def prepare(j):
    folder=ROOT/'data'/j['domain']
    pool,dev_pool,test_pool=[read(folder/(s+'.jsonl')) for s in ['train','dev','test']]
    capped=subsample(pool,j['train_cap'])
    order=list(range(len(capped)));random.Random(j['seed']).shuffle(order)
    train=[capped[i] for i in order[:max(1,int(len(capped)*j['frac']))]]
    dev=subsample(dev_pool,j['dev_cap']);test=subsample(test_pool,j['test_cap'])
    if not dev or not test:raise ValueError('Empty historical dev/evaluation split')
    audit=dict(train_count=len(train),capped_count=len(capped),dev=unchanged_audit(dev),evaluation=unchanged_audit(test),
               train_hash=digest(train),dev_hash=digest(dev),evaluation_hash=digest(test),
               training_unique=len({key(t) for t,g in train}),
               same_boundary_extra_labels=sum(len(set(g))-len({(a,b) for a,b,l in g}) for t,g in train),
               selection='historical protocol: independently cap train/dev/test with seed 42; keep overlap, duplicates and conflicts unchanged',
               evaluation_name='labeled-development-derived split (not official hidden test)' if j['domain']=='medical' else 'processed benchmark evaluation split; already inspected during research')
    # Preserve original type list construction; do not silently change classifier size.
    return train,dev,test,capped+dev,audit

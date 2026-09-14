"""Exact span metrics and fixed-model paired, text-cluster inference."""
import numpy as np
from protocol import digest
from data_policy import key


def score(counts):
    tp,fp,fn=map(float,counts)
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
    return dict(P=p,R=r,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,tp=int(tp),fp=int(fp),fn=int(fn))


def record(text,gold,pred):
    gold=set(map(tuple,gold));pred=set(map(tuple,pred))
    return dict(sample_id=digest(key(text)),text=text,gold=sorted(gold),predicted=sorted(pred),
                tp=len(gold&pred),fp=len(pred-gold),fn=len(gold-pred))


def summarize(records):
    return score([sum(r[k] for r in records) for k in ['tp','fp','fn']])


def per_type(records):
    counts={}
    for r in records:
        gold=set(map(tuple,r['gold']));pred=set(map(tuple,r['predicted']))
        for entities,column in [(gold&pred,0),(pred-gold,1),(gold-pred,2)]:
            for a,b,label in entities:counts.setdefault(label,[0,0,0])[column]+=1
    return {k:score(v) for k,v in sorted(counts.items())}


def _f1(a):
    tp,fp,fn=np.moveaxis(a,-1,0)
    den=2*tp+fp+fn
    return np.divide(2*tp,den,out=np.zeros_like(den,dtype=float),where=den!=0)


def paired(left,right,replicates=2000):
    """left/right: equally ordered seed lists of records. right minus left.

    One bootstrap draw is shared across methods AND seeds. Inference is over
    normalized text clusters conditional on fitted models, not unseen seeds.
    Raw p is two-sided paired cluster randomization, not uncentered bootstrap.
    """
    if not left or len(left)!=len(right):raise ValueError('Missing paired seeds')
    def clustered(group):
        counts={};gold={}
        for r in group:
            k=r['sample_id'];counts.setdefault(k,[0,0,0]);gold.setdefault(k,[])
            for i,v in enumerate(['tp','fp','fn']):counts[k][i]+=r[v]
            gold[k].append(tuple(tuple(e) for e in r['gold']))
        return counts,{k:sorted(v) for k,v in gold.items()}
    reference_counts,reference_gold=clustered(left[0])
    ids=sorted(reference_counts)
    if not ids:raise ValueError('Empty evaluation IDs')
    def array(groups):
        out=[]
        for group in groups:
            counts,gold=clustered(group)
            if sorted(counts)!=ids:raise ValueError('Unpaired evaluation sets')
            if gold!=reference_gold:raise ValueError('Different paired gold')
            out.append([counts[k] for k in ids])
        return np.array(out,dtype=np.int64)
    a,b=array(left),array(right)
    seed_delta=_f1(b.sum(1))-_f1(a.sum(1));observed=float(seed_delta.mean())
    rng=np.random.default_rng(20260911);boots=[];extreme=0
    for _ in range(replicates):
        ix=rng.integers(len(ids),size=len(ids))
        boots.append(float((_f1(b[:,ix].sum(1))-_f1(a[:,ix].sum(1))).mean()))
        swap=rng.integers(2,size=len(ids)).astype(bool)[None,:,None]
        aa=np.where(swap,b,a);bb=np.where(swap,a,b)
        perm=float((_f1(bb.sum(1))-_f1(aa.sum(1))).mean())
        extreme+=abs(perm)>=abs(observed)-1e-15
    return dict(delta=observed,ci95=np.quantile(boots,[.025,.975]).tolist(),
                p_randomization=(extreme+1)/(replicates+1),per_seed_delta=seed_delta.tolist(),
                clusters=len(ids),evaluation_rows=len(left[0]),seeds=len(left),replicates=replicates,
                scope='fixed fitted models; normalized-text clusters; not independent training-seed inference')


def holm(rows,family_size=None):
    """Use total prespecified family size even when some comparisons are absent."""
    size=family_size or len(rows);running=0.
    for rank,row in enumerate(sorted(rows,key=lambda r:r['p_randomization'])):
        running=max(running,min(1.,row['p_randomization']*(size-rank)))
        row['p_holm']=running

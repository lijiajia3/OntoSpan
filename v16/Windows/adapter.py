"""Extensions around an unmodified historical implementation."""
import copy
import random
import sys
from pathlib import Path
from types import MethodType
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import ontospan_experiment as legacy
import baselines
from signed_baselines import SignedGCN,SiGAT


def indices(model,n,device):
    cache=model.__dict__.setdefault('_span_index_cache',{})
    key=(n,str(device))
    if key not in cache:
        a=torch.arange(n,device=device)[:,None].expand(n,model.lmax)
        w=torch.arange(1,model.lmax+1,device=device)[None,:].expand(n,model.lmax)
        b=a+w-1;valid=b<n
        cache[key]=(a[valid],b[valid],w[valid])
    return cache[key]


def features(model,reps,a,b,w):
    prefix=torch.cat([reps.new_zeros((1,reps.shape[1])),reps.cumsum(0)],0)
    mean=(prefix[b+1]-prefix[a])/w[:,None].to(reps.dtype)
    return model.span_proj(torch.cat([reps[a],reps[b],mean,model.width_emb.weight[w]],-1))


def fast_span_reps(model,reps,offs):
    a,b,w=indices(model,len(reps),reps.device)
    spans=[(offs[i][0],offs[k][1]) for i,k in zip(a.tolist(),b.tolist())]
    return features(model,reps,a,b,w),spans


def enable_fast(model):
    model.span_reps=MethodType(fast_span_reps,model)
    model._v16_fast=True


def weighted_forward(self,x,pos_nbr,neg_nbr):
    for _ in range(self.depth):
        Wx=self.W(x)
        pos=self._aggregate(x,Wx,pos_nbr,self.a_pos)
        neg=self._aggregate(x,Wx,neg_nbr,self.a_neg)
        agg=pos-self.me_weight*neg
        g=torch.sigmoid(self.gate(torch.cat([x,agg],1)))
        x=self.ln(g*agg+(1-g)*x)
    return x


def edges(model):
    def rows(adj):return sorted({(i,k) for i,ns in enumerate(adj) for k in ns if i<k})
    return dict(positive=rows(model.pos_nbr),negative=rows(model.neg_nbr),
                node_names=model.types+model.anchor_names if model.anchor is not None else model.types,
                me_weight=getattr(model.gnn,'me_weight',1.),
                graph_class=type(model.gnn).__name__ if model.gnn is not None else None,
                graph_depth=getattr(model.gnn,'depth',1) if model.gnn is not None else 0)


def make_model(j,types,ontology,encoder):
    onto=copy.deepcopy(ontology)
    if j['type_mutex']: onto['tmutex'].append(('病因','症状'))
    if j['perturb'] in ('removed','rewired'):
        onto=legacy.perturb_ontology(onto,j['perturb'],j['seed'])
    if j['mode'] in ('biaffine','global_pointer','bert_crf','w2ner'):
        return baselines.build_model(j['mode'],encoder,types,j['lmax'])
    mode='explicit' if j['mode'] in ('sgcn','sigat') else j['mode']
    model=legacy.OntoSpan(encoder,types,onto,mode=mode,lmax=j['lmax'],gnn_depth=j['gnn_depth'])
    if j['mode']=='sgcn':model.gnn=SignedGCN(encoder.hidden,depth=j['gnn_depth'])
    elif j['mode']=='sigat':model.gnn=SiGAT(encoder.hidden)
    elif mode=='explicit':
        model.gnn.me_weight=j['me_weight']
        model.gnn.forward=MethodType(weighted_forward,model.gnn)
    if j['perturb']=='random_sign':
        graph=edges(model);pos=graph['positive'];neg=graph['negative'];all_edges=pos+neg
        if len(set(all_edges))!=len(all_edges):raise ValueError('Parallel/opposite duplicate edge prevents fixed-topology random-sign control')
        rng=random.Random(j['seed']);original=set(neg)
        for _ in range(1000):
            rng.shuffle(all_edges)
            if set(all_edges[:len(neg)])!=original:break
        else:raise ValueError('No non-identity sign assignment possible')
        model.pos_nbr=[[] for _ in model.pos_nbr];model.neg_nbr=[[] for _ in model.neg_nbr]
        for es,adj in [(all_edges[:len(neg)],model.neg_nbr),(all_edges[len(neg):],model.pos_nbr)]:
            for a,b in es:adj[a].append(b);adj[b].append(a)
    if j['fast_spans']: enable_fast(model)
    return model


def train_epoch(model,data,optim,device,rho=5,lam=.1,gamma=1.,bs=8,accum=1):
    if not getattr(model,'_v16_fast',False):
        return legacy.run_epoch(model,data,optim,device,rho,lam,gamma,bs,accum=accum)
    model.train();cew=torch.ones(model.K,device=device);cew[0]=.1
    order=list(range(len(data)));random.shuffle(order)
    n_batches=(len(data)+bs-1)//bs;total=0.;pending=0;optim.zero_grad()
    for bi,k in enumerate(range(0,len(data),bs)):
        batch=[data[i] for i in order[k:k+bs]]
        reps_list,offs_list=model.encoder([t for t,g in batch]);treps=model.type_representations()
        logits_all=[];ys=[]
        for (text,gold),reps,offs in zip(batch,reps_list,offs_list):
            a,b,w=indices(model,len(reps),device)
            if not len(a):continue
            spans=[(offs[i][0],offs[q][1]) for i,q in zip(a.tolist(),b.tolist())]
            y=legacy.gold_token_spans(spans,gold,model.t2i).to(device)
            pos=(y!=0).nonzero(as_tuple=True)[0];neg=(y==0).nonzero(as_tuple=True)[0]
            keep=neg[torch.randperm(len(neg),device=device)[:max(1,int(rho*max(1,len(pos))))]]
            sel=torch.cat([pos,keep])
            logits_all.append(model.score_with(features(model,reps,a[sel],b[sel],w[sel]),treps));ys.append(y[sel])
        if logits_all:
            loss=F.cross_entropy(torch.cat(logits_all),torch.cat(ys),weight=cew)
            if model.mode=='implicit':loss=loss+lam*model.ontology_loss(gamma=gamma)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            total+=float(loss.detach());(loss/accum).backward();pending+=1
        if ((bi+1)%accum==0 or bi+1==n_batches) and pending:
            norms=[p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
            if norms and not torch.isfinite(torch.stack(norms)).all():raise FloatingPointError('Nonfinite gradients')
            optim.step();optim.zero_grad();pending=0
    return total

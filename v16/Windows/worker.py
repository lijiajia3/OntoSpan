"""Isolated training process, historical learning objective + complete artifacts."""
import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from collections import Counter
import torch
import adapter
from adapter import legacy,baselines
from data_policy import prepare
from metrics import record,summarize,per_type
from protocol import file_hash,digest


def now():return time.strftime('%Y-%m-%d %H:%M:%S')


def console_utf8():
    for stream in [sys.stdout,sys.stderr]:
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')


def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def geometry(model):
    if not hasattr(model,'type_emb'):return {}
    was=model.training;model.eval()
    with torch.no_grad():
        raw=torch.nn.functional.normalize(model.type_emb.weight,dim=1)
        h=torch.nn.functional.normalize(model.type_representations(),dim=1)
        pairs=[dict(pair=[model.types[a],model.types[b]],raw_cos=float((raw[a]*raw[b]).sum()),
                    effective_cos=float((h[a]*h[b]).sum())) for a,b in model.type_mutex]
        reg=float(model.ontology_loss())
    model.train(was)
    return dict(type_pairs=pairs,implicit_regularizer=reg,
                interpretation='prototype geometry, not proof of separation in task decision space')


def support(model,rows,method):
    """No model forward/RNG consumption: audit original tokenizer/span support."""
    labels=Counter();total=reachable=nested=0
    for start in range(0,len(rows),256):
        batch=rows[start:start+256]
        if hasattr(model.encoder,'tok'):
            offsets=model.encoder.tok([t for t,g in batch],truncation=True,max_length=160,
                                      return_offsets_mapping=True)['offset_mapping']
        else:offsets=[[(i,i+1) for i in range(len(t))] for t,g in batch]
        for (text,gold),offs in zip(batch,offsets):
            offs=[(a,b) for a,b in offs if b>a]
            if not offs:raise ValueError('Tokenizer produces no content tokens; refusing silent CRF/NER failure')
            starts={};ends={}
            for i,(a,b) in enumerate(offs):starts.setdefault(a,[]).append(i);ends.setdefault(b,[]).append(i)
            gs=set(gold);total+=len(gs);labels.update(l for a,b,l in gs)
            width=None if method in ('bert_crf','w2ner') else model.lmax
            reachable+=sum(any(j>=i and (width is None or j-i<width) for i in starts.get(a,[]) for j in ends.get(b,[])) for a,b,l in gs)
            nested+=sum(any((a,b)!=(c,d) and a<=c<d<=b for c,d,_ in gs) for a,b,l in gs)
    return dict(sentences=len(rows),gold=total,candidate_reachable=reachable,unreachable=total-reachable,
                label_counts=dict(labels),entities_per_sentence=total/max(1,len(rows)),
                gold_spans_containing_another=nested)


@torch.no_grad()
def collect(model,rows,device,bs,method):
    model.eval();records=[];reachable=gold_total=0
    for start in range(0,len(rows),bs):
        batch=rows[start:start+bs]
        if method=='bert_crf':
            emissions,mask,offs_list,lens=model._emit([t for t,g in batch]);mask[:,0]=True
            if not torch.isfinite(emissions).all():raise FloatingPointError('Nonfinite CRF emissions')
            paths=model.crf.decode(emissions,mask=mask)
            reps_list=[None]*len(batch)
        else:
            reps_list,offs_list=model.encoder([t for t,g in batch])
            treps=model.type_representations() if method!='w2ner' else None
        for bi,((text,gold),reps,offs) in enumerate(zip(batch,reps_list,offs_list)):
            if method=='bert_crf':pred=model.decode_spans(paths[bi][:len(offs)],offs)
            elif method=='w2ner':
                pred=set()
                if len(reps):
                    logits=model._grid_logits(reps)
                    if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite W2NER logits')
                    pred=model.decode_spans(logits.argmax(-1),offs)
            else:
                sr,spans=model.span_reps(reps,offs);pred=set()
                if len(sr):
                    logits=model.score_with(sr,treps)
                    if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite evaluation scores')
                    for (a,b),label in zip(spans,logits.argmax(1).tolist()):
                        if label:pred.add((a,b,model.types[label]))
                candidate=set(spans)
                reachable+=sum((a,b) in candidate for a,b,l in set(gold))
            gold_total+=len(set(gold));records.append(record(text,gold,pred))
    audit=dict(gold=gold_total,candidate_reachable=reachable if method not in ('bert_crf','w2ner') else None,
               unreachable_policy='all gold remains in denominator, including truncation/width misses')
    return records,audit


def train(j,out,device='cuda',encoder_factory=None,data=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    if device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no automatic CPU full run')
        capacity=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.,j['memory_limit_gib']*2**30/capacity))
        torch.cuda.reset_peak_memory_stats()
    train_rows,dev_rows,eval_rows,type_rows,audit=data if data is not None else prepare(j)
    onto=legacy.load_ontology(j['domain']);types=legacy.build_types(type_rows,onto)
    legacy.set_seed(j['seed'])
    # Historical train_one consumes this Python RNG shuffle before model creation.
    discarded=list(range(audit['capped_count']));random.shuffle(discarded)
    if encoder_factory is None:
        from huggingface_hub import snapshot_download
        if not j.get('encoder_revision'):raise ValueError('Missing pinned model revision; check environment before running')
        source=snapshot_download(j['encoder'],revision=j['encoder_revision'],local_files_only=True)
        encoder_factory=lambda:legacy.HFEncoder(source)
    encoder=encoder_factory().to(device)
    model=adapter.make_model(j,types,onto,encoder).to(device)
    graph=adapter.edges(model) if hasattr(model,'type_emb') else {}
    initial=geometry(model)
    support_audit={name:support(model,rows,j['mode']) for name,rows in [('train',train_rows),('dev',dev_rows),('evaluation',eval_rows)]}
    optim=torch.optim.AdamW(model.parameters(),lr=j['lr'])
    startup=time.perf_counter()-started
    print(f'[{now()}] {j["id"]}: train={len(train_rows)} dev={len(dev_rows)} eval={len(eval_rows)}; batch={j["batch_size"]} accum={j["accum"]}; startup={startup:.1f}s',flush=True)
    best=-1.;best_epoch=0;history=[];best_path=out/'best.pt'
    config=dict(job=j,types=types,ontology=onto,graph=graph,splits=audit,support=support_audit,
                selection='historical dev split F1; latest epoch on ties, historical rule',
                python=platform.python_version(),torch=torch.__version__,device=device,
                gpu=torch.cuda.get_device_name(0) if device=='cuda' else None)
    dump(out/'config.json',config)
    for ep in range(1,j['epochs']+1):
        before=time.perf_counter()
        if j['mode']=='bert_crf':
            loss=baselines.train_crf(model,train_rows,optim,device,bs=j['batch_size'],accum=j['accum'])
        elif j['mode']=='w2ner':
            loss=baselines.train_w2ner(model,train_rows,optim,device,bs=j['batch_size'],accum=j['accum'],rho=j['rho'])
        else:
            loss=adapter.train_epoch(model,train_rows,optim,device,bs=j['batch_size'],accum=j['accum'],rho=j['rho'],lam=j['lam'],gamma=j['gamma'])
        if not math.isfinite(loss):raise FloatingPointError('Nonfinite epoch loss')
        if device=='cuda':torch.cuda.synchronize()
        train_seconds=time.perf_counter()-before;before=time.perf_counter()
        records,_=collect(model,dev_rows,device,j['batch_size'],j['mode']);ev=summarize(records)
        eval_seconds=time.perf_counter()-before
        entry=dict(epoch=ep,loss=loss,dev=ev,train_seconds=train_seconds,dev_seconds=eval_seconds,
                   optimizer_steps=math.ceil(math.ceil(len(train_rows)/j['batch_size'])/j['accum']))
        history.append(entry)
        if ev['F1']>=best:
            best=ev['F1'];best_epoch=ep
            torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},config=config,epoch=ep),best_path)
        dump(out/'history.json',history)
        print(f'[{now()}] epoch {ep}/{j["epochs"]}: loss={loss:.5f}; dev F1={ev["F1"]:.4f}; train={train_seconds:.1f}s eval={eval_seconds:.1f}s total={time.perf_counter()-started:.1f}s',flush=True)
    saved=torch.load(best_path,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict']);del saved
    best_geometry=geometry(model)
    eval_records,coverage=collect(model,eval_rows,device,j['batch_size'],j['mode'])
    dev_records,_=collect(model,dev_rows,device,j['batch_size'],j['mode'])
    if abs(summarize(dev_records)['F1']-best)>1e-10:raise ValueError('Best checkpoint replay mismatch')
    for name,records in [('evaluation',eval_records),('dev',dev_records)]:
        with (out/(name+'.jsonl')).open('w',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    result=dict(job=j,metrics=summarize(eval_records),per_type=per_type(eval_records),best_epoch=best_epoch,dev_best=best,
                splits=audit,support=support_audit,coverage=coverage,graph=graph,geometry=dict(initial=initial,best=best_geometry),
                history=history,timing=dict(startup_seconds=startup,total_seconds=time.perf_counter()-started,
                    train_seconds=sum(h['train_seconds'] for h in history),dev_seconds=sum(h['dev_seconds'] for h in history)),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if device=='cuda' else None)
    dump(out/'result.json',result)
    artifact={p.name:file_hash(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('COMPLETE.json','failure.json') and not p.name.endswith('.tmp')}
    dump(out/'COMPLETE.json',dict(job_digest=digest(j),files=artifact))
    print(f'[{now()}] complete F1={result["metrics"]["F1"]:.4f}; total={result["timing"]["total_seconds"]:.1f}s',flush=True)
    return result


def main():
    console_utf8()
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();j=json.loads(Path(a.job).read_text(encoding='utf-8'))
    train(j,a.out)


if __name__=='__main__':main()

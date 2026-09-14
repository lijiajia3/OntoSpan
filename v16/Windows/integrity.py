"""Validate data-bearing artifacts before skipping a completed training job."""
import json
import math
from pathlib import Path
from metrics import record,summarize
from protocol import digest,file_hash

REQUIRED={'best.pt','config.json','history.json','dev.jsonl','evaluation.jsonl','result.json'}
_HASH_CACHE={}


def cached_hash(path):
    s=path.stat();key=(str(path.resolve()),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if key not in _HASH_CACHE:_HASH_CACHE[key]=file_hash(path)
    return _HASH_CACHE[key]


def records(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]


def validate(folder,j):
    folder=Path(folder)
    try:
        marker=json.loads((folder/'COMPLETE.json').read_text(encoding='utf-8'))
        if marker['job_digest']!=digest(j):raise ValueError('different job configuration')
        if set(marker['files'])!=REQUIRED:raise ValueError('missing/unexpected artifact list')
        for name,expected in marker['files'].items():
            p=folder/name
            if p.is_symlink() or not p.is_file() or cached_hash(p)!=expected:raise ValueError('missing/changed '+name)
        result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        if result['job']!=j or config['job']!=j:raise ValueError('result/config job mismatch')
        evaluation=records(folder/'evaluation.jsonl')
        if not evaluation:raise ValueError('empty evaluation records')
        for r in evaluation:
            recalculated=record(r['text'],r['gold'],r['predicted'])
            if any(r[k]!=recalculated[k] for k in ['sample_id','tp','fp','fn']):raise ValueError('prediction counts mismatch')
        measured=summarize(evaluation)
        if any(not math.isclose(measured[k],result['metrics'][k],abs_tol=1e-10) for k in measured):raise ValueError('F1/count mismatch')
        if len(evaluation)!=result['support']['evaluation']['sentences']:raise ValueError('evaluation sentence count mismatch')
        if measured['tp']+measured['fn']!=result['support']['evaluation']['gold']:raise ValueError('evaluation gold count mismatch')
        dev=records(folder/'dev.jsonl')
        if not dev or not math.isclose(summarize(dev)['F1'],result['dev_best'],abs_tol=1e-10):raise ValueError('saved dev/best mismatch')
        history=json.loads((folder/'history.json').read_text(encoding='utf-8'))
        if len(history)!=j['epochs'] or [h['epoch'] for h in history]!=list(range(1,j['epochs']+1)):raise ValueError('incomplete epoch history')
        if any(not math.isfinite(h['loss']) for h in history):raise ValueError('nonfinite training history')
        chosen=max(history,key=lambda h:(h['dev']['F1'],h['epoch']))
        if chosen['epoch']!=result['best_epoch'] or not math.isclose(chosen['dev']['F1'],result['dev_best'],abs_tol=1e-10):raise ValueError('checkpoint selection mismatch')
        if result['best_epoch'] not in range(1,j['epochs']+1):raise ValueError('invalid best epoch')
        return result,None
    except (OSError,ValueError,KeyError,TypeError) as e:return None,str(e)

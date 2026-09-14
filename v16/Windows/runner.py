"""Sequential Windows execution; immutable namespace, validated resume."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from protocol import ROOT,manifest,select,digest,fingerprints
from worker import dump,now,console_utf8
from integrity import validate


def preflight(download=False):
    console_utf8()
    import torch
    import numpy
    import transformers
    import torchcrf
    from transformers import AutoConfig,AutoTokenizer
    from huggingface_hub import snapshot_download
    from protocol import ENCODER,SECOND
    print(f'Python {sys.version.split()[0]} | torch {torch.__version__} | transformers {transformers.__version__} | numpy {numpy.__version__}',flush=True)
    if not torch.cuda.is_available():raise RuntimeError('未检测到CUDA；不自动改用CPU运行正式实验。先安装匹配驱动的CUDA PyTorch。')
    print('GPU:',torch.cuda.get_device_name(0),'capacity GiB:',torch.cuda.get_device_properties(0).total_memory/2**30,flush=True)
    for name in [ENCODER,SECOND]:
        print('Checking model files:',name,flush=True)
        cached=Path(snapshot_download(name,local_files_only=not download,allow_patterns=['*.json','*.txt','*.model','*.safetensors','pytorch_model*.bin']))
        if not any((cached/n).is_file() and (cached/n).stat().st_size>0 for n in ['model.safetensors','pytorch_model.bin']):
            indexes=[cached/n for n in ['model.safetensors.index.json','pytorch_model.bin.index.json'] if (cached/n).is_file()]
            if not indexes:raise FileNotFoundError('模型权重未缓存：'+name)
            weight_map=json.loads(indexes[0].read_text(encoding='utf-8'))['weight_map']
            if not weight_map or not all((cached/n).is_file() and (cached/n).stat().st_size>0 for n in set(weight_map.values())):
                raise FileNotFoundError('分片权重不完整：'+name)
        AutoConfig.from_pretrained(name,local_files_only=True)
        AutoTokenizer.from_pretrained(name,use_fast=True,local_files_only=True)
    for domain in ['medical','tcm','law']:
        for split in ['train','dev','test']:
            if not (ROOT/'data'/domain/(split+'.jsonl')).is_file():raise FileNotFoundError(domain+'/'+split)
    print('环境检查完成；没有训练，也没有自动冒烟测试。',flush=True)


def run(stage='all',download=False):
    from report import refresh
    preflight(download)
    m=manifest();folder=ROOT/'outputs'/('v16-'+m['namespace']);folder.mkdir(parents=True,exist_ok=True)
    if any(not j['encoder_revision'] for j in m['jobs']):raise RuntimeError('Cannot freeze model revisions')
    lock=folder/'RUNNING.lock'
    try:handle=lock.open('x',encoding='utf-8')
    except FileExistsError:raise RuntimeError('本输出目录已有运行锁。确认另一进程已退出后，手动删除 '+str(lock))
    with handle:handle.write(str(os.getpid()))
    started=time.perf_counter();completed_times=[]
    try:
        dump(folder/'MANIFEST.json',m)
        selected=select(m['jobs'],stage)
        for i,j in enumerate(selected,1):
            if fingerprints()!=m['fingerprints']:raise RuntimeError('运行期间代码/数据/配置改变，请重新启动以生成新命名空间')
            out=folder/'jobs'/j['id'];result,error=validate(out,j)
            if result:
                print(f'[{i}/{len(selected)}] 已校验，跳过 {j["id"]}',flush=True);continue
            if out.exists():
                # Never overwrite failed/invalid artifacts; keep a recoverable copy.
                backup=folder/'incomplete'/f'{j["id"]}-{time.time_ns()}'
                backup.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(out),str(backup))
            if shutil.disk_usage(ROOT).free<4*2**30:raise RuntimeError('剩余磁盘不足4GiB，停止新任务')
            (folder/'requests').mkdir(exist_ok=True);request=folder/'requests'/(j['id']+'.json');dump(request,j)
            (folder/'logs').mkdir(exist_ok=True)
            print(f'[{now()}] [{i}/{len(selected)}] running {j["id"]}',flush=True)
            env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUTF8='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            before=time.perf_counter()
            with (folder/'logs'/(j['id']+'.log')).open('w',encoding='utf-8') as log:
                proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'worker.py'),'--job',str(request),'--out',str(out)],
                                      cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                try:
                    for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                    rc=proc.wait()
                except BaseException:
                    proc.terminate();proc.wait();raise
            elapsed=time.perf_counter()-before
            result,error=validate(out,j)
            if rc or not result:
                refresh(m,folder)
                raise RuntimeError(f'任务失败，未改batch/精度：{j["id"]}; rc={rc}; {error}')
            completed_times.append(elapsed)
            print(f'[{now()}] job={elapsed/60:.2f}min；本阶段已耗时={(time.perf_counter()-started)/60:.2f}min；本阶段剩余配置={len(selected)-i}（各组数据量不同，不用单一均速承诺总时长）',flush=True)
            refresh(m,folder)
        refresh(m,folder,with_stats=True)
        print('本阶段技术校验完成；无F1胜出门槛。论文与逐条回复仍需依据结果完成。',flush=True)
    finally:
        lock.unlink(missing_ok=True)


def main(default='all'):
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','key','main','review'],default=default)
    ap.add_argument('--download',action='store_true',help='明确允许下载模型；默认只用本地缓存')
    a=ap.parse_args();run(a.stage,a.download)


if __name__=='__main__':main()

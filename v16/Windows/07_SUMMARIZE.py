"""Verify saved artifacts and recompute paired statistics. Never retrains."""
import argparse
import json
from pathlib import Path


if __name__=='__main__':
    from windows_launcher import ensure_runtime
    ensure_runtime()
    from protocol import ROOT,manifest
    from report import refresh
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path)
    a=ap.parse_args()
    if a.output:
        folder=a.output.resolve();m=json.loads((folder/'MANIFEST.json').read_text(encoding='utf-8'))
    else:
        m=manifest();folder=ROOT/'outputs'/('v16-'+m['namespace'])
    if not (folder/'MANIFEST.json').exists():raise SystemExit('没有此配置的结果。配置修改后请用 --output 指向旧输出目录。')
    refresh(m,folder,with_stats=True)

#!/usr/bin/env bash
# =====================================================================================
# run_paper_cloud.sh —— OntoSpan 论文主实验「硬版」云端 GPU 命令(多 seed + 全量 + 8 epochs)
# =====================================================================================
# 用 run_grid_safe.py:每个配置一个独立子进程(本机 MPS 防累积 OOM;GPU 上虽无此问题,
# 隔离亦无害,且复用同一聚合格式 → 直接喂 fill_tables.py / make_figures.py)。
#
# 前置:① 一块 CUDA GPU(A100/H100/4090 皆可);② torch(cuda 版)+ transformers + numpy;
#       ③ 有网(首跑自动下载 hfl/chinese-macbert-base)或已把模型缓存进 ~/.cache/huggingface。
# 用法:bash run_code/run_paper_cloud.sh
#       (若环境只有 python3,把下面的 `python` 换成 `python3`)
# -------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."          # 切到项目根(data/ 与 run_code/results/ 在此)

PY=python                         # 视环境改成 python3

# ===== 主结果:2 域 × 3 方法 × 低资源比例 × 3 seed,全量数据,AMP fp16 =====
$PY -u run_code/run_grid_safe.py \
  --domains tcm,medical --modes none,implicit,explicit \
  --fracs 0.1,0.3,1.0 --seeds 42,1,2 \
  --epochs 8 --batch-size 16 --accum 1 --gnn-depth 2 \
  --train-cap 0 --dev-cap 500 --test-cap 0 \
  --device cuda --amp

# 产物:run_code/results/{tcm,medical}_paper.json(每 cell 3 个 seed 的 run 列表)。
# 出图表:
#   python run_code/fill_tables.py        # §6 Table 3/4(现取 3 seed 均值)
#   python run_code/make_figures.py       # Fig3 低资源曲线 / Fig4 每类 F1
#   python run_code/make_schematics.py    # Fig1 架构 / Fig2 有符号图(与数据无关)

# ===== 可选:随机字符遮挡增强消融(验证三项「涨点」之一),单列另存 =====
# $PY -u run_code/run_grid_safe.py --domains tcm,medical --modes explicit \
#   --fracs 0.1,1.0 --seeds 42,1,2 --epochs 8 --batch-size 16 --gnn-depth 2 \
#   --train-cap 0 --dev-cap 500 --test-cap 0 --device cuda --amp \
#   --aug-mask 0.1 --save-dir run_code/results_aug

# =====================================================================================
# 注意 / 取舍
# · 算力:全量 CMeEE(18.4k 句)× 3 frac × 3 mode × 3 seed × 8 ep,单卡 A100 约 1–2 天。
#   赶时间可:--seeds 42,1;medicine 用 --train-cap 8000;或按域分两次跑(分卡并行)。
# · best-dev 选模型用 dev;--dev-cap 500 足够且省时,要纯净评测设 --dev-cap 0(全量 dev)。
# · 多 seed 的 mean±std / 配对显著性检验:fill_tables.py 目前只出均值;要 std 与显著性,
#   告诉我,我再补一版聚合(bootstrap 95% CI + 配对检验)。
# · 扰动对照(P3:真互斥 / 随机重连 / 去互斥)目前未实现为 mode——需给引擎(build_datasets
#   的 MUTEX_INST/MUTEX_TYPE)加「重连/移除」开关再接入 run_grid_safe;要做我可代加。
# =====================================================================================

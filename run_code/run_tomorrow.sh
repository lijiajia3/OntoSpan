#!/usr/bin/env bash
# =====================================================================================
# run_tomorrow.sh —— 过夜多 seed 补跑(P3 优先,再主表)。一条命令即可:
#     bash run_code/run_tomorrow.sh
# 完整说明、判读、优先级见项目根 EXPERIMENTS_TODO.md。
# 用已测基础设施 run_grid_safe.py(子进程隔离防 MPS OOM + HF 离线)。P3 约 6h 先出。
# 注:implicit 已证逐字节 = none,主表只跑 none,explicit(省 1/3)。
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # 项目根
PY=python3
COMMON="--epochs 5 --batch-size 4 --accum 2 --gnn-depth 2 --train-cap 600 --dev-cap 100 --test-cap 300"
LOG=run_code/run_tomorrow.log
echo "=== START $(date) ===" | tee "$LOG"

# ---- 优先级1a:P3 多 seed(explicit × real/rewired/removed × {TCM0.1, med1.0} × seeds 42,1,2)----
for PERT in real rewired removed; do
  echo "### P3 ${PERT} | $(date) ###" | tee -a "$LOG"
  $PY -u run_code/run_grid_safe.py --domains tcm     --fracs 0.1 --modes explicit --seeds 42,1,2 \
     --mutex-perturb "$PERT" --save-dir "run_code/results_p3_${PERT}" $COMMON 2>&1 | tee -a "$LOG"
  $PY -u run_code/run_grid_safe.py --domains medical --fracs 1.0 --modes explicit --seeds 42,1,2 \
     --mutex-perturb "$PERT" --save-dir "run_code/results_p3_${PERT}" $COMMON 2>&1 | tee -a "$LOG"
done

# ---- 优先级1b:主表多 seed(none,explicit × 2域 × 3frac × seeds 42,1,2)----
echo "### MAIN TABLE 3-seed | $(date) ###" | tee -a "$LOG"
$PY -u run_code/run_grid_safe.py --domains tcm,medical --modes none,explicit --fracs 0.1,0.3,1.0 --seeds 42,1,2 \
   --save-dir run_code/results_3seed $COMMON 2>&1 | tee -a "$LOG"

echo "=== DONE $(date) ===" | tee -a "$LOG"
echo "下一步:聚合对照 + 灌 §6.3 Table 5 / Table 3-4 mean±range(见 EXPERIMENTS_TODO.md §5-6)。"

#!/usr/bin/env bash
# =====================================================================================
# run_p3_local.sh —— P3 因果对照(本地 MPS,~2h):real / rewired / removed
# =====================================================================================
# 目的:把论文 §6.3 deferred-to-cloud 的扰动对照在本地补上,钉死 explicit 增益的
#       因果来源(是「互斥语义」还是「只是多了 N 条负边」)。零改码——只用已接好的
#       --mutex-perturb 开关 + run_grid_safe.py 子进程隔离(防 MPS 累积 OOM)。
#
# 选 2 个旗舰 cell(单 seed=42,与主结果同采样/同超参):
#   · TCM     frac=0.1  —— 全文最大增益(+7.9),低资源旗舰
#   · medical frac=1.0  —— 最大医疗增益(+5.5),药物类型 +9.5(per-type 故事)
# 每个 cell 跑 explicit × {real, rewired, removed},real 为同码新参照(不依赖旧 results/)。
#
# 产物:run_code/results_p3_{real,rewired,removed}/{tcm,medical}_paper.json
# 判读:对同一 cell,若 rewired/removed 的 explicit-F1 显著 < real 的 explicit-F1,
#       则增益来自互斥语义(P3 成立);若三者相当,则来自结构/容量(P3 证伪)。
# -------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."          # 项目根(data/ 与 run_code/ 在此)

PY=python3
COMMON="--modes explicit --epochs 5 --batch-size 4 --accum 2 --gnn-depth 2 \
        --lam 0.1 --gamma 1.0 --rho 5 --lmax 10 --lr 2e-5 \
        --train-cap 600 --dev-cap 100 --test-cap 300 --seeds 42"

for PERT in real rewired removed; do
  OUT="run_code/results_p3_${PERT}"
  echo ""
  echo "############################ mutex-perturb=${PERT} -> ${OUT} ############################"
  $PY -u run_code/run_grid_safe.py --domains tcm     --fracs 0.1 $COMMON --mutex-perturb "$PERT" --save-dir "$OUT"
  $PY -u run_code/run_grid_safe.py --domains medical --fracs 1.0 $COMMON --mutex-perturb "$PERT" --save-dir "$OUT"
done

echo ""
echo "=== P3 local done. Compare explicit F1 across real/rewired/removed below ==="
$PY - <<'PYEOF'
import json, os
root="run_code"
for dom in ("tcm","medical"):
    print(f"\n[{dom}]  explicit F1 (real vs rewired vs removed)")
    for pert in ("real","rewired","removed"):
        p=os.path.join(root,f"results_p3_{pert}",f"{dom}_paper.json")
        if not os.path.exists(p):
            print(f"  {pert:8s}: (missing)"); continue
        d=json.load(open(p)); res=d["results"]
        for key,runs in res.items():
            f1=runs[0]["F1"]
            print(f"  {pert:8s} {key:14s} F1={f1:.4f}")
PYEOF

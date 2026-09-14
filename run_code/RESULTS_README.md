# 结果目录 & 跑器索引

> **2026-09-14 更新：**返修后的 174/174 配置、47 项统一 Holm 校正统计和新增的
> random-sign、第二 backbone、SGCN/SiGAT、lambda/ME-weight、uncapped、TCM
> type-edge 与训练成本实验，以
> [`result/revision_20260914/`](../result/revision_20260914/) 为当前论文数字真源。
> 下方 2026-06-16 索引保留用于解释早期结果目录，不再覆盖返修结果。

> 2026-06-16 整理。先前 `results/` 与 `results_3seed/` 共用 `{domain}_paper.json` 文件名、
> 且 zip 多次解压产生过 ` 2` 副本目录,导致工具可能读到**过期单 seed** 数据。本文件钉死真源。

## 结果目录:哪个是真源

| 目录 | 内容 | 谁写的 | 状态 |
|---|---|---|---|
| **`results_3seed/`** | 主表:none/explicit × {tcm,medical} × frac{0.1,0.3,1.0} × 3seed,**fp32 depth2 cap600** | `run_fast_4070.py` | ✅ **主表真源** |
| **`results_p3_real/` `_rewired/` `_removed/`** | P3 因果对照(explicit × 三种扰动) | `run_fast_4070.py` | ✅ P3 真源 |
| **`results_baselines/`** | 外部 baseline(`{domain}_baselines.json`:biaffine/GP/bert_crf)+ DeepSeek LLM(`{domain}_llm_deepseek.json`) | `run_baselines_4070.py` / `baseline_llm_deepseek.py` | ✅ baseline 真源 |
| `results_scarce/` | 稀缺档补点(frac 0.05/0.03),若跑 | `run_scarce_4070.py` | ⏳ 可选 |
| `results_A_anchors/` | A 实验(--score-anchors),若为论文正式跑 | (A 决策后定) | ⏳ 可选 |
| `results_OLD_singleseed_20260614/` | **旧·单 seed 主表(6/14)** | 历史 | 🗄️ **归档,勿用**(噪声不可靠,已被 results_3seed 取代) |
| `results_scratch/` | `run_grid_safe.py` 独立 driver 的临时落盘 | 临时 | 🧪 scratch,非真源 |

**铁律:做图/表/论文数字一律读 `results_3seed/`(及 `results_p3_*`、`results_baselines/`)。永远不要读 `results_OLD_singleseed_*`。**
- `make_figures.py` / `fill_tables.py` 默认已改指 `results_3seed/`。
- `collect_results.py` 已**不再**打包旧 `results/`(避免把过期数据带回 Mac 覆盖真源)。

## ⚠️ macOS 解压注意
Finder 解压 zip 遇同名目录会建 `xxx 2/` 副本(而非覆盖)。**正确做法:解压前先删掉 Mac 上的旧同名目录,或用 `unzip -o` 覆盖。** 出现 ` 2/` 副本时,先 `md5` 对比再删冗余(本次已清)。

## 跑器索引(4070/Windows 上跑)

| 脚本 | 作用 | 落盘 |
|---|---|---|
| `run_fast_4070.py` | 主表 + P3 + P1(并行栅格,fp32) | `results_3seed/` `results_p3_*/` `P1_cosine*.json` |
| `run_baselines_4070.py` | 3 个外部 baseline(同 macbert/同设定) | `results_baselines/*_baselines.json` |
| `baseline_llm_deepseek.py` | DeepSeek few-shot LLM baseline(API,不吃 GPU,Mac 可跑) | `results_baselines/*_llm_deepseek.json` |
| `run_A_tcm.py` | A 实验:none/explicit/explicit+anchors 三臂对照(决策用) | `.A_tmp/`(打印 ΔF1 表) |
| `run_scarce_4070.py` | 稀缺档(frac 0.05/0.03)补点,默认 medical | `results_scarce/` |
| `run_4070.py` | 顺序版一键跑器(run_fast 的串行前身) | 同上 |

引擎:`run_code/run_grid_safe.py`(每配置一子进程,防显存累积)、`run_ablation.py`(train_one)、`ontospan_experiment.py`(模型/数据/评测)。
跑完用 `collect_results.py` 打 `ontospan_results.zip` 拷回 Mac。

# v16 Implementation Plan

> For agentic workers: implement task-by-task inline. The requested executing-plans skill is unavailable; use the approved design and explicit tests. No new subagents or git commits are requested.

**Goal:** 独立、可审计的旧方法Windows返修包，默认batch8。
**Architecture:** vendor保存旧源码；adapter只加可审计扩展；worker负责训练/保存；runner负责固定作业及验证；report负责统计和材料。
**Tech Stack:** Python3.11, PyTorch, Transformers, NumPy, pytorch-crf。
**Spec:** DESIGN.md。

## Global Constraints

- 不改旧文件；文本读写UTF-8；默认8×1，旧配置4×2可选。
- 174配置；所有正/零/负结果保留；不使用F1通过门槛。
- 不混入新BCE、残差0.1、margin、字符窗口或校准阈值。

## Task 1: 冻结来源和配置

- [x] 创建vendor旧源码、原数据副本和来源hash；固定174 grid。
- [x] 测试 `len(jobs()) == len({j['id'] for j in jobs()}) == 174`，默认8×1，03关键15属于全表。

## Task 2: 包装旧模型

- [x] adapter提供 `make_model(job, types, ontology, encoder)`、`train_epoch`；旧版本保留用于对拍。
- [x] 同state同seed测试原/快速span输出、loss、梯度，测试负权0/1、随机符号、辅助类型、原CE。

## Task 3: 数据与worker

- [x] `prepare(job)`复现原论文train/dev/test独立固定抽样并返回audit；不去重、不排除重叠或冲突，保留候选不可达计数。
- [x] worker保存最佳权重、每句预测、配置、分割、hash、每轮耗时和训练显存。
- [x] StubEncoder端到端跑none/explicit/implicit/SGCN/SiGAT及四baseline；测试污染隔离。

## Task 4: 运行与汇总

- [x] runner支持阶段、配置变更命名空间、校验后跳过、失败不跳过；01安装与02检查分离。
- [x] report列缺项/坏项/正负结果、配对统计、审稿逐条证据状态，打包轻量摘要不含pt。
- [x] 测试文件缺失/篡改不能成功续跑；统计对称/恒等/配对样本匹配；UTF8 AST扫描。

## Task 5: 交付验证

- [x] 全部CPU测试和语法检查；真实缓存backbone小样本前后向及checkpoint复评（不冒充GPU验收）。
- [x] 生成Windows zip，确认包中无输出/缓存/绝对机器路径依赖；README写清174、batch8和未测Windows GPU。

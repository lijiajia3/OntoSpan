# 实现来源与边界

- 原OntoSpan及四传统baseline：vendor/下为Git f020ba916f8682fcc0430d524a4ec54871fa2b61原文件；SOURCE_PROVENANCE.json记录hash。只由adapter包装，不直接编辑。
- 四传统baseline是原论文仓库实现，不声称是各论文官方完整复现。尤其原W2NER解码采用THW端点输出，保留旧实现以对齐历史表；需在论文说明实现细节，不能借“官方W2NER”名义掩盖适配差异。
- SGCN：Derr et al., Signed Graph Convolutional Networks, ICDM 2018。继承balanced/unbalanced双通道，用投影回到NER原型维度。
- SiGAT：Signed Graph Attention Networks, 2019。6个符号/方向通道加32种三角模体，双向展开无向本体；一个模体块，不把gnn_depth=2描述成两个SiGAT块。signed_baselines.py为任务适配版本，沿用已单独实现的v11模块，不引入v11训练目标。
- SGCN/SiGAT只替换显式图模块，其他NER打分、CE、训练预算一致；记录初始化，不声称不同架构训练随机轨迹相同。
- 随机符号在固定节点、拓扑与负边数量下进行非恒等重排；使用独立Python RNG。原rewired算法仍是旧算法，可能保留部分真实边；报告实际边而非假定完全随机无重合。
- 原P3 removed在实例域删除锚点（旧实现行为），有架构混杂；新random-sign是更严格固定拓扑对照。不能把所有P3差值唯一归因于语义。
- type_mutex TCM控制只增加病因/症状边，保留实例层，遵循Git旧入口。历史JSON缺少注入参数完整记录，v16明确记录，不承诺旧TCM人工对照逐位重现。
- 批大小8×1与4×2不是严格等价：旧weighted CE每microbatch单独归一，dropout/RNG也不同。
- 安装命令依据PyTorch官方历史版本页：https://pytorch.org/get-started/previous-versions/ 。01恢复v12式本地.venv管理，默认在本包.venv安装2.9.1 CUDA12.8轮子；不向系统Python或旧版本.venv安装，驱动仍需兼容。02–08统一使用本包环境。

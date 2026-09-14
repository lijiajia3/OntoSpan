# v16 Windows：旧方法 + 审稿补充

默认 `batch_size=8, accum=1, fp32`，PyTorch进程分配上限8GiB。旧方法核心保留，不是v12 margin版。不要与v11/v12结果拼表。完整174配置；先关键15，再剩余。当前没有自动把w=0与removed合并，明确保留174，不声称已完成171项去重。

## 启动

在Windows解压到普通本地目录，用Python3.11运行（VS Code右上角运行按钮或终端均可）。不要在zip内部运行。

1. 直接运行 `python 01_SETUP_WINDOWS.py`：像v12一样自动创建本目录 `.venv`，在里面安装CUDA版PyTorch和其他依赖，并补齐两个backbone的缓存、检查环境。无需手动激活，也无需再加 `--install-torch`。不修改系统Python和旧v12环境。
2. 02–08会自动切换到同一个 `.venv`，即使VS Code运行按钮选的是系统Python，也不会直接在缺少torch的系统环境中加载训练代码。安装后不要移动/复制 `.venv`；需要换目录时在新目录重新运行01。
3. `python 02_CHECK_ENVIRONMENT.py`：只检查CUDA、依赖、本地模型文件和数据，不训练、不自动冒烟。
4. `python 03_KEY_15.py`：医疗10% none/explicit；法律100% none/explicit/removed，各3seed。这15项包含在174中。
   跑完显示15/174是正常进度，其余未运行不是这15项失败；没有“必须F1涨分才通过”的门槛。
5. `python 04_MAIN_54.py`：补完原主表，已完成项校验后跳过。
6. `python 05_REVIEW_120.py`：只运行主表之外的审稿补充，已有关键15中的3项删边会跳过。
7. 或直接 `python 06_ALL_174.py`：全部按同一配置续跑。
8. `python 07_SUMMARIZE.py`：不重训，重新校验/统计并生成 `FULL_SUMMARY.zip`。回传这个zip即可，模型pt留在本机。

01默认允许安装依赖和下载缺失模型；模型已缓存时复用。`--offline-models` 可禁止01下载模型（依赖安装仍需联网）；原 `--install-torch --download` 命令仍兼容。02和训练入口默认离线；缺模型时重新运行01，或训练入口明确加`--download`。失败/显存不足时停止，不自动降batch、不自动FP16。可选 `08_TESTS_ONLY.py` 是开发者CPU测试，02从不调用它。

本次启动修复只涉及环境管理和编号入口，模型、数据、174项配置及batch8均不变。请解压到新文件夹，不要覆盖正在运行的旧目录。源码指纹改变会生成新的输出命名空间；若先前已有v16结果，保留原目录，不要直接搬来冒充本包已完成项。

## 哪个文件改batch

只改 `settings.json`。当前8×1。需要忠实检查旧批处理设置时改为4×2。**有效batch都为8不代表严格相同训练轨迹**：原weighted CE每microbatch独立平均，dropout也不同。修改配置/源码/数据会产生新输出命名空间，不能混用旧结果。不建议一边运行一边编辑。

## 与原论文保持/变化

保留：原SignedGAT（包括孤立节点变换）、softmax CE、背景权重0.1、rho5、argmax、17类+背景等原类型构造、160 token截断、10 token候选、5epochs、所有参数lr2e-5。没有BCE、0.1图残差、96字符窗口、分组新LR或校准阈值。

仅模型工程优化：同顺序枚举候选、先随机负采样再投影、前缀和计算span均值。浮点求和顺序会变化，不保证bitwise一致。可将fast_spans设false回到原run_epoch。随机采样公式/次数、损失归一化与原实现保留。

数据协议已恢复原论文：train/dev/test分别按原代码用seed 42独立抽到600/100/300；不排除重叠、不去重、不删除冲突行。全量任务使用对应split全部原始行。不另抽校准集。唯一主动保留的批处理变化是默认batch8×1；CMeEE仍须如实说明是标注开发集派生评估，不是官方隐藏test。

原标签同边界多标签覆盖/截断不可达等限制被保留并计数，不偷偷改成另一种任务。TCM人工类型边保留实例层。新增SGCN/SiGAT为适配版本，详METHOD_SOURCES.md。

## 结果与安全续跑

源码、数据、配置、运行库版本、GPU型号和本地预训练模型revision共同确定结果命名空间；升级环境不会无声拼接旧结果。早期结果保留在仓库根目录的 `result/` 历史子目录中，仅供溯源，不计入v16完成数。

每个job保存best.pt、config、每轮history、dev/evaluation逐句预测、图与几何、指标及hash。完整性检查通过才跳过；不完整旧目录移到incomplete中保留。模型权重总量可能超过70GiB，请准备100GiB以上空闲空间；每任务不足4GiB停止。FULL_SUMMARY.zip不含权重。

一个输出目录一次只运行一个入口。异常退出留下RUNNING.lock时，确认没有训练进程后再手动删除该锁。多个入口不会自动并行。修改配置后查旧结果：`python 07_SUMMARIZE.py --output "outputs/旧目录名"`。

统计是47项预设比较，报告差值、paired文本簇bootstrap CI、paired随机置换p和统一Holm校正；不足三seed标未完成，不把seed×句子当独立样本。没有文档ID时文本簇不能完全处理来源文档相关性，论文需说明。

**“技术完成”不是“所有收益成立”，更不是保证录用。** 按REVIEWER_CHECKLIST.md逐条填真实结果和修改页码。没有F1胜出门槛，不会为了涨分自动调参。

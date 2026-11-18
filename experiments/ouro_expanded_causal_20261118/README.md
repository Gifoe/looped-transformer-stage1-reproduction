# Ouro-1.4B 扩展因果实验

基线提交为 `ce0ee7d98a35fb284b9af41ed6014bfe0388a2ef`。实验沿用上轮的 [权重、环境、计分和捕获实现](../ouro_cross_task_20261118/README.md)。所有运行使用同一张 RTX 5090、BF16、原生 SDPA 循环、K1–K4 和 batch size 1。多个独立进程在同一张卡上并行。运行时间是每个进程模型加载后到同步结束的墙钟时间，重叠时段会出现在多个进程的计时中。

## 样本

| 任务 | 主自然队列 | 独立主因果确认 | 额外队列 |
| --- | ---: | ---: | --- |
| 关系 | 240 组 | 120 组 | A、B、实体重命名各 90 组 |
| 派生 PrOntoQA | 180 组 | 120 组 | 角色定位 90 组，简化演绎另有开发、确认和定位，布尔规则只做开发 |
| 算术 | 240 组 | 120 组 | 角色定位 90 组，双加法另有开发、确认和定位 |
| PrOntoQA 发布数据 | 270 条 | — | 三个独立的 90 项文件队列 |
| ARC-Challenge validation | 299 条 | — | 完整唯一 ID |
| HellaSwag validation | 150 条 | — | 与上轮 50 个 ID 不重复 |

每个受控组包含四种因子世界和两个无关扰动世界。按组划分开发与确认，以 BPE 完全对齐作为入组的技术条件。两个相反方向使用相同的最终真值供体。替换后的预测来自供体第一阶段和接收方后段规则的组合。不同结构包括三记录正向查找、四记录逆向叙述、含额外仓库映射的五记录链、源自 PrOntoQA 证明种子的条件演绎、以及加加、减加和数轴往返算术。所有实际模板在 `factory.py`，原始输入清单完整保留。

补充能力队列保留开发期失败。另一项独立能力筛选先固定 600 个生成式逻辑组、210 个加法组和选择规则，再只用四世界原生预测筛选。合格组按固定哈希顺序选 90 组进入因果运行。该估计以自然四世界做对为条件。各队列原始样本、筛选失败和选择规则全部公开在实验目录。

## 证据定义

- A 比较相同 token ID 的箱子词、发生变化的归属人名和整段前缀。重命名队列改写供体的全部人名，保持被编辑箱子词的词元与位置，在接收方比较原名、异名、随机和无关噪声。
- B 在 K1/L5 注入来源变量。原生 attention mask 只切断指定轮次的 source→mapping 或 source→query/options/readout 直接边。对照在同一个 mask 下无注入地运行，以差分估计路由作用。之后在 K1 post-norm 恢复受体的部分区域，或在 K2 post-norm 向阻断分支恢复开放条件的映射或问题区域。
- C 分别干预输入信号、定义中间计算的桥接区域、整个第一阶段前缀。独立的中间问题与最终分类分别输出。主要效应保持全部组，四个自然世界全对的分层是敏感性分析。筛过的独立队列单独标记。
- 组内两个方向先平均，随后按组做 30,000 次成组 Bootstrap。K4 主位点以及 A 的相同 token 主要对比用 98.33% 区间。其余区间为描述性 95%。每个干预保留全候选分数、自然供体和接收方、无关方向、随机对照、自替换和观察到的后续状态差。全分数并列保留为 `-1`。
- `trajectory_witnesses.csv` 的 L2 是区域 token 上均方根的向量差范数。Attention、MLP 是模块输出，物理层是残差更新后的输出，`norm` 是循环末归一化。B 的“恢复”保留其它位置既有状态，解释受限于局部读出干预。

## 复现

用 [上轮 README](../ouro_cross_task_20261118/README.md) 的命令获取相同固定模型和三份源文件。验证环境为 Python 3.12、PyTorch 2.7.1+cu128、Transformers 4.55.4。其他版本见同一服务器上轮的 `logs/pip_freeze.txt`。在仓库根目录运行以下命令，参数采用绝对路径。输出目录须为新目录。

```bash
EXP=experiments/ouro_expanded_causal_20261118
PRIOR=experiments/ouro_cross_task_20261118/semantic_interventions.jsonl
export OURO_PYTHON=/absolute/path/to/venv/bin/python
CUDA_VISIBLE_DEVICES=0 bash "$EXP/reproduce.sh" /absolute/path/to/new/study \
  /absolute/path/to/model /absolute/path/to/sources "$PWD/$PRIOR"
```

用现成逐组数据重算统计、审计和图。

```bash
"$OURO_PYTHON" "$EXP/summarize.py" "$EXP/results"
"$OURO_PYTHON" "$EXP/audit_expanded.py" "$EXP/results"
"$OURO_PYTHON" "$EXP/plots.py" "$EXP/results"
```

`results/cross_task_causal_matrix.csv` 汇总跨任务证据，`results/causal_effects.csv` 含轮次、分层、对比及区间。`results/inputs.jsonl`、阶段 `rows.jsonl`、`results/case_witnesses.json` 与 `results/trajectory_witnesses.csv` 可联查。图为 PDF 和 PNG。源数据来自 [PrOntoQA](https://github.com/asaparov/prontoqa/tree/0a6412b6fddf46324a1cb96e066dd7b3d89b87d6)、[ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453/ARC-Challenge) 和 [HellaSwag](https://huggingface.co/datasets/allenai/hellaswag/tree/55aa56627db098098f50549e650a00be2684f4bf)。PrOntoQA 的零样本选项分数采用“证明结论 vs 其否定”的诊断协议。

运行时服务器只加载了上轮所需的两个底层模块。完整仓库复现入口显式先导入所需的上轮分析模块，以防同名模块被其它旧实验的路径遮蔽。校验用的原始执行版保存在 `provenance/executed_collect.py`，其字节仍与清单绑定。

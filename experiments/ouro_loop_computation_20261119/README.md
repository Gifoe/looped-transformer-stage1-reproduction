# What each additional Ouro loop computes

[中文报告](REPORT_ZH.md) · [文献与代码](LITERATURE.md) · [冻结协议](PROTOCOL.md) · [窗口留出复核](FOLLOWUP_PROTOCOL.md)

本目录从 f525239970b70c7ce28fe9fe91d95859653096cc 开始。使用固定 ByteDance/Ouro-1.4B Base 修订 574fa66cb8bf5abdc979642d01cf2b79b16bfab1，权重 SHA-256 为 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af。复用上游原生 hook 与模型校验。运行使用 RTX 4090、BF16、native SDPA、无 KV cache，权重保持冻结。

## 复现

1. 按[组织实验](../ouro_latent_organization_20261118/README.md)准备修正版组清单，并按[跨任务实验](../ouro_cross_task_20261118/README.md)取得 ARC Challenge validation 与 HellaSwag validation 的 parquet。PrOntoQA 筛选从修正版清单包含的官方证明数据抽取。另取[官方 GSM8K 测试集](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl)，SHA-256 为 3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14。
2. 使用[作者固定版本的 attention.py](https://github.com/HughBlayney/looped-mechanistic-analysis/blob/3429a8d61fdae95d441cb18eb3bd695268d086f4/src/looped_internals/analysis/attention.py)，SHA-256 为 8994f5d6d3f7c5a96a989847a9813cbde371a2e952df3399c70059dee725a1c1。参考复现直接调用其中三个注意力指标。运行环境见[依赖清单](logs/dependencies.txt)。
3. 在有一张 CUDA BF16 GPU 的 Linux 上运行以下命令。输出目录必须是新目录。

```bash
export OURO_PYTHON=/absolute/path/to/venv/bin/python
bash experiments/ouro_loop_computation_20261119/reproduce.sh \
  /absolute/path/to/model-snapshot \
  /absolute/path/to/public-parquet-directory \
  /absolute/path/to/corrected-parent-groups.jsonl \
  /absolute/path/to/gsm8k-test.jsonl \
  /absolute/path/to/author-attention.py \
  /absolute/path/to/new-output-directory
```

## 计分与记录

- **受控题候选准确率。** 带选项用 A–D 的原生单 token 分数。开放提示正文不显示选项，评价时对四组语义答案排序。算术采用完整两位数的条件 log-likelihood。单次自由输出以完整词名或数字开头的准确率另在 [free_generation](free_generation) 报告。开放算术因果差分仍用四个可区分首位数字的原生 margin。
- **公开开发筛选。** ARC 与 HellaSwag 用候选完整续写的总 log-likelihood。PrOntoQA 是证明结论和其否定的诊断力。GSM8K 用生成数字和含受控干扰的候选计分两个口径。样本为每任务 64 道的开发筛选，GSM8K 生成 32 道。公开筛选与 256 道的宏观注意力复现分别记录。
- **原始数据。** [data/groups.jsonl](data/groups.jsonl) 是全部语义世界和区域字节位置。各任务目录中的 rows.jsonl 存同次真实 forward 的 K1–K4 分数和分词坐标。arms.jsonl 存位点、供体、K1–K4 答案分数与剂量。哈希留在各子目录的 complete.json。
- **张量。** 完整服务器输出位于 /root/autodl-tmp/ouro-loop-computation-20261119。自然 trajectory.npy 的轴是样例、4 轮、24 物理层加 1 归一化、5 个角色、48 维固定正交投影。boundary.npy 记录四个边界的 2,048 维角色均值。attention.npy 记录 r 世界全部 96 物理层的混合、列集中与查询级 sink proxy。论文的头级 sink rate 在 [paper_reference](paper_reference) 中独立记录。服务器数组均有公开哈希，可按脚本再生。

## 交付入口

- [六组图](results/figures)和[逐组因果对比](results/causal_by_group.csv)
- [语义读出图谱](results/semantic.csv)和[答案循环轨迹](results/behavior_worlds.csv)
- [进阶判定](results/progression.json)和[执行审计](results/execution_audit.json)
- [案例](results/case_studies.json)、[单题轨迹](results/single_item_trajectories.csv)和[日志](logs)

预检中保留两类失败。在加入开放式算术前，分词检查发现 Ouro 会拆开多位数，因此实现了完整续写计分。候选矩阵乘法在 BF16 的舍入边界偶有偏差，因此全部 post-norm 输出以原生全词表 head 为准。此前非正式运行不加入正式结果。

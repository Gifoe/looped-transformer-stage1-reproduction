# Ouro-1.4B 跨任务潜在计算

本目录从研究分支的 `3c13d4e8ba026a6b617b61ccf3a95c88e88a6251` 建立独立实验。模型为 `ByteDance/Ouro-1.4B` Base，固定修订为 `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`。GPU 为单张 RTX 5090。执行条件为 BF16、原生 SDPA、batch size 1、关闭缓存和 TF32、K1–K4。分析早期位置状态的替换如何改变受体后续的选择，模型参数保持冻结。

## 环境与输入

Linux x86_64、Python 3.12、支持 CUDA 12.8 的 NVIDIA 驱动。验证环境的完整包版本见 `logs/pip_freeze.txt`。以下命令从本研究分支的仓库根目录执行。

```bash
python3.12 -m venv .venv-ouro-cross-task
source .venv-ouro-cross-task/bin/activate
python -m pip install 'torch==2.7.1+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.55.4 huggingface_hub==0.34.4 \
  accelerate==1.10.1 numpy==1.26.4 pyarrow==21.0.0 scipy==1.15.3 matplotlib==3.9.0
export OURO_PYTHON="$(command -v python)"
EXP=experiments/ouro_cross_task_20261118
ROOT=/absolute/path/to/empty/ouro-run
mkdir -p "$ROOT/sources"
hf download ByteDance/Ouro-1.4B --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 \
  --local-dir "$ROOT/model"
curl -fL https://huggingface.co/datasets/allenai/ai2_arc/resolve/210d026faf9955653af8916fad021475a3f00453/ARC-Challenge/validation-00000-of-00001.parquet -o "$ROOT/sources/arc.parquet"
curl -fL https://huggingface.co/datasets/allenai/hellaswag/resolve/55aa56627db098098f50549e650a00be2684f4bf/data/validation-00000-of-00001.parquet -o "$ROOT/sources/hellaswag.parquet"
curl -fL https://raw.githubusercontent.com/asaparov/prontoqa/0a6412b6fddf46324a1cb96e066dd7b3d89b87d6/generated_ood_data.zip -o "$ROOT/sources/prontoqa.zip"
(cd "$ROOT/sources" && sha256sum -c -) <<'EOF'
395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05  arc.parquet
cfcd787567d3a7feb1345dab8e32abb7ec67f370d075554deb8fdc189ec33352  hellaswag.parquet
0becba04e1e1eb80593709c6a3db2a46badd5587a29587ffaeb68ed5068c9de9  prontoqa.zip
EOF
for phase in prepare pilot dev freeze confirm trajectory results; do
  bash "$EXP/reproduce.sh" "$phase" "$ROOT/results" "$ROOT/model" "$ROOT/sources" || exit "$?"
done
```

每个 GPU 阶段写入独立输出目录并校验权重及 8 项实现/配置/词表文件。模型权重的完整 SHA-256 为 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`。相同快照下，早期 kit 的哈希文本漏了首位 `5`。本次沿用该 kit 的 `native_capture.capture` 和独立逐候选计分校验。不同 GPU、CUDA 和底层矩阵乘法内核可能造成 BF16 数值差异。

将实验目录中的参考输出直接交给 CPU 分析脚本可复算标量统计与图，无需模型。

```bash
python "$EXP/analyze.py" results "$EXP"
python "$EXP/figures.py" "$EXP"
```

## 样本与计分

| 数据 | 正式轨迹样本 | 配对控制 |
| --- | ---: | --- |
| 受控两跳关系 | 80 组 | 改变第一跳、第二跳、两跳同时改变、无关记录 |
| PrOntoQA 官方发布样本 | 90 条 | 3-hop ProofsOnly 测试条目，二选一证明结论与其否定 |
| PrOntoQA 派生反事实 | 60 组 | 2-hop ModusPonens 样本的主体及第一步证明词汇，重构为两支匹配世界 |
| 受控算术 | 80 组 | 加法中间值，第二步乘加可补偿，干扰项为独立算式 |
| ARC-Challenge validation | 100 条 | 按固定种子抽样 |
| HellaSwag validation | 50 条 | 按固定种子抽样，作为一般语境对照 |

所有任务使用零样本提示和单字符选项标签。每个候选的分数取同一个读出位置的原生 BF16 输出头，经全词表 FP32 log-softmax 后索引对应单 token。概率并列保留，`predictions=-1` 表示最高分并列。ARC 使用当前实验的零样本相同选项表，历史 25-shot ARC 的分类结果采用另一提示条件。PrOntoQA 使用明确标记的证明结论二选一改造，HellaSwag 使用选项标签提示，因此两者的准确率都对应这里的诊断协议。

受控组内固定选项顺序。第一步状态在第二步文本之前采样，并对每组各世界约束相同的总 token 数、干预 token 和绝对位置。准备阶段使用确定性重试满足对齐。第一批每任务 8 个组或条目用于吞吐与一致性检查。受控任务各 16 组开发、30 组确认，按组切分。确认世界从开发选点数据中隔离到规则冻结之后。

## 干预与统计

- 在第一阶段结束处的 post-layer 向量做整向量移植。开发搜索范围预先限制在 K1/K2 的零基 L5、L11、L17。
- 三个主要对比为同最终答案供体相对原生受体、配幅无关方向和配幅随机方向。确认前按各对比 K4 平均效应的最小值选出每任务一个位置。
- 每组评估两个互反方向。先在组内平均，再按独立组做 30,000 次 Bootstrap。每个任务的三项主要对比均需通过区间判据。任务间做 Bonferroni 校正。其余区间是描述性的 95% 区间。
- 对照记录实际 BF16 施加幅度和数值身份。确认还测量后续读出与注意力、MLP 输出变化，将稍后读出向量恢复为受体自然值，并在单独的中间问题提示中复用移植向量。

## 文件

- `REPORT_ZH.md` 为结论、机制证据、病例和局限。
- `semantic_interventions.jsonl` 保存提示、候选、真实答案、反事实答案和组划分。
- `trajectory_results/` 保存逐世界分数、token 角色、每物理层和循环的轨迹量及成对比较。`geometry.npz` 的几何轴为世界、转移或轮次、物理层/归一化、角色。角色顺序为 anchor、readout，层 0–23 的输出之后接 post-loop norm。可选 `semantic_bf16.npz` 是精选事件向量，以 int16 原始位保存 BF16。
- `dev/`、`site_freeze.json` 与 `confirmation/` 保存开发选择、冻结哈希、确认逐方向数据和运行元数据。
- `causal_effects.csv` 与 `cross_task_mechanisms.csv` 保存配对效应、区间、自然行为支持、方向翻转与配幅偏差。
- `case_witnesses.json` 给出报告病例的选择规则、读出和残差分支见证。
- `logs/`、各阶段 `completed.json` 与 `figures/` 保存运行记录和图。GPU 阶段的 `work_seconds` 是模型加载后至阶段结束的同步墙钟时间，包含该阶段的捕获和输出开销。

数据来源为 [PrOntoQA](https://github.com/asaparov/prontoqa/tree/0a6412b6fddf46324a1cb96e066dd7b3d89b87d6)、[ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453/ARC-Challenge)、[HellaSwag](https://huggingface.co/datasets/allenai/hellaswag/tree/55aa56627db098098f50549e650a00be2684f4bf) 和 [Ouro-1.4B 固定快照](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1)。

## 实验后的诊断与新组复现

在主流程完成后执行以下三项，输出目录与主研究相同。前缀移植沿用主开发确定的每任务物理层，并在每任务 30 个新组上比较 K1 和 K2。两跳记录实验使用再一批 30 个新组，固定 K2/L5，区分被查询记录和非查询记录，另在 K2/L11 对读出做局部恢复。算术诊断使用已有 80 组。

```bash
"$OURO_PYTHON" "$EXP/arithmetic_diagnostic.py" "$ROOT/results" "$ROOT/model"
"$OURO_PYTHON" "$EXP/prefix_followup.py" "$ROOT/results" --model "$ROOT/model" --prontoqa-zip "$ROOT/sources/prontoqa.zip"
"$OURO_PYTHON" "$EXP/record_followup.py" "$ROOT/results" --model "$ROOT/model"
"$OURO_PYTHON" "$EXP/analyze.py" results "$ROOT/results"
"$OURO_PYTHON" "$EXP/audit.py" "$ROOT/results"
"$OURO_PYTHON" "$EXP/figures.py" "$ROOT/results"
```

在已取得的标量参考目录上，用 `prefix_followup.py` 或 `record_followup.py` 的 `--analyze-only` 可复算扩展对照的效应与区间。精选 BF16 原始位数组保留在本地工作区与服务器输出。Git 中保留逐世界分数、逐层统计和全部实际干预的标量结果。

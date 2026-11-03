# Ouro weight-space repair V2

实验结果与结论见 [REPORT_ZH.md](REPORT_ZH.md)。本目录与 V1 隔离，实际测试两类免梯度权重编辑，最终固定一个真实 BF16 编辑版本，并在完整 ARC-Challenge 1,172 题上比较普通 K3–K6 及 K4 Fixed/Adaptive LoopCD。

- 原始模型：`ByteDance/Ouro-1.4B`，revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`。
- 主普通比较：Edited K4 vs Original K4。
- 主组合比较：同一编辑权重 + Fixed LoopCD（omega=.5，K4，参考 K1）vs Original Fixed LoopCD。
- Adaptive cap1 与普通 K3/K5/K6 为次要比较。所有 unsupported LoopCD 深度为 NA。
- test 是前序使用过的锁定回归基准，不是新盲测；禁止用其结果调整候选、深度或系数。

## 结果与文件布局

- `selected_edit_manifest.json`：实际选定参数、BF16 坐标计数、范数、原始/编辑哈希和验证集排序。
- `edit_search_results.csv`：14 screen、4 selection、3 validation、1 final，共 22 行。
- `final_comparison.csv`：六种模型/读出组合的准确率矩阵；`paired_statistics.csv`：六项配对统计。
- `paired_transitions.csv`：1,172×6 条逐题比较；`changed_answer_cases.csv`：全部改答，包括错误→不同错误。
- `accuracy_by_loop.pdf` 与 `figures/`：矢量及 PNG 图表。
- `configs/`：数据、方向、候选、实际物化权重、晋级、验证与最终方法锁。
- `edited_weights/statistics/`：训练分组、48 个 site 排名、列排名、方向因子及原始统计文件 SHA。
- `edited_weights/selected/manifest.json`：选定 bundle 的明文索引；精确目标张量和 patch 存于分块压缩归档，解包后恢复到同一路径。
- `artifacts/reproduction_records.zip`：完整渲染数据、逐题评分、实验 stdout/stderr、完成收据和环境记录。
- `artifacts/exact_weight_bundles.tar.xz.part*`：全部 14 候选及最终选定版本的精确 BF16 目标张量、替换 patch 和因子。分块 SHA、整体 SHA 见 `ARTIFACT_INDEX.json`。
- `astra/`：独立代码/统计审查与纯 CPU 合成检验。
- `REMOTE_STATE_INDEX.json`：大体积原始训练前向状态/输入输出统计的服务器保留清单；这些 `.pt` 不放进 Git，未冒充已上传。重跑 `collect_statistics.py` 可重新采集。

原始完整模型、环境和 cache 不随仓库重复发布。服务器原始逐题状态保留在 `/home/newuser1/experiments/ouro_weight_space_repair_v2/results/training_statistics/`。V1 全部受追踪数据/代码/结果的 size、mtime 与原始 safetensors SHA 在 `ARTIFACT_AUDIT.json` 中核验。

## 恢复公开记录与精确 BF16 patch

从本分支克隆仓库，进入本实验目录：

```bash
cd experiments/ouro_weight_space_repair_v2
python3 scripts/unpack_artifacts.py
```

此命令只用标准库，验证归档及分块 SHA，拒绝目录穿越、链接和覆盖内容不同的既有文件。`edited_weights/selected` 将包含 manifest、完整 BF16 目标张量、精确替换 patch 及适用的 U 因子。重放以精确 patch 为准，避免重算低秩 GEMM 带来二次 BF16 舍入差异。

复用与实验相同的 GB10/Python3.12/PyTorch2.11.0+cu130/Transformers4.55.4 环境（详见解包后的 `logs/ENVIRONMENT.json`、`requirements_frozen.txt`）。若已有原始模型，可在 `models/Ouro-1.4B` 使用只读链接；否则在线准备固定 revision：

```bash
env/bin/python scripts/prepare_replay_model.py
```

原始 safetensors 必须为 SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`。推理默认离线，原始权重不被写回。

先做训练样本重放，或者直接重放完整官方 test：

```bash
env/bin/python scripts/supervise.py --name replay_smoke --seconds 600 -- replay_selected.py --split discovery_screen --n 1
env/bin/python scripts/supervise.py --name replay_test --seconds 7200 -- replay_selected.py --split test
```

GPU 监督器读取本目录独立的 6 小时 ledger，计费加载、运行与失败；不会清空旧 ledger。重复复现应在独立副本保留历史记录并自行设定新的研究预算，不要修改原实验计费证据。重放写新结果目录，不覆盖已发布评分。非相同硬件/软件环境可能改变 BF16 数值，脚本会报告与公开逐题结果的最大误差，不自动放宽阈值。

## 从训练统计重新构造候选

`build_data_v2.py` 复用 V1 的固定示范与独立 selection，并从未用训练池扩展 discovery 至384。`collect_statistics.py` 只在训练题记录 K3–K6 状态和投影输入二阶矩/raw output 均值；随后 `build_candidates.py` CPU 构造固定14候选。`run_edit_pipeline.py` 依冻结家族配额完成 screen→selection→validation→final；它为每个候选物化并保存一次真实 BF16 权重，之后始终读取同一 patch。

若从头重跑，应在新实验副本生成新的执行收据；本目录的冻结锁与已有文件会拒绝混用不同代码、数据或方向。不要为了绕过检查修改已发布锁文件。原始发现统计不随 Git 发布，但全部训练标签、渲染请求、算法、列/层排名、实际候选因子和精确已测权重均提供。

## 方法来源与权重许可

方法适配 [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) 的权重方向/稀疏列修改思路；不是原数学基准的逐项复现。最终权重是本实验修改后的研究产物，不是 ByteDance 发布的新官方 checkpoint。上游模型卡标注 Apache-2.0，见 [固定 revision 模型卡](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/README.md)、`edited_weights/NOTICE.txt` 和 `third_party/APACHE-2.0.txt`。代码与数值结果没有证明新的普适失效机制。

ARC 问题及渲染 prompt 数据按[上游数据卡](https://huggingface.co/datasets/allenai/ai2_arc)标注的 CC BY-SA 4.0 提供；来源、许可链接和格式改动见 [DATA_SOURCE_NOTICE.txt](data/DATA_SOURCE_NOTICE.txt)。该数据说明与原始 Ouro 权重许可分别保留。

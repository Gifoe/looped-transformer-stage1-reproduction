# 单卡复现

在 Linux 上使用冻结 Ouro‑1.4B Base、BF16 和与先前研究一致的 Python、PyTorch、Transformers。`MODEL_CONFIG.json` 中的旧模型哈希仍适用。ARC Corpus 包含原生协议用 `train.parquet`、`validation.parquet`、`test.parquet`，测试 OpenBookQA 用先前留档的规范化 parquet 和相邻 JSON 回执。开放题评测沿用现有 Parquet 和哈希。 [原环境与数据](../ouro_functional_controllability_20261123/REPRODUCE.md)

原候选基线来自当前分支的固定父提交。上一轮内容生成记录及数据准备代码来自独立的补测提交。`FOLLOWUP_RESULTS` 可以是该补测的已解压结果，也可以是原始 `.jsonl.gz` 目录。`FOLLOWUP_DATA` 放之前的 `gsm8k`、`mmlu_pro` parquet 和 JSON 校验回执。 [补测数据与复现](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/c0efc9d01b7fedc3f485156228a6253eb272d2e2/experiments/ouro_minimal_functional_feedback_20261123/generation_followup/REPRODUCE.md)

```bash
cd experiments/ouro_rank1_functional_theory_generation_20261123
sha256sum -c --quiet ARTIFACTS.sha256
CUDA_VISIBLE_DEVICES=0 bash reproduce.sh /data/ouro /data/arc /data/openbookqa.parquet \
  /data/followup-data /data/followup-results /scratch/rank1-reproduction
```

生成候选在开发集选择一次，测试不会反选。完整候选结果在 `BENCHMARK_RESULTS/INHERITED_*.jsonl`，推理权重和路径不变。生成逐题结果、时延、停止类型和过往基线的配对统计也在 `BENCHMARK_RESULTS`。`results/*/RAW_EVIDENCE.tar.gz` 保存原始分片。`results/frozen/ACCEPTANCE.json` 指示新生成模式是否达到逐数据集部署门槛。所有 GPU 作业都计入同一硬上限。

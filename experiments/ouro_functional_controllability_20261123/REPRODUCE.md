# 复现实验

在 Linux 使用同一单张 BF16 GPU。基础模型、训练数据和验证数据沿用固定参考提交的配置。把官方 ARC 的 `train.parquet`、`validation.parquet`、`test.parquet` 放到同一目录。OpenBookQA 需要前一轮使用的规范化 Parquet 和相邻同名 JSON 校验回执。完整仓库内的上一轮研究目录须可见。 [基础环境和语料](../ouro_minimal_functional_feedback_20261123/REPRODUCE.md)

```bash
cd experiments/ouro_functional_controllability_20261123
sha256sum -c --quiet ARTIFACTS.sha256
python audit_release.py . --out /tmp/ouro-control-result-audit.json
CUDA_VISIBLE_DEVICES=0 bash reproduce.sh /data/ouro-base /data/arc /data/openbookqa.parquet /scratch/control-replay
```

输出目录和版本库分开。脚本复用上一轮发布的完整候选和逐题生成记录，重建按方法与深度划分的生成分片，执行新控制器训练、Validation 冻结、两个完整候选集、真实贪心生成以及 A–E 与独立迁移。C 用 Train 筛选后在完整 ARC Validation 确认，D 使用固定 K4 和三处入口。CommonsenseQA 输入在主控制器权重锁定后才会读取。

`BENCHMARK_RESULTS` 与 `GENERATION_RESULTS` 的合并逐题 JSONL 可直接读取。后者的 `RUN_FRAGMENTS.tar.gz` 保留按任务、方法和深度拆分的原始记录。A–E、独立迁移和候选选型的 `results/*/RAW_EVIDENCE.tar.gz` 保留各自目录下的完整原始输出和分片。可先用 `tar -tzf` 检查成员名，再解到同一结果目录运行分项分析。CPU 审计直接重算已归档结果的样本覆盖和主要正确数，并校验权重与 GPU 台账。

整个 GPU 序列使用同一张卡，台账上限是 28,800 秒。可用 `PYTHON` 指向已配置的 Python 3.12 环境。外部数据接口若受网络限制，实验 E 的转移入口支持 `--csqa-file` 指向具有同一 SHA-256 的作者验证文件。

若在复现运行中七任务门槛通过，可用 `STAGE2_ASSETS` 指定上一轮七任务的复现目录。它应包含 `code/stage2_score.py`、冻结的 `inputs/`、及旧分数和匹配计分锁所在的 `scores/`。新增评测仅读历史基线分数。 [七任务输入复现](../ouro_minimal_functional_feedback_20261123/stage2/REPRODUCE.md)

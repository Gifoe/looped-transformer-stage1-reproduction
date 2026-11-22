# 单卡复现

克隆完整研究分支，以保留同仓库中的固定 Ouro 加载器、ARC 原版候选评分代码和历史逐题分数。使用 Python 3.12、PyTorch 2.7.1+cu128、Transformers 4.55.4、NumPy、SciPy、PyArrow 和一张支持 BF16 的 CUDA GPU。原始运行使用 RTX 4090 D，SDPA、BF16、禁用 KV cache。权重和 tokenizer 由继承的加载器按官方固定修订核验。 [固定 Ouro 检查点](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1)

```bash
hf download ByteDance/Ouro-1.4B --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 --local-dir /data/ouro
cd experiments/ouro_semantic_amplification_arc_20261122
sha256sum -c ARTIFACTS.sha256
bash reproduce.sh frozen /data/ouro /data/arc /data/arc-amplification-frozen
bash reproduce.sh train /data/ouro /data/arc /data/arc-amplification-train
```

`frozen` 从已发布权重直接运行四预算的全部新模型和消融。`train` 先重新采集 Train/Validation 状态并训练再运行，需给状态缓存预留约 4 GB。两个入口都会核对官方 Parquet SHA-256，历史基线只读。两类运行都会记录新的实测计时、峰值显存、候选分数、配对统计和 GPU 台账。运行目录需为空或未曾开始评分。

- `FREEZE.json` 保存 Validation 选中的权重、剂量和源代码哈希。
- `TEST_PREDICTIONS.jsonl` 保存完整候选原始 log probability 总和。`GEOMETRY_PREDICTIONS.jsonl` 保存同剂量几何控制。
- `TEST_PAIRED.jsonl` 保存两种归一化口径下新旧方法的逐题选择。官方选项序和正确选项索引随行保存。
- `METRICS.json` 保存原生及历史 LoopCD 的恢复、退化、置信区间和配对检验。
- `COST.json` 保存新方法的完整 Test 单独运行成本。`HISTORICAL_COST.json` 是旧硬件实测数据的归档引用。
- `VALIDATION_TRAINING.json` 和 `VALIDATION_CALIBRATION.json` 保留所有选型，包括低分及退化的配置。

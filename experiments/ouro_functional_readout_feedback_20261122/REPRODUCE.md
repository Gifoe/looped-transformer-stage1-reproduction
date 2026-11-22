# 单卡复现

完整克隆研究分支，保留三个同仓库实验目录的代码、历史预测和训练权重。Python 3.12、PyTorch 2.7.1+cu128、Transformers 4.55.4、NumPy、SciPy 和 PyArrow，单张支持 BF16 的 CUDA GPU。运行环境为 BF16、SDPA、禁用 KV cache。模型加载器核对官方模型、配置和 tokenizer 哈希。[固定 Ouro-1.4B 检查点](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1)

```bash
hf download ByteDance/Ouro-1.4B --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 --local-dir /data/ouro
cd experiments/ouro_functional_readout_feedback_20261122
sha256sum -c ARTIFACTS.sha256
bash reproduce.sh frozen /data/ouro /data/arc /data/feedback-frozen
bash reproduce.sh train /data/ouro /data/arc /data/feedback-train
```

`frozen` 载入锁定的读出、效用门控和反馈权重，重算完整 ARC/OpenBookQA 与写回对照。`train` 另从固定 Train/Validation 采集约 4 GB 缓存，重跑双目标、两折 OOF、校准器、反馈训练和功能方向审计。两个入口均只读历史 ARC 分数，自动获取缺失的固定数据文件，并维护不超过五小时的 GPU 台账。公开状态文件中的绝对运行路径可在不改变权重哈希的前提下重定位。

`FEEDBACK_FIT_CODE_HISTORY.txt` 保留训练启动时的源码指纹。路径统计脚本随后增加配对分析，并随确认配置一同锁定在 `CONFIRMATION_LOCK.sha256` 中。反馈拟合源码在此期间哈希一致。

- `results/ARC_READOUT.jsonl.gz`、`results/feedback/TEST_PREDICTIONS.jsonl.gz` 及其 `TEST_CONTROL_PREDICTIONS` 记录完整 ARC 新方法候选分数。
- `OPENBOOKQA_READOUT.jsonl.gz` 和 `feedback/EXTERNAL_PREDICTIONS.jsonl.gz` 记录同配置、无外部调参的独立测试。
- `ARC_PAIRED.jsonl.gz`、`METRICS.json`、`FAILURES.json` 保存两种口径的逐题选择、配对统计和失败样本 ID。
- `READOUT_FREEZE.json`、`CALIBRATION_FREEZE.json`、`feedback/FREEZE.json` 保存锁定配置。`OOF_UTILITY.json.gz` 含折次、训练效用标签及候选分数，便于独立审计门控。
- `DIRECTION_VALIDATION_SCORES.jsonl.gz`、`SEMANTIC_SENSITIVITY_SCORES.jsonl.gz` 和对应审计为方向交换与语义对照。解压到工作输出目录后可运行 `python direction_replay.py /path/to/result`。
- `feedback/MECHANISM*`、`feedback/GENERATION_DIAGNOSTIC.json`、`feedback/ISOLATED_FEEDBACK_COST.json` 保存路径干预、自由生成诊断和单词表头计时。
- `feedback/GENERATION_INITIAL_DIAGNOSTIC.json` 和 `diagnostics_initial.py` 保留诊断初跑。最终诊断统一展示字母标签后重跑，完整候选评分未改变。

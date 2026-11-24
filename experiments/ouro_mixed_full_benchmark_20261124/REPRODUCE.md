# 单卡复现

实验使用 Ouro-1.4B Base。模型、分词器与归档请求哈希见冻结回执。Mixed 权重来自提交 59cc8e5330bd16b6dbf40d687b70ff68e518e3d6 的 Ouro Shared Functional Advantage 实验。

在 Linux CUDA 上使用继承的 Python 环境，保留 PyTorch 2.7.1+cu128、Transformers 4.55.4 与 BF16。参照 external_env.sh 配置已归档的 Shared Functional Advantage、Minimal Functional Feedback、Unified Rank-1、Stage2 输入、MMLU-Pro Parquet 和历史分片。五套零样本候选输入来自 Stage2 Zero-shot 子目录，SciQ 与 PIQA 使用父目录中已有的零样本输入与结果。MMLU-Pro 使用原始零样本贪心生成。

在新的运行目录放置本目录的代码、配置和权重。已有运行保留自己的冻结回执和分片即可从断点继续。

    sha256sum -c WEIGHTS.sha256
    export OURO_MIXED_WEIGHT="$PWD/weights/controller.pt"
    bash mixed_full/run_mixed.sh
    source external_env.sh
    "$py" -u mixed_full/aggregate_mixed.py --final --out mixed_full/results \
      --minimal "$feedback" --stage2 mixed_full/archive --unified "$unified" \
      --followup "$feedback/generation_followup" --model "$model"

结果目录有八个任务的逐题压缩 JSONL，以及对应原始 SHA-256 和实际成本回执。Native 和 LoopCD 直接复用历史逐题分数。MMLU-Pro 的回执校验了全部 197 批及其原始生成设置。

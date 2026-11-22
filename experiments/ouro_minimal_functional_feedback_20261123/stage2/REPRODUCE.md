# 七项 few-shot 复现

模型、主环境和历史仓库参见上一级 [复现说明](../REPRODUCE.md)。固定主 checkpoint 见 `../results/FREEZE.json`。七项数据的固定修订、示例 seed 和每次导出请求的 SHA-256 见 `inputs/`。

```bash
# 在主实验目录执行，另外建立含 lm_eval==0.4.13 的 CPU 环境
here="$PWD"
archive="$(realpath ../ouro_arc_pareto_20261119)"
prior="$(realpath ../ouro_semantic_amplification_arc_20261122)"
export PYTHONPATH="$here"
out=/scratch/ouro-fewshot
model=/data/ouro-base
mkdir -p "$out/inputs"
for task in arc_challenge arc_easy sciq mmlu hellaswag winogrande piqa; do
    data="$out/inputs/$task.jsonl"
    /path/to/cpu-venv/bin/python stage2/code/stage2_requests.py --tasks "$task" --output "$data"
    CUDA_VISIBLE_DEVICES=0 python stage2/code/stage2_score.py \
        --model "$model" --archive "$archive" --previous "$prior" \
        --input "$data" --manifest "$data.manifest.json" --out "$out/scores/$task" \
        --freeze results/FREEZE.json --arm linear8_t1 --depths 4 --benchmarks "$task" --shared-prefix
done
python stage2/code/stage2_report.py "$out"
```

每项任务串行使用同一张 GPU。可重复执行评分器，已完整写出且校验通过的分片会跳过。评分器启用共享前缀前在实际输入上核对比特级隐藏状态和评分一致性。复现输入需要网络下载固定数据版本。若环境访问 Hugging Face 官方域名受限，可以通过 `HF_ENDPOINT` 使用可访问的镜像，并核对清单里的输入哈希。

本次 MMLU 的通用导出在逐科加载时超时，改用固定 revision 内 114 个官方分科 dev/test Parquet，核对每个数据文件的 SHA-256 后交给同一 Harness 请求代码。完整 57 科及其题目顺序保留，独立核对已缓存的 24 科逐行一致。传输回执见 `recovery/canonical_mmlu_receipt.json`。 [恢复代码](recovery/mmlu_recovery.py)

`scores/*/RESULTS.jsonl.gz` 包含完整逐题结果。`COMPARISON.json` 记录全部任务的配对统计。`SHARD_COSTS.tar.gz` 里的实际分片计时与峰值显存用于共同评测成本。`GPU_LEDGER.tsv` 记录两个阶段共用的单卡时间预算。

## 同题目 zero-shot

沿用上文变量。对 SciQ 和 PIQA 使用已经产出的原始 0-shot 文件。其它五项用相同语料版本、评分器和锁定主权重，以 0 个示例构造题面。

```bash
mkdir -p "$out/zero_shot/inputs"
for task in arc_challenge arc_easy mmlu hellaswag winogrande; do
    data="$out/zero_shot/inputs/$task.jsonl"
    /path/to/cpu-venv/bin/python stage2/code/stage2_zero_requests.py --tasks "$task" --output "$data"
    CUDA_VISIBLE_DEVICES=0 python stage2/code/stage2_zero_score.py \
        --model "$model" --archive "$archive" --previous "$prior" \
        --input "$data" --manifest "$data.manifest.json" --out "$out/zero_shot/scores/$task" \
        --freeze results/FREEZE.json --arm linear8_t1 --depths 4 --benchmarks "$task" --shared-prefix
done
python stage2/code/stage2_zero_report.py "$out"
```

MMLU 分科，也可在独立新工作目录使用 `stage2/code/stage2_zero_requests.py --canonical-mmlu --source-cache "$HF_HOME" --workdir "$out/mmlu-canonical"`。把生成的 `mmlu.jsonl` 和相邻清单放在 `zero_shot/inputs/`，再运行同一评分命令。

# 生成评测复现

使用主实验冻结的 Ouro-1.4B Base 与反馈权重。运行位置是主实验目录，模型及 Python 依赖见 [主实验复现](../REPRODUCE.md)。本阶段还使用 `requests` 下载数据。代码评分在独立 CPU 环境中固定 `evalplus==0.3.1`。

```bash
taskdir="$PWD/generation_followup"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
model=/data/ouro-base
archive="$(realpath ../ouro_arc_pareto_20261119)"
prior="$(realpath ../ouro_semantic_amplification_arc_20261122)"
out=/scratch/ouro-generation
mkdir -p "$out"
python "$taskdir/generation_followup_data.py" "$out/data"
for task in gsm8k mmlu_pro humaneval mbpp; do
    script="$taskdir/generation_followup_run.py"
    if [[ "$task" == gsm8k ]]; then script="$taskdir/generation_followup_run_gsm8k.py"; fi
    if [[ "$task" == mmlu_pro || "$task" == humaneval ]]; then script="$taskdir/generation_followup_run_reference.py"; fi
    CUDA_VISIBLE_DEVICES=0 python "$script" \
        --model "$model" --archive "$archive" --previous "$prior" \
        --freeze results/FREEZE.json --data "$out/data" --out "$out/results" --task "$task"
done
```

每个任务固定同一题面和生成上限。所有方案用同样的当轮 K4，固定和自适应 LoopCD 使用 K1 作为参考。脚本以独立 KV 位置存储每层及每次循环。它在实际短、中、长提示上核对缓存与全序列重算的前几个贪心 token，逐任务保存最大 logit 数值差。分片的保存和恢复均检查题目 ID 及文件指纹。

MBPP 题面附题目提供的原始公开断言，确定目标函数接口。该设置包含零个已解示例。GSM8K、MMLU-Pro 和 HumanEval 复用各自治理锁定的源码副本。MBPP 全量评分采用同一缓存策略，预检在 9 个不同长度的样本上记录缓存和全序列重算间的近并列 token 变化。

运行 EvalPlus 需要能建立 chroot、分离进程 UID 和安装 seccomp 过滤器的 Linux 主机。基准只在设置隔离且通过已知答案和网络调用检查后执行模型生成代码。

```bash
for task in humaneval mbpp; do
    OPENBLAS_NUM_THREADS=1 /path/to/evalplus-venv/bin/python \
        "$taskdir/generation_followup_evalplus.py" --jail "$out/jail" --results "$out/results" --task "$task"
done
python "$taskdir/generation_followup_report.py" \
    --results "$out/results" --out "$out" --ledger /path/to/GPU_LEDGER.tsv
```

逐题文本、token、停止原因和输出均在 `results/<task>/<method>.jsonl.gz`，EvalPlus 基础及扩展测试的逐题状态在相邻的 `*.evalplus.jsonl.gz`。仓库内的台账记录原始串行单卡运行，分片计时可由 `SHARD_COSTS.tar.gz` 核对。用 `sha256sum -c --quiet ARTIFACTS.sha256` 校验归档。

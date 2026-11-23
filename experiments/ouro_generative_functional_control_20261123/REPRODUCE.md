# 单卡复现

使用 Linux、Python 3.12.3、PyTorch 2.7.1+cu128、Transformers 4.55.4、NumPy 1.26.4、SciPy 1.15.3 和 PyArrow 21.0.0。RTX 4090 D 实跑。Ouro-1.4B Base 的官方修订为 `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`，`model.safetensors` SHA-256 为 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`。原生模型代码、tokenizer 和其余依赖见父研究数值审计。 [官方模型快照](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1) · [模型运行签名](../ouro_functional_transferability_20261123/results/PREFLIGHT.json)

## 核对已发布结果

在实验发布目录验证压缩文件、权重和报告，再从原始预测重算逐题指标。`confirmation.tar.gz` 内含每个冻结条件的预测与原始计时文件。`development_rescored.tar.gz` 对应选型所用的解析规则。`development.tar.gz` 同时保留最初的解析结果与相同的生成文本。生成通过曾经写入 JSON 的完整解码 token 和停止原因审计。候选分数保留字符数、选项标签和逐选项 log-likelihood。

```bash
sha256sum -c ARTIFACTS.sha256
tar -xzf results/confirmation.tar.gz
tar -xzf results/development_rescored.tar.gz
export PYTHONPATH="$(realpath ../ouro_minimal_functional_feedback_20261123):${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES='' python paired_analysis.py --phase confirmation
CUDA_VISIBLE_DEVICES='' python paired_analysis.py --phase development_rescored
CUDA_VISIBLE_DEVICES='' python analyze_transfer_depth.py
python result_tables.py
```

## 重新训练并运行确认

将新实验源码复制到一个空工作目录。保留同一提交的父研究目录，用 `GEN_TRANSFER_ROOT` 指向 `ouro_functional_transferability_20261123`，并用 `dynamics` 指向 `ouro_dynamical_transport_20261123`。`published` 为本实验的只读归档目录。`GEN_EXPERIMENTS_ROOT` 指向包含原生 scorer 和其依赖的 `experiments` 目录。若当前克隆缺少旧 scorer，从 `a07698897bfa705cf3d31deaa2d8c00050c5c84c` 补齐对应目录，细节见父研究复现说明。`GEN_MODEL` 指向已核验的模型快照，`GEN_GPU_LOCK` 指向当前进程可写的单卡互斥锁，`GEN_PYTHON` 可指定 Python 路径。官方训练语料通过数据源实时获取，发布包提供题号、类型、数据哈希与全部采样代码。

```bash
source control_env.sh
mkdir -p data results
sha256sum PREREGISTRATION.md > PREREGISTRATION.sha256
"$py" fetch_sources.py --prior "$prior_transfer" --published-data "$published/data"
CUDA_VISIBLE_DEVICES='' "$py" prepare_baseline.py "$prior_transfer" data/csqa_matched.jsonl
CUDA_VISIBLE_DEVICES='' "$py" prepare_pool.py --model "$model" --inputs inputs \
    --prior "$prior_transfer" --dynamics "$dynamics" --out data
"$py" fetch_sources.py --prior "$prior_transfer" --published-data "$published/data" --extras
CUDA_VISIBLE_DEVICES='' "$py" expand_pool.py --model "$model" --inputs inputs --data data
CUDA_VISIBLE_DEVICES='' "$py" overlap_audit.py --data data --train data/csqa_matched.jsonl \
    --train data/full_mixed_train.jsonl --out data/filtered
bash run_core_b.sh > run_core_b.log 2>&1
bash run_development.sh > run_development.log 2>&1
bash run_scale_up.sh > run_scale_up.log 2>&1
bash run_round4_and_freeze.sh > run_round4_and_freeze.log 2>&1
chosen=$("$py" -c 'import json; print(json.load(open("results/FINAL_PROTOCOL.json"))["selected_arm"])')
bash control_budget.sh "test_k4_$chosen" evaluate_suite.py --archive "$archive" --previous "$previous" \
    --model "$model" --protocol results/FINAL_PROTOCOL.json --arm "$chosen" --stage k4 \
    --out results/confirmation > "confirmation_$chosen.log" 2>&1
bash run_confirmation_priority.sh > run_confirmation.log 2>&1
bash run_analysis.sh
```

复跑时以重新生成的训练、开发集封存和数据指纹为准。切勿覆盖发布的 Test 权重锁。保留 `data/filtered/CONTAMINATION.json`、`results/FINAL_PROTOCOL.json` 和 `GPU_LEDGER.tsv`。全套比较共用同一候选、深度、batch 和最大生成长度，同一任务使用同一套停止规则。生成长度可以改变端到端成本。单卡累计 18,000 秒上限由 `control_budget.sh` 计算。

实跑在测出长输出用时后使用 `run_confirmation_priority.sh` 调整队列。保存核心 K4 回执后暂停当前模型，通过相同的封存解码器完成各模型的问答深度迁移，再续跑开放指令。暂停和模型重载均记账。该调度沿用原始协议、权重和匹配题号。覆盖情况见 `results/CONFIRMATION_COVERAGE.json`。

若五小时预算实际用尽，可使用 `bash finalize_budget_limited.sh` 统计全部已有回执并列出未完成条件。没有完整回执的任务不进入统计。

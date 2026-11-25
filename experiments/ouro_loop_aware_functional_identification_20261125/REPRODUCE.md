# 单卡复现

阶段快照包含已核验结果、采集与预测代码。完整十二项评测和新控制器权重仍待生成。

在 Linux GPU 主机保留历史实验目录，并将本目录放在独立路径。所需的既有工作目录由 `external_env.sh` 配置。使用历史 Ouro-1.4B Base 检查点，模型文件 SHA-256 为 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`。加载相同的 PyTorch、Transformers、PyArrow 环境及历史导入路径。当前实测来自一张 RTX 4090 D。

```bash
study="$PWD"
joint=${OURO_EXTERNAL_ROOT:-/root/ouro-external-data-joint-training-20261124}
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$PYTHONPATH"
mkdir -p logs
t1="$controllability/results/weights/rank1_t1.pt"
t2="$controllability/results/weights/rank1_t2.pt"
mixed="$advantage/results/fits/mixed/controller.pt"
runtime=(--archive "$archive" --previous "$previous" --model "$model")
operators=(--operator "t1=$t1" --operator "t2=$t2" --operator "mixed=$mixed")
sha256sum "$t1" "$t2" "$mixed"
```

T1、T2 和 Mixed 的 SHA-256 依次为 `91d4919308bf9060ce0c44e4046e4851e7e59f65b9692c77045535f96534efd2`、`117f5f2fe9c5afe363ba8c032f42cd25155f7a88cb465e4e79bfb3bf496c8032`、`ec2e55a64564f608cfeb245da079015db4347a9391072cba7f2949db05185b87`。原始 ARC 训练输入是历史 Transferability 的 `data/arc_train.jsonl`。它没有应用本轮的排重过滤。

```bash
for k in 2 3 4; do
    "$py" -u collect_identification.py "${runtime[@]}" "${operators[@]}" \
        --input "$prior_transfer/data/arc_train.jsonl" --dataset arc_challenge \
        --mode candidate --depth "$k" --out "results/moments/arc_challenge/candidate/k$k"
done
"$py" -u fetch_reading.py --out data/reading
"$py" -u build_pools.py --transfer "$prior_transfer" --generation "$gen" --external "$joint" \
    --model "$model" --feedback "$feedback" --advantage "$advantage" --reading data/reading \
    --out data/pools_complete --sources arc_challenge csqa obqa qasc scienceqa boolq social_iqa race squad gsm8k
```

用候选池启动已排队的第一批完整 K4 矩阵及有限剂量预测确认。每折的矩阵和逐题记录均有断点，配置改变会拒绝读旧断点。

```bash
domains=(arc_challenge csqa obqa qasc scienceqa boolq)
tags=()
for source in "${domains[@]}"; do
    tags+=("$source:candidate")
    "$py" -u collect_identification.py "${runtime[@]}" "${operators[@]}" \
        --input "data/pools_complete/$source.jsonl" --dataset "$source" \
        --mode candidate --depth 4 --batch 6 --token-budget 2600 --full-context \
        --out "results/filtered_moments/$source/candidate/k4"
done
forecast=results/forecast_candidate
"$py" -u freeze_forecast.py --moments results/filtered_moments --data data/pools_complete \
    --sources "${tags[@]}" --scale-weight "$t1" --out "$forecast"
"$py" -u test_forecast.py "${runtime[@]}" --data data/pools_complete --forecast "$forecast" \
    --depths 4 3 5 --n 48 --out "$forecast/confirmation"
"$py" -u score_forecast.py --forecast "$forecast/FORECAST.json" --results "$forecast/confirmation" \
    --out "$forecast/AUDIT.json"
# 仅在上述独立确认命令成功后写入训练作业的就绪标记。
printf 'DONE\n' > run_forecast.log
bash run_fits.sh
```

`bash run_remaining_moments.sh` 继承前六个矩阵，再采集 SocialIQA、RACE、SQuAD、GSM8K，另建十来源预测。它使用与此前相同的固定折号，且不会覆盖首批候选预测。教师强制生成效用的逐词元正确率单独解释，不等同于自由生成的整题准确率。

初轮训练和开发完成后，可用 `bash run_fit_refinement.sh` 比较匹配计算预算的单源对照以及 T1 初始化的因子切空间加权。按题数和按预算的模型结果分别分析。

```bash
"$py" -u score_source_training.py --forecast results/forecast_candidate/FORECAST.json \
    --development results/dev/SELECTION.json --work-development results/dev_refinement/SELECTION.json \
    --out results/forecast_candidate/TRAINED_VALUE.json
"$py" -u forecast_direction_contrast.py --input results/forecast_candidate/confirmation \
    --out results/forecast_candidate/DIRECTION_CONTRAST.json
"$py" -u audit_cohort_independence.py --pools data/pools_complete \
    --mmlu "$feedback/stage2/zero_shot/mmlu_cache" --confirmation results/forecast_candidate/confirmation \
    --out results/COHORT_INDEPENDENCE.json
```

有限剂量诊断的输入应为官方 ARC-Challenge Validation。脚本按固定哈希选取 24 题，记录所有真实剂量、有效写入线性估计和逐题正确性。

```bash
"$py" -u finite_identification.py "${runtime[@]}" "${operators[@]}" \
    --input "$prior_transfer/data/arc_challenge_validation.jsonl" \
    --dataset arc_challenge --mode candidate --depth 4 --limit 24 --out results/finite/arc_k4
"$py" -u audit_depth_spectra.py --input results/moments/arc_challenge/candidate \
    --out results/ARC_DEPTH_SPECTRA.json
python plot_identification.py --observed EARLY_MECHANISM.json --out figures
```

历史全量参考可从归档提交 `93b80141e1c81b4a632d4d8a696786c525cde697` 恢复。详见 [结果表](RESULTS.md)。

## 正式评测入口

`run_full.py` 读取已锁定的独立开发获胜者。首八项仅跑新权重，要求与历史 Mixed 全量 K4 使用相同的请求、模型和评分器。新增四项对新权重及 Native、T1、T2、Mixed、Rank-8、LoopCD Fixed、LoopCD Adaptive 跑完整官方有标签集合。CommonsenseQA 的 1,221 题是官方公开 Validation，SQuAD 1.1 的 10,570 题是官方公开 Dev。脚本保存原评分及另计的标准指标。SQuAD 列标准 EM 与最大参考词元 F1，GSM8K 同时保存标准 `####` 精确匹配与历史宽松结果。LoopCD 系数按历史候选和生成协议分别复用。

```bash
release="$joint/mixed_full"
"$py" -u run_full.py "${runtime[@]}" --selection results/dev/SELECTION.json \
    --reference-freeze "$release/results/FREEZE.json" --stage2 "$release/archive" \
    --minimal-code "$feedback" --unified "$unified" \
    --followup "$feedback/generation_followup" --historical "$historical" \
    --panel "$joint/data/panel" --t1 "$t1" --t2 "$t2" --mixed "$mixed" \
    --rank8 "$feedback/results/linear8_t1.pt" \
    --out results/official
```

以 `--block freeze` 只核对文件和拟合选择，以 `--block legacy` 或 `--block extra` 分别恢复任务块。生成分片与正式输出由历史评测器逐题校验。

等价的串行入口为 `bash run_final_confirm.sh /absolute/path/to/SELECTION.json`。它对官方测试使用全局 GPU 锁，并在全部完成后执行逐题配对。发布目录由 `package_research.py` 从正式预测与封存回执生成。

```bash
"$py" -u aggregate_full.py --out results/official --release "$joint" --panel "$joint/data/panel" \
    --stage2 "$release/archive" --minimal "$feedback" --model "$model" \
    --followup "$feedback/generation_followup" --unified "$unified" --historical "$historical"
```

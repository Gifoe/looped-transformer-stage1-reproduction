# 单卡复现与原始结果

父研究版本为 `e4d2d219a995b0380b875da547f2db3f40353e8d`。本轮使用相同的 Ouro-1.4B Base、同一 BF16/SDPA 候选评分器、相同的历史 T1、ARC K2-only 和 QASC 权重。没有训练新参数。数据准备读取父研究保存的原始官方来源快照。服务器路径和可覆盖入口位于 [control_env.sh](control_env.sh)。

## 验证发布结果

```bash
sha256sum -c ARTIFACTS.sha256
source control_env.sh
CUDA_VISIBLE_DEVICES="" "$py" audit_predictions.py .
```

后者会重新计算预测误差、逐题配对统计和硬 GPU 上限校验。路径恢复、前向伴随关系、固定时域、上下文交换和置换都有各自的阶段目录。原始 `.jsonl` 含完整候选分数、处理组和逐题动力学指标。E1 的独立确认另外保存每题每预测位置的双因子投影，足以重建已报告的置换统计。

## 从模型执行

使用 Linux、Python 3.12.3、PyTorch 2.7.1+cu128、Transformers 4.55.4、NumPy 1.26.4、SciPy 1.15.3、PyArrow 21，单张已获授权的 RTX 4090 D 或兼容 CUDA 卡。依据旧研究的 [实际模型清单](../ouro_functional_transferability_20261123/results/PREFLIGHT.json) 核验官方 revision、modeling 代码、tokenizer 和权重。复用父研究的完整未压缩功能矩。在新工作目录运行，保留本发布的原始统计。将父研究的冻结路径设在 `TRANSPORT_ARCHIVE` 及 `TRANSPORT_T1`，并按父研究复现说明配置原生 scorer、模型和旧软件环境。

```bash
source control_env.sh
mkdir -p results
sha256sum PREREGISTRATION.md > PREREGISTRATION.sha256
CUDA_VISIBLE_DEVICES="" "$py" archive_audit.py --archive "$transport_archive" --t1 "$t1" --out results
bash control_budget.sh e1 e1_coupling.py --archive "$archive" --previous "$previous" --model "$model" \
  --transport "$transport_archive" --t1 "$t1" --out results/E1
bash run_e2.sh > run_e2.log
bash run_e3_discovery.sh > run_e3_discovery.log
bash run_confirmation.sh > run_confirmation.log
bash run_context.sh > run_context.log
CUDA_VISIBLE_DEVICES="" "$py" seal_mixed.py --parent-study "$transport_archive" \
  --core-forecast results/FORECAST_LOCK.json --out results/MIXED_LOCK.json
sha256sum results/MIXED_LOCK.json > MIXED_LOCK.sha256
bash run_mixed_check.sh > run_mixed_check.log
bash run_negative_path.sh > run_negative_path.log
bash seal_path.sh
bash run_path.sh > run_path.log
CUDA_VISIBLE_DEVICES="" "$py" audit_predictions.py .
CUDA_VISIBLE_DEVICES="" "$py" plot_dynamics.py
```

确认用的两份官方 Train 留出问题已随发布冻结。需从父研究原始输入重新划分时，在新目录使用 `freeze_cohorts.py --transport "$transport_archive" --out data`。正式读确认标签前，必须先按发现集重新封存预测及控制器哈希。数学 JVP 使用同一模型的 math SDPA 高阶导数，独立中心差和生产 dtype 一致性检验与结果一起发布。 [完整 GPU 台账](GPU_LEDGER.tsv)

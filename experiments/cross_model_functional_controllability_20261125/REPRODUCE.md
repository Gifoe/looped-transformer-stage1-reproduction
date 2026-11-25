# Ouro 跨版本 K4 复现

本目录先评测 Ouro-2.6B、Ouro-1.4B-Thinking 和 Ouro-2.6B-Thinking。三个模型都对比自身原生 K4 与自身 K4 加同一份历史 Rank-1 T1。模型权重与 T1 在整个评测中冻结。

T1 来源提交为 `f1027ac176d0f47495fad43b72b4b42fe96a673a`，路径为 `experiments/ouro_mixed_full_benchmark_20261124/t1_full/weights/rank1_t1.pt`，SHA-256 为 `91d4919308bf9060ce0c44e4046e4851e7e59f65b9692c77045535f96534efd2`。写入位置、秩、相对 RMS 上限 0.45 和剂量 1.0 均沿用历史 T1。

模型 revision 和权重摘要在 [MODELS.json](MODELS.json)。下载同时验证官方 tokenizer 与远程模型实现的 Git blob ID。每个模型用自己的 tokenizer，两个对照共享原有零样本输入。MMLU-Pro 复用归档的真自回归贪心解码与解析，保持原有生成上限。48 层 Ouro 的按批 KV 容量调整为 24 层版本的一半，避免超出显存，两组对照设置相同。

```bash
study="$PWD"
joint=/root/ouro-external-data-joint-training-20261124
cd "$joint"
source external_env.sh
cd "$study"
export PYTHONPATH="$study:$PYTHONPATH"
model_root=/dev/shm/ouro-cross-model-20261125
for name in ouro-2.6b ouro-1.4b-thinking ouro-2.6b-thinking; do
    "$py" fetch_ouro.py --plan MODELS.json --name "$name" --out "$model_root"
done
OURO_TRANSFER_MODELS="$model_root" bash run_cross_models.sh
```

每个模型先完成 ARC-Challenge 的两个配对条件，再运行其余七项。全部预测、配对置信区间、正式样本数、生成合法率与严格格式、实际时间和 CUDA 峰值分别保存在模型结果目录。中断后重跑相同命令会逐分片核对冻结口径、ID、预测摘要及指标。

历史 Ouro-1.4B Base 的八项全量成绩单独复用。跨架构的 Huginn 权重训练与评测是后续阶段。

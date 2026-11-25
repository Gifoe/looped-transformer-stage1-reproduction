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

正式评测单卡串行。云端并行下载检查点时，优先评测已通过完整权重验证的版本。每个模型先完成 ARC-Challenge 的两个配对条件，再运行其余七项。全部预测、配对置信区间、正式样本数、生成合法率与严格格式、实际时间和 CUDA 峰值分别保存在模型结果目录。中断后重跑相同命令会逐分片核对冻结口径、ID、预测摘要及指标。

历史 Ouro-1.4B Base 的八项全量成绩从原归档读取。

## 跨架构 Huginn-0125

Huginn 使用官方发布版 `bb6621b65e90b6a4b9b29ef88dc83866d450470c`。源码、模型配置、tokenizer 和四个权重分片在启动时校验。Huginn 保持其 Prelude、输入注入、随机初态与 Coda，控制器写入第 32 次循环的入口。评测的两个方法共享模型、批次、随机种子、解码协议和 K32。训练覆盖 ARC-Challenge Train 1,119 题，执行 280 次更新，按 8、16、32 轮交替训练 10,560 个控制器参数。目标沿用历史 T1 的原始和字符归一化排序损失。

运行前三个 Ouro 版本后，在研究目录中执行以下命令。Huginn 使用单独的 Transformers 4.49.0 导入路径，其官方 KV-cache 在这个版本通过动态与静态缓存一致性检查。

```bash
study="$PWD"
joint=/root/ouro-external-data-joint-training-20261124
cd "$joint"
source external_env.sh
cd "$study"
"$py" -m pip install --no-deps --target "$study/huginn_deps" 'transformers==4.49.0'
"$py" fetch_huginn.py --metadata HUGINN_RELEASE.json --out /dev/shm/ouro-cross-model-20261125/huginn-0125
bash run_transfer_mechanism.sh
bash run_huginn.sh
bash run_loopcd.sh
"$py" report_transfer.py --results results --historical "$joint/t1_full/results/RESULTS.json" \
    --out REPORT_ZH.md --final --require-huginn
```

功能谱从同一批 ARC Train 题估计。有限剂量对照使用 ARC Validation 中固定的 24 题。Ouro 各版本可比较读写子空间，Huginn 的跨架构比较使用各自归一化的谱和功能响应。

LoopCD 补测复用相同零样本请求和原生循环预算。[LOOPCD_PAPER.json](LOOPCD_PAPER.json) 记录原文可复用的零样本汇总。Ouro 的选择题同次主模型前向计算 Fixed 和 Adaptive。MMLU-Pro 各方法独立生成，Huginn 用单独的早期 Coda 缓存读取第一轮状态。

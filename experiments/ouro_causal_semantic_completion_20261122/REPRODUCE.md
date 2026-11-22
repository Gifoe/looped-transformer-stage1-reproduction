# 单卡复现

从仓库根目录运行。需要完整 checkout，本实验导入同一研究分支里归档的原生前向、整数生成和干预代码。使用 1 张 CUDA GPU。本次实测为 RTX 4090 D 24 GB，Python 3.12.3，PyTorch 2.7.1+cu128，Transformers 4.55.4，NumPy 1.26.4。权重使用冻结的 Ouro-1.4B Base。复现程序在加载前核对权重 SHA-256。[ByteDance 检查点](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1)

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install 'transformers==4.55.4' 'numpy==1.26.4' 'huggingface_hub[cli]>=0.34,<1'
hf download ByteDance/Ouro-1.4B \
  --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 \
  --local-dir /data/ouro-base
exp=experiments/ouro_causal_semantic_completion_20261122
python "$exp/reproduce.py" --verify-artifacts
python "$exp/reproduce.py" /data/ouro-base /data/ouro-csc-run
```

最少预留 4 GB 运行输出和缓存空间。首次复现使用归档的冻结题组 `results/DECK.json`。`csc_data.build` 也记录了题组生成方法、种子及排除的历史输入。Oracle 用当前题的真实第二轮更新，仅用于因果测量；达到 `ORACLE_SUMMARY.json` 中预注册门槛才启动学习阶段。GPU 时间按子进程占用计费，累计上限 18,000 秒。正式生成按首个非空行的完整整数判分，数字生成过程中没有正确前缀或候选过滤。

- `NATIVE.json` 包含 00、10、01、11 和同答案无操作的 11 在 K1/K2 的完整输出。
- `ORACLE_DISCOVERY.json`、`ORACLE_RANK_DEV.json` 和 `ORACLE_CONFIRM.json` 保留逐题的 token、文本、解析后的整数及回滚记录。`ORACLE_LOCK.json` 冻结开发集选择。
- `BANK_INDEX.json` 记录中间缓存的输入和每块 SHA-256。大型缓存由代码重建。`ORACLE_BASIS.pt` 是训练分组拟合的固定正交基。
- `ORACLE_COORDS.pt` 给出提示末尾无生成前缀时的投影坐标、向量范数和一组按组固定取样的全量更新。前 $r$ 个坐标乘以正交基的前 $r$ 行可恢复秩 $r$ 的残差干预。`ORACLE_ATLAS.json` 给出第 14–17 层的写入量和四角交互范数。
- `FORMATION_RAW.json` 和 `FORMATION_SUMMARY.json` 记录同位置、同范数比较 K1/K2 的 MLP 非线性重放项，及 K2 生成项的反向扣减。
- 若学习阶段启动，`LEARNED_SCAN_K*.json`、`LEARNED_DEV_K*.json`、`LEARNED_RAW_K*.json` 和 `LEARNED_SUMMARY_K*.json` 保留选型、逐题结果和计时。已训练的 `operator_*_K*.pt` 由 `csc_learn.deploy` 加载，生成只接收当前提示和冻结模型。
- `LOOKUP_DECK.json` 独立冻结查表。`LOOKUP_RAW.json` 和 `OPERATOR_ABLATION_RAW.json` 保留可能启动的迁移和消融评测。若初始 K1 算子未达到迁移门槛，`TEMPLATE_*` 记录一维方向和可学习门控的后续实验。其实际部署权重可由 `csc_template.load_template` 加载。
- `LOOKUP_LATE_*` 将两种查表题面各前 96 题移到物理第 20 层做 Attention 和 MLP Oracle 控制，未用这些结果重新选择算术算子。
- `PRIOR_AUDIT.json` 记录独立核查过的历史原始数据指纹。
- `ARTIFACTS.sha256` 校验 GitHub 发布件。`GPU_LEDGER.tsv` 记录原始占用，包含失败尝试。

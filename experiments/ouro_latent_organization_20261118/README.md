# Ouro-1.4B 潜在组织与轮间演化

基线为 `5502634381e8a0c532f0a91a9b5cafe8bd7617b4`。模型复用 `ByteDance/Ouro-1.4B` Base 固定修订 `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`，运行条件为单张 RTX 5090、PyTorch 2.7.1+cu128、Transformers 4.55.4、BF16、原生 SDPA、无 KV 缓存。L0–L23 为零基物理层，数组第 24 位单独记录 post-loop norm。模型权重 SHA-256 为 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`，动态代码、词表及配置也做内容校验。

## 输入与判据

每类受控任务 1,000 组，其中拟合 300、探索 100、同结构确认 300、结构留出 300。前三级只使用 family 0/1。留出集专用 family 2，对应更多记录和仓库第三跳、不同前提形式、以及数轴往返。每组完整运行四个因子世界，另外运行预定三分之一组的无关变化世界。三类基准为官方 PrOntoQA 包中的 500 个去重证明条目、全部 299 个 ARC-Challenge validation 唯一 ID、按 ID 去重的 500 个 HellaSwag validation 条目。PrOntoQA 使用证明结论与其否定做诊断性零样本二选一。派生逻辑使用独立合成的 PrOntoQA 形式证明种子。

因素 r/x/y/d 依次为原始、改第一阶段、改后段映射、同时改变两者。r/d 最终真值相同而中间变量不同。r/y 中间变量相同而最终真值不同。n 为无关变化。输入 JSONL 保存完整提示、真值、中间变量、局部文本区间和因子。独立输出标签排列在语义场景构造完成后执行。第一轮采集发现继承的生成器把选项表和语义生成列表共同重排，关系和逻辑的 r/d 正确选项全部固定为 A。新运行保留语义场景，独立重排选项并验证每个真值和 token 角色。上轮涉及该生成器的绝对分数需要重新核对。

所有轨迹均取自完整原题的一次原生 K4 forward。诊断标签、备选中间值和训练的线性探针都没有加入推理提示。技术预检将每个循环投影得到的全词表 logits 与独立 K forward 逐位比较。最高分并列记为 `-1`。组级置信区间按组重抽样，单组内多个因子世界先汇总。

## 捕获与分析

- `natural/<task>/rows.jsonl` 含 token IDs、角色位置和逐轮候选分数。逐世界的 `stats.npy` 轴为世界、轮、物理层或归一化、角色、通道。角色依次为来源、桥接、映射、问题、读出。几何通道依次为角色池化向量 RMS、相邻物理事件的余弦、同层相邻循环余弦、同层相邻循环相对 L2。基准的来源和映射按位置定义，作为几何参照。
- `vectors.npy` 在原生浮点池化后，采用同一个固定正交随机投影保留 96 个特征，所有层与轮共享该基。全宽控制另存 2,048 维 BF16 池化状态，以相同训练组、测试组和正则强度对比 96 维与全宽。完整 token 子样本保留 BF16 bit pattern、每层和归一化、以及指定层的 attention/MLP 输出。
- 线性读出只使用 fit 组。中间变量和最终变量以真实语义值评分，类别在该样本候选语义之间比较，算术使用连续数值回归后选最近候选。除各位置独立估计外，还把同一组固定权重跨循环或跨角色使用。保留 hash-token 表面基线、未见人名与数值、同组受控距离、带 token Hamming 协变量的距离分解、因子能量和稳定选择下的几何变化。
- 只用发现集选择因果位点。用关系和逻辑发现集选出的共享预测在算术确认和结构留出上独立检查。因果位点释放后再读取确认结果。每类 90 组，含 60 个同结构确认、30 个新结构。完整同真值供体、去除线性最终读出子空间后的中间读出方向、反向、正交、随机、无关、映射变化和自替换均保留。干预后的候选评分在接收方上下文重新运行。
- 另取与探针干预互斥的 60 个关系组，在纠正选项后复核旧路径。K1/L5 注入，K2 切断来源→映射边并在 K2 末恢复映射向量。开放和切断各自有无注入运行，使用交互差分。样本无需原生答对才参与。

## 复现

环境与源文件下载命令见 [前序实验](../ouro_cross_task_20261118/README.md)。在仓库根目录运行。根输出目录须为新目录。

```bash
E=experiments/ouro_latent_organization_20261118
P=experiments/ouro_expanded_causal_20261118/results
export OURO_PYTHON=/absolute/path/to/venv/bin/python
CUDA_VISIBLE_DEVICES=0 bash "$E/reproduce.sh" /absolute/path/to/new/study \
  /absolute/path/to/model /absolute/path/to/sources "$PWD/$P"
```

运行 `audit_org.py results` 可对提交中的数值与逐组记录做离线审计。全量投影向量、BF16 全宽控制数组和完整 token 样本按 SHA-256 保存在实验服务器的 `/root/autodl-tmp/ouro-latent-organization-balanced-20261118`，文件索引为 `results/raw_tensor_inventory.json`。在服务器给审计加 `--include-server-tensors` 可逐个核验。Git 保存脚本、语义清单、逐世界分数与几何通道、诊断和因果统计、因果原始记录、图和日志。

- `results/shared_dynamics.csv` 是冻结的跨任务检验。
- `results/readouts.csv`、`cross_loop.csv`、`factorial_variance.csv` 和 `paired_distances.csv` 覆盖语义组织与跨轮泛化。
- `results/success_failure.csv`、`correction_association.csv` 和 `prediction_stability.csv` 提供成功/失败轨迹。
- `results/causal`、`route_retest` 和 `width_control` 含对照结果。
- `results/figures` 包含六组 PNG 和 PDF。

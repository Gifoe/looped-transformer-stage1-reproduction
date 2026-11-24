# Ouro 单一 Rank-1 外部数据联合训练

本目录锁定 A/B/C/D 的训练输入、题级调度和正式评测口径。数据与 CPU 调度检查通过。GPU 小样本已在 K2、K3、K4 各完成一次更新并执行真实生成，正式评测结果尚未产生。 [GPU 验证回执](results/SMOKE_RECEIPT.json)

## 对照

| 编号 | 拟合数据 | 训练目标 |
| --- | --- | --- |
| A | CSQA 1,000、OBQA 1,000、GSM8K 2,000 | 完整回答 NLL |
| B | 与 A 同一批题、视图、初始化和更新调度 | 完整回答 NLL + 0.25 候选排序 |
| C | CSQA 和 GSM8K 保持不变，OBQA 换成 1,000 道 ScienceQA，监督为短答 | 用开发集在 A/B 中选择 |
| D | 与 C 同一批原题，一半 ScienceQA 替换为原始解释加最终答案 | 与 C 相同 |

每个有效更新包含 2 道 CSQA、2 道科学题、4 道 GSM8K。生成损失在每题目标词元内归一，再在有效更新的题目上平均。骨干冻结，仅优化 4,096 个 FP32 控制参数，沿用倒数第二圈 RMSNorm 前的写回与 0.45 相对 RMS 限制。K2/K3/K4 顺序轮换。初筛按 1、2、3 个真实 epoch 保留权重，只有预先规定的连续改善才会增加到 4 或 5。两种入围配方各使用三种种子，配方由三种种子的开发指标中位数决定。最终只归档一套正式部署权重。

如果 D 在三轮等题数比较中超过 C 且多消耗至少 10% 骨干 token×loop，会加做同种子的 D 训练，达到 C 的实际工作量后在首个完整更新处停止，记录剩余差异和总更新。该对照只用于核对计算成本的解释。

开发选择优先减少比 Native 低 2 个百分点且配对区间上界低于零的端点数，再比较相对 Native 的最弱差值、相对事前强参照的最弱差值和等权任务类平均。原七项采用表中已锁定的 Stage2 请求，主指标在配置里固定。MMLU-Pro 复用原有零样本全词表生成与答案解析。可核验的 Native、T1、T2 逐题历史回执直接复用。历史 T1/T2 使用完整 ARC Train 1,119 题、280 步，分别是候选排序和内容／字母生成加排序。Retention 源于 CSQA/OBQA 成对短答训练，普通 mixed 来自本系列前一轮的四源训练。

## 单卡运行

在上游模型、历史评分器和官方输入均可用的服务器中配置 `external_env.sh`。以 `data/MANIFEST.json` 的固定 revision 和 SHA 取得 ScienceQA 两份源文件，置于 `inputs/`，保留 `inputs/SCIENCEQA_SOURCE.json`。

```bash
source external_env.sh
CUDA_VISIBLE_DEVICES='' "$py" fetch_scienceqa.py
bash prepare_data.sh
CUDA_VISIBLE_DEVICES='' "$py" prepare_panel.py --data data --gen "$gen" --transfer "$prior_transfer" --advantage "$advantage" --out data/panel
bash run_stage1.sh
bash run_after_stage1.sh
```

训练入口会保存真实更新、每题与每种视图覆盖、监督词元、骨干 token×loop 和实际 GPU 时间。正式评测从已经冻结的 `results/FREEZE.json` 校验模型、检查点、评测输入及代码。逐题文件按分片写入，可复用同指纹的已封存分片。开发集结果用于选型，K5/K6 和正式测试留待冻结后评估。 [训练数据与测试人口](DATASET.md) · [配置与固定指标](configs/study.json)

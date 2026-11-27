# Ouro ICML v5 实验结果

截至 2026-11-27 15:25（北京时间），Phase 1 已按预设门控收尾，裁决为 **PHASE2_HOLD**。反重力另行运行的 Phase 2 探索脚本完成四个模块。原计划的七模型纯语料自举与全量确认仍未完成。

## Phase 1

- 严格 1A 完成 24 个诊断单元并判定 FAIL。最接近门槛的配方有 239 道有效四选题，过滤后原句恢复率 94.87%，读出差 +23.08 个百分点，预计保留 476 条，低于 1,000 条门槛。严格 1B 和后续对称补测按门控停止。
- Track B 的 60 份同源同种子回执及权重均已核验。全层 LoRA 和全层 IA3 在原 K4 协议下各评测 43,769 题。
- 另行开展的 CSBP v2 使用 371 条过滤后的训练探针。WikiText 送测权重按外部开发集 Macro5 和 NLL 约束选择，原协议要求仅在无标签留出折选型。其正式测试覆盖全部八项，研究结论应按探索结果解释。

| 43,769 题，八项等权 | 可训练参数 | Macro8 |
| --- | ---: | ---: |
| Native Ouro-1.4B 历史锚点 | 0 | 66.96% |
| V4 零人工标签 Rank-1，使用 ARC 题干 | 4,096 | 73.14% |
| 全层 IA3 | 147,456 | 72.79% |
| FixW_MeanMu | 2,048 | 72.58% |
| Pareto_ZeroNLL | 4,096 | 72.53% |
| 全层 LoRA Q/V | 196,608 | 72.50% |
| ZeroQA_WikiText，探索性选型 | 4,096 | 69.45% |

纯 WikiText 比 Native 高 2.48 个百分点，离 72.0% 预设目标差 2.55 个百分点。五个本轮送测方法共 218,845 条逐题预测。每组评测回执、配对统计、MMLU-Pro 生成和实际成本均随本分支发布。

## Phase 2

四份开发集结果覆盖 Qwen2.5-1.5B、Ouro-2.6B Base、Huginn-0125 和 Ouro-1.4B Base 深度对照。主要的零人工标签臂使用 ARC 题干，监督对照另列。原计划中的 Ouro-1.4B-Thinking、Ouro-2.6B-Thinking 和 Llama-3.2 缺少对应结果，四个已完成探索模块也缺少同协议的全量官方确认。脚本结束标记代表这四个模块完成，正式准入仍为 PHASE2_HOLD。

参阅 PHASE1_ICML_GATE_REPORT.md、PHASE1_MASTER_SUMMARY.json、RELEASE_STATUS.json 和 phase2_cross_model 下的模块报告。官方逐题文件以原始 JSONL 的无损 gzip 形式保存，配套 JSON 回执包含原始 JSONL SHA-256。


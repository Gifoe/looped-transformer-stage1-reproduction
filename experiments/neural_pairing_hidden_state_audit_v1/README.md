> GitHub delivery: [complete report](REPORT.md) · [code/results/source ZIP](https://github.com/Gifoe/looped-transformer-stage1-reproduction/releases/download/neural-pairing-nl33w-arc-audit-20261030/neural_pairing_audit_code_results_20261030.zip) · [raw states/data](https://github.com/Gifoe/looped-transformer-stage1-reproduction/releases/download/neural-pairing-nl33w-arc-audit-20261030/neural_pairing_audit_raw_20261030.tar.gz) · [release](https://github.com/Gifoe/looped-transformer-stage1-reproduction/releases/tag/neural-pairing-nl33w-arc-audit-20261030).
> Extract raw data first, then code/results; the release ZIP contains source archives omitted from this Git tree.

# Neural Pairing: nl33w ARC causal audit

本轮已完成：完整1165题BF16 ON/CUT逐题分析、16题探索、95题原始批量8层干预、16个新目标的验证（原批量含240题）。三阶段共343个不同问题；335题恢复原始评估形状。因果worker合计44.09分钟。严格nl23重训练保持暂缓，准备数据和历史日志保留。

**研究结论：未确认足够稳健的新隐藏状态机制。** 正确伙伴也会造成伤害；存在局部reverse路径候选，但16个新案例中，范数匹配K/V修复在native与FP64输出点积下同时成立的为0/16。FP64点积并非全FP32模型。Gate参数为常量，不能解释为学到的输入依赖检测器。没有Loop依据。

- [完整报告](REPORT.md)：指标、干预、数值失败、对照和研究结论。
- [失败及纠错案例](CASE_STUDIES.md)：原题、选项、逐层margin及官方nl23独立历史对照。
- [最新范围](CURRENT_SCOPE.md)、[两阶段干预计划](CAUSAL_FOLLOWUP_PROTOCOL.md)、[验证计划](CAUSAL_VALIDATION_PROTOCOL.md)。
- [复算说明](REPRODUCE.md)、[竞争假设](hypothesis_matrix.md)、[后续判断](NEXT_STEPS.md)。

| nl33w / ARC1165 | CUT正确 | ON正确 | 纠错 / 损害 |
|---|---:|---:|---:|
| Quill | 1010 | 1003 | 19 / 26 |
| Ember | 967 | 987 | 42 / 22 |

`fresh/arc_first/`保存全量题目、两侧四选项logits/概率和转移；`fresh/causal_pilot_16/`保留不稳定探索样本；`fresh/causal_original_shape/`为95题精确原始批量扫描；`fresh/feedback_validation/`为全部剩余16个正确伙伴伤害目标，含成功/失败结果。根CSV按study标识合并7370行干预和4064行逐层诊断；不能把选择性病例修复率称为benchmark收益。

官方nl23的964/83/50/68分组与本次nl33w的902/108/65/90分组分开保存。`checkpoint_separated_question_comparison.csv`逐题对齐但不混合权重、分母或机制归因。作者nl23记录仅有正误，缺少错误字母、概率及状态。

`artifacts/neural_pairing_audit_raw_20261030.tar.gz`保存完整fresh输出及选定位置的状态张量。原始SHA见sources/RAW_ARCHIVE_MANIFEST.json；下载后解压到本目录可离线重算。代码/报告包不含模型权重、SSH凭证或签名下载链接。`STATUS.json`是当前状态；sources/historical/中的早期报告不代表当前进度。

次要margin结果：固定FP32 Write范数的CUT K/V在16个新目标中13个提升margin（native均值+0.1641；FP64-dot均值+0.1825），但没有新增两种读出均成立的纠错。13个保留原始转移的目标中11个margin提升。该效应值得用匹配随机方向和语义值置换继续检验；不能把失败的原定纠错标准事后改成成功，也不能把0纠错说成没有margin因果效应。见secondary_margin_effects.json。

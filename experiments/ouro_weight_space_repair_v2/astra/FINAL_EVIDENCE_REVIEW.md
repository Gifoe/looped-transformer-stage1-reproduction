# V2 最终证据核对

范围：全部 1,172 题推理完成后，对已下载最终报告、CSV、FINAL_ANALYSIS、权重审计和预算收据进行只读核对。没有运行 GPU，没有修改方法、原始结果或源脚本。本文件不引入任何新候选、深度选择或剔除样本分析。

**结论：数值一致；本次选定编辑没有建立两项主比较的提升。** 两项主比较都是负向点估计，但差异均不显著，不能将其表述为已证明的真实退化或等效性。建议报告开头及第 5 节的“准确率下降”明确写为“点估计下降，差异不显著；未建立提升”。这只是统计结论措辞，不改变数值。

## 七项核对

1. **实际编辑与选择。** 最终候选是 B05：物理第 17 层 attention o_proj 的 rank1、alpha=0.5 稠密方向投影。A/B 两个家族均实际运行；14 个 screen、4 个 selection、3 个 validation、1 个 final，共 22 条候选阶段记录。验证集 B05 为 168/299，A08 为 163/299，A02 为 161/299；选定 B05 与规则一致。14 候选报告表中的全部改变坐标数及各阶段 K4 正确数均与 edit_search_results.csv 一致。

2. **普通推理主比较。** 原始 K4 702/1172（59.898%），编辑 K4 698/1172（59.556%）：17 题纠错、21 题退化，净 -4 题，-0.341 pp。配对 95% 区间 [-1.365, 0.683] pp；exact McNemar p=0.62710257，两项主比较 Holm p=1。没有改善证据，也没有显著下降证据。

3. **LoopCD 主组合。** 原始 Fixed 717/1172，编辑 Fixed 715/1172：17 题纠错、19 题退化，净 -2 题，-0.171 pp。区间 [-1.195, 0.853] pp，exact p=0.86793940，主比较 Holm p=1。Adaptive 均为 716/1172，23 题纠错和 23 题退化，净零。两种组合使用同一选定编辑的自身 K1/K4 状态；固定组合没有胜过对应原始组合。

4. **跨深度及泛化边界。** 普通 K3/K4/K5/K6 的净变化依次为 -7/-4/+6/+3 题（-0.597/-0.341/+0.512/+0.256 pp）。所有六项比较的 95% 区间都跨零，all-six Holm p 都为 1。K5/K6 小增不能称作稳定跨深度改善，不能用它们替换预定 K4 主结论。报告四个数据用途的正确数和差值均与 FINAL_ANALYSIS.generalization 一致；selection/validation 参与选择，其增益不是独立泛化证据。

5. **重复样本与测试性质。** 审计与报告均披露规范化问题文本 discovery/test 重叠 3、validation/test 重叠 2；selection/test 和 demos/test 为零。官方全量测试保留 1,172 题，没有事后删除重复题或另做择优口径。涉及重叠的题目不能视为独立于训练发现或验证选择的新证据。该 test 亦有历史使用，报告正确限定为冻结后的回归基准，不是新盲测。

6. **实际 BF16 改变量及重放。** B05 改变 3,570,385 / 1,434,652,673 = 0.248868% 个 BF16 坐标；参数为 `model.layers.17.self_attn.o_proj.weight`，shape [2048,2048]；实际 delta Frobenius 3.9742782116、相对范数约 0.08982358。报告、selected manifest、ARTIFACT_AUDIT 一致。成功 selected gate 收据覆盖 269 个参数张量；全部八种训练读出分数重放误差为零，own-edited Fixed/Adaptive 误差为零，独立深度检查和精确恢复/patch 再载入通过。该 gate 证明执行与重放，不证明准确率改善。

7. **预算和失败透明性。** ledger 总计 8427.875679 秒 = 2.34107658 小时，小于 21,600 秒 / 6 小时。逐项为统计 819.540033 秒、主 pipeline 7531.545412 秒、首次 selected gate 失败 34.196038 秒、retry 成功 42.594196 秒；求和与报告和 ARTIFACT_AUDIT 一致。首次失败被保留且计费，没有从成本中删除。根据执行方说明，该失败来自新增 observer hook 后未 invalidate 的保护检查；当前代码已在 observer 注册和移除后 invalidate，成功 retry 收据已核验。这里未将该工程失败解释为模型效果失败。

## 交叉核验与限定

程序化核对了全部六行 readout/depth 准确率表、六个 paired 统计条目、14 候选表、四个数据用途表、参数占比、269 张量与八方法 gate、预算求和及重叠披露。Unsupported LoopCD 深度为 NA。报告的精度舍入与 CSV 一致。

结论只约束这 14 个预先定义的候选和本次选择过程，不能升级为所有训练免梯度稀疏/低秩编辑均无效。另一方面，真实完成两个编辑家族和严格重放也不能补足缺失的主指标提升。

核对时文件 SHA256：

| 文件 | SHA256 |
|---|---|
| REPORT_ZH.md | 99487b98b912bdc21454dcd0b27dafe39d04d58adf95a0d8eab02e5a2e416542 |
| final_comparison.csv | 67e589a5c313dc5c76dbf6c7ea9c31330c57628898a8921090c256340357bd33 |
| paired_statistics.csv | 2d844d8523cafd4eed29f642f2a198bf0e2756c26df5241aaf0bfc90afca90ef |
| edit_search_results.csv | 4704873b7110aab5787b1bb7d9d4a01dbbb49000ff44f4d5e115489ac86f7799 |
| results/FINAL_ANALYSIS.json | 9ccbab6701096de925ed586c0756ed9fe76faf4f609dce844ab15cb6459e054e |
| ARTIFACT_AUDIT.json | 8f9fa39e9d723f3a97a6d6dd06822bf14a392a2b7505c3e8f51d194f8872a511 |
| results/SELECTED_CHECKPOINT_AUDIT.json | 6540553b2f21b2aaa9f49a106225c3b1ee9a2a1e66ad6a85f9acbcc5d04dd481 |
| logs/gpu_budget.json | ff7892695bba667ba98ce5ebfe21ed2fb8e52b3e335ef12ccf2c5ef3bced1edb |

上述报告哈希记录的是本次审阅快照；后续仅统计措辞或图表版式更新应重新记录发布哈希，保留原始数字不变。

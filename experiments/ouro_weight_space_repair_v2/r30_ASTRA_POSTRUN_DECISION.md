# R30 Astra完整结果决策

固定C关闭。完整官方解封及Root跨raw/代表公式独立检查支持：primary13/24，历史/matched R6各14，均0 rescue、1 damage，唯一旧收益保留0/1；strong与cost_review均false。236/421 unique prefix、306/548 use的alpha活跃，不是未实施或全部零修正。不能改clock、符号、来源、reference、强度来挽救这次假设，也不能提升control。

依据完整r30_ANALYSIS/COMPLETE/SCORES、r30_ROOT_POSTRUN_VERIFICATION和r30_ROOT_ALL24_MARGIN_INVENTORY。Root报告C相对A1 gold-bestwrong margin 9增/2零/13降，平均负；与A的方向和数值并不相同，却没有预测翻转。这不支持“方向普遍正确，只是力度太小”。785 bodies、778CF、7194nativeheads及完整parity意味着这次有限C的失败不应被解释为cache失效。

本轮另经授权执行的CPU piece分解见r31_CPU_MARGIN_DIAGNOSTIC_PROTOCOL.md、scripts/r31_cpu_margin_decomposition.py、results/r31_CPU_MARGIN_DECOMPOSITION.json：全部24上，R28和R30均primary pre-late正确13、post-late正确13，late veto/rescue均0，Fixed→primary的pre-late rescue/damage也均0。没有“本来救对，却被固定late取消”的实际案例。相同gold/wrong pair的修正增量不会因加同一late项变号；本诊断不是去late候选，不授权调整晚期项。

数值说明：该脚本的Fraction late ratio取 `Fraction(native4.sum)/char−Fraction(native5.sum)/char`，是理想精确组件分解，不是 `Fraction(stored rounded late)`；其符号不是实际FP64最终score的逐位表示或部署可达证书。上述zero-veto结论另由实际保存float的P+late、最终scores及winner exact核对支撑，不依赖将理想ratio误作实际评分。

解释边界：R27只证明限定lookup对应信息可以由V传递；R28与R30说明两种不同来源反事实没有转化为本pilot的纠错。尚不能据此推全部因果先验或无训练修正不可能，也不能将任何新增向量当成新颖原理。来源干预、校准、function vectors、CAA的先例已在r30_ROOT_PRIOR_ART_NOTES.md列明。

18ledger累计20755.46525352029/21600，余额844.5347464797087，不追加或重置。目标未达。下一步选择CPU表达能力与价值识别的分离挑战，见r31_EXPLORATORY_IDENTIFIABILITY_DESIGN.md；它是诊断，不冒充方法改进。R29 canonical没有被验证，不计为可用加速或方法结果。

UTC: 2026-11-11T10:48:28Z

# NATURAL_L18_DISCLOSURE_ORDER_AND_TRAIN_ANALYSIS_AMENDMENT_20261111
train仅在1119全题保存且只读CPU preliminary审计通过后探索，主GPU继续val/test；读取任何val/test L18效果之前，基于完整train冻结新增预测/判据，之后各split完整再揭示。若最终独立CUDA重放尚未执行，train只能标CPU_SOURCE_BASELINE_IDENTITY_PASS_INDEPENDENT_CUDA_PENDING；collector逐题CUDA回读不能冒充独立审计。CPU预审须精确1119 IDs和原source/meta/tensor SHA、6必需+2norm状态一致完整键、BF16[1,V]有限、原B1/B2及I1/I2位等、全部原生scores/predictions/ties/fixedwrong一致、实际source x/a hashes；CPU近似logsoftmax只诊断，绝不能放宽最终CUDA零差或由近似分数重定胜者。norm recipe先CPU诊断，不用CPU重算误差冒充同CUDA相等。无须暂停主GPU做早期重放。
训练集1119与非demo1094分别报告；四自然组由原B1/B2 gold correctness冻结，不按treatment再分组替代；natural无top-tie另分列，所有干预引入/解除的top tie都记录。若另报各condition无tie子集，标为post-treatment选择敏感性，不能替代全体。
逐题冻结wrong=原B1最高wrong(first argmax)，主gap=gold对全部nongold的平均（先题内再题间等权），fixedwrong及dynamic gold-vs-current strongestwrong都保留，完整scores和target原BF16 logits保留。主量L2=G(B2)−G(T21)，G1=G(T12)−G(B1)；norm版本各自同定义。N/mean/median/q05,q25,q75,q95/正负零，联合(L2,G1)四象限+含零；不只报两个均值。报告两gap方向冲突，勿把allwrong增益等同预测变好。
决策：每条件各自accuracy、相对其receiver baseline的全候选预测保持/改动、correct→wrong、wrong→correct、wrong→differentwrong。T21净损失=correct→wrong−wrong→correct；T12净增益=wrong→correct−correct→wrong。同时在原natural rescue组数T21是否撤销B2救回、T12是否复制B2的gold胜出；在damage组数T21是否恢复gold、T12是否复制原damage；稳定组完整报告。原natural预测相同也仍保留全部score变化，不把“终点同另一baseline”笼统叫成功。
控制：I1/I2必须零；norm与full swap逐题比较效应差、方向一致/反向/零及预测差异，记录实际BF16范数误差分布。UNDEFINED控制N及ID列明，主swap仍覆盖全cohort，不静默删除。反例按各自然组L2/G1最正/最负与两方向矛盾事先规则列出，排序值后ID。
解释/下阶段：只单向T21掉分而T12无改善，最多支持该receiver对替换敏感，不证明K2信息充分；双向均符合且范数控制保留则支持L18 a_R所携带差异的可转移因果作用，但不是K1→K2总增益的唯一中介。norm控制改变结论则限制到幅度或混合态效应；negative/zero如实反证，不能换层/换gap/调控制救假说。K1与K2上游不同、T是非自然混合，局部do效应不等于天然推理机制；后续MLP参与不可剥离。完整train结果到后再冻结最少跨split符号/行为预测；不在当前预先指定未观察到的结果。

真实UTC记录：2026-11-11T09:58:49Z

# NATURAL_BRANCH_TRAIN_INFORMED_L18_PROSPECTIVE_CHECKS
依据/披露：使用已完成train1119的初步CPU锚审计（CUDA独立重放待完成），主要探索为非demo1094；projection SHA1cf627697be7c3f8d6f4c1904242b8b921b926ebb6ffdaaa3ee58fe762874ec0，源rows SHAc280f4ee3719c96d04e18742050b012e36a7361c197e3affd702c5ddc84bf4af，coverage SHA61132388ee5403a042cc1ab8fbca17c59192e168866ea6cbac084a49c368440f，disclosure SHA95ef8aa7cbd8aadd7faa23ee42f5ec67fa80c075e25555dd330f98e86b6e57b0。L18从完整自然层轨迹探索及train分支分解选择，非首次运行前注册。此前三split自然最终表现/状态已揭示，val/test分支效果尚未揭示，因此仅项目内train-informed前瞻分支比较，不称独立盲验证。
定义：物理layer18(index17)，每轮k：A_k=g(y_k)−g(x_k)，M_k=g(z_k)−g(y_k)，dA=A2−A1，dM=M2−M1；g为相同FP64末端RMSNorm/head候选gold-vs-wrong gap，实际a为attention输出经过input_layernorm_2后的残差增量。主要gap=所有non-gold逐gap题内等权；固定K1最高score wrong gap为预定敏感性，first argmax不变。
主预测（validation299与test1172各自单独判定；完整cohort和natural无top-tie子集分别判定，不合并）：P1 mean(dA)>0 且 mean(dA−dM)>0。P2 个体优势 median(|dA|−|dM|)>0，同时原样报告大于/等于/小于0数量；不以两组median相减代替逐题差。两条件任一不成立则对应split/子集未支持该预测，零为未支持而非正向。P2幅度不代表有益方向。
条件组预期（次级，不能挽救主预测失败）：自然rescue中mean(dA)>0且正数多于负数；damage中mean(dA)<0且负数多于正数。稳定正确/稳定错误同样完整报告，不预设必须符号；所有组保留N、mean/median/q05/q25/q75/q95、正负零、dA/dM联合符号、候选gap方向冲突和固定规则极端反例。固定wrong/no-tie若与主要结果不一致，明确限制适用范围，不择优替换主要gap。
完整报告：依旧所有24层的A/M/T，不用L18掩盖L21/23/24或反例；报告起点+ΣdA+ΣdM=终点。统计不调参、不按valtest结果重选层/方向/分组/阈值。bootstrap如做仅题级描述性不确定性，非挽救上述符号判据。
解释边界：支持只意味着自然候选读出坐标的L18变化主要位于attention残差步骤；不证明其因果必要/充分、不定位Q/K/V/头/语义来源，也不排除attention之后本地norm或R初态变化。mean dM小不等于MLP无用，后续层及MLP的attention输入依赖仍存在。若主预测反转，放弃跨split稳定L18签名解释；即使通过也须进一步检验实际a与归一化/分母，而非宣布机制已找到。

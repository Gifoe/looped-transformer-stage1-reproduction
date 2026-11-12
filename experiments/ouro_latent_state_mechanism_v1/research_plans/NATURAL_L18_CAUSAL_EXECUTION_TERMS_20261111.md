UTC: 2026-11-11T10:27:32Z

# NATURAL_L18_CAUSAL_EXECUTION_TERMS_20261111
(1) 主终点必须最终原生产候选sum_loglikelihood计算的gap/预测，不用FP64 lens替代；g固定wrong=自然K1最强wrong，allwrong逐题均值；全部candidate原始BF16词表logits保存及同CUDA score独立回放。L2=B2−T21、G1=T12−B1分别报告，禁止把二者相加当自然总增益的唯一分解。
(2) norm控制固定算法：在CUDA上source BF16 a转FP32，以torch.linalg.vector_norm L2计算 donor/receiver norm，a_ctrl32=a_donor32*(norm_receiver/norm_donor)，转回BF16一次；记录原/目标/实际BF16 norm及误差、不自选容差当相等；计算只用于新控制，baseline/donor原值不变。norm_donor==0时该控制UNDEFINED并保留主条件，该题不剔除；不可加epsilon改控制定义。
(3) hook作用只ut/layer/R精确一次，site actual input_layernorm_2；capture在目标层prehook核实际current_ut与physical identity，不靠调用序号猜。patch输出clone后仅R赋值；现场核outside-R输出与原输出位等、receiver residual x_R在注入前与本条件无patch基线位等。I1/I2全词表及全部候选零差；若donor天然等receiver，cross条件也须等baseline。所有hooks/depth finally恢复。不得把T21等B1、T12等B2作为门槛。
(4) 用既定三题A/B/C（两同长、最长2032）作技术锚，不能按效应选题。先CPU合成24层×2轮：错loop/layer/R/题/source拒绝、受控a替换及完整MLP后缀oracle、identity/zero-difference；再真实GPU三题核B1/B2原48状态/postnorm/logits/score、I1/I2、outside-R、注入值及后续重算。不要求干预非零/正确率升降。通过才同冻结代码完整2590，不另加大工程；继续复用现有全局锁/原子分片/恢复审计/40GiB同进程策略和一次真实资源测量。
(5) 预测解释预先冻结：全体各split mean L2>0、mean G1>0是双向支持；rescue组两者偏正、damage偏负为次级，报告个体正负零和全候选预测翻转/净救伤、no-tie/25demo分列。仅单向掉分最多说明脆弱性；有无norm匹配若方向不同，限制到幅度或混合态效应。主条件不支持不换层补救。该实验有local do输出替换的因果效应，但不能单独证明自然K1→K2总改善的中介比例或attention语义来源。
(6) 当前全部自然/分支结果已看，实验是据全cohort观察提出的新假说，不将这些split称独立验证；运行前冻结此事实。首三题真实锚请发快照给我只审关键数据/匹配后直接继续全量，无需再向用户求权。

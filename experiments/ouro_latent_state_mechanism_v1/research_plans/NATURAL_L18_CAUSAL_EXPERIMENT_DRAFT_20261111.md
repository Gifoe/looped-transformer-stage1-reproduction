UTC: 2026-11-11T10:24:20Z

# NATURAL_L18_CAUSAL_EXPERIMENT_DRAFT
请先把下列草案落盘、准备复用现有branch hook/core，待test独立复核与正式执行条款确定再跑GPU，不扩工程。核心是“固定receiver轨迹的同题分支输出替换”，不是声称K1/K2层输入相同。
site：physical L18/index17，真实 `input_layernorm_2` 输出a_R（attention后、加到residual x_R前），ut0=K1、ut1=K2。R为同题真实predecessor；同输入/target/原25-shot/BF16/SDPA/no-cache。来源为已全覆盖自然分支保存的同题a1_R/a2_R；逐题source metadata/tensor hash、ut/layer/R绑定。禁止把a1/a2当匹配上游输入，它们来自不同自然x和context。
最小6条件：B1原生K1；B2原生K2；I1同receiver a1→a1；I2同receiver a2→a2；T21在K2 ut1 L18仅R用自然a1替换a2；T12在K1 ut0 L18仅R用自然a2替换a1。所有其它token、receiver x_R和该层输入完整保留；随后真实BF16残差加法、MLP以及L19–24全部由receiver重新计算。I1/B1、I2/B2必须全词表logits零差；B1/B2与原自然完整状态和候选scores零差；hook exact1且匹配真实current_ut/index。替换后不得要求结果等于另一轮天然终点，也不得强行替换MLP。
两项预定幅度敏感性（共8条件，建议保留而非另造随机对照）：T21_norm使用a1方向、receiver a2的L2范数；T12_norm使用a2方向、receiver a1的L2范数。实际转换方法/设备/dtype先冻结，记录BF16后的范数误差；零donor范数判UNDEFINED控制但不删除主条件/该题，不私自epsilon改值。这仅区分该输出幅度混淆，不等同on-manifold，也不替代完整swap主效应。
所有候选完整BF16 logits和原score逐candidate保存；gold/固定wrong从原K1冻结，另全non-gold题内等权及动态margin/argmax/top-ties。定义L2=G(B2)−G(T21)（去掉K2原a的损失）；G1=G(T12)−G(B1)（给K1 a2的增益）；norm版本同样单列。三split完整执行、train25demo另列、四自然组+no-tie+正负零/分位/净救伤与反例，不能按效果筛题/层。
待冻结假说：rescue组L2/G1偏正，damage组偏负；全体期望mean L2>0且mean G1>0作为双向证据，单向损伤不能声称信息传递而可能是扰动脆弱性。若L2或G1不支持，不换layer/换候选定义救结果；a幅度控制与原swap不同则限制解释。因果结论最多为该receiver下该节点替换改变最终决策；混合状态OOD、同题donor仍受上游状态差异影响，不证明自然总增益由L18唯一生成或attention源信息已定位。

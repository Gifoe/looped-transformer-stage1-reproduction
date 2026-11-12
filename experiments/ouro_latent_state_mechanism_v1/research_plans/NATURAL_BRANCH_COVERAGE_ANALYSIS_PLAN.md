独立coverage：以原自然coverage精确ID集合1119/299/1172验每题原metadata/tensor SHA及本次配对文件SHA，拒重复/缺失/额外题；核gold/targets/R绝对位置与saved_row、原单token各候选输入同一性。每题240个x/a/y/b/z均严格BF16[1,2048]且有限；两norm同形状、两logits BF16[1,49152]，无多缺key。逐位核48个z、两postnorm、两logits与原自然锚；检查每轮i>0的x_i=z_(i−1)，loop2 x0=loop1 postnorm，y=BF16(x+a)、z=BF16(y+b)。loop1 x0不错误地要求等于某层旧态。attention48事件恰ut0/1×layer0..23，核必需实际记录字段、cache/past、positions/cache_position、mask/RoPE摘要与同题原验收规则一致；None mask允许，不声称kernel已知。独立用保存raw x/a/y/b/z+固定gamma/W重新算lens及分解，逐字段对照；最终候选scores重放保持原CUDA FP32 logsoftmax路径，不能用CPU舍入容差替代原零差。各项通过后才把PENDING升级为完整coverage。

分析：每题固定gold与K1最强wrong，另对全部non-gold逐gap再题内等权；全候选z与预测另保留。对每轮k、每物理层l，A_kl=g(y)-g(x)，M_kl=g(z)-g(y)，T=A+M；主量dA=A_2l−A_1l、dM=M_2l−M_1l、dT=dA+dM。所有24层、三split分别报告，train同时全1119和非demo1094；自然K1/K2 rescue/damage/stable-correct/stable-wrong及natural-no-top-tie分列。报告N/mean/median/q05,q25,q75,q95/正负零计数，dA与dM同正/同负/相反/含零的联合计数；不得只比较均值或用dA/dT近零比值。层18/21/23/24作为此前探索提示，须同时展示全24层，不能称独立预选验证。

每个A/M再按已验对称分子/分母恒等式拆分；另保存线性直接投影w_gap·gamma·a/b、真实残差位移w_gap·gamma·(y−x)/(z−y)和BF16加法残差项，区分分支输出与归一化后读出。原生最终margin与这些量关系只作描述；以完整候选score/prediction/ties校验，不把mean-allwrong等同argmax。

反例按预定规则每split每组列最负/最正dA、dM与dT及相消例，按ID解tie，完整表保留。候选均值/固定wrong方向冲突也计数。解释限定：自然轨迹的头坐标分解；A包含attention输出后的本地norm，M包含MLP后的本地norm；M输入由attention产生，不从M大推出attention无关。没有干预必要性/充分性结论，不把跨轮同层差等同该层唯一新增计算。

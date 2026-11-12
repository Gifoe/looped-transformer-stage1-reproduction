UTC: 2026-11-11T11:27:12Z

# TRAIN-INFORMED L18 NATURAL ATTENTION SWAP PROSPECTIVE CHECKS
范围：physical L18/index17、自然a1/a2在R的双向交换，后续MLP/层由receiver重算；由完整train1119（主分析非demo1094）结果引导。validation299、test1172分别评估，完整各split及仅B1/B2自然无top-tie子集分别报告；当前L18效应尚未揭示，但原自然及先前干预已经披露，不能称独立盲验证。完整split经独立coverage后报告最终结论，提前CPU结果须标CUDA_PENDING。
主要连续终点：每题先计算gold对每个非gold的原生产score gap，再题内等权平均；L2=G(B2)-G(T21)，G1=G(T12)-G(B1)，题间等权。P1：每split全体与自然无tie子集分别mean(L2)>0且mean(G1)>0；任一不满足即该范围未复现，不跨split合并挽救。固定B1最强错gap为并行敏感性，不替代主终点。
主要决策终点P2：每split全体T12相对B1的救回数减损伤数>0；自然rescue子集中，T12达到gold的题数>T21离开gold的题数。分别报告分母、比例和配对差，不把不同receiver比较称必要性/充分性比例；自然无tie重复检验。任一方向不满足如实列为未复现。
控制敏感性P3：采用现有receiver范数匹配控制，不追加实验或选择阈值。分别检验mean(L2_norm)>0、mean(G1_norm)>0与T12_norm净救回>0，并列主干预与norm逐题差、正负零/翻转。此为方向保持检验，不预设两者相等、效应保留百分比或纯方向因果。
分组与反证：预先固定自然rescue/damage/stable_correct/stable_wrong。次要预期rescue组mean(L2),mean(G1)>0，damage组二者<0；报告所有四组和个体反例，不删除与均值反向题。完整逐题联合L2/G1符号表、预测转换、动态最强错gap、原生及处理后的top ties。
数值敏感性：保留原生产score端点，同时用保存的BF16 target logits以Fraction精确计算相同逐非goldgap及题内均值，报告生产正/负/零与精确正/负/零冲突；无epsilon修正。精确零不叫严格正/负因果效应。所有候选完整保留。
解释边界：a2由K2全上游计算产生，移入K1有效不等于省略K2能算出它；a1非消融零信号，a2与a1来自不同自然输入，双向不对称不能直接推出a2对自然K2必要、冗余或独立充分。结论最多为同题R位该冻结attention增量在指定receiver中的可转移/可替换因果效应，不能推断语义来源、prefix/Q路线或attention独占机制。

Source train rows SHA256: c5270ee6798f5893914c936c009469c5a89ea25fb486dfef7c9c4fbc8aea91e3
Source train summary SHA256: 994711edecab29aa033d6f8571034b1141b7b214fc84bfd50da6361b43044273
Source train preliminary coverage SHA256: 32185c796d722f937c93845d46a4b8d9fb7a4066f5892b82ce12f2a9e9d3d2e7

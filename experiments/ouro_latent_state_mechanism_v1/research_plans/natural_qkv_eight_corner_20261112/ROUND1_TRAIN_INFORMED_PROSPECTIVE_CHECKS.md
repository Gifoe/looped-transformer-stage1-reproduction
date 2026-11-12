# TRAIN_INFORMED_PROSPECTIVE_CHECKS

Frozen UTC: 2026-11-12T11:11:25.317232+00:00
Train summary SHA256: 3606ffab45fc80beec9da4962fcacfe4a6d2000e4e070f1966b0b2446b06ca95
Train coverage SHA256: d257487676f31d0d122a913caad2c85369629ce1acca21bdabb5b2de0603c381
Train rows SHA256: a8ac7510ec4bbad1a5b8e745a8d41d523ac984f4e2e3bdbc8c0d11b06b5a635e
Analysis code SHA256: dd25ee7df9de6fd25737cafe9f7708aff723c55ef75a6aa2f6c3e665a687e10f

本判据由完整train非demo1094探索形成；完整val299/test1172分别检验，另分别在预处理固定的自然B1/B2无top-tie子集重复，不汇总跨split掩盖失败。F为原生产候选分数gold−各非gold题内均值，题间等权；固定K1 receiver、八角定义及推理协议不变。P1（KV条件互补）：I1、I2各自mean>0、median>0，且BF16 target-logit Fraction精确I>0的题数超过该子集N/2。三种指标逐项报告，任何一项不满足均明确记失败；不把I重复当两个独立K/V证据。P2（独立可转移效应）：每个Q背景i，K-only=Gi21−Gi11、V-only=Gi12−Gi11平均gap各>0；四条对应决策边各净救回>0。分别报告gap与决策失败，不要求每题有效。P3（联合决策收益）：每个i，correct(Gi22)同时大于correct(Gi21)及correct(Gi12)；correct(Gi21)>correct(Gi12)作为预先声明的决策排序预期，不把其推为gap排序。次要、较弱预测：H1=G121−G112的mean>0，H2=G221−G212的mean<0；J=I2−I1的mean>0。H_i同时等于两种条件效应之差，仅算同一个对比。自然rescue组预期mean(H1),mean(H2)>0且mean(J)>0；stable_correct组预期mean(H1),mean(H2)<0且mean(J)<0；其他两组不指定方向，全部报告。对所有条目并列原生产/固定K1最强错/动态最强错、Fraction精确正负零、原预测ties与损伤/反例；不以精确零的浮点残差宣称有效，若代理/预测冲突保留。报告效应大小/分位/个体比例，无事后epsilon或以显著性阈值改判据。已暴露旧四角及自然ARC结果，故为项目内train-informed检验，非独立盲测。对s/a内部交互另报向量差与norm，不预先等同终端I。完成本轮后仅一条Next Scientific Question，不自动扩展。

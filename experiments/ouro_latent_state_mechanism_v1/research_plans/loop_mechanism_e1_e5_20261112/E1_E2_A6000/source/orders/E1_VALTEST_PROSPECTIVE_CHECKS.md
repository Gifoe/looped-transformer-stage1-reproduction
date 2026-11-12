# E1 TRAIN64-INFORMED VALIDATION AND TEST CHECKS

Frozen UTC 2026-11-12T14:48:30.949871+00:00
E1 order SHA256 4e9db345ba47dbf35012d37a7843d515ea61b727fe8ac676db9c1c36803c1541
64 selection SHA256 6faad196db958744780ee6477af8839c693798b8b24d2a01a2d450c856c720e1
64 audit SHA256 df93e79df963ab9e1aa1f5fb6ffe18d11330f48fa759830f3f9c261afd2f6bf4
64 summary SHA256 cd0ad0b46a14514a93adc3df646aac1996dc924ed55f143a87077663da46c977
Train remainder forecast SHA256 56a2e957841479a55746fba03312ef73a8e7f98e4995825bd40c7a3ee93fe664

Val299与Test1172分别检验：每q自然rescue组 mean(G_ID−G_ADD)>0 且 mean[(G_ID−G_ADD)−(G_ID−G_UNRELATED)]>0；damage组mean(G_ID−G_ADD)<0；rescue ADD_NORM损失>0为辅助，注明RMS标量尺度近不变的结构局限。allwrong主、fixedwrong/dynamic/Fraction/no-natural-tie敏感性，四自然组及全部题均完整报告，无决策伤害优于unrelated的预测。stage全部13站报告，不按结果挑最大站。采集固定原14suffix、同Train donor规则、原生/旧角/restore锚不变，不调参。Train完整后同锁可自动Val采集→audit→Test采集→audit；新分区效果可供检验E1，但禁止用于E3首次靶点/cut选择。每分区独立成败，不因均值或上一split结果改下一split。此前ARC自然及干预结果已暴露，属于项目内64-informed前瞻比较，非独立盲测。

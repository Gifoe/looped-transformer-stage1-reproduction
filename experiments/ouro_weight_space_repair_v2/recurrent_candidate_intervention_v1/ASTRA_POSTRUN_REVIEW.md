# Astra 独立postrun审查

2026-11-05。只新增本文件和hypothesis_comparison.md；未改冻结42输入、源码、协议、语义或采集前审查。未读尚在下载的NPZ，未模型/SSH/Git。

读取官方ANALYSIS.json、numerical_validation.json、question_level_statistics.csv、native_predictions.csv、permutation_transitions.csv、distractor_effects.csv、content_letter_readout.csv及constructor的固定ID集合。独立CPU从native_predictions重算P0/P1/P2/P3各深度计数、全部相邻救伤、同4题B5/B6baseline；从distractor_effects核J与raw logit差，避免把LP舍入当科学效应。Root另行独立核大文件/原始fullV，本审查不冒充该验证。

关键纠偏：P0=10/20/19/20；1→2救11伤1而非“后续始终单调变好”。Placeholder K4 gold DID=+.617187表示原始margin更高，不是替换有益；matched4 DID=−.671875，样本与替代不同，不宣称全24符号反转。B5同4 P0是1/2/2/2而非主24的10/20/19/20。B6两条件均从K1满分，缺少检验一般late-gain的余地。

数值：官方LP/FP64参考最大误差约1.02e−6；CSV独立J/raw-logit-J差Q≤4.77e−7、N≤2.39e−7。约0.1–1的均值效应不能仅归于这些舍入；shared-normalizer取消不排除输出尺度放大，非零I/ΔI/J不能自动转为语义比较。

科学结论详见hypothesis_comparison.md：有内容条件响应和binding行为证据，三个内部机制仍未被独立识别。K1→K2可作本样本行为区间，但没有有根据的token-position类别；**STOP Experiment C，不启动状态干预。** 多重比较未校正，4题子矩阵只描述，不夸大为全ARC因果或泛化证据。

完整B用293.1106862188317 GPU-worker秒，累计21账21182.437382869888，余417.5626171301119。该结论不申请或授权额外计算。

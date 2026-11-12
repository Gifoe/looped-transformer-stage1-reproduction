# 八角条件效应图与表

在包含 `train_analysis/`、`validation_analysis/`、`test_analysis/` 的实验目录运行 `python figures/build_inputs.py`，随后运行 `python figures/plot_eight.py`。第一步核查三分区共 2,590 条不重复题目，每题八角 scores、target logits、prediction、tie、固定错误项与动态/全部错误项 gap，并将重算的正确数、12边及交互均值与 summary 对照；它不是独立 CUDA 重放或完整来源审计。第二步生成 PNG、矢量 PDF 与八角正确数表。`source_sha256.csv` 记录输入 summary 和压缩逐题 rows 的SHA，`corners.csv`/`effects.csv` 是绘图数据。

每个点是先在每题内对全部非 gold 候选等权平均的 gold–wrong 生产候选分数 gap，再对题目等权计算的差分均值；不是所有题的共同效应，图未展示置信区间，也不作显著性结论。训练集排除25个固定示例，validation/test 在本项目中有此前暴露，不称独立盲测。固定 K1 receiver 上每条边只改变所示分量：K边比较 K2−K1，V边比较 V2−V1，Q边比较 Q2−Q1；横轴相应背景固定。`I(Qi)=Gi22−Gi21−Gi12+Gi11`，`I(Q2)−I(Q1)` 是跨Q背景的交互差。终端候选gap交互与内部 SDPA 或 attention向量交互是不同量。图不把人为混合状态效应当作信息来源、自然路径必要性或完整机制解释；细分、负例及审阅以正式报告和逐题结果为准。

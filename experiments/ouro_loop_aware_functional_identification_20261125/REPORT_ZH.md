# Loop-Aware Functional Identification 阶段记录

T1 在同协议八项全量基线上的等权平均为 72.48%，相比 Native 高 5.52 个百分点。Rank-8 为 72.58%，Mixed 为 67.61%。新方法还没有正式 Benchmark 成绩。

目前已完成原版 ARC-Challenge Train 1,119 题 K2–K4、排重后 ARC 1,013 题 K4、CommonsenseQA 8,786 题 K4 的共同接收状态梯度采集，以及 ARC Validation 24 题的真实正负剂量重放。OpenBookQA 已进入全量采集。十个来源的合格训练池已写入服务器。随后原服务器在 SSH 握手前关闭连接，后续 GPU 作业状态暂无法核验。

两个测到的边界很明确。其一，ARC 上 T1 的循环后效用斜率为 2.360，Mixed 为 0.916，较深的冻结原生计算确实改变了排序响应。其二，跨到 CSQA 后，循环交互项与绝对循环斜率给 T1/T2 的排序相反。实际 24 题有限重放还发现正交互与绝对损害可以同时出现。下一阶段冻结联合得分后在独立折检验跨来源排序，随后运行匹配训练对照，再进入完整的 12 项评测。

[数学推导](THEORY.md) · [已核验机制](MECHANISM.md) · [来源及预测口径](DATA_SELECTION.md) · [全量历史参照](RESULTS.md) · [单卡复现](REPRODUCE.md) · [阶段机制图](figures/early_mechanism.svg)

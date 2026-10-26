# 完整 ARC-Challenge：Ouro-1.4B / LoopCD

已完成全部1,172题、25-shot、固定四轮循环的三组评测。主指标为
lm-eval `acc_norm`，以第1轮 logits 为参考。Fixed 使用ω=0.5；Adaptive
按原文概率间隔公式逐token计算强度，上限1.0。未训练、微调或量化模型。
官方 checkpoint revision：`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`。

| 配置 | 正确题数 | 本次 Accuracy | 原文 Accuracy | 纠错 | 改错 | 净纠错 | 配对提升 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline |707|60.32%|60.41%|—|—|—|—|
| Fixed LoopCD |722|61.60%|62.29%|38|23|+15|+1.28 pp|
| Adaptive LoopCD |727|62.03%|61.69%|50|30|+20|+1.71 pp|

原文为 [LoopCD Table2及AppendixB](https://arxiv.org/html/2610.02185v1#S4.T2)。
其1.88pp Fixed提升约对应净增22题；本次净增15题，少7题。原文未发布
逐题预测，因此无法核对作者各自的纠错、改错总数。

| 配对比较 |10,000次问题级bootstrap 95% CI|McNemar精确双侧p|
| --- | --- | ---: |
| Fixed − Baseline |[0.00,2.56] pp|0.07218|
| Adaptive − Baseline |[0.17,3.24] pp|0.03299|

Fixed 的提升方向为正，量级接近原文，但本次未达到0.05显著性水平。
不能仅因三组绝对准确率均在原文±1pp内，就声称原文Fixed收益已得到
强验证。Adaptive的区间在本次点对点比较中排除零；上述区间和p值均
未做多重比较调整。两种指导方法的排名与原文相反，也应保留这一差异。

运行前固定的操作性复现判据已通过：三组绝对准确率偏差均≤1pp，
两种指导的净纠错均为正、配对提升偏差均≤1pp，且原文提升位于本次
配对95%区间内。这个判据用于决定是否继续GSM8K，不等同于证明
Fixed具有显著收益，也不等同于逐题复现作者未公开的预测。

评测使用seed2021、lm-eval0.4.9、Transformers4.57.1、BF16。原文没有
公开确切few-shot随机状态或完整运行环境，因此这些实现条件均显式
固定并保留提示清单。所有方法共享同一份完整测试题、提示、候选项及
评分规则；超参数未根据本次测试结果调整。

GPU子进程与监督进程均以退出码0结束。新增收费GPU运行时间为
659.078401800012秒，即0.183077小时，含该worker的模型加载和正确性
检查。下载的43文件证据包已通过ZIP CRC、SHA256和逐文件哈希验证；
本地Python3.10从原始记录重建后，两张表及复现判据与服务器结果一致。
原始Stage1的175个封存文件哈希仍全部匹配。

可审查文件：

- `results/arc_summary.csv`：三组准确率、纠错/改错、净提升、区间和p值。
- `results/arc_paired_transitions.csv`：两组各1,172条配对记录，共2,344行。
- `trajectories/arc_three_methods.jsonl`：1,172题、3,516个readout的完整候选评分。
- `data/arc_full_test.jsonl`：完整题目、25-shot提示、候选项及输入哈希。
- `results/ARC_INDEPENDENT_CPU_REBUILD.json`：本地独立CPU重建记录。
- `logs/arc_controller/`：实际进程、亲和性、加载源码及退出证据。

本次GitHub上传的完成范围是ARC实验。GSM8K已按先前授权另行启动，
其完整1,319题生成与结果审核尚未完成，不能将这里的ARC结论转写为
自回归推理收益已经复现。

# Minimal Functional Recurrent Feedback

Ouro-1.4B Base 保持冻结。K2–K4 在倒数第二圈末尾、共享 RMSNorm 之前更新残差，让最后一整圈读取修改后的状态。K1 在第 21 个物理层之后更新，保留最后三层。写入位置是当前序列中用于预测答案的 token 位置。候选评分不读取选项正确性，贪心生成每步只使用已生成前缀。

对于当前残差 $h\in\mathbb R^{2048}$，令 $N(h)=h/\operatorname{RMS}(h)$。线性、非线性与历史变体分别为

$$
\begin{aligned}
u_A(h)&=UVh,\\
u_B(h)&=\operatorname{RMS}(h)\,U\operatorname{SiLU}(VN(h)),\\
u_C(h,h_{\rm prev})&=\operatorname{RMS}(h)\,U\operatorname{SiLU}\big(V_hN(h)+V_\Delta N(h-h_{\rm prev})\big).
\end{aligned}
$$

在第一圈末尾及 K1 的物理层写入中，历史项为零。实际状态更新为 $h+d\operatorname{Bound}_{0.45}(u)$，其中 Bound 保证更新 RMS 至多为当前残差 RMS 的 45%，$d$ 是 Validation 锁定的共享剂量。仅使用原生词表头。线性与非线性各用 $4096r$ 个参数，历史变体用 $6144r$ 个参数。所有预算共用一个 checkpoint。具体选中方法、剂量和哈希在 `results/FREEZE.json`。

T1 使用原始和字符归一化候选排序各一半，原始分数温度为 3，归一化分数乘 16。T2 使用 0.35 权重的正确候选内容 token NLL、0.65 权重的统一字母提示下的正确字母 token NLL，以及 0.30 权重的双排序损失。T2-KL 额外用 0.10 权重的原生词表分布 KL。两种目标都使用官方 ARC Train 的 1,119 题，更新步数为 280，T2 训练额外执行字母提示前向。全部方法推理不增加 Loop。

等容量普通 Adapter 与线性版本共享注入位置和排序目标，取消幅度上限。状态蒸馏让校正后的注入点残差经 RMSNorm 直接回归同预算原生后缀末端状态，推理时则把该更新交给原生后缀。该对照测量直接状态回归作为反馈训练的效果。常量对照学习一条共享向量，随残差 RMS 缩放并使用相同的幅度限制。这些对照用于区分下游任务训练、范数限制和按题的系数预测。

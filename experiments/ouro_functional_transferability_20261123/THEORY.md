# 冻结循环后缀上的任务条件控制

## 状态、干预和真正被优化的量

令 $h_k$ 是第 k 圈所有 24 个原生层执行完、边界 RMSNorm 执行前的残差。第 k 圈的读出及下一圈的输入均为 $N(h_k)$。将冻结的 24 层计算块记为 $B_\theta$，原生边界映射即 $F=B_\theta\circ N$。对固定 token 上下文和无缓存的候选评分，循环使用同一权重和位置结构。真实后缀为 $G_m=R_\theta\circ N\circ F^m$，其中 $R_\theta$ 是 LM head 和对原始目标 token 的评分。$m=0$ 仍先执行一次边界归一化。

对候选 $i$，$s_i$ 为答案 token 对数似然之和，$z_i=s_i/c_i$，$c_i$ 为答案字符串长度。连续效用 $q=z_y-\log\sum_{i\ne y}\exp(z_i)$ 是正确对错误候选的归一化 log odds。另报告 $z_y-\max_{i\ne y}z_i$、acc 与 acc_norm。控制器训练沿用历史混合排序损失，效用预测和梯度矩使用 $q$。

## 归一化和实际限幅

对宽度 $d=2048$，令 $s(h)=\max(10^{-6},\sqrt{d^{-1}h^\top h})$，$u$ 是写轴，$w$ 是读轴。控制器先以 FP32 计算写入。

$$
x=u(w^\top h),\qquad
d(h)=x\min\left(1,\frac{0.45s(h)}
{\max(10^{-6},\operatorname{RMS}(x))}\right).
$$

令 $a=\operatorname{RMS}(u)w^\top(h/s(h))$ 和 $\widehat u=u/\operatorname{RMS}(u)$。忽略舍入，更新为 $s(h)\widehat u\,a\min(1,0.45s(h)/\max(10^{-6},s(h)|a|))$。更新 RMS 高于下限时可用 $[-0.45,0.45]$ 对 $a$ 裁剪。写入后残差，随后进行 BF16 舍入。远离两个下限时，限幅更新为正齐次。RMSNorm 的 epsilon 和 BF16 舍入会破坏完全尺度不变性。

设 $r^2=d^{-1}h^\top h+\epsilon$，$N(h)=\gamma\odot h/r$。在实数运算的近似下，原生边界的微分为下式。

$$
D N(h)[d_h]=\frac{\gamma}{r}\odot
\left[d_h-h\frac{h^\top d_h}{d\,r^2}\right].
$$

写轴在每个状态产生不同的切向分量。纯径向 $d_h=bh$ 的导数为 $b\epsilon\gamma\odot h/r^3$，切向 $h^\top d_h=0$ 的导数为 $\gamma\odot d_h/r$。切向相对于未加权输入定义。逐通道权重可以改变输出空间的正交关系。有限的 $N(h+d_h)-N(h)$ 还包含方向和分母的交互。代码分别计算有限差、线性化、径向和切向干预及 BF16 实际响应，再比较它们经同一后缀的效用。

## 一阶共同作用和有限剂量充分条件

令 $g_t=\nabla_{h_t}q_D(G_m(H))$，在真实候选预测位置累积梯度与状态的外积。

$$
M_c=\mathbb E_c\left[\sum_t g_t h_t^\top\right],\quad
S_c(u,w)=u^\top M_cw.
$$

小幅未限更新的期望一阶效用为 $\eta S_c$。每个任务、状态和后缀长 $c=(D,k,m)$ 有自己的矩。真实有限更新的连续路径积分如下。

$$
\Delta q(H)=\int_0^1\!Dq_G(H+tD(H))[D(H)]\,dt.
$$

这里 $D(H)$ 在起点计算，积分路径不重新运行控制器。令未限写入 $D_0=\eta(uw^\top)H$，真实数学裁剪差 $e_{\mathrm{clip}}=D-D_0$，BF16 写入误差 $e_{\mathrm{bf16}}=\widetilde D-D$。设冻结后缀的任务效用梯度沿该连续路径的 Lipschitz 常数至多为 $L$。

$$
\left|\mathbb E\Delta q-\eta S_c\right|
\le \mathbb E\left[
 \|g\|(\|e_\mathrm{clip}\|+\|e_\mathrm{bf16}\|)
 +\frac{L}{2}\|\widetilde D\|^2\right].
$$

内部 BF16 运算与连续代理之间还需计入下游误差。未测得全局 $L$，因此用真实有限更新独立检查符号、排序及绝对误差。

对单位 $u,w$，若已测源满足 $\min_c S_c=\gamma_0>0$ 且未见条件满足 $\|M_*-M_c\|_2<\gamma_0$，则 $S_*>0$。若有限剂量误差小于 $\eta(\gamma_0-\|M_*-M_c\|_2)$，也得到正有限效用。一个更弱的充分条件只约束投影 $u^\top(M_*-M_c)w$ 的负偏移。拟合的共享 $\Gamma_1$ 给出有限训练条件的可行下界。求解器无法找到共同正方向时，较高维空间仍可能存在可行解。

## 传播量、输出响应和任务对齐

比较 $F(C(H))$ 与 $C(F(H))$。定义 $d_e=F(C(H))-F(H)$ 与 $d_l=C(F(H))-F(H)$。差 $d_e-d_l$ 度量相同 pre-RMSNorm 接收位置上的非交换效应。两个干预都保存所有有效 token 的扰动。匹配每个候选的实际序列 RMS 后，在同一边界和相同剩余计算上重放。对照 $\|d\|$、输出候选分数变化 $\|\Delta z\|$、任务对齐 $g^\top d$ 及实际 $\Delta q$。Attention 与 MLP 的恢复效应可相互作用。

## 可推翻的事前预测

1. 从零训练的至少一个 Rank-1 可在一个真实未见深度产生正配对平均效用。用 K2、K3、K4 单组和多组训练，并在 K5/K6 单列。
2. 从控制器实际允许训练的深度拟合 $M$。未见目标只用可准入源矩、边界与后缀距离及目标无标签状态均值。跨任务预测无需目标标签。K5/K6 未进入任何矩训练。
3. signed source moment 及其 RMS 比例修正的排序应优于与标签目标无关的候选写入范数基线。只修正标量剪裁比例的代理无法更换符号；若错误集中于大剪裁、极端后缀或新任务，支持局部模型失效的边界，是否真的由裁剪造成要看同题小剂量和传播重放。
4. 预测在读确认集标签之前，记录源矩、目标无标签开发输入及全部检查点哈希。以同一题为重采样单位，分别审计连续效用与实际改对/改错，负转移完整保留。

[预测冻结文件](results/forecast/FORECAST_LOCK.json) · [预测值](results/forecast/PREDICTIONS.json) · [独立预测检验](results/PREDICTION_AUDIT/AUDIT.json)

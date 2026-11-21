# 自然状态依赖计算的数学定义

冻结 Ouro 的共享映射以完整带 mask 序列 $H$ 为输入。第 $l$ 层执行 $T_l=(I+m_l)\circ(I+a_l)$。两个分支各含原生输入及输出归一化。设 $N$ 为共享循环末端实际 RMSNorm，则

$$F=N\circ T_{24}\circ\cdots\circ T_1,\qquad H_k=F(H_{k-1}).$$

$H_0$ 为 token embedding。固定序列布局、RoPE 和 mask 时，no-cache 的循环编号无额外映射参数。此项用固定输入在不同原生轮次实际重放并逐位比较。

## Attention

对固定头及两个对应自然调用，$A_i=\operatorname{softmax}(Q_iK_i^\top/\sqrt{d}+M)$。

$$
O_2-O_1=(A_2-A_1)V_1+A_1(V_2-V_1)+(A_2-A_1)(V_2-V_1).
$$

线性输出投影保持该和，反向参考改变三项的记账方式。项之间可抵消，范数不能解释成实际份额。将 $v_j$ 记为三个投影项，非线性归一化 $P$ 的剩余定义为

$$P(x+\sum_jv_j)-P(x)-\sum_j[P(x+v_j)-P(x)].$$

softmax 每行的局部算子为 $\operatorname{diag}(a)-aa^\top$。相同 Q 或 K 方向在两种状态上产生的图变化会随当前 Q、K 及 A 改变，而 Value 差也受当前 A 加权。本文主结果直接执行原生有限扰动。

## SwiGLU 与 RMSNorm

设 $g=\mathrm{SiLU}(W_g h)$，$u=W_u h$，$z=g\odot u$，其中 $h$ 是实际分支 norm 后状态。自然差为

$$\Delta z=(\Delta g)\odot u_1+g_1\odot(\Delta u)+(\Delta g)\odot(\Delta u).$$

固定输入方向 $v$ 的局部响应为

$$D_z(h)v=[\mathrm{SiLU}'(W_gh)\odot W_gv]\odot u+g\odot W_uv.$$

该响应受当前门、Up 内容及门导数共同决定。后续是否改变任务结果取决于原生 down、norm 与整个真实后缀。对于 $N(x)=\operatorname{diag}(\gamma)x/s$，$s=\sqrt{\operatorname{mean}(x^2)+\epsilon}$，

$$D_N(x)v=\operatorname{diag}(\gamma)\bigl[v/s-x\langle x,v\rangle/(d s^3)\bigr].$$

因此即使匹配各 token 的范数，当前方向仍能改变响应。范数配平在本实验中是诊断性反事实。

## 原生有限响应与上游中介

$R(H,\delta)=F(H+\delta)-F(H)$。输入向量在对应自然起点分别执行完整 $F$，在 lm head 之前测量响应的角度、增益及传播路径。只改读出无法解释这一测量中的方向变化。

指定上一轮天然写入 $w$ 与下一轮候选因子 $M$，比较原生 $Y(w,M_w)$、删除写入 $Y(0,M_0)$、在删除背景恢复中介 $Y(0,M_w)$ 和在正常写入背景换入删除中介 $Y(w,M_0)$。四项保留相同预算与算子。多中介相互作用时，$Y(0,M_w)-Y(0,M_0)$ 只描述指定背景下的条件作用。行为归因还需要受体规则目标、同剂量方向对照和未经干预的独立评分协议。

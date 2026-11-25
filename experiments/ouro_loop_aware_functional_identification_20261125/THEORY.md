# Loop-Aware Functional Identification

## 共同接收状态

设 $H\in\mathbb R^{T\times d}$ 为固定序列在 Ouro 循环边界 RMSNorm 前的完整残差。$N$ 为原生 RMSNorm，$B$ 为接下来的一次完整共享循环。$F=B\circ N$ 把边界前残差送到下一处边界前残差。$q=R\circ N$ 完成相同的边界归一化与原生 LM head 评分。直接分支用 $q(H)$，循环分支用 $q(F(H))$。有干预时两条路径在同一个边界接收 $V$。

$$
D(H,V)=q(H+V)-q(H),\qquad
L(H,V)=q(F(H+V))-q(F(H)),\qquad C(H,V)=L(H,V)-D(H,V).
$$

写入与原生循环构成配对四格。一个正交互可与部署负收益并存，例如 $D=-2,L=-1$。因此每次干预同时保留 $D,L,C$、准确率及逐题救回和退化。

## 一阶识别与跨词元项

实数代理中，设两条路径对 $H$ 局部可微，其梯度是

$$
g_d=\nabla_Hq(H),\qquad
g_\ell=J_F(H)^\top\nabla q(F(H)).
$$

令 $S$ 为实际允许写入的位置集合，$V_t=\epsilon u(w^\top h_t)\mathbf 1[t\in S]$。若两个效用在干预点附近梯度 Lipschitz，且相应的期望余项有限，则

$$
\begin{aligned}
C(H,V)&=\langle g_\ell-g_d,V\rangle+O(\|V\|_F^2),\\
\mathbb E_D[C(H,V)]&=\epsilon u^\top C_D w+O(\epsilon^2),\\
C_D&=\mathbb E_D\left[\sum_{t\in S}(g_{\ell,t}-g_{d,t})h_t^\top\right].
\end{aligned}
$$

实现将完整边界状态设为同一梯度叶片。读出边界后的状态及再执行一轮的状态，分别把效用梯度回传到同一个前归一化张量。$J_F$ 的位置间块 $\partial F_s/\partial H_t$ 可非零。多个位置的输出梯度可以贡献给较早的同序列位置。没有直接计分的词元可因此获得非零循环梯度。实现分别储存历史 T1 写入位置和全部有效上下文位置对应的矩。

### 效用、剂量与限幅

选择题给每个候选定义 $z_i=16\sum_j\log p(y_{ij}\mid x,y_{i,<j})/\operatorname{chars}(y_i)$，效用为 $z_y-\log(\mathrm{mean}_{i\ne y}e^{z_i})$。归一化去掉仅由选项数导致的对数平移。错误候选的组成仍会改变效用和梯度。生成效用采用正确词元相对最佳错误非 EOS 与 EOS 的对数优势，终止位置把 EOS 当作正确目标。对每题的监督词元先平均，再跨题平均，因此生成长度不直接增大该题矩权重。参考前缀上的词元指标与整段自由生成分别报告。

若先写未限幅的 $V$ 再令 $\epsilon\to0$，正 RMS 状态在足够小的剂量下不会触限。若先将历史写入限幅再乘剂量，局部斜率应沿各题各位置实际限幅方向计算。保存相对 RMS 0.45 限幅的真实干预、BF16 后的有效写回，以及有效写回对应的线性预测。BF16 的量化台阶和生成 top-k 并列点会限制局部线性外推，因此另跑工作剂量的真实干预。

## 共享成分与可证伪预测

各来源矩按训练折原生 K4 优势的稳健尺度归一。用其余来源矩的归一化平均估计单位 Frobenius 模板 $A_{-D}$。令 $\lambda_D=\langle C_D,A_{-D}\rangle_F$，报告相对残差 $\|C_D-\lambda_D A_{-D}\|_F/\|C_D\|_F$。投影分解本身是恒等式。跨独立训练折的有符号稳定性、去掉来源后的投影和独立有限剂量排序，可以检验共享性。普通矩 $M_D=\mathbb E_D[\sum_{t\in S}g_{\ell,t}h_t^\top]$ 使用同样的权重和谱算法。另保留直接读出矩。

每来源用训练折 0–2 的 $C_D$ 和 $M_D$ 各取首个奇异向量对。两种方向用相同因子范数乘积和限幅。对候选源方向 $(u,w)$，参考环境 $r$ 提供 $c_r=u^\top C_rw$ 与 $m_r=u^\top M_rw$。联合评分对 $\min(c_r,m_r)$ 取均值减半个标准差，再乘非负折间稳定度。对照包括随机、原生难度、普通矩余弦、普通矩投影和源域强度。留一来源和留一任务类型都将预测环境排除在参考矩外，实测使用独立第 3 折。固定 K4 方向在 K3、K5 上复测，剂量为 0.1、0.5 和 1.0。

### 秩一因子切空间

设 $A=uw^\top$，任务对矩 $A$ 的代理梯度为 $G$，则该梯度拉回到秩一因子为

$$
T_{u,w}(G)=(Gw,G^\top u).
$$

同一初值 $\theta_0=(u,w)$ 上，沿源任务代理梯度做无动量单步更新，目标环境的一阶变化为

$$
\Delta q_r \approx \eta \left\langle T_{u,w}(G_r),\,T_{u,w}(G_D)\right\rangle.
$$

分别以 $M_r$ 和 $C_r$ 作目标代理，要求预测的绝对与循环交互收益同向。冷启动 $u=0$ 时首步只更新写轴，可达方向依赖初始读轴 $w_0$。T1 初始化时两个因子都能更新。实际训练还涉及 AdamW、限幅、有限步长和冻结历史双排序损失，切空间结果只作为能够被真实训练反驳的预测。

### 判定证据

预测文件在第 3 折干预前封存。按环境报告 Spearman、Kendall、留环境校准误差、选型后悔值及最弱条件收益。训练对照按原生正确转移和输入长度分层，保持可比较的题目暴露、初始化、预算和目标。测试只在开发模型锁定后执行。使用同一 K4 主模型和统一推理协议。

历史结果中，单用源矩预测收益的 Spearman 为 −0.011。加入目标域 Train 功能矩可改善连续效用排序，目标准确率排序仍未解决。接收边界和完整后续循环的配对干预用于辨别这些解释。 [历史跨域验证](../ouro_functional_transferability_20261123/REPORT_ZH.md) [原生循环传输](../ouro_dynamical_transport_20261123/DYNAMICAL_THEORY.md) [Ouro](https://arxiv.org/abs/2510.25741) [梯度影响方法](https://papers.neurips.cc/paper_files/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html)

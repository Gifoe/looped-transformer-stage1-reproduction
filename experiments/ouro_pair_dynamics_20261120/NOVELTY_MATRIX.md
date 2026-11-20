# 近邻工作与本轮可检验贡献

| 一手工作 | 已完成的工作 | 本轮需要独立检验的区别 |
| --- | --- | --- |
| [Training-Free Looped Transformers](https://arxiv.org/abs/2605.23872) | 用阻尼子步把非循环预训练 Transformer 的中段重复应用解释为同一 ODE 近似的细化。 | Ouro 已原生共享计算。本轮直接用原生幅度的有序离散 pair 和语义反事实作靶标。 |
| [A Mechanistic Analysis of Looped Reasoning Language Models](https://arxiv.org/html/2604.11791v2) | 报告多个循环语言模型的循环潜在轨迹、部分层固定点与推理阶段。 | 本轮关注来源、规则和联合目标的有符号混合响应，以及冻结预测与实际原生干预是否吻合。 |
| [Pretraining Builds Cooperation, Not Components](https://aclanthology.org/2026.findings-aacl.149/) | 用真实 Jacobian 的 Lie bracket 衡量相邻层协作，考察预训练阶段和层复制。 | 单层内 A/M 顺序反转的 bracket 预测须对照四世界混合响应；两种量的意义不同。 |
| [Log Neural Controlled Differential Equations](https://proceedings.mlr.press/v235/walker24a.html) | 用 log-ODE 方法及 Lie 工具训练处理长时间序列的连续系统。 | 本轮不训练模型动力学，从冻结的真实残差分支构造和检验小幅修正。 |
| [LoopFormer](https://arxiv.org/abs/2602.11451) | 用时间与步长条件、跨预算 shortcut consistency 训练弹性循环深度。 | Ouro 的本轮参数保持冻结，所有继续轮共享相同的场和系数。 |
| [Rectified Looped Transformer](https://arxiv.org/abs/2611.16265) | 用流启发的线性反馈插值重设计循环连接，考察状态稳定性和深度外推。 | 本轮使用真实原生顺序作为干预靶标，并测试冻结结构组合和方向补偿。 |

使用的已有数学工具包括显式 Euler 的后向误差、交换子、JVP、固定网格 RK4、因果三角映射和求解时钟的重参数化。RMSNorm 的分段流给出算术精确的参考端点。

机制主张需要在未见结构的四世界任务上用预测先行协议复现：完整模型对有符号 Iq 超过强对照；原生 A/M 实验的剂量方向也匹配预测；结果跨来源组及循环边界。小步长阶数、充分积分的路径平滑和既有 Lie 工具单独不足以支持这样的主张。

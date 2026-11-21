# Ouro 原生跨轮计算的新颖性审计

核查截至 2026-11-21。研究对象是权重冻结的 ByteDance/Ouro-1.4B Base，模型版本为 `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`。下列数值均来自已有工作。本轮证据以正式分析为准。 [原始模型论文](https://arxiv.org/html/2510.25741v1) · [模型快照](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1) · [本项目继承证据](BASELINE_AUDIT.md)

## 最接近的研究

| 研究 | 已有证据 | 本轮可检验的区别 |
|---|---|---|
| Wu、Hu、Zhang（2026），*Shared Weights, Selected Computations* | 训练的入口控制量可改变复用层的路由。附录 D 在任务适配的 Ouro-2.6B 字母传递任务中，通过自然运行间的 16 头注意力图移植，在 182/256 对中得到按受体规则计算的目标。输出移植在 240/256 对中复制供体答案，值移植在 24/256 对中得到重路由目标。 | 固定权重下的计算随状态改变，以及区分注意力图路由与值复制，均已有直接先例。当前区别是冻结的原始 1.4B 中，定位造成下一轮路由变化的上一轮真实写入，并在自然轨迹上验证作用。 [论文 §4、附录 D/F](https://arxiv.org/html/2609.39892v1) · [作者代码](https://github.com/wjjpku/howloop/tree/ec56df3fc548f3b0321475002ed302fc10169e35) |
| Mu、Wang（2026），*Cross-Loop Query Handoff* | 同一 Ouro-1.4B Base。在 600 道合成题中选取 K1 错、K3 对的 245 题。用同题 K3 查询跨所有物理层改写 K1 最后提示位置，受限读出恢复率 89.8%，打乱供体为 23.7%。冻结和交叉供体实验识别查询分量在合成任务中的作用。自然 QA 迁移和 K1 自由生成未给出相同结论。 | 同模型的查询更换、必要性及供体交叉已有先例。文章把桥变量在哪里写入留作后续问题。当前可定位具体 K1 写入，并检验原生 K2 Q/K/V、注意力输出或门控是否传递相应变化。 [论文 §§4–9](https://arxiv.org/html/2610.34639v1) · [作者代码](https://github.com/Muuzzziiy/cql-handoff/tree/75b52dbfdb160eb5da59b7e7897bbeb0bb16ec91) |
| Kale、Baser、Vishwanath（2026），*The Key Handoff* | 在 12 个稠密或混合模型上，用交叉供体区分计算出的桥、答案和读取位置。在混合模型上置换早先循环层的实际输出，保留供体答案和行位置，用匹配层对照识别谁向注意力提供桥，并用独立题的钳制和头输出注入验证下游路径。 | 具体写入→读头→按受体上下文改变答案的总体设计已有先例。区别在于 Ouro 原生深度循环中，由上一次调用造成同一权重块下一次调用对任务变量的不同响应。 [论文 §§2、4，附录 D/E](https://arxiv.org/html/2609.32942v1) |
| Shalev Merin（2026），*Present but Hard to Read* | Ouro-1.4B 在答案末位前的指定位置可读出部分中间加法信号。在测试子空间从第一轮起钳制为另一道题的中间值，未使末位朝供体偏移；钳制答案则使 91% 的配对朝供体偏移。 | 轨迹可读、状态含有中间结果、信息被原生续算利用是三个独立命题。当前检验要求自然写入的删除和下游中介恢复与预先规定的答案变化一起成立。 [论文](https://arxiv.org/html/2610.28498v2) · [作者代码](https://github.com/Aur-Shalev-Merin/present-but-hard-to-read/tree/fecf75a698b735d8782556aef693ec631b6f29b1) |
| Blayney 等（2026），*Mechanistic Analysis of Looped Reasoning Language Models*；Olszowiec 等（2026）；Wang、Reid（2026） | 多模型跨轮机制观测、循环头行为、状态携带与输入注入架构的区别，以及 Ouro-2.6B 的分轮线性读出与 Jacobian 传输。已有证据覆盖宏观阶段、状态可读性和局部线性响应。 | 物理层同权重时按轮分析、探针显示语义变化、或不同轮 Jacobian 响应不同，都只能先用作候选定位。可区分之处是自然来源、局部中介与任务输出可被同一组受控干预解释。 [Blayney 等](https://arxiv.org/html/2604.11791v2) · [作者代码](https://github.com/HughBlayney/looped-mechanistic-analysis/tree/3429a8d61fdae95d441cb18eb3bd695268d086f4) · [Olszowiec 等](https://arxiv.org/html/2610.18127v1) · [Wang、Reid](https://arxiv.org/html/2609.01924v1) |
| Liang 等（2026）；Zhou 等（2026） | 在专门训练的循环模型中，已有针对绑定或多跳计算的逐轮干预、已读出状态的冻结/恢复，以及路径移植。两项工作的设计都能区分部分既存信息与后来实际完成的任务。 | 当前模型保持通用预训练权重，原生执行形态和问题分布与这些专门训练设置不同。单凭冻结、恢复或逐轮进度不能主张方法新颖。 [Liang 等](https://arxiv.org/html/2609.33144v1) · [Zhou 等](https://arxiv.org/html/2610.19768v1) |
| Tuomi（2026）；Kim、Lee（2026） | 常规 Transformer 已有对自然运行所携带方向与 MLP 门控工作点的因果分析。跨块顺序和非加性交互也已有独立研究。 | 下一轮 gate 非线性、早期残差改变 gate、或任意块顺序效应均不足。需要识别与具体任务变量及原生后续计算相关的写入，并用同幅度、同结构干预区分一般增益变化。 [Tuomi](https://arxiv.org/html/2609.31956v1) · [Kim、Lee](https://aclanthology.org/2026.findings-aacl.149/) |
| Wan、Chen（2026），*Attention Routing Stabilizes Early* | 在 Huginn 和 Recurrent-Llama-T32 上发现整体注意力支持集先于状态和输出稳定。后期重用支持集、继续计算该集合内权重和值可保留多跳 QA 的大部分表现。 | 相同支持集上的值、边权和读出仍可细化。平均注意力距离和单题的指定边效应要分别报告，避免用前者推断所有头在后轮重路由或全部保持同一功能。 [论文 §§7.2–7.3](https://arxiv.org/html/2609.27373v1) · [作者代码](https://github.com/tbn5pj/WISE_code/tree/eaccf51090dc3c841e931012168906bfb80613e5) |

两份相关匿名投稿的 OpenReview 正文在核查时未能直接读取。据 *Cross-Loop Query Handoff* §9，*Same Head, Different Calls?* 发现 Ouro-2.6B 某个头输出移植的效果可在又一次调用后消失。*What Does Another Loop Buy?* 的证据包括在 Ouro-2.6B 冻结跨轮注意力后 GSM8K 表现退化。这两条只作检索线索。 [Same Head, Different Calls?](https://openreview.net/forum?id=ipTUVMFcAY) · [What Does Another Loop Buy?](https://openreview.net/forum?id=mQUR2BQm7r) · [可核查的转述](https://arxiv.org/html/2610.34639v1#S9)

## 本项目已经完成的近邻试验

- **E1–E3** 已经用同题自然 Q 和 KV 做析因移植，并在后续试验中拆为 Q×K×V 八角。E3 对上游 L10–L16 进行独立干预，在 Train1030 与 Validation299 上分别通过六项冻结判据。接收位置的 L18 行为测量属于诊断，未隔离新 L-only 干预到特定下游中介的边。此前的 L18 亲本方案也有 P/PK/P+A/P+A+key 的条件 key-to-R 组合，已存分数的交互总结属于事后分析。 [综合报告](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/3c13d4e8ba026a6b617b61ccf3a95c88e88a6251/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E1_E5_SYNTHESIS/FINAL_RESEARCH_REPORT.md) · [自然八角](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/3c13d4e8ba026a6b617b61ccf3a95c88e88a6251/experiments/ouro_latent_state_mechanism_v1/research_plans/natural_qkv_eight_corner_20261112/README.md)
- **E4/E5** 已在同题 K2 受体中把末位置残差回拨至 K1，并用原生 K2 实际注意力写入恢复或反向交换。Validation299 的 104 个自然 K2 修正中，回拨损伤 29 个，恢复救回 22 个，11 个满足四臂无并列条件。独立 128 题中恢复 6/9 个受损修正，2 个满足严格条件，连续固定错误对手的均值区间含零。原先产生相关残差的具体写入仍未确定。 [主报告](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/3c13d4e8ba026a6b617b61ccf3a95c88e88a6251/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E4_E5_4090/FINAL_RESEARCH_REPORT.md) · [中介分析](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/3c13d4e8ba026a6b617b61ccf3a95c88e88a6251/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E4_E5_4090/E4_MEDIATION_ANALYSIS.md)
- **近期项目** 已验证 K1 自然供体角色态联合移植的组合效应、原生后缀合成四世界的非加性，以及同输入 AM/MA 换序和后缀 JVP。结构留出中，分角色移植 K4 五候选 63/96，联合自然提示 56/96，两者全词表贪心首 token 均为 0/96。结构留出的 K2 路径切断特异性和 K3 再切断区间均含零。因而当前贡献不能建立在重复整状态移植、证明四世界交互非零或观察轮间算子响应上。 [逐项证据与源码](BASELINE_AUDIT.md) · [已发表的路径对照](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/521d71b00d09c8cafafa7541b9db9c43d3341719/experiments/ouro_hidden_computation_20261120/HIDDEN_COMPUTATION.md)

## 哪些结果可以改变判断

| 拟检验的命题 | 判别所需的结果 |
|---|---|
| H1，历史信息积累 | 在同提示自然轨迹量化写入及可读性。观测到轮间状态或分数变化，只支持自然计算继续。即使注意力图和 gate 变了，也可能只是输入变了以后同一固定计算的平滑响应。 |
| H2，状态改变有效计算 | 用同一有限扰动跨轮、跨上下文比较响应，保持物理层、掩码和剩余执行深度一致。预先规定与任务规则有关的响应方向或目标。独立问题上的方向差异须超过同幅随机、无关位置或头、以及供体答案复制对照。仅有响应尺度、显著交互或 Jacobian 变化不足以主张离散的算法切换。 |
| H2，指定路径的因果作用 | 从原生轨迹取出一个**实际产生的写入**。切换该写入后同时追踪 K1 轮末状态、K2 指定 Q/K/V/注意力或 gate/up 与输出。用受体自身的原生中介恢复，并对照中介单独更换以及无关边恢复。在相同余下深度、原生归一化和匹配幅度下检验交互及方向。只有独立样本同时复现特定中介与任务输出，才可在该模型、任务和路径范围内主张机制。 |
| H3，读出边界 | 固定词表或选项分数、完整答案的 teacher-forced 得分、自由生成和字母读出分别报告。若变化只在受限读出出现，就将行为结论限制在该读出。若内部因果方向复现而自由生成稀少，可同时报告内部机制证据和生成证据的边界。 |

执行时，`Δ(AV)=ΔA·V₁+A₁·ΔV+ΔA·ΔV` 与 SwiGLU 的 `Δ(G⊙U)` 展开是恒等式。逐头、逐通道分量可在线性输出投影处相加核算，归一化后的每项作用需用整条原生续算重新测量。析因的 `Y₁₁−Y₁₀−Y₀₁+Y₀₀` 只估计所选干预和受体基线下的交互。共享后缀、残差旁路和其他中介可以共同影响它。路径特异结论需明示中介取自哪一次执行、其他节点从哪一分支继续以及恢复/相反方向的结果。无答案标签的发现只保证筛选未直接使用正确性，仍须冻结选择和在独立样本上确认。 [路径移植](https://arxiv.org/abs/2304.05969) · [因果抽象](https://www.jmlr.org/papers/v26/23-0058.html) · [移植指标与损坏分布](https://proceedings.iclr.cc/paper_files/paper/2024/file/06a52a54c8ee03cd86771136bc91eb1f-Paper-Conference.pdf) · [线性近似误差](https://arxiv.org/abs/2403.00745) · [多中介交互](https://arxiv.org/html/2606.27510v1)

本轮最有辨别力的贡献是确定**哪一次原生写入通过哪一个下一轮算子改变了什么任务变量**。若独立样本只重复相异状态、输出变化或候选交互，合理的结论是原生计算随状态改变，以及目前哪些路径尚无法确定。

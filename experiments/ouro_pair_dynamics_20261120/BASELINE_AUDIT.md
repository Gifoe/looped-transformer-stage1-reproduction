# Ouro-1.4B 历史基线审计

冻结模型为 ByteDance/Ouro-1.4B Base，模型 revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`。审计从基准提交 `bfca32210a0d08d6bc03f070f69e94212ffb88f8` 读取历史源码与结果。逐行重新评分和分组 bootstrap 使用 CPU，无模型前向。其余起点陈述通过实现、原始结果及报告核对。 [基准提交](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/bfca32210a0d08d6bc03f070f69e94212ffb88f8)

## K1 联合移植与原生 K4

从 `causal.jsonl` 选取 `split=new_structure`、`fork=open`、`kind=interchange`、`loop=1`、`arm=tableau` 和两个非零对角 `(s,m)=(1,1),(2,2)`。48 组各有两个事先确定的目标。分别从接收世界 `(0,0)`、来源供体 `(s,0)`、规则供体 `(0,m)`、原生新世界 `(s,m)` 读取同组记录。原生新世界对照取自 `natural.jsonl` 的 `kind=natural`，它直接编码新世界提示。历史开发集 24 组、新实体集 48 组均未计入这里的分母。 [设计](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/bfca32210a0d08d6bc03f070f69e94212ffb88f8/experiments/ouro_hidden_computation_20261120/mechanism_design.py) · [评分与聚合](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/bfca32210a0d08d6bc03f070f69e94212ffb88f8/experiments/ouro_hidden_computation_20261120/mechanism_analyze.py)

结构留出题增加固定的 token→sign 桥规则，三个 sign→object 记录随 `m` 循环平移。令正确对象索引 `y=(s+m) mod 3`。对角 `(1,1)` 的新目标为 2，接收方为 0，两个单因素供体均为 1。对角 `(2,2)` 的新目标为 1，接收方为 0，两个单因素供体均为 2。所有入选组的历史自然记录均有 1 个来源 token 和 24 个规则 token。两个块不重叠，检查了接收方与联合世界的 token 长度、角色位点和块外 token 一致性。`mechanism_design.py` 对全 3×3 做相同布局检查。

`mechanism_run.py` 用无 cache 的 BF16 SDPA 前向，在 K1 的输出 RMSNorm hook 将供体同位置、全通道的状态写入接收方来源和规则块。桥接、读出及其他位置沿用接收方，K2、K3、K4 继续运行原生循环。来源位置早于规则文本。因果掩码使来源位置无法访问后面的规则，规则块可以携带其供体的上游来源上下文。因此，这组干预验证角色块的可组合使用，状态内容是否纯粹编码单因素仍需单独检验。 [干预实现](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/bfca32210a0d08d6bc03f070f69e94212ffb88f8/experiments/ouro_hidden_computation_20261120/mechanism_run.py)

开放提示只显示题面与 `Answer:`。评分从最后一个提示位置的原生完整词表 logits 以 FP32 做 `log_softmax`，只比较五个事先确定、各占一个续写 token 的对象词，候选并列按未命中。复算直接从 `scores[3][0]` 取唯一最大值，核验其与历史 `pred_content` 一致。线性 margin 使用 `log p(y_joint) - mean(log p(y_receiver), log p(y_single))`，两个单因素供体目标相同，只计算一次。共同的词表归一化项在对比中抵消。原报告的 margin 区间先组内平均两个目标，再对 48 组执行 3,000 次 bootstrap。

| K4 历史条件 | 五候选唯一首选 | 并列 | 全词表贪心首 token 命中 | 联合移植减该条件的命中率差及 95% 区间 |
|---|---:|---:|---:|---:|
| 分拆供体联合移植 | 63/96 | 2 | 0/96 | — |
| 原生新世界 | 56/96 | 1 | 0/96 | +7.29 [0.00, 14.58] 个百分点 |
| 同块等范数随机方向 | 46/96 | 1 | 0/96 | +17.71 [4.17, 30.21] 个百分点 |
| 同 Y 供体规则块 | 34/96 | 1 | 0/96 | +30.21 [22.92, 37.50] 个百分点 |
| 联合世界提供两个角色块 | 61/96 | 1 | 0/96 | +2.08 [−2.08, 6.25] 个百分点 |
| 联合世界完整边界移植 | 54/96 | 3 | 0/96 | +9.38 [2.08, 16.67] 个百分点 |

命中差区间是本次对配对组重新抽样 20,000 次得到的描述性区间。原报告各自的组 bootstrap 95% 区间为联合移植 56.25%–73.96%，原生新世界 50.00%–67.71%。与原生新世界配对时，10 个目标仅联合命中、3 个仅原生命中、53 个均命中、30 个均未命中。组内相关和比较区间触及零限制了对二者差距的结论。原统计的 K4 固定目标 margin 差为联合对接收方 +1.595 [1.312, 1.868] nats、对等范数随机 +1.067 [0.577, 1.532]、对同 Y +1.911 [1.611, 2.205]、对仅来源 +2.013 [1.704, 2.336]、对仅规则 +1.763 [1.476, 2.065]。这些值是分别对比各臂的目标 margin，四世界的混合差量需要另算。 [历史报告](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/bfca32210a0d08d6bc03f070f69e94212ffb88f8/experiments/ouro_hidden_computation_20261120/HIDDEN_COMPUTATION.md)

**供体与批布局对照。** `same_y_mapping_control` 固定使用首个与接收方同 Y 且 `s!=0` 的 `(1,2)` 世界。来源块沿用正常移植，规则块方向改用该世界，输入差向量按正常规则移植的范数缩放后回写 BF16。对于 `(1,1)`，这会改变规则文本对应的索引。对于 `(2,2)`，供体仍为 `m=2`，但其来源上下文不同。同 Y 指供体的答案与接收方相同。该对照的两个目标并非对称的规则文本替换。完整边界移植通过同一 20 路干预批中的原生比较及 20 路供体库的逐位核对。单独归档的 `natural` 来自九世界批。对这两份实际记录，96/96 个 K4 五候选分数向量有差别，6/96 个首选索引不同。差异与批布局变化同时出现，归档无法分离具体数值原因。因而 54/96 和 56/96 各自保留原口径，完整移植控制的数值比较需使用同批的原生对照。

**可引用范围。** 在这批旧的、单一增跳模版题组中，来源和规则角色的联合移植能改变原生后续计算对预定新目标的候选排序。历史 63/96 用于回归检查。候选排序、开放生成、有限混合响应和顺序干预需分别评价。

## 其余五项起点

### 1. 整轮流和逐分支流具有不同的细化参考终点

`solver.py` 的整轮 A 以 `F(x)-x` 为场，原生 `F` 已包括轮末 RMSNorm。B 依次积分 48 个原生残差分支场，终点另做一次 RMSNorm。两种构造在各自单次单位 Euler 配置下复现原生离散映射。细化会实际引入额外的自相互作用项。FP32 高阶参考在每类各一题的相同初始 K1 上用 RK4、`m=128`，以 Euler `m=512` 独立比较。数字均为有效 token 全状态相对 L2，分母为比较对的第二个状态。

| 队列 | A 的 Euler–RK4 | B 的 Euler–RK4 | A 与 B 后归一化距离 | A 至原生 K2 | B 至原生 K2 |
|---|---:|---:|---:|---:|---:|
| 3×3 新结构 | 0.000445 | 0.01164 | 1.1705 | 0.33675 | 0.82783 |
| 双跳新结构 | 0.000445 | 0.00771 | 1.0092 | 0.32938 | 0.82011 |
| ARC 验证题 | 0.000372 | 0.00328 | 0.72907 | 0.27936 | 0.66118 |

相同方法的 RK4 64→128 终点改变量约 `2.7e-7–5.4e-6`。终点差在测得的求解器误差之外；实数光滑场的有限时间解与 BF16 量化场的数值行为须分别表述。A 与 B 的时间及 Norm 放置不同，分支非交换性只是差异来源之一。 [实现](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/bfca32210a0d08d6bc03f070f69e94212ffb88f8/experiments/ouro_continuum_refinement_20261120/solver.py) · [细化结果](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/bfca32210a0d08d6bc03f070f69e94212ffb88f8/experiments/ouro_continuum_refinement_20261120/DISCRETE_VS_CONTINUOUS.md)

### 2. 单分支 BEA 的改善集中在小 α，单位幅度仍有误差

`modified` 的单分支 BEA2 用 `f−α J_f f/2`。BEA3 用真实 JVP 及中心差分高阶项。每类一题，每题十个分支，局部积分从原生分支输入开始，导数与局部目标用 math SDPA。α=.125 时无修正、BEA2 和 BEA3 的末态相对 L2 中位数依次为 `7.25e-4`、`4.31e-5`、`3.39e-6`。α=1 时分别为 `0.04466`、`0.02072`、`0.01368`，三阶最坏 `0.15981`。开发整轮 BEA2 细化后平均误差 `0.39688`，普通配置 `0.40376`。这组证据将小幅局部展开与单位幅度整轮近似分开。 [源码](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/bfca32210a0d08d6bc03f070f69e94212ffb88f8/experiments/ouro_continuous_embedding_20261120/dynamics_embedding.py) · [局部结果](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/bfca32210a0d08d6bc03f070f69e94212ffb88f8/experiments/ouro_continuous_embedding_20261120/MODIFIED_EQUATION.md)

### 3. Chart 与 Anderson 改善端点，hybrid 大量调用原生残差步

Chart 从预设的分支路径 `Ψ_t(x)=x+t f(x)` 定义 `v(z,t)=f(Ψ_t⁻¹(z))`。沿精确解的分支轨迹因此是直线。`field_step` 先在当前输入调用 `f(x)`，构造 `x+.5 f(x)`，再用 6 次 Picard 或 Anderson 近似求逆，最后调用分支计算输出斜率。路径和端点都可由输入及原生分支查询得到，纯方法沿自己的状态串联。FP32 留出 24 题上，K1→K2 的全状态平均相对 L2 从 Picard `0.14553` 降至 Anderson `0.13002`，K2→K3 从 `0.05548` 降至 `0.02048`。Hybrid 用 `2e−5` 全序列相对逆残差阈值，在两轮界分别有 86.979% 和 85.503% 的分支直接执行残差加法，误差为 `7.12e−5` 和 `5.05e−5`。BF16 两轮界均全量回退，回退使用 math SDPA，历史原生参考使用默认后端。

Anderson 将同一样本全部有效 token 的残差展平后构造 Gram 矩阵，一套权重供全序列共享。Hybrid 同样使用跨 token 范数和样本级门控。后缀能通过共享权重或门控影响较早 token，代码存在这种依赖路径。历史 `prefix_control.py` 实现的是因果前缀 ridge 的补充对照，实际泄漏大小应由新轮的等长共同前缀审计测量。 [重建报告](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/bfca32210a0d08d6bc03f070f69e94212ffb88f8/experiments/ouro_continuous_embedding_20261120/CONTINUOUS_RECONSTRUCTION.md) · [推进与回退实现](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/bfca32210a0d08d6bc03f070f69e94212ffb88f8/experiments/ouro_continuous_embedding_20261120/embedding_experiments.py)

### 4. Bracket 传播依赖干净原生边界与真实原生后缀

`audit_layers` 以 `op.actual[i]` 为原生层输入，以 `op.actual[i+2]` 为原序 pair 输出。在原序减倒序 convention 下，`C=J_m a−J_a m`。实际干预为倒序减原序，因此在原序输出调用真实 `native_suffix` 的 JVP，方向为 `−C`。开发两轮界的末态方向余弦中位数为 `0.87833`、`0.90954`。六道独立留出复核为 `0.87293`、`0.90909`。开发原生改选命中分别为 9/13、0/2，分母仅含原生实际改选且无并列的条件。开发随机和替代层方向都用实际交换向量范数配平，属于事后诊断。该实验支持给定原生边界和后缀时的方向近似。逐层真实起点及后缀依赖决定其应归入条件式解释。 [算子实验](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/bfca32210a0d08d6bc03f070f69e94212ffb88f8/experiments/ouro_continuous_embedding_20261120/OPERATOR_INTERACTIONS.md)

### 5. 最近的 `causal_phase` 只运行单臂编辑

实现保存 `clean`、`source`、`rule`、两种 `random` 和两种 `unrelated`，`target` 绑定各自单因素供体的目标。两跳的 `rule` 实际编辑 `bridge`，tableau 编辑 `mapping`，ARC 供体交换选项位置并记录槽位到内容的映射。独立反事实样本为 14 个受控组和 6 个 ARC 组，两种单因素干预合计每轮界 40 个观察，独立统计单元仍为组。`freeze_confirmation.py` 固化 ARC 试题、代码及清单哈希、跨槽位内容对齐和 gate。它没有定义 H11、独立的 y11，或同一固定评分下的四世界混合差。历史 K1→K2 的 Anderson 改选命中 10/12 中，ARC 占 10/10，受控两族合计 0/2。 [因果报告](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/bfca32210a0d08d6bc03f070f69e94212ffb88f8/experiments/ouro_continuous_embedding_20261120/CAUSAL_VALIDATION.md) · [冻结实现](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/bfca32210a0d08d6bc03f070f69e94212ffb88f8/experiments/ouro_continuous_embedding_20261120/freeze_confirmation.py)

## 复算入口

在本地持有目标 Git 对象的仓库执行 `python historical_audit.py <repository>`。脚本使用 `git show <commit>:<path>`，校验候选顺序、角色布局、并列规则、两个联合目标和历史汇总值，并打印归档 SHA-256 及组级配对区间。

| 原始输入 | SHA-256 |
|---|---|
| `causal.jsonl` | `3725bba31ba94fb86da0fb745f76303429bee9194a043d1894be05ccf0650d0e` |
| `natural.jsonl` | `48cbf1f403f8af99139a0e17e04dd66811479cb82fabccaaa36ed8bb0ee62a58` |
| `mechanism_tableaux.jsonl` | `d059a5db0af6db0702766ba09d77fe2adc0e7e4b486e5cb0a912ab98db15c63f` |
| `mechanism_summary.json` | `8cc2ec24551b927b955639279df442d49da8e1e25781f7f9bf7a2714c11318ac` |

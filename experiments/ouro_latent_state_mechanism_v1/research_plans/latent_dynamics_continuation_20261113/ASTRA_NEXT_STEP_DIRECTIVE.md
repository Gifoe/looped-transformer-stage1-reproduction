# Astra 下一步研究指令：先判别失败原因，再决定是否增加内部信号

日期：2026-11-13。依据：goal-objective.md，以及 Sol 已核实的 E6A/B Joint 结果。角色严格分工：Astra 只制定假设、实验范围、预算与科学门槛；Sol 负责全部数据分析、实现、执行、数值复核和报告。当前权重保持不变。本指令不授权 E6C hidden-state control，也不启动外部大规模评估。

本指令采用 ccf-experiment-designer 的最小充分证据原则。它是前瞻研究协议，不是结果报告。FullAnswer Test 在指令制定时尚未完成，任何下述未知结果均不得预填。

## 1. 当前判决与禁止替换的基线

当前不能声称 latent-aware 方法改善了强 output-only/K4 的准确率或效率。主 latent rollback 与匹配 output-only 都是 1018/1172，低于当前 5090 K4 的 1022。Early latent 与 output-only 同为 1021，但前者平均 3.490 loops，后者约 3.105；真实 Validation online 也同为 260 正确，而 latent 的 0.1659s 高于 output-only 的 0.1474s。现有 B 特征/控制器被实际比较支配，E6C 保持 **NO-GO**。

Agreement 局部规则的 +15 题是一个真实受条件限制的效果，但 latent 1016 仍低于 K4，不能用较弱 plain 1001 替代最强对照。监督诊断的 output→combined F1 下降同样保留。以上结果否定的是已测试的特征/策略增量，不等于所有潜在内部信息都无用。

保留 E6A 的原始 primary ZL weighted4=1016、原 Val-selected Cal=1014 及全部 32 规则/2 个仓库 LoopCD 对照身份。不能把 Test 上较好的 Calearly.99=1022 或 early.9=1021 追认为新 primary。历史 A6000 K4=1020 与当前 5090 K4=1022 分开；推论使用同硬件同协议原生基线。

## 2. 立即执行：完成冻结转移，禁止用其 Test 修方法

Sol 继续完成已经授权的 FullAnswer 收集、完整性核验和冻结评估，不中断、不重选参数。主分数是预注册的完整答案 char-normalized likelihood；sum/tokenmean 是次指标。Joint Cal 身份保持冻结，B 为 transfer-zero-fit。字母评分与完整答案评分分别报告自己的 K4、最强预先选定 output-only、主方法和成本。

保存 uniform234/logprob 平均等理论等价规则的有限精度 near-tie 差异；不得通过换算顺序、tie 规则或重新排名来改善原始结果。方法间差异必须指向原始候选分数和冻结实现。

FullAnswer 结果仅裁决转移，不用于本轮选择新特征、阈值、层、控制器或假设。完成后 Sol 给一份核实摘要，Astra 可追加一次转移结论；不能依据 Test 哪个规则赢了而换 primary。相同 ARC 问题换评分方式不是独立数据确认。

## 3. 立即执行：仅 Train/dev 缓存 CPU 判别

本阶段不新增 GPU 前向。Sol 先审计已有自然输出、16 维特征和可复用中间张量的覆盖、来源、采集时点及成本。不得先下载大文件或默认重采所有隐藏状态。已有完整的 oracle、norm/delta、层间 coordination、四 utility group 与监督诊断结果直接引用其审计产物；只有缺少 question-disjoint、output-conditional 这一特定判别时才补做。这里不要求重跑已完成的低维分析套餐。

所有新拟合、特征选择和假设更新只使用 Train/dev。以 question ID 的固定哈希在 Train 非示例 1094 内建立 question-disjoint 的 development folds；同一题所有深度必须在同一 fold，近重复/相同输入也不能跨 fold。保存规则与 manifest。历史 ARC Train/Val/Test 均有暴露，这只是新算子和预测头的训练/评价分离，不是全局盲测。Test 文件不能进入新的拟合/选型进程。既有 299 可在冻结后作暴露过的复核，不能再次充当自由搜索池。

按以下顺序回答三个竞争解释，报告正反证据，不再计算一批无关联相关系数：

| 假设 | Sol 的缓存判别 | 结果如何改变方向 |
|---|---|---|
| H0：现有循环输出中缺少可利用的纠错空间 | 逐题列 K4 错但较早某轮正确、K4 对但早停会错，以及全部可选动作的 label-oracle 上界；把 retrospective 选输出和 prospective 早停分开 | 上界不是方法。若准确率可恢复空间过小，则停止把当前输出选择包装为显著超 K4 的路线；效率问题单独保留 |
| H1：B 的几何信号主要重复输出轨迹，或其增量不足以支付采集成本 | 在固定 output-only 预测器的 out-of-fold 错误/残差上评价 B 的增量，复用原 B 的标签定义、监督头和固定正则化集合；比较同复杂度 output、output+B、output+L2–8 控制及 shuffle | 若没有稳定条件增量，不继续调整原几何阈值或增加搜索规则；保留已经得到的 NO-GO |
| H2：原几何压缩丢掉了与候选竞争有关的方向，而非所有 latent 信息无用 | 只测试第 4 节预先定义的一个 candidate-aligned feature family，保持监督头、预算和控制器不变 | 它若仍无增量，关闭本轮 latent-selector 路线；不得立即换另一批特征寻找正数 |

Label-oracle 仅供离线诊断，不进入推理、不作为可达到方法。按 Sol 核实的原 B 接口，预测增量主诊断固定为 prospective 四类目标的 Brier score：沿用 class-balanced ridge，alpha={0.1,1,10}，对 ridge logits 做温度固定为 1 的 softmax 后计算；log-loss 与 retrospective 目标为次诊断。采用 outer5/inner3 question-group folds，exact/Train near duplicates 同组，所有变体使用相同划分与头。此次从初稿 log-loss 主指标改为 Brier，是在新 OOF outcomes 前根据原头接口进行的明确修订，不是依据结果选指标。不要增加温度搜索或换标签定义。另报告校准、实际 action utility、最终 rescue/damage 和成本；F1 只是附加诊断。不要用某个类别的 F1 上升代替闭环方法超过强对照。

CPU 预算：一次固定分析包，最多 60 分钟 CPU wall；复用原 B 的小正则化集合，不增加 ensemble/宽模型/大网格。到预算仍未完成时保存已完成覆盖和瓶颈，不用缩小且挑过的样本声称全量结论。

## 4. 唯一允许的新特征候选：固定的候选竞争投影

这是对“几何压缩丢失方向”的可证伪检验，不是新的因果机制结论。内部状态不被修改；所有计算来自自然前向，禁止 donor、未来 loop、当前 gold 或正确答案选择。

先由 Sol 确认是否缓存了 L10–16 读出位置 R 的实际 attention residual-add 前状态 x 和实际 post-localnorm attention 增量 a。使用原模型冻结的 final normalization 和对应候选输出行，定义一个**诊断 readout lens**：对自然 x+a 与 x 分别计算候选向量之差，去除候选共同均值，记为 b(l,k)。x+a 应使用实际捕获的 attention residual-add 后状态，避免另一次加法舍入改变它。Lens 统一在 FP32 副本上计算，沿用 checkpoint 的 normalization 权重/epsilon 与实际候选输出行；不得搜索不同精度、norm、token 映射或投影形式。该 lens 不是模型从此处到终端的真实后缀，不能称为该层的因果 logit contribution。

在已经完成 loop k 后、k≥2，取 d(l,k)=b(l,k)−b(l,k−1)。候选 t/u 由当前已得到的 output-only 第一/第二候选确定，使用冻结 tie 规则；不读取标签。每个 L10–16 层只取两项：d_t−d_u 与候选中心化 d 的 RMS，共 14 项；再加七层中 d_t−d_u 为正的比例、七层 d 之和与当前相对上一轮输出分数变化的 cosine，共 16 项。零范数显式处理并记录；k=1 的 unavailable 状态不得伪装成零证据。

同样构造 L2–8 的 16 维控制。与原 B 的 16 维几何块、固定 shuffle 控制分开。Output-only 的输入、监督头、正则化集合和控制器结构完全相同；只替换追加的 feature block。不增加层选择、层权重、投影训练或新阈值网格。保留原 B 作为失败基线，不覆盖结果。

Retrospective rollback 只能使用截至当前 loop 的量；prospective continue/stop 只能预测尚未执行的下一轮效用。拟合目标必须明确对应哪个动作，不能用观察下一轮之后的特征预测同一轮的收益。凡读取了当前完整 loop 输出的特征，其成本至少包含该 loop。

这一检验最初限于 Joint 的预定义候选输出头。若 FullAnswer 不能使用同一特征语义和冻结映射，明确写“尚不具备 zero-fit transfer”；不得把字母 token 投影冒充完整答案特征，也不得为迎合已见 FullAnswer Test 结果重新定义特征。

## 5. 新 GPU 收集的准入：只为冻结诊断补必要自然量

缓存核实修订：现有 v3 NPZ 中 latent/unrelated attention increments 为 (4,7,2048)，但 *_x 是每 loop 七层 x 的均值 (4,2048)。它不能忠实计算第 4 节的逐层 lens。禁止用均值 x 冒充各层 x，也不把仅投影 a 的另一假设临时替换进来。需要补采的唯一原因是实际逐层 residual-add 前 x 与实际加法后状态缺失，而不是旧实验失败就重采全部状态。

仅当缓存不足，且 Train/dev CPU 证据表明仍有超过最强 output-only 的可恢复空间、H2 的检验确实缺少上述必要状态时，Sol 才可准备一次最小自然前向补采；这不是 E6C 内部干预。

启动前保存完整 feature/operator 定义、Train/dev folds、来源 hash、零值规则、成本估算及执行命令。常规技术核验由 Sol/主协调者完成，无需频繁调用 Astra。只核验改变的采集路径：两个预定 Train 问题，实际 repeated-layer/current_ut、原生 full-vocab BF16 endpoint hash、候选映射、feature 时序和真实成本。任何 endpoint 不一致先排错，不允许带差异跑科学实验。

预算上限：新自然采集累计不超过 30 GPU 分钟；不保留完整 QKV/大激活，只保存可复算紧凑中间量与特征。优先仅采所需 Train/dev。若全量超预算，先固定 label-blind Train256 的 pilot（fold/ID 在新 outcomes 前冻结），明确它只能决定是否继续本检验，不能当全量确认。超过上限须回报实际覆盖与估算，由新的科学指令决定，不能自动扩张。本指令不授权新 Test 收集、外部 GPU 任务或无限追加 pilot。

## 6. Go / No-Go：增量、实际开销和独立确认分开

**G0：协议有效。** 时间可用性、source/endpoint identity、样本分离或成本核算失败则停止该分支；不能用更好准确率抵消失效。

**G1：有条件信息增量。** 在 question-disjoint Train/dev 预测中，candidate-aligned block 相对同复杂度最强 output-only 必须改善事先选定的 proper predictive loss，并在固定 question-bootstrap 95% 配对区间中支持正向改善；不能只在一个事后筛出的 group 上成立。它还必须优于同维低层/打乱控制，并保留所有 fold 和副指标。若失败，H2 不获支持，本轮 latent feature 搜索结束，E6C NO-GO。通过 G1 只说明预测诊断有希望，不是可用推理算法。

**G2：实际方法有价值。** 冻结原控制器中的该 feature 替换后，才可在预先固定的 Train/dev online 样本上验证同一预测的实际 exit/rollback；不能再改规则。主对照是按 Train/dev 预先选定、同计算约束的强 output-only frontier 方法和当前 K4，而不是较弱 plain agreement。必须包含特征提取、控制器、同步和输出处理的端到端成本。

下列两种成功类型须分开报告，不能互换：

- 准确率路线：在不增加超过固定 K4 的实际计算预算下，相对 K4 与强 output-only 都有正的配对准确率差，且预先规定的 question-bootstrap 95% 下界均大于 0。报告 rescue/damage，不能仅用 margin 或 F1 晋级。
- 效率路线：相对强 output-only，观察准确率不下降，且配对准确率差的 95% 下界不低于预注册的 −0.5 percentage point 非劣界，同时实测平均端到端 latency 至少降低 10%。这是有界非劣效率结论，不叫“超 K4 准确率”。该容差是现在冻结的工程门槛，不能随结果放宽。

若只通过 G1 而 G2 失败，结论是“存在离线可预测性但未转化为值得支付的控制收益”，不得推进内部状态控制。若原 output-only early-exit 在冻结 FullAnswer 上表现可靠，可保留它为独立效率结果，但不能归因于 latent mechanism。

**G3：一般化。** G2 通过之后冻结一个候选及主对照，再提出一份独立数据确认的具体清单与小成本计划；独立数据须在观察其方法 outcomes 前选定，并审计训练/开发/重复问题暴露。ARC Test 已暴露，FullAnswer 仍是相同问题，它们都不能替代这一环节。本轮不自动启动外部大规模收集。未经独立确认，不写“可靠普适改进”或“已完成 Goal”。

**E6C 仍 NO-GO。** G1/G2 通过最多允许进一步确认一个自然状态控制器。内部 suppression/restoration 还需一个具体、可区分替代解释的因果操作及成本理由，另发指令；不能因为早期 L10–16 bundle 的条件因果作用而跳过真实增量门槛。

## 7. Sol 交付与停止条件

立即交付 TRAIN_DEV_FAILURE_DIAGNOSIS.md：缓存覆盖、动作上界、fold 外输出预测错误、B 的条件增量、完整负例与推理时序。随后交付唯一 candidate-aligned 检验的冻结 manifest、特征定义/来源、同复杂度对照表、全 fold 预测/动作结果、bootstrap、真实或尚未实测的成本标记。所有实现和分析代码均由 Sol 编写并审计。

最终只允许四种明确状态：①原 B 失败且未发现增量；②新特征有离线增量但在线不划算；③候选通过开发门槛、尚待独立确认；④独立确认支持预先定义的准确率/效率结论。不得把①或②重新命名为方法成功。

FullAnswer 冻结结果完成后，最多追加一次本指令的转移裁决。若新特征 family 失败，不再在本轮自动增加层、特征或规则。Astra 重新判断的是研究假设是否值得改变，Sol 不用尝试凑出一个赢的指标。

## 8. 唯一一次 FullAnswer 完成后追加裁决

以下为 Sol 完整冻结评估和数值核实提供的结果；Astra 未运行统计、代码或 GPU，仅据其核实摘要作科学裁决。E6A/B 既定实验现已完成，收据/逐题结果/原始分数/SHA/replay 由 Sol 汇总保存。

| FullAnswer 口径 | 当前 K4 | 原 primary weighted4 | 原 Joint 选定 LoopCD adaptive Cal |
|---|---:|---:|---:|
| Char-normalized，预注册主指标 | 705/1172 | 693/1172 | 721/1172 |
| Raw sum，次指标 | 658/1172 | 640/1172 | 679/1172 |
| Token mean，次指标 | 696/1172 | 690/1172 | 724/1172 |

Char primary 下，weighted4 相对同协议 K4 为 18 rescue / 30 damage，差 −1.0239 percentage points，配对 95% 区间 [−2.2184,+0.17065]。它没有实现准确率提升；不能用次指标或另一条规则替换其 primary 地位。

原 Joint-Cal LoopCD adaptive 没有重新拟合，在 char primary 下为 49 rescue / 33 damage，差 +1.3652 percentage points，区间 [−0.08532,+2.9010]。这是冻结 output-only 对照的正向转移点估计，但区间跨零，尚不能宣称确定的总体提升。它原本就是预先选定的 Cal，报告这个结果不需要换 primary；也不能把它的收益归给 latent features。将它与 K4 保留为以后独立确认的强对照是合理的，但本轮不自动启动外部实验。

B 的 FullAnswer zero-fit 仍不支持内部信号增量：ZL/Cal 的 output rollback 为 702，latent rollback 为 700，均低于 K4 的 705。Early/stability 的两类方法均为 705；early 平均 loops 为 output 3.99147、latent 3.99232，几乎没有计算节省。Joint 中 output-only 的明显早停节省没有在这一冻结评分转移中保留。不可把“正确数持平”单独称为效率成功，也不可推断这种协议差异已经定位了某个内部计算机制。

该结果加强原判断：现有 latent 几何控制器既未超过强 output-only，也没有可靠跨评分效率收益；E6C 继续 NO-GO。FullAnswer 与 Joint 的评分/任务形式不同，不能用 705 对比 Joint 的 1022 来声称隐藏表示发生了某种因果退化。两者使用同一 ARC 问题，依旧不是独立数据确认。

不据这些 Test 结果改变第 3–6 节的特征定义、层、监督头、Brier 主诊断或门槛。下一步只由 Sol 执行已经批准的 Train/dev 缓存 H0/H1 判别；不重复已完成分析，有必要时再按原技术/预算门槛补采唯一 H2 lens。候选若不能通过 output-conditional 增量与实际成本门槛，就接受失败，不把上述 LoopCD 点估计或局部 agreement 收益改写为 latent 方法成功。

## 9. 持续 Research Goal 下的 H2 实际执行与 G2 映射澄清

用户现已明确启动持续研究 Goal。失败 family 的停止不等于整个 Goal 完成；必须根据失败结果选择新的可证伪问题或明确当前实证缺口，不能以提交计划替代执行。当前 Sol 正执行本指令已经准入的唯一 H2 自然补采/诊断，不改其特征或指标定义。所有分析、代码和 GPU 执行仍由 Sol 完成，Astra 仅指挥。

Sol 已核实 H0/H1：Train1094 当前 K4=966、循环输出 oracle=1014，有48题可恢复；这只是离线输出集合上界。旧几何 prospective Brier improvement=−0.000464，95% CI[−0.001689,+0.000771]。Retrospective improvement=+0.001137，但低层控制=+0.001852、shuffle=+0.001141；固定 action 的 output/combined 均959，低于 K4 的966，prospective combined943低于output948。原几何路线没有获得新的增量支持，不能反复重选阈值。H2 的真实逐层状态确实缺失且输出有可恢复空间，因此原技术2和≤30GPU分钟自然补采门槛适用，不需要再等待一轮计划批准。

**G2 唯一允许的控制器解释：独立标注的 supervised 分支。** 旧无训练 B controller 的 index11 relative_delta/index5 coordination 与新方向特征没有相同语义；禁止将新向量盲塞旧索引，也不临时在旧 gate 上加 direction criterion。

若 G1 通过，沿用上一轮已经固定的监督 class→action mapping、类别语义和预测时点，只替换预测头追加的16维 feature block。Output-only 和 combined 使用相同的头、alpha选择目标、class→action规则、可执行深度与成本定义。Sol 在查看 H2 outcomes 前保存原 mapping 的可读表和来源 hash；若旧 mapping 不具备所声称的推理时点条件，报告不适用，不能偷偷另造一个映射。标签只用于 Train 拟合/离线评价，推理不见 gold；此分支不称 training-free。

当前仅执行 G1，不修改控制器。Train/dev folds 用于拟合/选择；Val299 可以自然采集，但在头和规则冻结后只作历史已暴露复核。G1 后依据实际证据裁决，科学分支见 ASTRA_H2_CONTINGENCY.md；该文件不提前准入新的 GPU 操作或 E6C。

### 用户更新：没有总研究时间限制

用户进一步明确没有时间限制。不能以单轮预算、预期完成时间或一项特征失败结束持续 Goal，也不能为赶时长缩小科学样本。当前 H2 已按冻结定义启动完整 Train1094+Val299，自然采集期间不改定义、源码或 cohort。此前30 GPU分钟是该轮事前资源估计/核算门槛，不是整个研究的期限；如果实际需要更长，Astra依据实测成本和科学必要性延展执行指令，保持结果完整，不把超时等同于科学失败。后续实验按辨识力与完整证据设计，不要求压进这一旧时间窗。

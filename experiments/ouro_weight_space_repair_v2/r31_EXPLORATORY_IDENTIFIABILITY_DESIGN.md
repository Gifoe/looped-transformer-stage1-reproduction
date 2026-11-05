# R31 exploratory：先区分局部信号容量与价值识别

状态：Astra新设计，供Root审查、Sol CPU实现；没有新GPU授权。当前唯一执行建议是下面的prefix共享二元oracle挑战，不再凭活跃范数或小概率margin变化构造新部署系数。它不改变已关闭A/C，不是方法成功。

## 四张证据卡

1. R27：固定V通路能传递lookup内容，但该结果不识别ARC nuisance或truth。
2. R28/R30：空间重排和时间冻结都产生非零且不同响应，却0 rescue/1 damage。C多数题margin变化不利，不能以放大修正作为独立机制。
3. 新CPU全24分解：两者pre-late和post-late均13，late veto=0。后续固定late抹掉已获得正确排序的解释，在这批实际预测中没有实例。
4. 旧R21的59/216仅是native六深度候选分数族容量；它没有检查固定C的prefix局部输出是否包含可用的纠错信号。Root prior-art notes中的LoopCD/校准/function-vector/CAA已覆盖通用框架，不能把新的oracle或patching名称宣传为原创机制。

## 三个机制不同的候选假设

| 假设 | 观测支持/未识别点 | 真正不同的后续机制与证伪问题 | 当前选择 |
|---|---|---|---|
| H1：局部修正有可用内容，但当前投影缺乏prefix价值识别 | C有少量题margin向好且多位置活跃；这远非方向有效证据 | 未来方法需要识别何时保留局部修正。先问：即使oracle可在每个共享prefix只选现成Fixed或C，是否能救R6错误并保旧收益？若不能，研究selector没有本次信号容量基础 | **主challenge：CPU二元oracle，仅容量上界** |
| H2：问题缺失参考的来源效应不是当前问题的nuisance | A/C在无问题reference上有影响，但无效用；R20的question-value零消融也未给价值证据 | 有意义的新机制应测真实问题与自然来源内容的因果交互，而非无条件来源响应。需预定义不破坏字段/位置的独立两因素操作和交互反证；不能仅给C增加Q项或改参考就称新方法 | 不执行；natural factorial尚缺可识别操作，不复活已关reference变体 |
| H3：晚期readout中的线性扣除丢失了来源信息的非线性利用 | R27在内部V replay上可影响映射，但外部logit扣除无收益；二者任务不同，迁移证据弱 | 潜在机制是冻结模型后续计算对一个已定位内容的非线性响应，而不是output空间再投影。需先证明原ARC中对应功能、指定合法fullV→内部作用映射；不能直接把CAA/function-vector输出注入当新理论 | 与H1不同的fallback研究问题；目前无完整target-free映射，拒绝无依据指定注入层/幅度 |

重叠挑战：H1看起来可能只是旧scalar controller再包装。修订为**不训练、不拟合、不部署controller，只对固定已观察分布做oracle上界**；不扫alpha、不寻找最佳符号/强度。H2/H3没有获准实现，不能按oracle结果自动开另一条GPU路线。

证据挑战：题级C margin有9增不意味着某些prefix可用，且独立按候选选token会作弊。修订为强制同一实际Q prefix任务共用一个二元选择，使用全部24全部候选，不筛正例；oracle若正面仍只说明容量，不说明label-free selector存在或可学习。

## 唯一CPU实验：共享prefix二元路由容量

输入只读完整已解封r30_SCORES/ANALYSIS/COMPLETE、R28 task graph和sanitized requests、既有gold与R6保护集。先Root冻结本设计/实现/合成receipt后Sol才运行真实oracle。不给模型输入，不读新heldout，不使用其他数据集。

对一题每个候选c的位置t，取现有FP32完整分布产生的两个target logprob：f_ct=Fixed，p_ct=primary(C)。让k(c,t)是图中实际Q prefix task key，共享prefix必须同key。定义二元x_k∈{0,1}：x=0选择原Fixed整分布，x=1选择原C整分布。输出词表已在原实验完整归一化；因为是离散选择已有分布，不需要也不允许从candidate LL伪造词表归一化。

```
a_c = sum_t f_ct / chars_c + saved_native_late_c
d_ck = sum_{t:k(c,t)=k} (p_ct-f_ct) / chars_c
s_c(x) = a_c + sum_k d_ck*x_k
maximize m
subject to m <= (a_gold-a_j) + sum_k(d_gold,k-d_jk)*x_k   for every wrong j
x_k binary; all candidates and all shared keys included
```

这是oracle、不是合法推理算法：gold决定解，不能部署、报告accuracy提升或与原方法排名。模型source干预为零；张量只需候选×prefix稀疏系数矩阵，不访问BF16缓存/head，不修改clock。x全0为A1，全1为原C（考虑下面FP32归约证书）；前缀全零变化时该变量无作用并固定0，仅去除完全相同的零列，不按gold筛变量。

这里同题的共享prefix被耦合；跨题若Q task key完全相同也必须报告并全局共享，不能假定不同。CPU预审先核跨题key无交集；若有交集，不能直接用独立题oracle称联合可达，只报告单题上界并将联合结论标undetermined，另审联合约束。不按目标token或候选索引另造key。

## 数值证书与互斥类别

以存储FP32 LP的精确binary Fraction构建a,d；原字符整数分母和已保存late作为精确输入。求解只用于提出witness/dual，最终结论独立验证。预定tau=1e-8（原normalized score单位），不按结果改tieband。

原评分用FP32 sum而线性表达式用实数sum，因此必须给每候选全x一致的保守误差B_c。设n为token数、u=2^-24、S=sum_t max(|f_ct|,|p_ct|)，gamma=(n-1)u/(1-(n-1)u)。可取
`B_c=(gamma*S+n*2^-149)/chars + 16*eps64*(S/chars+abs(late)+1)`，全部上取保守界；检查n*u<1、无非有限/溢出。FP32次正规项纳入绝对误差；最终除法/加late舍入由后一项覆盖。不得将Python/NumPy不同reduction直接当GPUexact。

- **certified reachable**：MILP给出严格0/1 witness，独立Fraction验证所有gold-wrong差减(B_gold+B_j)均>tau。存所有key选择和证书，但不得作为新controller复用。
- **certified unreachable for strict margin**：允许对x∈[0,1]的松弛LP求dual。对任一实际非负simplex λ_j，令A_j=a_gold-a_j+B_gold+B_j，D_jk=d_gold,k-d_jk，则有效上界为 `U=sum_j λ_j*A_j + sum_k max(sum_j λ_j*D_jk,0)`。将solver dual非负化仅限既定微小数值误差，随后转Fraction并精确归一化，独立算U；U<=tau才认证。这个上界甚至放宽二元选择，故结论安全。
- **undetermined**：无已认证witness且U>tau，包括MILP超时/松弛间隙/near-tie，不强行当unreachable。Solver自身optimal/status/dualbound不直接当数学证明。

需要单独报告strict margin与原argmax tie规则的差异；unreachable仅针对>tau，不声称不可通过tie拿到正确。不能浮点松容差把不可行点变证据。

## Synthetic/资源/事前决策

CPU合成覆盖：共享key耦合反例、全0/全1端点、只有混合二元可达、所有不可达、LP松弛可行而整数不可行、tieband、dual符号错误/非simplex拒绝、候选重排、重复候选、FP32舍入界。每题MILP最多3秒，单线程；全部CPU job cap180秒，不调用torch模型/CUDA，不改GPU ledger。保存源/输入/solver版本/原始解/精确证书。无真实oracle前的效果选择。

事前报告全部24四个背景：R6正确、R6错误、唯一R6-only收益题、其余；类别仍只有上面三互斥类。R6错误中>=1 certified reachable只表示存在本分布族未被利用的局部容量；必须同时报告唯一旧收益题是否可达。若全部R6错误certified unreachable，则停止围绕C的局部selector研究；若部分undetermined，不得宣称没有容量。若有容量也不自动开GPU，更不训练gate；下一设计须另给无标签的、独立于这些oracle选择的价值证据。

这项挑战不能完成原ARC目标，选择它是为了在剩844.5347464797087 GPU-worker秒下避免无依据继续试向量。完整576+517+299+1172不缩，CPU oracle结果不能替代全链。当前没有证据充分且完整可实现的新部署方法，因此不提供伪造的推理公式或耗掉余额的GPU任务。若要求直接再跑方法，缺口是H2的可识别自然干预或H3的合法内容映射，及完整全链预算；不是更大的同一alpha。

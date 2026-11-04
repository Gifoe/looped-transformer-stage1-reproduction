# R21 Astra设计决策：新增CPU可达性证据后不启动GPU候选

状态：DESIGN ONLY / NO EXECUTABLE CANDIDATE。基于已完整解封的R20 E2，不是未完成v0/E1的部分效果。新增GPU为0；无SSH/Git/heldout访问。唯一新增工作是读取完整24题结果并进行CPU集合与标量诊断。用户目标尚未达到，本文件不声称完成目标。

## 新结果改变了什么判断

R20不是没有实施干预：548个预测位置中415个主修正能量大于0；alpha均值0.39643225788803904，133个为0、57个为1，平均修正P范数平方0.06937977258870448。它在这个样本上改变了分布，但**24题预测全部与A1/Fixed相同**，相对R6只改变1题且是损害。不能再将失败优先归于零干预、错误readout或没有执行到路径。

这也不是只有总准确率相等：逐题预测exact一致说明本次问题V路径参考没有形成任何A1之外的新候选决策。不能因此声称所有问题信息都不能隔离，也不能把分布变化较大说成语义方向有效。

唯一R6相对Fixed独有纠错LEAP_2001_4_10241，在固定gold3对primary rival1的分数差：primary=-0.008428516842070022，demo=0.030048290888468387，raw-demo C=0.010419187091645798，shuffle=0.36831209773109075，oldM2=0.0395450478508359。它说明本例问题位置消融不比这些控制更会保护原收益；不能据此挑demo/shuffle晋级或调primary强度跨过单题阈值。

## 新增且可精确证明的路由上界

令S={R6, primary, demo, raw-demo C, shuffle, oldM2, Fixed, A1}为这些方法已经实际输出的单个答案。对题i令C_i为S中至少一个方法答对的事件。CPU逐题核对得到：

- R6正确14题、错误10题；
- R6错误的10题中，S的其余每个方法都错误；
- 因而Σ_i 1[C_i]=14。

即使给一个选择器提供gold，让它每题从S已有答案中任意挑选，其正确数上限也只有14/24。这不是可部署方法，而是事后oracle上界，**否定了用这些已输出答案之间的更聪明门控在此样本获得新净纠错的可能性**。它不依赖门控阈值、训练与否或是否知道正确答案。

适用范围严格限定：不否定对连续score混合后产生新候选排序的可能性，不否定新forward信息，也不否定全部ARC题。多类别任务即使所有端点答错，组合score仍可能换赢家，故不能把上述选择上界冒充凸包或全参数空间证明。控制不能晋级，已有锚点/强度/轴/metric路线的关闭规则照旧。

## 下一方案审查：没有足够合理的唯一可执行设计

R13/R14已表明撤销候选级R6会丢失原收益，单纯加法锚点不解决；R7旧方向oracle对216个R6开发错误仅32可达；R15没有可靠价值代理；R16—R19的优化、metric、loop保护没有获得净收益。R20进一步在结构匹配的因果参考上产生真实非零干预，却未产生新A1决策，并未超过位置/幅度控制。

因此下一项若只是替换参考、扩大/缩小消融范围、改变层/循环、保护轴、门控、系数或把几种已有预测路由起来，其理由仍然只是“也许这次不同”。现在还有上述路由上界的明确反证，不应拿剩余预算继续做这类搜索。

真正不同的方向需要产生新的、与事实正确性有关的证据，而非再次估计与已有guidance的对齐。例如外部检索、辅助语义验证任务或学习校准都能引入不同证据，但前者改变当前限定的信息来源/任务，后者涉及已明确禁止的强辅助任务支配或训练；不能擅自把它们包装为原目标下的新修复算法。内部因果干预本身不是正确性监督，R20也不能给出下一种内部干预应当向哪侧修改的准则。

这并不逻辑证明不存在符合限制的改进方法。结论是：在现有信息与已关闭路线下，我不能给出一个机制明确、区别实质且优于任意猜测的新公式。强造一个会把设计问题变成未申报的多轮超参数搜索。故本轮不向Sol交付GPU公式；不得将这里的例子解释为开展新任务的许可。

## 给Sol的具体交接与CPU复核合同

仅允许复核本次完整结果的CPU诊断：

```text
load complete E2 analysis and its bound raw scores
assert complete24 and both old overlaps passed and final params equal
assert IDs exactly frozen24
for each i: union_correct[i] = any(method_prediction[i] == gold[i] for method in S)
assert sum(union_correct) == 14
assert no R6-wrong question has union_correct
assert all(primary_prediction == A1_prediction == Fixed_prediction)
recompute alpha counts, nonzero energy counts, fixed unique-case margins
write diagnostic receipt marked post-hoc development upper bound
```

gold只用于已解封结果的事后评估，不能进入任何部署方向/系数。无需新forward、缓存读取或全V重建；原raw scalar无法恢复新logit路径。不能据此挑出题目重新设计小cohort，不能做test逐题搜索，也不能把oracle当真实算法。

本设计的GPU停止门槛已触发：无可执行且有理由的不同机制。没有新pilot、无新控制排名晋级、无扩预算请求。剩余3127.7056477041915 GPU-worker秒为父任务已审计余额；原6小时不重置。预算不是必须花完的额度。若日后出现具体不同机制，必须重新提供完整公式、无标签prefix条件、精确parity、可反证的matched controls及完整后续预算，而不是从此文件自动启动。

## 对用户的准确说明

本轮验证了一个确实不同的辅助参考机制，结果13/24低于R6的14，救0伤1；完整效应已复核，不能继续称它只是执行未完成。干预虽非零，却没有改变A1的任何最终答案；全部既有候选输出的oracle路由也至多14/24。当前依据支持停止这一修复家族，而不支持承诺再换一个公式就更有效。R11的742/1172及p=.307保持原状，前缀机制与751目标仍未实现。

## R21正式下一实验：CPU六深度读出可达性审计（优先于前述仅复核交接）

Root要求提供能区分方向容量与选择问题的实际实验。本地原R2 discovery/selection结果已确认保存native_k1、k2、k3、k4、k5、k6每候选normalized_scores，因此可以0 GPU执行，而不读取hidden states、不重跑head、不发明新的prefix公式。前述No-Go仅指没有合理新部署候选；本节冻结**唯一下一实验**，交Sol CPU实现。

### 研究问题与可识别范围

在原576题中，现有六个native深度的**候选读出分数**是否已经足以使正确答案成为赢家？如果能，具体是某个实际深度已经答对，还是只有各深度分数的凸组合才存在正确排序？原R6错误中有多少仍在这个有限score族外？

它区分读出层的有限族容量和oracle选择空间，不测试新解码器。不能回答全部hidden representation是否含事实信息；凸包外也不代表信息本身缺失。不能将oracle存在直接归因于某个未测试无标签选择器失败，更不能把按gold求的权重用于实际推理。

### 精确输入及两个嵌套集合

只读results/slcd_r2_discovery_development.jsonl与selection对应文件中的原始native_k1..k6 normalized_scores，及R7开发记录的baseline_scores.M0_R6、gold、ID、候选顺序/字符长度元数据。384/192各自保持原划分，合并576为主表。两套来源ID/prompt hash、候选数/顺序、字符长度一致，并对双方共有k1/k4/k5分数逐元素exact核验；无法一致就停止，不放容差。代码及所有输入hash在计算新oracle结果前冻结。

每题构造S=[s1,...,s6]，s_k是原native深度候选分数向量。H6=conv(s1,...,s6)。扩展集合H7=conv(s1,...,s6,s_R6)。原记录中晚期项/R6定义保持原样；本诊断对已有完整score对象做容量上界，不声称这些凸组合对应合法全Vprefix干预。

不得添加Fixed/M2/控制或探索其他集合，不求训练集共享权重，不在selection上选权重，不将任何求得的权重命名为新方法。H7只评估R6锚点增加的候选排序容量，不复活已关闭的混合方案。

### LP与离散oracle

对每题、每个可能标签y（不仅gold），解：

maximize m
subject to sum_k w_k=1, w_k>=0,
           sum_k w_k*(s_k[y]−s_k[j]) >= m  for every j!=y.

H6使用6个权重，H7使用7个。m是最大最小候选margin；有限simplex保证有界。用float64 scipy.optimize.linprog(method='highs') CPU求解。为数值稳定，每题全部score先减各列第一候选分数（不变margin），再除同一正尺度 max(1,max_abs_pair_difference)，主LP用尺度化margin，原分数margin另存。不得因结果更好选择尺度。

严格可达：经原始约束独立重算且归一化margin>1e-8；严格不可达：dual上界<-1e-8；其余near-boundary/uncertain单列，不计入任何成功数。不能使用求解器success flag代替证书。保留primal weights/margins、sum/非负/不等式残差、dual变量及独立dual gap；最大残差或gap>1e-8则uncertain并标注。遇solver异常停止本诊断输出最终解释，保留错误记录，不调solver/presolve/阈值搜结果。

离散D6 oracle直接按原argmax tie-breaking计算六个实际深度是否至少一个答对；另报告是否存在strict正margin的native深度，区分tie-only。不把LP的strict标准与原argmax混成同一指标。

### 固定输出分类

在R6错误题中，按顺序分四类：

A. 某个实际native深度预测正确（D6可选答案已存在）。
B. D6全错，但H6严格可达（只有组合排序容量的证据）。
C. D6全错且H6严格不可达，H7严格可达（只有含R6的扩展score族容量证据）。
D. H7严格不可达（该有限score族不能纠错）。

其余边界/证书不确定单列U；若D6因tie答对，明确tie-only。H6严格不可达与H7严格不可达不要求负margin一定明显，near0不可武断归D。验证H7最大margin>=H6在证书界内；D6存在strict赢家必须H6可达，违反即实现审查失败。

报告discovery、selection、合并的数量和比例；对R6正确题报告同类可达率作背景，但不得通过背景分布改门槛。对每题枚举全部候选标签的可达数，可以描述这个convex hull是否几乎任何答案都能实现：若大量候选均可达，gold可达只是容量，不是语义选择证据。该数不作为显著性p值；尤其R6错误条件下的gold不是均匀随机标签，不做错误的随机标签显著性推断。

### 事前判读门槛（不是模型晋级）

主判据为R6错误题中H6严格可达比例。若>=20%（实际错误216时至少44题），两个划分各自>=10%，且U<=全部576的1%，则结论仅为“六深度读出有足够非零纠错容量，值得设计独立无标签价值识别诊断”；不授权部署或GPU。若不满足，结论为“这个有限score族未达到预设值得继续的容量门槛”，不是模型没知识。

A/B/C/D分布给出下一设计依据：A多提示未来重点应是怎样无标签辨别既有不同答案；B多提示存在读出组合容量但没有正确性识别方法；C多仅说明R6扩展上界，不能复开混合；D多限制的是该有限方向空间。任何情况都不能自动发布新的weighted-score模型。

R7的32/216旧单方向oracle仅作历史参照，两者可行集合不同，不作显著优越性宣称。所有门槛只决定研究诊断结论，不称独立验证或统计显著，且本轮不用test纠错选赢家。

### CPU执行与交付

Sol可写scripts/r21_readout_oracle.py、configs/r21_READOUT_ORACLE_LOCK.json及results/r21_*。先用CPU合成构造：已有strict赢家、两个端点都错但中间gold赢、整个凸包gold不可达、全同/tie、候选置换、列公共平移、正缩放；同时验证primal/dual证书。完整最多2*sum候选数个小LP，通常约4600次，每题一次求全部标签；禁止重跑模型或加载权重。CPU墙钟硬限20分钟，超过则保存incomplete，不自动改抽样。

交付包含冻结输入hash、每题每标签H6/H7 primal/dual诊断、四类表、各划分汇总、上述固定门槛结果及可识别范围说明。原已有548位置诊断可附为背景，不参与LP选择。

GPU预算0秒；原累计与3127.7056477041915秒余额不变。另两数据集继续暂停，517/299/test逐题继续关闭。该实验提供的最有价值结果是严格限定的读出容量证据，不能越界宣称native6 hidden信息是否存在。

### Root收口后的唯一主分组（替代上面的A/B/C/D主表）

对全576每题使用H6与实际六深度预测，按以下互斥顺序分组；R6正确/错误只作分层汇总，不用于选题：

1. discrete_reachable：六个实际native深度至少一个按原argmax规则答对。另标strict或tie-only，tie-only不冒充严格正margin。
2. only_convex_reachable：不属于1，且H6有经认证primal lower bound>1e-8。
3. hull_unreachable：不属于1，且H6有经认证dual upper bound<-1e-8。
4. numerically_undetermined：其余，包括near-zero tie band、证书跨阈值及数值容差无法定性的情况。solver异常则整个运行停止为incomplete，不以剔除题目继续作最终结论。

H7只作为单独的第二扩展upperbound列与H6交叉表，不再形成主分类C，不将它用作晋级判据或部署mix。H6/H7的primal下界与dual上界而非点估计用于全部可达判断；主门槛统计H6严格可达（包括1中strict可证部分与2），不把tie-only计入。原>=20%、两个划分>=10%、U<=1%保持不变。完整576主表包括所有题，没有深度、样本或子集搜索。

本节为最终交Sol合同。R21是一项已明确可执行的CPU诊断；文件开头“No executable candidate”仅指没有新部署算法，不指本CPU实验。Root review后冻结再执行；Astra本轮没有运行LP或查看新oracle结果。

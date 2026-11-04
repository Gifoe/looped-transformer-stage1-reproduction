# R22 Astra设计：递归轨迹收缩是否提供无标签价值信息

状态：唯一候选为CPU机制诊断，不是部署解码器；尚未运行。需Root审查、Sol实现及事前公开锁后执行。新增GPU预算0。

## R21给出的具体问题

R6错误216题中54题已有某个native深度答对，5题仅在六深度score凸包内可达，157题在该有限族不可达。这支持研究“54个已经出现的正确答案是否带有可观察的价值信号”，而不支持直接用oracle权重部署。H7对R6错误只增加1个可达，故不选择R6混合路线。

本实验仅检验一个物理计算假设：同一共享权重递归反复作用时，相对更稳定的候选前缀轨迹，是否更可能对应正确的native候选答案。固定Ouro源码每轮结束的norm输出继续进入下一轮，因此可直接比较同一坐标系的已保存normalized states。这里的“稳定”只指观测步长衰减，不等于Jacobian收缩、固定点存在或语义正确。

这不同于多数投票、熵或用guidance保护轴：不数答案出现次数，不计算概率熵，不拟合参考方向，也不输出新修正。理论风险明确：错误答案也可能是稳定吸引结果；反复递归的数值收敛不蕴含真值。本诊断就是以正确/错误挑战对检验这个风险，失败后不能改层、窗口、norm或信号方向。

## 冻结无标签信号

仅使用原576题main native normalized BF16 states h1..h6；不进行任何新forward。每候选每预测位置t，读取CPUfloat64，定义：

E45(t)=||h5(t)−h4(t)||² / hidden_dimension；
E56(t)=||h6(t)−h5(t)||² / hidden_dimension；
R(t)=E45(t)/(E45(t)+E56(t))，分母严格为0时R(t)=0.5并记录degenerate。

每候选价值信号V(a)=mean_t R(t)，越大表示末轮步长相对前一轮更小。不加epsilon、温度、裁剪或token筛选；不是越晚状态越接近gold的预设结论。固定使用最后两个已存在递归步，避免搜索最有利深度；不复活loop-axis干预。

这是当前无标签轨迹特征：每h_k(t)来自预测该token之前已消费文本，特征不读取该位置target ID、gold或未来答案；候选级聚合完成之后只用于诊断，不承诺可直接变成候选扩展不变的prefix系数。

两个固定控制：C_energy(a)=−mean_t E56(t)（绝对末步能量）；C_margin(a)=native6原字符归一化score。前者检验相对衰减是否只是末步幅度，后者检验是否只是现有晚深度分数偏好。控制不可晋级，不加熵/投票或其他特征搜索。

## 全576题、预先确定的挑战集合

对每题，用旧六个native实际argmax的不同答案组成集合D_i（不看gold）；取所有a∈D_i且a≠R6赢家作为挑战者。native同一答案出现多次只计一次，防止变成隐含投票。先冻结每题挑战者列表及其hash，再计算特征。全576题均进入覆盖统计，不以R21 gold可达、margin、长度或正确性筛题。

每个挑战对(i,a)的信号为ΔV=V(a)−V(R6)，控制同样作差。计算特征的脚本不得读取gold；gold仅由后续独立评估脚本加载。

评估时：rescue类=R6错且挑战者对；damage类=R6对且挑战者错；双方错单独作为background，不放进主AUC。挑战者与R6不同，不存在双方对。每题在自己的主评估类内总权重1，多挑战者各权重1/该题该类pair数；各题而非token/挑战对作为独立重采样单位。必须报告R21的54题是否全部有rescue pair；理论上应当是，否则输入/候选索引错误。不能只在这54题报告信号而漏掉正确R6的损害机会。

主指标为加权AUC(ΔV区分rescue与damage)，分数越大越倾向rescue，符号事前固定；ties计0.5。同时报告每类样本题数、pair数、degenerate位置比例、两控制AUC及主减控制的AUC差。采用seed2021、10000次题级分层bootstrap（按rescue题/damage题重采样）获得percentile95%CI；同一次重采样用于三个信号，不能独立bootstrap后相减区间。分别报告discovery与selection及合并；重复开发数据，不能称独立验证。

## 固定判断门槛

必须全部满足才称“这一信号值得进一步研究”：

1. discovery与selection各自rescue题和damage题均>=10，否则underpowered/inconclusive，禁止扩样或换cohort。
2. 两划分主AUC均>=0.65；合并主AUC的95%CI下界>0.5。
3. 合并主AUC减C_energy、减C_margin的两个配对bootstrap差值95%CI下界均>0。
4. 全数据/状态审计通过，所有数值有限；degenerate比例如实报告，不能用其筛题或改规则。

不做正负号选择、阈值优化、学习controller、特征组合或层/窗口扫描。失败即关闭这一具体“末段相对步长衰减代表价值”的假设；不外推所有稳定性信号无效。通过也只是有条件关联证据，不是因果选择价值或可部署改进，下一步仍需另行设计prefix机制及成本审查。

## 精确输入/缓存与CPU可行性

依据R21冻结ID和原SOURCE_AUDIT，只读取discovery384/selection192对应状态文件；不得加载517/299/test状态、权重、模型或参考forward。逐文件验证已有hash，payload ID、candidate index、target IDs/位置数、normalized six states shape与原manifest/cache元数据对齐。计算时不把target IDs输入特征，它们只用于索引审计。保持完整候选token位置，mean是token等权，不重新使用字符分母修饰V；C_margin沿用原score字符分母。

状态在服务器时可由Sol经Root批准运行**CPU-only**脚本：CUDA_VISIBLE_DEVICES空、torch.load(map_location='cpu',weights_only=True)，逐候选流式算E45/E56，立即释放；不能调用load_model或将tensor移到CUDA。直接引用缓存，不需要新fullV。使用torch或NumPyfloat64独立实现对合成状态和首个实际候选数值核验；先冻结数学实现和容差再运行真实特征，不能见结果后调整。

CPU合成测试须包括静止状态→0.5、末步零→1、前步零且末步非零→0、两个步长相等→0.5、共同平移不变、共同正缩放不变、候选扩展不变、原shape/位置拒绝、非有限拒绝；bootstrap/AUC用人工可核对样本验证tie/题权重/配对差区间。first real NumPy/Torch float64最大差<=1e−12，不通过则停止。

墙钟硬限30分钟（含所有状态hash/IO，实际体积未知，不保证完成），主流程25分钟停止启动新文件、30分钟前终止并保存incomplete；不更换部分cohort得结论。不占GPU-worker预算，但CPU/IO成本单独报告。全部576特征完成且输入审计通过之后才进行标签关联分析；不报告部分AUC。各既有GPU账本字节不变，原余额3127.7056477041915秒不变。

## 对157题的解释与后续边界

157题在六深度候选score凸包内不可达，R22不试图用controller创造不存在于D_i的正确答案。若信号通过，它最多为54个discrete机会提供研究依据，不能凭5个convex-only或157个unreachable宣称收益覆盖。若信号失败，当前可达性与可靠性之间的缺口依然存在，不能用oracle54直接推算部署准确率。

此实验提供一个可反驳的无标签轨迹机制假设，成本为CPU缓存读取而非再跑数轮参考前向。它是调查，不是保证更有效的新方法；所有部署与完整ARC决定继续关闭，另两个数据集继续暂停。

## 最终事前scope澄清：长度混杂控制与CPU执行

Root提出的长度混杂成立：候选token平均值可能随长度及共享首位置权重变化。以下内容在任何真实特征计算前冻结，优先于上文未包括长度控制的门槛。

增加两个固定、均不可晋级的控制：L+(a)=log(T_a)，L−(a)=−log(T_a)，其中T_a为原候选完整预测位置数且必须>=1。对挑战者与R6取相同差分ΔL±；两个方向都事前报告，不能按效果择符号，也不能把长度与V组合拟合。

主信号、energy/margin控制、4→5与5→6窗口及正方向全部不变。完整“值得进一步研究”的门槛在原四条件外，增加主AUC减L+、减L−的两项配对bootstrap差值95%CI下界均>0；四个主减控制差均使用同一次题级重采样。若未超过任一长度控制，仅报告探索性关联/混杂未排除，不通过原机制继续门槛。即使通过这些控制，也不能声称已排除所有长度或语义混杂，更不能作因果结论。

描述性覆盖表必须报告：全部挑战对、等token长度对、双方均单token对的题数/pair数及rescue/damage/background组成。上述子组不用于筛选主样本、修改门槛、重新选信号或独立晋级；不针对小子组追加显著性检验。

对每题各候选首个预测位置，先核对原prompt prefix token IDs完全相同，再检查缓存h1..h6首位置跨候选逐元素exact equality，记录逐深度最大差与exact计数。若不相等，不静默归因于模型：记录缓存/历史causal计算路径审计失败并停止真实特征解释，不能删除首位置后继续。共享首位置exact应使首位置R(t)相同；双方单token时ΔV理论上为0，作为独立实现自检。等长度不代表前缀相同，不能要求后续位置也equal。

所有576仍来自原discovery384+selection192，不增加/重抽cohort。Sol实现后先CPU synthetic和静态审核；不得先跑真实特征。实际服务器CPU执行需要CUDA_VISIBLE_DEVICES为空、torch.load(map_location='cpu',weights_only=True)、2线程（torch.set_num_threads(2)，并设置OMP/MKL等线程上限），流式读取，禁止load_model、forward或CUDA设备操作。GPU账本保持字节不变。

首个真实候选NumPy/Torch FP64核对<=1e−12仍保留；其BF16输入先按原值精确提升FP64。所有新增控制、覆盖及首位置审计须纳入代码、协议和输出schema，在真实特征开始前冻结。

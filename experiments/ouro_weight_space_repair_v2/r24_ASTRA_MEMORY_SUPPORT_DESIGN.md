# R24 Astra设计：合法25demo经验记忆的0GPU内容支持审计

状态：唯一下一步为Phase A CPU内容支持审计，尚未执行。不是新部署方法，不授权key抽取、模型加载或GPU。保留R23负结果，不并行其他机制。

## 为什么这是不同机制，以及目前不能声称什么

R20/R23构造的新方向仍完全来自模型自身分布；R22只观测自身轨迹。经验记忆则将固定25demo中已经观察到的后继token作为value，以当前prefix的表征检索这些value，提供了可通过value置乱反证的内容来源。它不是用当前待评分答案token给方向，也不是训练controller；但25demo可能只提供常见词/格式，几乎没有新的事实证据。

不能仅凭“大模型kNN-LM有效”就假定25个示例足够；也不能把已有25demo重复利用误判为外部知识泄漏。这些示例已在原prompt内。本文先检查此有限value源是否有足够的、区别于通用词复制的潜在纠错内容，再决定是否值得花GPU抽key。

## Phase A：全576、固定token化与值集合

仅用原固定25demo文本、原pinned tokenizer、原discovery384/selection192完整候选及原R6开发记录。CPU-only，CUDA_VISIBLE_DEVICES空，不load_model、不forward、不读hidden缓存。全部输入/source hash在真实audit前冻结。

按结构字段复原共享25demo前缀，逐字节与原prompt前缀一致；定位每条demo的answer字符范围。使用原encode_pair对应token流与offset精确对齐；value只取**token字符区间完全落在该demo答案字段内**的token，避免混入下一question/分隔符。若首个answer token横跨leading space，定义答案字段起点包含原'Answer:'之后的那一个分隔空格，终点不包括随后换行；规则事前固定，边界token数量另报。不改变原token化。记录每个value token ID、demo ID、源token位置及下一阶段对应key预测位置i−1；token0无前驱时停止。

S_all为全部合法value ID集合，df(v)为含该value的demo数。预定义内容子集S_specific：df(v)<=2，且tokenizer.decode([v])去掉首尾空白后至少包含2个Unicode字母或数字。该规则仅用于Phase A内容支持评估，不用于未来检索词表过滤或value重加权。报告规则排除了哪些token；不能看到支持率后改df或字符条件。

对原576每候选，使用其实际原target token IDs，构造集合T_a。这些target ID仅用于离线支持诊断，绝不进入未来查询/系数计算。主输出先不读gold，保存每候选T_a∩S_all、T_a∩S_specific及相对R6赢家b的差集(T_a\T_b)∩S_specific。

该差集的含义很有限：记忆中有一个较少见答案token可区别候选a与R6答案。不是语义支持、不是序列准确率上界、不是可达性证明；全V归一化还会影响不在S中的目标token，故无差集不能证明所有记忆修复不可能。

## 固定admission及结果边界

全部576无标签支持表完成并校验后，单独评估脚本加载gold，统计R6错误题中gold相对R6的specific差集非空数。包括全216错误题，不只看R21的54个成功机会。分别报告discovery/selection/合并、所有错误候选的对应覆盖作为背景，以及只用S_all的通用token覆盖。

admission须同时满足：

- 合并R6错误中specific差集非空比例>=20%（若216，至少44题）；
- discovery与selection各自>=10%；
- S_specific包含至少10种token且来自至少5条demo；
- 所有原prompt/token/candidate顺序与R6 baseline来源审计通过。

阈值是低成本资源分配规则，不是显著性检验；数据重复使用，不称独立验证。不做p值，不比较挑阈值，不按支持题筛下一pilot。低于门槛关闭“仅25demo答案value”的具体路线，理由是限定记忆的预设内容支持不足，不声称外部大datastore或所有kNN都不可能。通过也不能说新方法有效，只意味着允许进一步审查一次固定实效pilot。

合成检查：重复value只增加出现数不改变set，df跨demo计数正确，空白/标点/单字母排除规则，跨answer边界token拒绝，候选顺序置换与target ID映射一致，R6赢家自身差集恒空。CPU墙钟上限10分钟，超时/incomplete不输出部分admission，不改cohort补跑。

## 如果admission通过：如何真正接到prefix修复（只作预先限定，不在本轮执行）

接法可以具体成立而不再做无尽proxy：对合法value位置，在原共享25demo前缀的一次native4 forward中取其前驱位置h4作为key，value就是该已知过去token；当前prefix query取原h4。只检索合法demo key，不纳入任何开发题/当前答案未来token。原候选prompt和token评分路径不变，memory分布是完整词表上的稀疏概率分布。

为避免后验调参，若进入后续设计，仅考虑**单位范数h4的单最近邻**，平方Euclidean距离、tie按源token位置；检索分布r_X为该邻居value的one-hot。zero/nonfinite query或key规则须在后续实现合同冻结，不能通过加epsilon任意改排名。单邻居是最小、无k/温度搜索的实现，不宣称优于论文默认。

唯一预期桥接形态是Z_M3 + [r_Q−mean(r_N0,r_N1,r_N2)]，再完整V log_softmax、原字符分母、原晚期项。单位步，不额外减R6。N0/N1/N2仍独立同已消费prefix forward；三参考用于扣除背景复制偏好。该形式仅作为路线限定，未获GPU执行批准，更不能看到Phase A结果后换成任意kernel/k/插值扫描。

这确实输出了当前prefix的经验fullV分布，且不读取当前target。但其检索权重来自hidden相似性，**若仍严格将原合同解释为一切推理权重只能由native fullV而不能由hidden query计算，则这一步需要Root明确确认辅助分布估计范围**。不能把“最后产出fullV”当作自动绕过输入来源限制。用户已允许冻结辅助forward和hidden诊断，并不自动等于所有hidden驱动部署均获准。Phase A不受这个歧义影响。

真实实效pilot须预先包括：同keys固定value置乱（保留频率）、频率分布、memory-only、q-only、旧M3/M2/R6及配对参考稳定控制；一次完整cohort、保留旧收益及主胜全部必要控制。不能在这里提前挑控制/超参。该对value含义的干预比R22相关性proxy更接近直接检验内容是否改善原评分。具体控制公式、zero/unmatched、pilot门槛及预算由Phase A通过后完整冻结，再由Astra/Root审查，当前不得执行。

## 成本与资源结论

Phase A GPU=0，CPU只有tokenizer与已有文本/score读入；原余额2958.2004723269492秒不变。若进入下一阶段，demo key只需一次shared-prefix native4，不需要每题重抽；576个Q h4已缓存，可CPU检索。没有保存的参考h4仍需fresh N forward，不能从LL标量恢复；完整ARC还需主模型，不能用“key只抽一次”掩盖约数千秒的完整链成本。

因此目前只授权准备/审查Phase A CPU程序；不请求泛泛预算、不提取key、不运行新数据集。若真实后续pilot正面，再按实际拆分成本审查完整链。该路线是否值得继续由预设支持审计回答，不因剩余预算或用户期望而硬造成功。

### 补充核验与解释边界

Root现已报告R23独立复算通过。P_latent有效支持中位数Q≈4010.96、N0≈6117.46、N1≈5182.56、N2≈7782.64；这不是“全部趋于uniform”。主与E候选score最大差约0.00020105而预测全同，仅作描述，不据此调整SLED支持或强度。

R24缓存约束已查源码：ouro_cached_scoring._run在191行只保留selected位置，score_uncached在309行将位置限定为候选预测区间，268行再把选中states存CPU。原576缓存不能提供demo key轨迹，统计量也不能重建它们；未来key抽取必须fresh一次原forward。Phase A不会装载或假复原这些key。

再次强调：gold token不在S_all或S_specific中不证明不可纠错；fullV normalizer仍会改变非support token的log概率。Phase A门槛只能表达预设“有区分性的经验内容机会”是否足够，不能作严格数学不可达性结论。

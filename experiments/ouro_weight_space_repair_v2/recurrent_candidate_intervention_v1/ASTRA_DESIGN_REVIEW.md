# Experiment B：最小完整试验与语义边界（采集前提案）

Astra，2026-11-05；ccf-experiment-designer / design / raw protocol。已全读用户Experiment B附件，核对A的历史行为与parity-only终态。本文件是待Root收口的设计建议，不是GPU授权。Sol实现，Root冻结与单独START；A/R11不修改。

## 1. 目标与证据

A的435救/48伤与47个“排列下两端一致错→一致对”支持继续问内容决策形成，但没有定位具体循环。B必须读取真实native K1..4，比较输入干预，不能用新calibration或控制器代替实验。H1/H2/H3可能同时成立；不能预设唯一胜者。

固定24 pilot是允许的最小完整覆盖；其选择应来自SHA256(seed2021, canonical question ID)排序的训练/dev64前24，与A的numpy抽样不同。精确字符串、编码、同hash时ID排序在构造前冻结。先生成全部64，再声明本轮24；不得看干预后改子集。保存私有含gold constructor manifest，导出不含gold的模型任务计划，二者以ID/hash关联。

## 2. 完整24主矩阵

P0原顺序；P1固定shift1循环；P2正确位置与hash排序的一个wrong位置交换；P3固定非cyclic排列。P3若与P2重复，按预定lexicographic非cyclic排列列表取首个未覆盖者，不按效果挑。3选项同样先列完整排列再检查资格。实际相同input/target/positions允许去重，逻辑条件记录仍保留。P2禁止一律把正确项移到A；构造后报告正确项新位置分布，不能为了平衡结果重抽。

B3固定同一个wrong槽位（直接用P2的hash partner，降低人为自由度）：全部24一项替换为固定“Not a valid answer”；secondary仅固定SHA序列前4题尝试事前独立审核的错误、完整prompt token-length-matched自然候选。此范围已经Root在效果出现前接受为预算有界子矩阵；其他20为未尝试而非语义拒绝。正确项与其他选项原字节不动。审核不通过仍保留该题的所有其他条件，secondary计NA，报告全部拒绝原因与覆盖；不得换题。

B4当前question字段固定“N/A”，保留所有原选项和25示例。为使H1不止停在“文本有影响”，对同一B3替换也在N/A下执行：N/P0、N/placeholder、N/matched。这是question-presence × distractor-content的配对交互，不是新method候选。

最终24主矩阵为每题7个letter-context（4个position、Q/placeholder、N/original、N/placeholder），前4额外Q/N matched共8个context；其余为完整B5/B6子矩阵。CPU实际总数为196逻辑groups、784候选consumers、218个unique body、872次native head、1448个head prediction positions。P1使用一个固定循环是本轮有界位置干预，不能称覆盖全部循环/所有排列。历史全部循环K1/K4可单列背景，不填补这些条件的K2/K3。

## 3. B5/B6有界但完整

B5预定24序列前4题，不按答案长度/模型效果筛。保留相同25demo IDs/order/题目/正确内容，只将demo输出letter改为完整正确答案。每题完整2×2：letter-demo/letter-target、letter-demo/fullanswer-target、fullanswer-demo/letter-target、fullanswer-demo/fullanswer-target。P0可复用，其他独立encode全部合法候选。

保存每候选完整FP32 token-LP sum及原target字符分母，报告sum排序与字符归一排序，事前指定各自用途，不择优。跨target格式不直接比较raw LL或margin幅度；比较各格式内K1→K4正确性/排名变化，并承认长度和分布失配。这个矩阵没有独立的“无joint choices”因子，因此不能单独识别joint availability的净因果作用。

B6也在固定前4题执行两类：固定短词列表的显式内容→字母查找、直接明确指定合法字母的复制指令。答案由预定hash/循环规则产生，与ARC gold无关；输入中出现任务要求的目标词/字母是人工任务定义，不是泄漏ARC答案。保持25demo背景且报告格式失配。目标位置不能全部A。它们若有late gains，支持一般输出/绑定难度参与，但不量化ARC收益有多少由H3解释。若无gain且K1已近天花板，不可据此反驳H3。

## 4. 语义、形状与数值边界

占位符会同时改变词义、词频、语法、token数及后续位置，不能称纯语义干预。自然错误替代必须由独立审查逐题核验：原正确项仍明确正确，替代明确错误，不引入第二正确、关系歧义、与原wrong语义相同或只改格式；记录理由。审核只看原题/候选/替代，不看模型输出。

“长度匹配”必须在实际完整prompt编码后检验：总token长度一致，替换span长度一致，之后的原后缀token IDs及位置一致，所有target encode边界一致。只比较孤立替代字符串token数不够。相同shape排除了R26已知长度位敏感的一类混淆，不能消除词频/可读性/attention变化，也不能将匹配组与placeholder效果相减称为精确长度效应。

固定N/A与真实question长度通常不同，Q×B3交互仍有question长度/结构混淆。每个Q内部的matched pair、每个N内部的matched pair应各自same-shape；该交互可称“问题存在条件相关的响应”，不能声称纯语义因果效应。无需为追求完全纯净增设无穷负控。

## 5. 统计与相斥观察

原始定义I_k(i,j)=S_original,k(i)−S_modified,k(i)，仅对未改变选项i报告；ΔI_k=I_(k+1)−I_k，全相邻阶段均列，禁止选最大阶段作唯一结果。S用完整字母target LL（另可存target原始logit供归因），不把候选LL当fullV logits。

同时报告未改选项之间的pairwise margin响应：J_k(i,l;j)=I_k(i,j)−I_k(l,j)。它消除同一prompt内共享的fullV log-normalizer项，比I非零更接近相对内容选择改变。Q/N交互为J_Q−J_N，及其相邻loop差；所有未改对、每题先平均再跨题，不让选项更多的题加权更多。方向性另列gold对所有未改wrong的margin变化，不因gold结果重定符号。所有效应可为负；非零不等于有益。

H1支持需要：相对候选影响随loop改变，且其Q/N差异、gold-relative方向与实际正确性变化共同提供内容相关性；若只有共同LL平移、同样影响N/A或损害正确答案，则不支持“有益比较”。H2支持看映回content的equivariance变化，而非一直选同一显示字母。H3看两种简单任务及B5格式交互，不能以一个失配任务失败认定早期隐藏无知识。

问题级配对bootstrap，所有该题conditions共同重采样。B5/B6仅4题，列个体及范围，不靠渐近显著性给强结论。secondary matched资格不是效果筛，但其子样本仍有选择偏差，不能外推全部24。

## 6. 可完成性与停止

原20账20889.326696651056/21600，余710.6733033489436秒。不得复用A的500上限当新授权，也不得重置六小时。最终CPU任务图为218 body，24 P0在前用于历史端点parity，余194为干预；无需另加重复parity。每个fullanswer continuation可能需独立forward；不能因多token而拿letter缓存冒充。

已存历史K1/K4用于端点parity与描述性复用，但没有对应K2/K3时不能省该条件depth4。不要把CPU LMhead重建叫免费前向。唯一同输入去重必须含input IDs、positions、actual depth和head shape；完整target sum/reduction保持。

64完整矩阵至少是24主矩阵的约2.67倍，且另有有界B5/B6；当前余额下应先按实际任务数与A warm component timings预注册24，而非先尝试64直到超时。Root应在新任务图完成后确定inclusive load/hash/head/body/IO预算与余量，检查共享进程，采用单次完整24入场规则。估计不是物理下界也非完成保证。若不能负担整个已定24，给具体资源要求，不删除N/A配对或B5/B6来冒称完整B。

若完整效果不支持任何假设，应停止，不制造Experiment C。即使阳性，本轮只能提出候选循环区间与token位置类别的后续因果问题；不能单凭input干预定位内部source/head角色。

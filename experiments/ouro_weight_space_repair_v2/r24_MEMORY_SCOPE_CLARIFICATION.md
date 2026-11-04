# R24 memory scope澄清：事前固定fullV Hellinger单邻居

状态：Phase A真实支持结果之前的范围与算法限定；尚未计算新支持、抽取key、计算query或运行GPU。本文件独立新增，不修改任何R23冻结文件，也不修改Sol已读取的R24设计。

## 决定

**接受作为严格兼容的唯一未来检索定义，但不据此批准模型实验或声称有效。** 用native4完整词表分布而非h4直接计算检索权重，消除了原“权重只能来自当前无标签fullV”文字上的实质歧义。此选择是事前scope约束，不是比较距离后选赢家。若这一定义无效，不自动回到hidden距离、cosine、其他温度/k或降维检索。

本澄清优先替代r24_ASTRA_MEMORY_SUPPORT_DESIGN.md中未来Phase B的“单位h4欧氏单邻居”设想。Phase A的合法value定义、specific集合、覆盖阈值和CPU-only权限不变。

## 唯一检索定义

每个合法demo value位置i，固定value v_i为原demo答案已观察到的token；key为预测该token之前的native4全词表概率p_i=softmax(z4_i)，sqrt向量a_i(v)=sqrt(p_i(v))。当前主/参考已消费prefix的query为相同native4全词表概率p_X，a_X=sqrt(p_X)。h4仅经原LM head得到z4，不直接参与相似度、加权或方向选择。

选i*=argmin_i Σ_v(a_X(v)−a_i(v))²，完全相等按原共享prefix中的源token位置升序。等价于按Bhattacharyya系数Σ_v sqrt(p_X(v)p_i(v))最大取邻居；因各概率归一，平方L2是常规定义Hellinger平方的2倍，不影响argmin。r_X为value v_i*的one-hot完整词表分布。

不学习距离、不调温度、不择k、不作topK词表裁剪、PCA、ANN或仅比较候选token；单邻居固定，不按余量切换查询策略。距离只依赖完整分布和合法历史memory；当前target/gold/未消费答案不得进入检索。候选列表增删不改变给定prompt与prefix的检索结果。

实际数值协议须在未来key提取前冻结：原BF16 head→FP32 logits，再FP64稳定softmax/sqrt及距离累加；分块只改变调度，不改变全词表支持。需独立NumPy/Torch误差与邻居ID一致性核验；近邻距离几乎相等时不能事后加容差挑有利value。当前仅事前限定数学距离，不提前宣称跨设备近tie一定exact。

## 科学合理性与限制

合法性不等于有用。此距离度量的是模型对“下一个token”的预测分布相似，而不是问题语义或事实支持相似。两个无关上下文若都高概率预测the、a、数词或标点，可能非常近；真正有关的两道题若答案词不同，反而很远。top概率词会影响距离，fullV支持不会自动消除这个问题。

此外，demo key往往处于短答案内，query可能来自错误候选续写的不同语法阶段；single-NN会把分布相近的一条已观察value作成one-hot，方差可能很大。该方法把模型相似性与独立观测value结合起来，确有不同于R23自构目标的信息链，但不能保证它新增事实能力，也不能把它称作原kNN-LM原设置的忠实复现。

因此Phase A覆盖只是资源admission；即使通过，未来仍必须用**相同keys、固定value置乱**保留几何与词频后检验内容是否带来收益。频率、memory-only、q-only及原M3/R6控制不可省，不能因Hellinger有标准名称而豁免。不存在凭fullV概率天然“更语义正确”的保证。

## Prefix、参考与基线合同

keys必须来自原共享25demo前缀的fresh一次原native4 forward，不能从现有target-position缓存或标量统计伪造。未来须验证共享prefix因果位置与独立原prompt读出一致；key的token v_i在其自身key预测位置尚未出现，但在当前question查询时已属于合法过去上下文，因此不是当前答案泄漏。

N0/N1/N2分别按原encode_pair及相同已消费候选前缀取得query分布，各自独立检索同一固定memory。主bridge仍限定为Z_M3 + r_Q − (r_N0+r_N1+r_N2)/3，单位一步，完整V重新log_softmax、原候选全文、字符分母及Q4−Q5晚期项不改；不再次减R6。

这有参考排列不变性及共同检索value抵消性质，但单邻居对微小分布变化可能跳变，所以不宣称参考稳定已成立。稳定性必须在未来固定控制与leave-reference-out检验中通过；失败不能换距离或扩大k。Q与N具有相同概率时应返回同neighbor/value，bridge新项严格0；candidate-extension与target-free签名须合成检查。

## 成本：不能把“只抽一次key”误写成便宜的完整链

现有576缓存只有h4预测位置，**没有query fullV概率**。每个query仍需一次原LM head/softmax，虽无需重跑主干；N1/N2等参考未存状态，仍需fresh forward。key前缀抽取一次节约的是memory构建，不是这些query/ref成本。

令M为合法value位置数、V为词表大小、T为待查询prefix数。FP64 sqrt-key存储约8MV字节；精确全词表单邻居约O(TMV)乘加，另有O(THV)的LM-head成本（H=hidden维度）。可以分块流式处理，不必保存全部T×V，但不能用压缩/ANN改变检索定义。M由未执行的Phase A得到，目前不报虚构的耗时或支持数量。

因此该选择比h4检索更贵；它换取的是严格遵守fullV信息来源，不是速度。未来若CPU全V投影过慢，需单独预算GPU head计算且计入原上限；不得把GPU head-only记作0GPU。完整ARC还需每题主forward，三参考成本也仍在；余2958.2004723269492秒不足以自动承诺全部后续。只有Phase A通过、具体实现与一次实效pilot设计获得审核，才会讨论该次预算；正面pilot后才有完整链预算review。

## 当前允许动作

Sol仅实现/合成/静态审查Phase A支持审计，并把本澄清加入事前公开输入锁。现在不抽key、不计算真实query距离、不装载模型、不运行新GPU、不读取heldout/test逐题。Phase A失败即按原门槛关闭这一路线，不以修改Hellinger、specific规则或cohort救回；通过只开启下一设计审查，不自动执行Phase B。

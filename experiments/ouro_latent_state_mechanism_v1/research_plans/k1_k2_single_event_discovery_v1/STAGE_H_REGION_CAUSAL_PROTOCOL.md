# L11接收区域与后续用途：事前协议

决策：优先于纯box/object/endpoint角图。当前allH结果混合问句关系材料、选项证据绑定与末位置直接读取；本协议分开实际计算用途，不要求纯地址变量。未运行新GPU或修改旧冻结，目标active。

## 完整分区与算子
H严格划为Q/O/R：Q是Question文本到首选项标签之前（含换行）；O从首选项标签至最终read token之前，包含选项和Answer提示中除末token以外的格式；R是既定q['R']一个token。用offsets验证无跨token边界，三个span互斥且并集恰H。无法对齐即技术失败，禁止猜边界。

bit1=Q、bit2=O、bit4=R，mask0..7。只改K2/L11，沿用V2R1理想paired-message加实际SDPA基底，完整G records/allheads。只回写开放行，其他立即输出bitwise不变；不声称原生KVcounter，不回填V1失败。

random保持原unit/arm/layer种子、完整H形状方向生成、冻结matcher。target来自同mask semantic实际BF16行dose，关闭行target0且不写。不得切短张量后抽随机数。mask7全logits与既有L11单层sem/random bitwise桥接；mask0与native exact。单层输入自然相同，开放行即时semantic写入应等于mask7对应行。保留allunmatched，QC沿用V2R1。

## 开发选择
旧16groups全512inputs，复用native/mask0和L11mask7，新增mask1..6各semantic/random=6144forward。无token/head/层/系数搜索。

Y=opposite rawlogit−mean(two decoys)。四graph strata j的E_m,j=mean(Y_sem−Y_random)。资格：每层E_m,j>0.1且>=0.8E_7,j；八firsthop×endpoint×selector点估计全正；doseQC通过。基准是L11单层，不是旧八层window。80%是指定干预效应保留阈值，不是推理贡献。

选开放region数最少的合格mask；再按min_j(E_m,j/E_7,j)最大；精确并列取整数mask较小者。mask7合法；只联合合格则保留联合，不强找单点，也不声称各区严格必要。无合格mask停止。全部结果公开，开发选择不当独立确认。

## 标签因果约束
对固定f/e/selector/order、仅label factor不同的两world，Q的两轮各层状态/KV及L11即时写入应bitwise一致。实际核查而非只看文本。Q在未来选项前，若Q-only写入改变两label编码下对应终点，支持pre-label材料被后续用于绑定；Q仍可能已计算终点，不能宣称尚未组合。O可含标签/终点，其作用可能是证据绑定；区域效应不证明具体算法。

## 新64确认
新64basegroups、seed2026111743、全32factorial，CPU去重所有旧R/C以及新集自身。现有已曝光64不是新确认。未运行任何模型的iso候选新集可在重新冻结本协议manifest后使用；不得在其曝光后称盲确认。within-family新组合不等于预训练未见。

每input native、selected sem/random、mask7 sem/random五条件=10240forward；selected7则去重三条件=6144。新确认不换mask。

固定18标量族：四E_selected、四(E_selected−0.8E_full)、四E_full、pooled Y_selected_sem−Y_native、pooled G_selected_sem−G_native、nativeK2accuracy、nativeK2−K1accuracy、两label-factor分层pooled E_selected。G=opposite−selected。100000basegroup bootstrap，seed2026111745，Bonferroni18双侧同时95%区间，所有32输入配对保留。

通过条件：四E_selected和四E_full下界>0.1；四保留差下界>=0；Y/G相对native下界>0.1；两label层E下界>0.1；八selector点方向正；nativeK2 observed accuracy>=.70且gain区间下界>0；全部数值/doseQC通过。accuracy区间公开但不额外要求其下界>=.70。selected7时保留差是0.2E_full构造依赖，不称第二份证据。失败不得换确认上更好的mask。

## 后续跨token问题：不自动追加
若selected不含R，L11的R没有直接写入，R后续变化只能经后续跨位置路径；仍须检验实际路径。下一独立版本可将选中U的actual semantic KV向所有U外下游读者替换为native参考。只关U→R会遗漏U→其他token→R，不够。U=Q/O/Q+O时是连续块，关闭其结束后的全部读者；U自身保持semantic轨迹KV需实际断言。所有其他source/receiver不编辑。

全关闭恢复native是封闭构造/数值校验，不是科学发现。科学对比可固定L12–16、L17–24两个窗口局部恢复和相反背景阻断。若selected含R，已有直接写入，全关闭不应恢复native；不得套用错误不变量。若只联合有效，保留联合结论，不把效应归给Q/O。R-only优先报告G→R直接读取，不虚构query中介。

本文件尚不冻结后续路径的全部算子、效应门槛与统计族，故不能直接运行其科学确认。默认先完成区域确认；看完这64再制定路径研究，则这64已曝光，必须另用fresh确认。若拟共享holdout，必须在任何该holdout结果出现前完成全部路径协议与联合多重比较冻结，不能事后补写。

## 信息增量与停止
旧Temporal B是全K2的S→D入口map/query/read；新实验固定L11的G→H接收用途，区分问句、选项、最终读出。标签因果约束和未来路径关闭可推进pre-label材料到后续binding的解释，不预设独立低维变量。

仍未解决第一跳如何写入G、G局部是否组合、自然纠错覆盖或唯一算法。目标active。任何失败均不能说没有机制，也不能无限细分token/head挽救。先完成既有raw/CPU审计，再新建本阶段source/input/统计manifest冻结及技术桥接；旧文件不改。

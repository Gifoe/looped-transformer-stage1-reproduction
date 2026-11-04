# R27 proposal：最小关联内容来源探针，而非 function vector 部署

状态：**proposal only，未冻结、未授权、未生成真实模型特征**。优先完成R26源码审核和基础验收。本提案不能在R26之前运行synthetic model，不恢复R20/R23/R25，不增加或重置6小时cap。它是功能定位的有限前提检验，不是新增benchmark验证或学习controller。

## 原论文支持什么，不支持什么

[Todd等官方project](https://functions.baulab.info/)先跨任务以causal mediation识别attention heads，再平均task-conditioned head outputs形成function vector。这与任意抽一个hidden向量或直接移植现成head名单不同。[官方intervention_utils.py](https://github.com/ericwtodd/function_vectors/blob/main/src/utils/intervention_utils.py)确实按(layer,head,token)将预计算平均head输出写入再做output projection；它不是Ouro上已验证的功能定位或天然label-free正确性控制。

[Hendel等论文页面](https://aclanthology.org/2023.findings-emnlp.624/)提出ICL示例可压缩成task vector；这里仅核了其作者/摘要，不将其未复核的具体实验细节当实施依据。[Olsson等原文](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)对小attention-only模型给出强机制证据，对大MLP模型的推广主要依赖较弱证据，并讨论attention/MLP交互使消融解释复杂。因此不能直接说Ouro存在同样的head、功能向量或事实修复通路。

真正值得检验的最小问题是：在Ouro原native4的**最后一次recurrent loop**，原位的某组source投影是否能携带“哪个nonce对应哪个label”的配对内容，且不是仅带来label词频/一般扰动。这是task-specific late-loop source transport，不是寻找一个最优layer/head、不宣称新原理。

## 信息合同：仅诊断例外，不是ARC推理机制

原25demo背景字节保持；新增当前question字段完全由人工规则生成，不读任何当前ARC gold、题目事实或test文本。背景中已有固定demo答案不是新增标签数据。所有synthetic对应关系都在forward前由固定规则定义，用于评价功能，绝不建成ARC答案库或拟合gate。

干预搬运的是donor中间投影，故本探针**本身不是满足“部署系数只取当前fullV”的ARC算法**。其作用是诊断一个因果信息路径，类似有规则真值的功能测试。若当前合同把这种辅助诊断也禁止，则直接No-Go，不以最后输出fullV绕过。即使它成功，未来ARC机制仍须另给只依赖当前prefix fullV的合法系数、无需synthetic答案的内容来源、R6旧收益保护与反证；不能把patch向量直接写进ARC。

## CPU阶段：完全确定的两张表、四个配对query

固定label文本池为A/B/C/D，不因模型偏好更换。用seed2021生成两组各4个互异nonce字符串（固定长度ASCII字母数字，生成器源码/具体文本先冻结）。每张表包含4条 `nonce -> label`，每个label恰好出现一次。查询一个表中已出现的nonce，答案由表唯一确定，不涉及自然知识。

使用原prompt template的当前question字段装入表、明确的lookup指令与query，末尾仍原 `Answer:`。禁止把synthetic答案拼到query前向。两个版本A/B的nonce、行顺序、query完全相同，只交换一对label：表1交换A/B，表2交换C/D。每张表分别查询被交换的两个nonce，共4个A/B配对实例。这样两侧label直方图完全相同，四个label均成为查询答案；不根据输出重新配对。

CPU tokenizer必须证明：每个表label位置和每个拟评分答案（含原边界空格处理）均为恰好单token，A/B长度、token位置完全相同，除交换的label位置外IDs完全相同；来源组只包含四个label token位置，query与旧25demo均在组外。若固定A/B/C/D或模板无法满足，记录proposal不具token准入条件，先回Root设计审查，不自动筛新label/template直到通过。

本阶段输出只含规则、文本、位置、hash与预期label，不运行模型。合成检查可先用人工tensor验证patch的mask/loop计数/identity/permutation，不装模型。

## Phase A：先证明模型使用映射，而非固定label偏好

每个配对实例对A、B各一次原native4 forward，共8次。原uncached/no外部mask/no改变backend/BF16，固定25demo；仅读取最终query位置完整V logits。评价只在事前4个合法单token label之间比较，同时保存全V读出hash和label概率质量，不据质量另设后验阈值。

预提准入：全部4配对中，A、B各自正确规则label均严格高于另3个label。没有平局宽容；不以fullV top1必须是label作额外门槛，因本测试是显式label集合任务。该8/8规则是极小功能存在性screen，不是总体准确率估计。失败即不进行Phase B，不扩大样本/调prompt/挑成功配对；也不声称Ouro不能ICL。

允许在这8次中以不改变输出的capture记录指定source的native-loop4、全physical-layer K/V投影供后续使用；capture本身必须CPU合成证明不改tensor，并以Phase B identity检验运行时exact。仍未按head或layer评价效果，不训练排序。

## Phase B：固定最后loop的source K/V重放，不把输入完整还原

每个B作为recipient，前3个loop完全原样。只在第4个loop的全部24 physical layers、固定4个source label位置，分别执行以下6次（每次只一种，不组合）:

1. B自身的V donor按原位置重放：identity，最终fullV必须exact等于原B。
2. A的V donor按相同source位置重放：clean V。
3. A的V donor按固定位置置换重放：shuffled V。
4. A的K donor按相同source位置重放：clean K。
5. A的K donor按相同固定位置置换重放：shuffled K。
6. B自身的K donor按原位置重放：K identity，最终fullV必须exact等于原B。

共24次，连Phase A最低32次原native4 forward，不进行额外逐head/逐layer/逐loop扫描。置换固定为四位置循环移位1位，不能重抽；每层/每loop保存的donor完整向量多重集合精确保留，但其与source nonce位置的关联被破坏。K在k_proj输出、RoPE之前重放，位置仍使用recipient原RoPE；V在v_proj输出重放。不得在identity之外偷换donor、按效果选K或V部署。K/V均有自己的identity检查，不能为了成本删掉；两条均完整报告。

最后loop固定是问题定义，不是认为它最优：检验的是在native4末段是否仍存在内容特异的source通路。patch不改token embeddings，不改前三loop，不写query residual，不直接替换attention输出或LM head，不同时还原K和V，因此不是完整清洁input重放的同义反复。另一方面，后续physical layers的K/Q/residual可能受前面V干预间接影响，所以只能称对指定V投影的干预，不声称所有其他变量始终固定。

它仍是广泛source组干预，而非compact circuit定位。四个source位置全部处理不能区分具体head，也不能证明哪些单独位置必要；不据成功给出head名单。固定置乱保留向量多重集合，不匹配所有下游激活分布，故其失败只支持位置—内容配对重要，不能排除所有off-manifold解释。

## 事前效果量、负控与结论边界

对每实例，令m_X=log p_X(y_A)−log p_X(y_B)，其中y_A/y_B是规则给定的两个不同label。这是事后功能评估，不回传、不参与干预强度。Phase A必须m_A>0、m_B<0。

分别报告V和K的transport Δ=m_clean−m_B、content specificity=m_clean−m_shuffled，及clean是否使m_clean>0。不取两条路径的最大值、不把较好者当候选。

拟议必要结论规则：只有身份重放全部exact、所有4实例V的Δ>0且specificity>0，并且至少3/4的clean V跨过0，才记“在这个固定末loop/source组上有有限内容对应关系传输证据”。K以完全相同规则独立报告，不作为V失败的替代成功入口。若K也通过，只说明不止V有此效应，不能声称V特有。若只有K通过，V假设不通过；可保留其描述但不能转成另一个ARC方案。

四实例很少，门槛不是显著性或泛化检验；此处正面仅允许讨论功能定位前提是否成立。即使全部通过，也没有任何ARC纠错、R6保护、无标签可靠性门控或参考稳定证据。**v2因果表述更正：同一张表的source位置在query之前，source donor应与后续query无关；要求同一组query-agnostic donor在不同query下恢复不同的label。** 原稿“donor必须随query对应关系变化”的表述错误，撤销。保频A/B和位置置乱分别排除单纯label直方图与不看source对应关系的解释，但不保证完全因果识别。

## 资源和当前是否值得做

32次长背景native4 forward，加一次模型加载、capture/identity/hash；现有每题/每候选成本只能给量级，不能声称已profile。建议若此proposal获认可，正式锁最多一次180秒、soft165/kill179/reap1、所有wall计费，无额外warmup/retry；若静态资源估计表明32次不可能容纳，就不启动，而不是删负控。R26实际计费后重新核余额，不能沿用未扣R26的数值。这个数字是待审提案，不是授权或必定够用的承诺。

我的判断：**可先做CPU结构可行性准备，暂不建议直接批准模型job**。必须先有R26结论、CPU同shape/单token准入、hook在Ouro实际重复调用上的身份/cleanup合同，以及Root确认“仅功能诊断”范围。缺任意一项就不执行。它比再造fullV相关性向量有更明确的干预含义，但距离当前ARC目标仍有实质缺口；若Root不接受这段信息价值相对成本，关闭proposal而不包装成新算法。

本文件未冻结；后续源码审查优先R26。没有synthetic模型执行、ARC新score、GPU、训练或外部benchmark。

## v2：query-agnostic donor与有界CPU生成澄清

本节修正原proposal的两项范围含混；没有新特征、模型或输入生成结果，尚未冻结。source位于后续query之前，数学因果性要求同一A表的source K/V不随两种query而变，B表亦然。不同长度kernel可能破坏位级相同，不能将其解释成task vector响应了未来query。

因此正式CPU准入再要求每张表的两个query构造出的完整输入token总长度相同、source位置完全相同；A/B之间也同shape。正式Phase A的既有8次capture中，对每张表、每种A/B版本、指定source全部layer/loop4的K与V，核两query下逐bit exact。如果不一致，记为source-query独立性前提未通过，不进行Phase B，不改长度/nonce重跑，也不归因功能变化；是否属长度或其他来源须按实际证据另定，不能仅因已有R26有限验收就跳过。

Phase B每张表统一使用第一query记录的donor，第二query只在exact验证后共享同一缓存；不能分别选择“适合各query”的donor。B identity也用这一经跨queryexact验证的B缓存。这样排除常量输出偏好的检验要求是同一个source donor经不同query访问产生不同label，而不是为不同query输送不同答案向量。此补充不增加32次forward预算。

随机字符串字符等长不代表token等长。一个可供CPU准备审核的**有限且不可按模型输出修改**生成规则是：seed2021，为每张表一次生成32个字符串，每个为固定`zx`前缀加6个由`abcdefghijklmnopqrstuvwxyz0123456789`等概率取出的字符；保留原生成顺序。若64个字符串出现重复则CPU no-admission，不重抽。只用tokenizer在固定lookup后缀中测token长度，取该表池中按索引顺序第一对等token长度字符串作为两个query key；其他两个key取池中未使用的最早两项。第一表query key放第1/2行，第二表放第3/4行，以对应预定A/B与C/D交换。若不存在这对或最终完整prompt验证不满足单tokenlabel、总长度和source位置合同，CPU no-admission，不扩大32池、不迭代改template。所有实际文本/池/选择索引/排除理由必须在任何model调用前公开冻结。

这个规则仅是token结构选择，不使用模型logit/正确率；不能称随机无条件抽样。若Root认为这样的CPU结构筛选改变了其欲检验任务，须先修改proposal而非执行后解释。

Phase A的准确名称为**限定四label子空间的对应关系辨别**，不是unconstrained全V任务完成。保存fullV不意味着fullV top1已正确；label集合总概率质量只描述，不能在结果后追加质量阈值或把集内8/8宣传成任意生成成功。

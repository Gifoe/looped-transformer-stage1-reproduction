# E7 正式执行指令：自然任务重编码、循环纠错与固定预算互补

Scientific Director：Astra。全部数据分析、代码、模型前向和独立复核：当前 Sol。研究 Goal 持续进行；本轮不是新的 latent feature 搜索，也不准入 E6C hidden-state control。原 checkpoint、推理精度、评分主定义与历史记录不改，不读取/新增 ARC Test outcomes。用户没有总时间限制，按完整科学范围执行，不因预期时长缩样本。

## 0. H2 裁决与新问题

Sol 已完成 H2 Train1094+Val299：技术 native endpoint、实际 BF16 residual-add 与独立 FP32 lens 检查通过；全1393题完成，无新 Test。Train prospective Brier output−H2 improvement 为 +0.00011810，95% CI[−0.00127645,+0.00152379]；低层−H2为−0.00050750，CI[−0.00215127,+0.00111208]；shuffle−H2为+0.00127383，CI[−0.00054504,+0.00308251]。五fold三好两坏；固定 prospective actions H2=943/output=948/K4=966。Val冻结后复核各增量区间也跨零。原 G1 **FAIL**，G2不启动，不能放宽门槛或再搜第三批摘要。

下一步检验：自然 K1→K2 纠错是否对相同题意下的选项位置/答案符号重编码稳定？若不稳定，哪种受控重编码改变了纠错、损害与候选竞争？这能区分语义候选区分与协议依赖的一部分替代解释，但不直接定位 attention/MLP writer。所有新版本均为原模型自然前向，不进行内部状态替换。

## 1. 完整样本与预冻结分层

使用全部 Train 非示例1094和Validation299。排除原25个演示题，保持它们仍作为固定25-shot上下文。四选题为主分析：Train1092、Val295。其余Train三选1/五选1，Val三选3/五选1共6题作为明确分开的扩展，不混入四选主效应或伪装成充分统计证据。按此设计应有22,280个题目×variant单元（包含1,393个identity），每单元保存四depth读出；这是预期覆盖，实际新增/复用前向次数另由Sol记录，不能把demo渲染检查计为研究样本。

Sol 已完成结构审计：全部原prompt parse→rebuild→tokenize逐token一致；25 demos均为四选，主选项labels=ABCD；宽引用筛查人工复核未发现选项顺序/字母交叉引用。固定这些审计文件及其hash，保留“词法和人工审查不是形式证明”的边界。不按gold、自然utility组、模型置信度或新outcomes选择样本。新语义问题若出现，作为协议有效性问题记账，不能依据是否有利静默删除题目。

所有拟合/方法选择只可用Train/dev。Val只有在本轮定义/分析/候选已冻结后作已暴露问题上的新干预响应复核。历史ARC全部有暴露，不能称独立盲测。本轮不打开Test来检查该重编码或选择规则。

## 2. 全16格 counterbalance，不抽四个视角

四选主样本每题运行 p,s∈{0,1,2,3} 的完整16格，使用相同自然K1–K4读出；不是只按哈希选择一个非零offset。四格2×2含义是把各因素分为identity/非identity，并在其所有非零offset上先作题内均值。保留全部16格，不能只报告最有利offset。

**P：仅待答题的语义候选内容左循环移动。** 当前显示位置 j 放原语义候选 i=(j+p) mod m。位置的基础label按原位置顺序分配。25 demos的候选内容顺序不变。

**S：全prompt一致的结构化答案符号置换。** 全局σ_s使 A→第s个循环字母，B/C/D同理，ABCD按模4循环，E保持不变。对25 demos和待答题的结构化候选label应用同一σ_s；对每个demo的已知示例答案label应用同一σ_s。题干、科学内容、选项文本、解释和其他字符不做普通字符串替换。当前待答题不出现gold答案，也不借gold决定任何变换。

四选时，原语义候选i的新label索引为(i−p+s) mod4。因此p=s对角线保留待答题每个语义候选的原答案符号，同时改变位置；但demo符号仍随s改变，这个对角线不是“整个prompt只改变位置”的纯效应。必须按此边界解释。

6题扩展仍使用全局ABCD循环/E恒等；P取其全部m个循环位置，S仍取4个值，分别12或20格。三选可能变成BCD等label集合，五选可能变成BCDAE；这是明确的非标准label-set扩展，单独报告。不要为了混入主层而修改全局映射。

每格保存原语义candidate ID、显示位置、当前label、实际候选token ID及其双向映射。原候选身份就是原输入中的序号，与gold无关。验证所有映射为双射、原候选文本完整且唯一对应、demo原正确候选语义不变。数字符号不用：其token长度不同会引入另一因素。也不加入V–Z或新标签搜索。

## 3. 结构/数值技术门槛与实际执行

先固定全部variant manifest、formatter/source/model hashes、原始/生成prompt文本和token IDs、排除/分层规则。Collector输入不含当前题gold；分析标签与semantic映射单独保存。

技术集：按预先固定ID hash从Train四选取2题，另纳入Train唯一三选和唯一五选，仅作扩展实现检查。全部指定格都核验；技术题后续纳入完整cohort，但不算额外独立样本。Sol审阅四套demo符号映射的结构化对应，确认不是改写题意。

- p=s=0必须恢复历史同硬件原input tokens；自然四depth full-vocabulary BF16 endpoint hash逐位一致。
- 所有格保持25-shot题目与内容、BF16/SDPA/eval/no-cache、实际四次body、原评分head shape和候选score实现不变。确切记录各格token长度；不要假设所有重编码token长度完全相等。
- 若用一次四loop body捕获K1–K4读出，沿用已验证的逐depth桥接；新增collector至少对技术集做独立固定depth endpoint核验。隐式default gate不得替代显式native depth。
- 保存每loop原始候选BF16 logits、FP32生产scores、全部top ties、原生预测、full-vocab hashes、实际layer/body/head计数和资源窗口。不要仅存accuracy。
- 分别保留“原生产first-displayed-index tie规则”的预测、映射后的semantic top-set，以及固定原语义序号打破tie的敏感性结果。不能靠换tie优先级消除或制造重编码效果。

全部门槛通过后，Sol直接推进完整Train与Val固定格，负效应不触发提前停止或改offset。先发布实际technical成本和全量估算，再记录实际成本；没有旧30分钟总期限。内存/磁盘压力用流式保存、可校验分片和断点恢复解决，不改变因果操作或缩小样本。复用已完成identity结果可以减少纯重复，但如果缺少本轮必要gate中间量，可明确补采并标注复用/新增范围。不要保存不需要的完整QKV或大激活。

## 4. 主要因果对比与可反驳结论

主终点是**语义候选正确性在K1→K2的题内变化**。每题每格 g(p,s)=I(correct at K2)−I(correct at K1)，先在完整offset格内计算，再对问题等权平均。主三个对比预先固定：

1. P对loop gain的影响：平均所有s后，非零p均值减p=0。
2. S对loop gain的影响：平均所有p后，非零s均值减s=0。
3. 二因素交互：以P/S是否非零的四个题内cell均值，g11−g10−g01+g00。

报告原始accuracy、rescue/damage及风险集，不把净gain当全部变化。各loop的P/S效应、K2→K3/K3→K4、全部candidate gaps（meanwrong、固定identity错候选、各格receiver固定错候选、dynamic margin）、语义top-set一致性和p=s对角线是预定次分析。固定错候选要用原语义身份跨格跟踪，不能把字母相同当候选内容相同。不同评分口径不互换。

Bootstrap以**题为block**重采样，题内全部offset/depth必须一起保留；seed20261113、10,000次。主三对比同时报告普通95%和Bonferroni调整的98.3333% percentile区间；后者用于主family的支持门槛。它们是经验问题重采样，不是训练seed复现或严格保证的全新数据泛化。Train、Val及3/5选扩展不混合。

若至少一个预注册主对比在Train调整区间排除0，并在Val同一对比的95%区间同方向排除0，可称“该循环gain对所测试重编码存在可复核的协议依赖”。同时报告无相关ties的敏感性、offset异质性与反例，不能只呈现该显著对比。若区间宽且跨0，结论是不确定；若所有主对比区间均落在预先定义的±1 percentage point实用等价带内，才支持这些平均gain对该因素较稳定。既非显著也未落入等价带时，不把未拒绝零效应写成不变性证明。

即使存在协议依赖，也不能归因于单一label bias或某个writer：P改变了各选项的因果上下文，S改变了demo与query的符号token。它们是保持任务语义的自然输入干预，不是纯内部模块消融。若K2跨格更一致，仍要区分更早就正确、相同错误的一致和真正纠错，不能把一致性当正确性。

## 5. 唯一预注册的实用候选：两个K2视角

为了让机制判别连接实际目标，现在就在新outcomes之前冻结一个候选，避免结果后挑最佳排列：四选用(identity p=0,s=0,K2)与(p=2,s=0,K2)，各占1/2。将FP32生产candidate log-prob按原语义ID对齐，在FP64分别去共同均值后等权相加；argmax，精确tie按原语义序号。只做这一固定pair，不搜索另120种pair或权重。3/5选扩展用p=floor(m/2)，仍单列。

该候选运行时不见gold，但它是**输入多视角/output-only**策略，不是latent-aware算法。其两个K2都必须实际计算；不能把有后见信息的最好视角或label-oracle称为方法。Full16格的正确答案oracle仅可作离线上界，不能和该固定pair混为一谈。

基线：同题identity K4、原E6预注册primary/Cal，以及仅按既有Train/dev结果、在查看新variant outcomes前固定的强output-only预算frontier对照。保持所有原E6规则身份，不借Test最佳规则更换primary。每个对照的参数、选择依据、预算和source冻结入manifest。

方法推进门槛独立于上面的输入因果主检验：固定pair在Train相对K4和预选强output-only都具有正的配对accuracy增量，两个95%区间下界都>0，才推进实际online成本确认；在此之前其完整Train/Val配对结果仍必须报告，不能只报赢的分层。用固定hash的Train128题做真实两次K2执行的profile，比较实际固定K4和强对照，包含第二份prompt处理、tokenization、body/head及聚合；报告模型计时和端到端计时，不能以“2+2=4”冒充同延迟。

若无准确率增量，则此固定pair方法失败，不再搜视角/权重。若有增量但成本超出匹配预算，只能报告实际accuracy–cost trade-off，不能宣称同时提高效率。只有在真实预算frontier上超过相应强对照，才能形成待独立数据确认的候选；若被对照支配，就停止该方法分支。Val仍是已暴露题的新响应复核，不替代后续独立数据。

无论固定pair成功与否，完整自然重编码实验都可提供机制边界；不能因方法失败抹去真实因果现象，也不能以因果现象代替方法收益。

## 6. 原生halting gate：补既有baseline，禁止变成新特征搜索

Sol只读审计确认checkpoint有gate权重、真实stick-breaking BF16逻辑和原项目baseline代码，但没有证明gate有效。默认threshold1.0是在四body完成后按token CDF选读出，不是真省算；forcedK4显式exit3与默认调用分开。

准许collector在自然前向中顺便保留各loop的实际gate_R logits、BF16 hazard/PDF/CDF及来源hash，不改变模型或native输出。先以技术集核验原项目实现与当前公开forward默认threshold1.0读出；不能仅凭数学等式替代BF16和full-shaped head parity。

在identity Train/dev/Val上评估以下**原有固定baseline**，与E7重编码效应分表：stock default threshold1.0的full-body readout；原项目已预定的CDF阈值{0.25,0.5,0.75,1.0}的R级readout。完整列出4个阈值，不添网格；任何阈值选择只用Train/dev并在Val前冻结。离线四body读出一律计为四body，不报虚构早停效率。不自动评估weighted-exit、训练新gate或把gate塞入H2。

若某个Train/dev预选gate策略在accuracy与估计执行深度上不被强output-only支配，并改善至少一个轴，才用原有真实prefix-stop实现进行技术/online验证实际成本：控制流必须停止后续body，实际layer calls==depth，endpoint与相同native depth精确一致；估计深度不等于实测latency，未经该验证不称真实效率收益。R策略只用于本Joint问题，不冒充FullAnswer逐token默认策略。此项是复核既有baseline，不是对失败H2的第三批任意feature搜索。若gate不具增量，完整记录负结果即可。

## 7. 交付、裁决与持续研究边界

Sol保存 E7_PROTOCOL_FREEZE.json、全部样本/variant/demo映射、semantic审计、技术报告、逐题全格四depth分数/预测、预定对比和interval、固定pair结果、baseline身份、gate来源/parity、实际资源和执行日志。单独列出与primary四选不同的6题扩展及所有tie/错误映射反例。

Astra在收到完整证据后做科学裁决：协议依赖/稳定性/不确定，固定pair可用/不划算/失败，以及gate是否补足既有baseline。没有任何分支自动准入E6C或新的Test/外部大规模实验。若本轮均未给出实用增量，持续Goal仍未完成；下一问题必须由实际未解释现象导出，而非继续枚举feature、offset或控制器。科学上必要的后续完整实验可以更长，不能用时间到点冒充达成目标。

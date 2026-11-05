# R32 H2/H3收口：自然字段交互的最小可执行问题

状态：proposal only，未冻结、无模型/GPU授权。R31已排除当前Fixed/C binary选择救回这批错误，不能再从该family造selector。本文件给出H2的具体有限探针，并明确它为何仍不足以成为ARC修正方法；H3不强造映射。

Root随后明确否决本文件norm-probe的GPU执行，Astra同意。以下模型调用/200秒仅保留为被审查方案的成本说明，不是待自动启动任务；余额844.53秒保持未动。即时可执行内容仅CPU反例，且不声称它解决原目标。

## 可执行的H2问题

在原ARC上下文中，固定25demo的完整question字段和完整answer字段，是否产生**大于同token数任意分组的输出非加性交互**？这检验自然字段的功能分组，而不是仅问单个source是否影响输出。没有gold、正确候选、未来答案token或synthetic答案输送到ARC。它不是又一个先验alpha，不改变A/C结论。

CPU先固定原24 cohort每题第一candidate的第一个预测输入（只有原prompt，尚未消费任何answer token）；按exact IDs去重，并加入原N1/N2的同第一预测输入。预期最多26 contexts，全部保留，不按模型输出筛选。各context在其所有条件内长度/position/backend/mask保持完全相同；不是R29 canonical。N1/N2只作reference描述，**不把不同长度Q与N的差称纯question因果效应**，因为R26已证跨shape位级敏感。

Qdemo集合定义为25个demo的question字段完整文本，Ademo沿用已核143-token完整answer字段。CPU用模板结构和token offsets独立核验两个source集合非空、互斥、全在当前query之前。若边界token跨字段无法完整归属，报告具体反例停止，不静默删边界、不裁切成固定尾部、不增选demo。source question不是当前待答问题。

在每个context以native4为00，新增10/01/11三条件：仅在最后loop4、所有24physical layers的V投影，将Qdemo/Ademo/其并集对应行置零；其他行保持rawexact。前三loop不改，K/RoPE/Q投影及原权重不改，实际输入文本不变。全token fields归零是一个明确的受控路径操作，不声称删除了其所有语义信息（K及剩余通路仍在）。

固定seed2021把Qdemo∪Ademo的source位置重新分成两组R、S，保持|R|=|Qdemo|、|S|=|Ademo|，互斥且并集相同。再做R-only/S-only消融；both条件与11是同一实际操作，共用exact输出，不为null另造方法。单context因此仅**6个不同forward**：native、Q、A、both、R、S。随机partition不可重抽；原25自然字段映射不能按norm或效果调整。

令z为FP32 native fullV readout，其他五个按同head[1,1,H]定义。FP64中计算

`u_field=C_p(z00-zQ-zA+zBoth)`，
`u_random=C_p(z00-zR-zS+zBoth)`，`p=softmax64(z00)`。

只测D_field=||u_field||p²、D_random=||u_random||p²；不减去u、不用它作部署方向，不读target。自然或随机一组空时理论交互为零；相同消融/00身份以及严格线性toy为零。非线性toy可以非零，见下面证据挑战。检查fullV finite、identity/nohook和前三loop、source位置/clock及其他行exact；不得把消融造成的响应称正确内容。

## 固定对照、互斥判断与成本

原24全部Q为主统计；若input去重，在计数时仍报告题级重复依赖，不称24独立样本。任一有效性/identity/finite/完整性失败=invalid，不作机制判断。

预先使用已有64eps形式检验FP64范数可分辨性，并只将严格`D_field>D_random`且差超过所记录数值界计为自然字段excess。一个可审的operational screen为24中至少18题excess；否则not-established。这不是显著性、准确率或泛化门槛；不按N1/N2选择赢家，两个reference全部描述。所有条件完整后统一解封norm，不能挑部分题。

最多26*6=156基础forward。加首context native repeat和identity各1、首N1/N2完整native repeat各1，总160。CPU必须冻结确切context/input/source/null图和检查计数；有重复只能exactinput去重并报告，不换样。原无KV/scorer/invalidate、完整参数前后哈希沿既有合同。

按R28约.5–.65秒/长context forward，模型本体约80–104秒；load/hash和fullV/证据IO至少另留50秒。可讨论一次hard200秒（soft185/kill199/reap1/reserve5），但这只是预算提案，尚未测算具体字段hook开销，也未授权。剩余844.5347464797087不追加，全部真实wall需另记ledger；无独立profile/补跑。这个probe无法替代576+517+299+1172全链，positive也不授权新ARC评分方法。

## 证据挑战：目前不建议仅凭此立刻花GPU

上述探针虽可执行，却仍有明确可识别性缺口：若输出是普通非线性函数f(x+y)，`f(x+y)-f(x)-f(y)+f(0)`可以非零而完全没有question-answer对应机制。例如标量f(s)=s²给出2xy。自然字段excess也可能来自向量和/相对位置/字段聚类差异；保持两组token数的随机partition没有同时匹配这些因素。因此positive最多说明该自然分组具有非加性响应，不能证明“binding nuisance”，更不能确定应扣或增强的符号。

在R27已有功能通路、A/C已有非零响应但无utility的情况下，这个有限excess本身仍不够接近新的纠错证据。**当前选择：先由Sol做CPU结构准入及上述非线性反例验证，Root据此审是否值得正式冻结GPU probe；本文件不推荐自动GPU START。** CPU反例输入固定x=1,y=2、f(s)=s²，identity/linear f(s)=3s+1作zero控制；它能直接否定“非零double difference就证明binding”的推理，不依赖任何ARC labels或模型。可同时构造同总和、相同组基数但不同组和的partition，验证norm差也不识别语义。CPU时间上限30秒，无solver/模型，不修改GPU账本。

若要使H2成为方法而不只是探针，缺的是一个同时控制普通非线性/来源位置而改变对应内容的自然操作，及其target-free价值依据。不能仅用新norm门槛填补，也不能把这个factorial探针的u直接代回旧projector称新算法。

## H3合法映射的具体缺口

### Assignment-preserving/breaking救援审查

三个容易提出的“对应关系控制”仍不足以支持此GPU probe：

1. 同时移动完整demo question和answer块，作为preserving；只移动answer作为breaking。自然两字段长度需分别匹配，CPU尚未证明存在足够这样的块；即使存在，contextual V已编码原位置/上文，preserving也并非单纯语义等价。不能把旧A单边移动结果当新accuracy主法，或只凭两组norm差证明binding。
2. 对最后一层目标attention已可见的source，配对置换post-RoPE K和V可保持加权和，V-only打破对应。这可以作实现负控，但保持条件的预测是attention代数恒等式；breaking改变结果仍不说明哪些内容对ARC正确，不解决R27后的价值缺口。
3. 将原demo问题再作为query，要求输出已知demo答案，配对交换后要求对应答案变化。其读出有预先正确标签，优于norm，但它只证明重复例子的检索/复述；fixed25已包含答案，不能将它称当前ARC未见问题的功能证据。若只挑单token demo，进一步不能证明全部多token source的足够性。

真正需要的是一个自然任务上的assignment-preserving变换，其正确输出关系由规则事前确定，且匹配位置/词面边际的breaking对照破坏该关系。原ARC科学知识问答通常不是任意实体重命名不变：改实体也可能改变外部事实，不能把机械替换冒充语义保持。当前材料没有提供这样的规则及来源位置匹配证明。本轮不凭想象添加一个200秒模型实验；保留这个具体缺口，停止norm-only救援。CPU非线性反例就是对现有识别论证可立即执行的反证，不是方法性能实验。

把全V方向r变成内部残差，例如`W_lm^T r`，数学上可实现、target-free，却不唯一也不保持原模型的隐藏表征尺度；经过RMSNorm/后续attention/MLP后的logit响应不是r。固定norm匹配仍只是选择一个幅度，不能由R27映射传输推出其价值。用伪逆/梯度找使输出等于r的隐藏向量会落回新的度量/优化，且本项目不授权backprop；用learned lens/task-vector训练则越合同。

因此本轮不给任意层/任意系数的activation-addition部署公式，也不以CAA/function-vector新名称装饰它。可保留的可执行前提是先找一个原自然上下文中已验证的donor内容与recipient作用关系；当前25demo不是当前ARC答案的查找表，不能把demo正确答案当recipient正确内容。这个缺口是内容识别，不是只剩844秒造成的技术不便。

结论：H2已有精确自然字段probe与有限成本，H3目前没有合法且有证据的内部内容映射。下一即时行动可以是CPU准入和反例审查；若Root选择进一步模型功能检验，须独立冻结/START，并承认它仍非新纠错方法。没有新性能保证、没有扩预算、没有复活C选择器。

# L11双补偿功能接口协议：冻结候选，执行前签署源码manifest

本文件在任何新C神经能力或iso角图结果前确定规则。不修改DISCOVERY_FREEZE。已实际读取LAYER_SELECTION.json：唯一选择物理L11，四分层最小效应0.136962890625，八selector方向正、doseQC通过；L10虽也合格但不是fallback。既有64确认groups已曝光，不作为本研究独立确认。完整机制目标未达，active。

## 1. 固定问题、数据和来源
仅K2、L11(index10)、两个完整等长G records、全部heads、全部H接收行。主算子为原生full-shaped SDPA混合Q/K/V重算，非V2理想差值。V1排列数值null失败永久保留，本研究不回填其状态。

每个basegroup仍完整32个f,e,s,label,order输入。R=f,e,s；A donor D_A=(1-f,1-e,s)；B donor D_B=(1-f,e,1-s)，label/order不变。二者均保最终终点与标签。每个R另有唯一samebinding C：使用same_binding_cpu/固定模板0/1/3词典，仅改当前S/G措辞，关系图、selector、options、总长度、G边界及H绝对位置不变，不改demonstrations。无额外措辞搜索。

先使用原16开发groups。新确认64groups，seed2026111743，objects仍E–L，既定三模板比例与32factorial不变。生成器须对全部旧R与其C文本及tokenhash去重；若随机候选重复，按预定同一PRNG顺序拒绝并生成下一候选，拒绝记录完整保留。新集生成后，所有新R/C之间也必须唯一。这里只是within-family新组合，不宣称预训练未见或新实体OOD。生成器CPU完成后冻结全部ID和输入hash，再开始确认。

## 2. 单recipient条件、反向与精确去重
来源位q,k,v：0为当前R，1为该family donor。每个R实际运行：
- native R及native C：2次，保存两轮heads，缓存K2/L11 actual Q_H、K_G、V_G。
- A八角除000：7次，来源1=D_A。
- B四角除000：100、010、110，V始终来自R，共3次，来源1=D_B。
- samebinding C八角除000：7次，来源1=C。
总计19次/输入，native000为三个family共同基线。D_A/D_B native由同basegroup的R全集提供，不重复运行。不得将B的110称为完整donor天然角；它保留recipient V。

完整32输入已包含每个R/D对的两个recipient方向，不再人工复制两方向。分析方向预定按recipient f=0/1区分，因为两gauge均翻f；每方向16输入/group。C只作当前R recipient下的控制，不额外研究C recipient；两个f方向各自均有对应C。结论不扩展到C背景。如果将来增加C recipient，必须新版本，不能混入本freeze。

无进一步理论去重：不同source IDs即使数值可能相同也保留；只共享明确同一native运行。旧开发16×32×19=9728forward，新确认64×32×19=38912forward，合计48640科学forward。技术重复独立计账。预算与ETA先实测；不得按结果缩减条件或选样本。

## 3. 主干预与技术门槛
每次只替换L11当前SDPA的H行Q、G列K/V；其他Q/K/V保持recipient实际current值，所有源为对应world的同层同轮自然actual post-RoPE张量。用完整形状原始SDPA调用重算，只把H输出写回；完整o_proj、残差、MLP及后缀自然执行，不clamp。不改input IDs、源G状态或其他位置输出。

每个混合run：48次原生SDPA调用加1次counter SDPA，1个编辑event，|H|接收行回写，allheads；native48次，无编辑。记录实际调用数而非层visit代理。194prefix检查、立即nonH保留、正确source/hash、完整shape/stride/dtype/scale/causal/cache设置均强制断言。缓存必须绑定model/source/input/硬件运行freeze。Q变化的counter全序列结果不得写入非H。

旧开发技术集冻结8world，覆盖两f方向和三模板；native、noop、self-copy必须full49152两轮bitwise一致。全部同源000在同recipient必须精确复现。非自源111不要求bitwise native，它进入科学兼容性检验。

技术保存全部实际Q/K/V、原/counter SDPA_H、投影写入、两轮full logits，独立CPU FP64 fullattention只作oracle诊断，不替换科学主输出。核对掩码、GQA、RoPE位置、scale与直接原生counter实现一致。FP64与BF16无bitwise要求，relativeL2<=1%（分母max(FP64norm,1e-12)）及绝对误差完整报告；失败暂停新版本诊断，不放宽。self/noop不能通过FP64误差界豁免。无joint排列代数null的科学要求，因为本实验不是排列算子。

## 4. 预定语义角图与基本指标
在每个输入当前标签编码下：Y=logit(opposite)-mean(logit(decoy1),logit(decoy2)); G=logit(opposite)-logit(selected)。均取K2原始head logits，不做概率或归一化替代。selected/opposite/decoy均由符号图事前计算。M=logit(selected)-max(other3 logits)，accuracy采用既定首索引ties规则，全部ties保留并单列。

预测0为原终点维持，预测1为向opposite定向改变：
|受限功能模型|A八角|B四角(V=R)|
|box-address|q XOR v|全部0|
|object-address|k XOR v|q XOR k|
|endpoint-address|k XOR v|全部0|

实际表征不必属于任何模型。allH是一整个接口，不推断每行Q都表达同一地址。

令E_X,c,Z=Z(X,c)-Z(native R)，X=A/B/C，Z=Y/G。定义同binding调整效应D_X,c,Z=E_X,c,Z-E_C,c,Z；B使用C中相同q,k,v=0角。所有值在输入内配对，按basegroup、方向f各自先平均，再对groups等权。不得把反向world当独立样本。

## 5. 完整统计族与判定界
开发和确认使用同样规则，各自100000次basegroup bootstrap，固定seed2026111744；保留全部32输入与所有条件配对。确认64groups一次性分析。冻结124个标量的共同族，双侧Bonferroni区间分位点0.025/124与1-0.025/124，familywise95%。这是保守诊断，不通过不换宽松检验。

族构成：
(1) A7+B3非self角 × 两方向f × Y/G × raw E及control-adjusted D：80项。
(2) C7非self角 × 两方向 × Y/G：28项。
(3) A111、B110、C111三种对齐角 × 两方向 × paired M变化和paired accuracy变化：12项。
(4) native R与C两种总体的K2 accuracy和K2-minus-K1 accuracy：4项。
合计124。先在CPU生成带唯一ID的contrast manifest，assert恰124；三个hypothesis共享同一组区间，不另行挑选最有利检验。所有指标raw数据和逐角色logit变化公开。

能力门槛：native R和C各自K2 observed accuracy>=0.70，且各自K2-minus-K1 accuracy的上述同时区间下界>0。D_A/B是R全集重排，同一总体，不伪造额外独立能力证据。另报告每模板能力，不筛native-correct。C失败即stopversion，不换词。

对齐兼容性：A111、B110、C111在每个f方向的paired M变化同时区间须完全落入[-0.20,0.20]rawlogit，paired accuracy变化同时区间须完全落入[-0.05,0.05]。这是预设实际等效界，非“不显著”；任何失败则不能分类该功能接口。界值是本研究设计的可容忍偏差，不是已由数据证明的自然尺度。

同binding控制：C全部7角、两方向、Y/G的E同时区间均须完全落入[-0.10,0.10]。不通过说明本控制下混合不够兼容，停止domain分类；仍完整报告，不改为“减去control就解决”。

对某个受限hypothesis：
- 每个预测1角、每个f方向，Y与G的raw E及adjusted D共四个区间下界均须>0.10。
- 每个预测0非self角、每个f方向，Y与G的raw E及adjusted D区间均须完全落入[-0.10,0.10]。
- 000是技术精确基线，无科学显著性计数。111或B110即使预测0，也仍须通过上述更严格Y/G等效以及独立M/accuracy兼容性。
这要求完整角图支持，不是一个均值或正交互。预测1要求Y和G同时定向，防止仅正确答案损伤或两个mapping终点等量改变；控制不能排除任意结构化不兼容，解释仍有限。

仅当能力、技术、兼容性、samebinding控制全通过，且恰一个hypothesis全部要求通过，才报告该接口在两补偿变换下符合该受限功能模型。无一通过=unidentified；多于一个通过=ambiguous并复查manifest，绝不nearest-fit或按最大效应分类。所有候选图样和失败角公开，不能删除“例外”角。

## 6. 顺序与停止
顺序：CPU manifest/source冻结→旧开发技术→旧开发R/C自然能力→旧开发完整角图→完整预定gate。开发全部满足且恰一模型通过后，冻结该结果作为开发结论，并按未改动协议执行新64确认。不得用开发结果调整等效界、词典、层、head、receiver、source或统计族。开发失败即停止本版本单事件domain识别，不运行确认、不回退L10、不用不同gauge组合救失败。

新确认生成与冻结可在CPU阶段完成，但任何新神经结果必须在开发gate后。新确认与开发分类不一致或确认无完整模型通过，报告未复现/未识别，不回开发重选。当前旧64确认只允许作为已曝光历史描述，不能复用。

本协议不保证成功；严格等效在16开发groups上可能因不确定性失败，这意味着本版本证据不足，不是证明机制不存在。通过也只支持L11/allH功能接口在有限任务族下的寻址图样，仍不能排除混合码、上游预计算或其他残差路径，不能称为完整K1→K2机制。所有运行需新源码与contrast/input manifest哈希绑定，当前文档是可执行规格候选，不等于已通过技术验证。

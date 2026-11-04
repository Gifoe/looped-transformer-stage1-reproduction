# R26 Astra执行后决定：内容不变性通过，跨长度位级prefix合同未通过

新执行后文件；不改冻结原文、方法、源码、mask、backend或dtype。已审读 `results/r26_ROOT_VERIFICATION.json` 和 `results/r26_ROOT_LENGTH_DISTRIBUTION_DIAGNOSTIC.json`。Root独立从已保存向量重算结果；本次Astra没有重复模型或新增GPU。原研究目标仍未达成。

## 完整结果及能成立的结论

完整5 contexts、30 forwards、126对比较，每对包括state和fullV head两域；269组参数未变。重复性42/42 exact，同shape后缀内容干预42/42 exact，长度截断36/42 exact。最终类别为 `length_bitwise_sensitive`。

Q、N1、N2的全部所测depth/cut均exact。6个非exact长度比较为N0的C1/C2 native4，以及key的C1（位置13）native1..4。最大state差0.1875，最大fullV head差0.1640625；N0 native4 head最大差0.125。key C2最后memory前驱的位置1084全部exact。因此既不能说所有输入有问题，也不能从Q/N1/N2通过推及所有reference/key前缀。

这些数据支持：所测原路径可重复、同shape改变未来token内容没有影响受保护位置；不同总输入长度会改变部分相同已知prefix的数值结果。没有发现本次同shape未来内容泄漏，但有限探针不是全局数学因果证明。R26不能识别哪个具体kernel、精度环节或调度造成长度差，也不能把它直接归咎于量化、SDPA、某层或某loop。

CPU分布诊断排除了“只差全V公共常数”的解释。以保存logits作FP64 softmax，N0 native4的C1/C2 TV约0.008011/0.001914；key C1 native1..4 TV为0.003827、0.008615、0.015028、0.017677。它们确实是这些readout的概率分布变化，而非仅logit gauge。该描述没有用target/gold，也没有计算旧ARC预测；不说明实际accuracy、R25邻居或其失败由此改变。

实际55.08359357668087 GPU-worker秒，15个ledger累计18874.135196831077/21600，余额2725.8648031689227秒。不能用内部forward耗时代替全worker计费。

## 严格prefix合同受到什么影响

必须区分三个层面：

1. 原评分规则在原teacher-forced路径下的历史结果仍是原路径的结果。本次未证实未来语义内容泄漏，未推翻旧parity、参数或score审计，不重写/撤回历史分数为另一实现的分数。
2. 理想causal函数只依赖已消费prefix，与位级实现长度不敏感不是同一断言。本次明确否定“原路径对所有所测context均与真prefix截断bit-exact”的强断言。
3. 因此，将原全文teacher-forced fullV直接称为已完全满足**严格执行意义的prefix-only分布/系数**，依据不足。某些位置的实际分布还取决于总shape，而总shape可能来自未来候选长度。不能因这不是语义泄漏就忽略此合同缺口。

这也不等于已证任何现存alpha/1NN结果发生改变：分布不同不必改变alpha或neighbor，更不必改变最终赢家。本轮没有重放R25检索，禁止把该发现当作R25失败的事后解释或调参许可证。R25保持关闭，NoNewMechanism判断不变。

如果要求同时“任意未来长度下原全文数值精确不变”与“等于单一严格prefix函数”，本次反例说明两者在这些配置上不能兼得。若未来采用canonical执行primitive，需要明确协议修订和匹配基线重算，不能把它命名为原方法的零成本修复。

## canonical-prefix primitive：需要另审，当前不实施

这是一项执行协议问题，不是新事实纠错方法。至少有两种待证选择：

* **真prefix执行**：每个待评分位置只输入已知prefix，保持原dtype/backend/no-KV/no外部mask。输出结构上不接收任何未来答案或未来长度。代价是每位置独立forward、旧全文score可能不exact，主/参考/late及基线都须按同一新primitive重新定义并核算。不能偷偷重用原全文cache并称匹配。
* **canonical固定shape/filler执行**：完整输入由已知question/reference prefix、已消费答案prefix及预先固定decode budget决定，其余填固定token。总长度与filler不得由待评分候选的未来长度、候选集合最大长度、gold或任何模型效果决定；溢出规则须预先固定，不可看候选剩余长度再扩容。它可使原语义内容和数值shape都成为已知prefix的确定函数，但改变了原实际输入及可能数值，同样需要新合同、causal/noise验收与匹配基线重算。

固定filler不是加padding mask；若改变mask、backend、dtype或KV就是另一项变更，不能混进同一次修复。即使canonical primitive通过，也只是执行定义满足prefix输入约束，不提供价值选择器、R6收益保护或ARC改进证据。其额外全链成本当前未知且可能显著高于原路径；余额不足以预先承诺重算完整576+517+299+1172。当前不选择哪个primitive、不跑试验、不修改原评分。

最小下一动作若Root要追此执行问题：先写清需要的强合同是哪一个、canonical输入函数及超长边界、匹配baseline/late/cache失效范围、保留原结果的版本方案和完整成本，再审是否值得一次有限纯执行验收。不能先改到“分数更好”再给理论解释。

## R27：允许CPU结构准入，尚不允许模型诊断

R26不是禁止一切后续source探针的理由。R27方案的关键比较是同shape、同label直方图、source位置完全相同的A/B，以及同一表两query间的source donor独立性；它可以主动排除本次已知的长度混淆。**Astra同意只做现有proposal v2限定的CPU结构准入准备**：

* 固定一次seed2021、两池各32 nonce、有界选择、原25demo和既定template；不使用ARC gold或模型输出筛选。
* 验证四label确为原边界下单token、每表两query与A/B的完整token总长度/source位置相同、变化仅在预定label位置；无合格输入则no-admission，不重抽或改模板。
* 输出sanitized manifest/来源与选择日志/hash，保持proposal未冻结、模型未加载、GPU=0。不能将CPU结构通过称为功能性通过。

其后Phase A仍必须在实际运行时证明同表两query的source K/V donor exact，并以同一份query-agnostic donor重放；K/V identity各自exact，不能从R26不同配置的通过结果替代这些检查。如果source跨query不exact，前提不成立，不能解释为source编码了未来任务变化。R26只说明在所测两个cut上同shape不变；没有证明R27任意source的位级独立。

R27任何真实功能模型job仍需独立Root审核、正式冻结和明确start；还须明确它是有人工规则真值的功能诊断，不能把patch或synthetic答案直接部署到ARC。当前允许CPU准备，不批准32 forwards或180秒job，不占用新的GPU ledger。

## 阶段决定

R26完成了真正的新基础检查，发现了有限但真实的长度数值依赖，而没有发现所测同shape未来内容依赖。保留原历史方法和结果，不自动修复或重跑。canonical-prefix执行另审；R27只推进有限CPU结构可行性。原研究目标active且未完成，15 ledger余额保持；两个外部数据集仍禁止。

# R30 来源时钟 C：唯一待审设计

状态：设计与缓存复用合同；无新模型执行、无GPU授权。A已关闭；本文件不修改其任何冻结结果。原24题、96候选、548位置、421 Q/389 reference组、true-prefix primitive全部保持。R29 canonical不进入本实验。

## 假设与区别

在不含真实问题内容的N1/N2中，demo-answer来源随loop演化的V贡献，可能比泛native loop guidance更窄地刻画可扣除先验。**这是假设，不是真值方向。** C保留全部143来源位置、自然token顺序和上下文，不作A的空间重排。R27支持V通路能传对应内容，R28否定了A在该pilot的效用；二者未证明早期V更好。C检验的是时间来源贡献，不能事后用结果声称A已被证明因位置破坏而失败。

N1/N2每个独立已知prefix只新增一次4loop CF forward：每physical layer在loop1（index0）capture该层143来源V的detach clone；同一次forward的loop4（index3），将相同层相同143位置V替换为保存的loop1 V。其余行不变，全部层实施。loop2/3不改，K/RoPE/mask/backend/BF16/权重/原prompt不变，真实Q和N0不干预。不把loop1末层V广播给其他层，不做跨prefix donor复用。

该响应称 repeated-source-refinement contribution；同时包含时间表征尺度、内容与分布变化，不能命名纯语义先验。它不同于R20问题V置零、R19保护轴、R14候选残差及R25词面近邻。投影器刻意沿用，新的实验变量只有prior来源。

## FullV与评分

每位置用合法缓存重放的native q1/q4/q5及n_j1/n_j4，CF新readout为c_j4，j=1,2。定义 e_j=float32(.5*(n_j4-c_j4))，其余直接沿R28冻结math_v2：p=softmax64(q4)，g=.5(q4−q1)，F=q4+g，center_p后 D=mean_j||ec_j||p²，alpha=clip(<gc,mean ec>/D,0,1)，64eps退化alpha0；Z=float32(F−alpha*mean ec)。alpha0严格原Fixed。e0不加入平均。

完整FP32 fullV log_softmax之后才gather当前target；原位置FP32 buffer一次sum、原字符分母、native Q4−Q5 late保持。系数不接target/gold/未来token/候选菜单。e为零时完整方法严格A1，不声称退回R6。禁二次减R6。

## Native cache另审后的使用合同

可使用R28的1588native任务/6773 BF16单位置state，不能从scalar恢复fullV，也不能使用旧teacher-forcing或KV。先冻结R28 COMPLETE、state NPZ/index、task graph、源码/模型/配置/SDPA/masking/torch/Transformers、原269参数及其逐项哈希、headshape和R28公开锁的精确SHA。

逐引用核dtype=BF16（uint16无损容器）、shape[1,1,2048]、rawhash、task inputIDs/context/depth/位置、weights与cache-contract。cache ABI明确为原true-prefix+单位置head，不按“文本相似”匹配。所有fullV从同一BF16 LM-head在相同[1,1,H]shape重算，不持久缓存全词表。

第一遍不读gold：重放全1588native任务所需heads，按原顺序重建native1/4/5、Fixed、adaptive、N0原分数、M2/M3/J2、matched R6，所有对应逐target FP32 LP、sum、normalized score和late对R28标签免费SCORES exact。新C主输出、G-matched/null/scale/LOO不应对A旧输出作parity。代表性已保存fullV heads另外逐位核验；原artifact没有保存的全词表不得宣称与历史fullV逐位比较完成。

重新load后269参数before/after必须等于R28；模型/runtime/config/source ABI任何差异即invalid，不放公差、不转为fresh全量native重跑。首次cache任务Q代表使用R28的exact_duplicate_Q_task，fresh无hook5loop state/head与缓存及其保存代表raw exact。可选state未来复用授权只限本次读出与比对，不扩展成任何协议都能复用。

## Runtime验收覆盖与固定调用数

基础CF为N1/N2各389，共778。每个CF的前三loop **last-position** rawstate均与对应R28native cache exact（2334对）；它不等同于2334个完整序列比较。

固定首N1/N2代表沿R28 no_hook_native_tasks选择，不看效果：各一次capture-only fresh native4，仅capture loop1来源，不修改任何output；完整4loop states与R28保存同输入native/nohook raw exact，native heads exact，并与单位置cache一致。该代表的CF前三loop完整sequence与这些fresh native exact；其loop1捕获source也须与capture-only donor exact。全source调用次数、位置/层/clock/其他行rawexact/finite和cleanup逐forward核验。

再对首N1/N2各一次clock-identity验收：loop4捕获当前source并原位写回同值，其余路径同新hook，完整native4 states/head exact；另各重复一次真实CF，全部捕获source/当前预测state/head exact。这两种条件只是执行验收，不是可选方法。

共778+2 capture-only+2 identity+2 CF repeat+1 fresh Q=**785 forwards**。不增加“择优clock”或层扫描。所有代表 raw与全任务lastpos证据无损保存并可独立核验；hook install/configure/invalidate/finally cleanup严格遵守scorer。CPU合成先覆盖错clock、错层、错误输入/source、缺调用、原异常清理、identity和未来token不可访问。

不要求重跑所有native fullseq3的原因：前三loop在代码上没有干预；对每个真实任务已有相同输入的lastpos对照，另有固定代表完整sequence及capture-only无扰动检查。这个证据范围足以审本次有限实现，不是假称所有中间token都被独立实测。若任一实际检查失败则invalid，不凭因果源码忽略数值差。

## Controls和效果门槛

全部沿R28：实际generic J2 guidance、M2/M3、G-matched原始generic方向按新h范数匹配、seed2021 vocabulary-null匹配新h、matched_scale匹配新residual guidance范数、N1/N2 LOO，均由新C同一readout构造。退化/unmatched规则原样；不能promotecontrol。clock-identity全零prior仅检验零限，不是有力科学null；现有方向/幅度对照联合限制解释，不单凭随机null很差称成功。

完整24、clean、参数/ABI/缓存/数值全pass且无unmatched后才解封一次gold。strong原逻辑保持：对历史及matched R6各rescue>=2、damage0、各自R6-only-vs-Fixed全保留；严格胜M3/J2/G-matched/vocab-null/matched_scale；两个LOO各对两R6净值>=0且旧收益全保留。cost_review_eligible仅全部integrity通过且对两R6净>=0/旧收益全保留，不叫positive，不授权下一GPU。R28 A分数仅历史描述，不改门槛追它。

失败关闭此固定C，不改时钟/符号/参考/强度或source范围。小pilot不具排除1.3pp微效应的充分功效；stage关闭仍按事前资源决策执行。没有partial效果、无补跑/另行profile。

## 实测组件预算与完整后续

R28真实N1/N2 patch forward共449.5199673334137秒；Q/nativeN0/N1/N2 forward约708.82245秒可由cache避免。原全部projection约33秒，load17.59、前后hash14.33，head和IO/检查另计。C增加loop1 source capture，不能保证等于A patch耗时，但不是再跑原1263秒全链。

规划预算：`1.20*449.5199673334137 + 33.1 + 32 + 40 = 644.5239608000964秒`；40秒是cache读取/head重放/7代表forward/IO检查/退出的规划余量，尚未实测。建议待CPU/source审查后冻结一次 **hard700、soft685 TERM、699 KILL、余1秒reap、reserve5**；不保证完成。总wall全部计第18ledger，继承17ledger20194.085665407125，余1405.914334592875；700后名义剩705.914334592875。无额外GPU授权来自此提案。

若CPU工程证明40秒余量不现实，应在任何新feature前向Root提出具体cap修订；不能边跑边延时。运行不完整或超过硬上限不看partial、不重试。计时分项仍是诊断，总账仅加一次supervisor actual。

缓存只覆盖这24题，不能把省下native成本外推到全链。后续保持576+517+299+1172，不缩样。按现每unique长参考CF约.578秒粗算，已读 r28_ROOT_DEV576_RESOURCE_INVENTORY.json：全576每N1/N2各8669个unique input，共17338个CF，仅CF约1万秒（按pilot每forward耗时外推，不是完成时间保证），还要未缓存native/Q和控制评分；明显超余额。未读取heldout/test逐题，不能伪造其精确费用。仅有预定cost资格后才能提出完整实际工作量与预算review；没有自动追加6小时。

下一动作限Sol CPU缓存/来源clock合成和完整任务/数值重建准备，Root最终审查与公开freeze/readback后才可能单独START。R29执行优化与C不在同一job混合。

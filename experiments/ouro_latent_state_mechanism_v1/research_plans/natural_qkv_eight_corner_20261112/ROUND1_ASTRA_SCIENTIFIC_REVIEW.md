# ROUND1_ASTRA_SCIENTIFIC_REVIEW

作者与审阅者：Astra。本文为最终可归档正文，取代消息中的排版草案；由 Astra 根据完整逐题结果撰写，Terra 负责原样归档与记录服务器落盘时间。

**结论：本轮固定接收轨迹的 Q×K×V 八角全队列实验完成，三分区共 2590 题的数值与来源审计支持结果使用。主要机制证据是 Key 与 Value 的条件配合，以及 Query 对这种配合的调节；数据不支持给 Key 或 Value 一个脱离背景的唯一贡献排名。预冻结判据并非全通过：Validation 全体的 Q1/K1 背景 V-only 净救回为 −1，必须保留这项反证。**

## 1. 实际审阅范围与数值身份

Astra 独立读取 Train 1119、Validation 299、Test 1172 的完整逐题结果，核对唯一 ID、25 个训练示例标记、八角候选 scores/target logits、首索引 argmax、top ties、自然四组、固定错误候选、全部非 gold gap、12 条边、两个 K×V 交互及三阶差。独立重算生产与 Fraction 精确 BF16 target-logit 对比，核对四个旧角的逐题 scores/target logits 与既有四角结果，以及 coverage/receipt/ledger 哈希对应。全部公式一致，未发现 15 个线性对比的 raw-versus-production 严格正负相反。Test 最初采用 publication 已完成副本，随后确认其 SHA 与 executor 官方快照一致。

本轮 GPU 审计覆盖八角完整词表 logits 的 CUDA 分数回放，以及每题四旧角的 logits、SDPA_R、attention increment、scores 和实际捕获来源零差桥接。新独立进程评分器另写 float/log-softmax/gather 逻辑，来源恢复仍复用冻结 adapter；它不是另一套独立模型前向实现。Astra 对固定三题技术产物和训练字典序首题的原始 BF16 张量作了直接核查；没有声称在本机重新执行 2590 次模型前向，或逐个重算全部原始内部向量。全队列内部向量统计的可信范围是冻结实现、完整审计及抽样独立重算。

原始 natural 部分整文件此前依授权删除。本轮从已绑定 ZIP 恢复原 metadata，并从仍完整且有历史独立 coverage 的 natural-branch 产物读取 48 个 R 状态、两轮 norm/logits 及 L18 x/a，实际重验现存文件。历史原 natural 整文件 SHA 是继承证据，不是本轮对已删除整文件重新扫描。该范围已写入恢复和审计记录。

冻结模型、权重、BF16、SDPA、Joint-Option Letter 原评分、25-shot、eval/batch1/no-cache 协议保持一致，无训练。API 参数及位级输出锚已核验；底层具体 CUDA kernel 身份没有独立确定，不作此声明。

## 2. 条件、指标与解释单位

Gijk 的 i/j/k 分别指定同题自然 K1 或 K2 的 Q/K/V 来源；接收者始终为自然 K1 在物理第 18 层（index 17，ut0）的轨迹。只换 Q 的 R 行，K/V 换完整序列；SDPA 使用原完整形状与参数，只有输出 R 行回写，其后 o_proj、局部 norm、残差、MLP 与完整后缀按接收轨迹重新计算。G111=B1；G222=既有 T12，**不是自然 B2**。

主要 F 为每题 gold 对所有非 gold 候选 gap 的题内均值，再按题等权汇总。固定错误候选来自自然 K1 的最强非 gold，另保留动态最强错误 margin、所有候选分数、预测与平局。三类因子边是：Q_KjVk=G2jk−G1jk；K_QiVk=Gi2k−Gi1k；V_QiKj=Gij2−Gij1。I1=G122−G121−G112+G111；I2=G222−G221−G212+G211；J=I2−I1。这里的 I/J 是终端候选 gap 交互，不等于 SDPA 内部向量交互。

以下表格的效应单位是生产 log-probability gap 的变化。数字是有限队列的完整描述与预冻结方向检验，不附加未预定的显著性声明。

| 分区 | N | G111 | G112 | G121 | G122 | G211 | G212 | G221 | G222 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train 全体 |1119|609|614|724|823|686|711|762|875|
| Train 非示例，主要探索 |1094|589|594|700|800|663|689|737|850|
| Validation |299|150|149|178|202|172|180|195|225|
| Test |1172|626|639|726|818|678|702|753|882|

表为正确题数。25 个示例单列，不用于主要训练发现。自然 B1/B2 无 top tie 子集分别为训练非示例 1018、Validation 287、Test 1107；八角全无 tie 是另一种干预后筛选，只作敏感性分析，不替代主分区。

## Finding 1 — Key 的效应依赖 Value 与 Query 背景

四条 Key 边的平均 F 改变：

| 分区 | Q1V1 | Q1V2 | Q2V1 | Q2V2 |
|---|---:|---:|---:|---:|
| Train 非示例 | .100272 | .409668 | .029341 | .365607 |
| Validation | .140085 | .425272 | .057448 | .382124 |
| Test | .113935 | .406730 | .056238 | .373951 |

K2 的条件效应在 V2 背景更大，三个完整分区一致。Test 四条 Key 边救回/损伤分别为 125/25、215/36、101/26、206/26。Key-only 的正确数在两个 Query 背景、三个分区均高于 Value-only；但该决策排序不能改写为所有指标、所有题目上的 Key 主导。

## Finding 2 — Value 有可转移效应，但单独收益并不稳定

四条 Value 边的平均 F 改变：

| 分区 | Q1K1 | Q1K2 | Q2K1 | Q2K2 |
|---|---:|---:|---:|---:|
| Train 非示例 | .090103 | .399499 | .102915 | .439181 |
| Validation | .072081 | .357267 | .092026 | .416702 |
| Test | .075161 | .367956 | .087960 | .405672 |

Test 四条 Value 边救回/损伤为 29/16、114/22、46/22、142/13。Validation 的 V_Q1K1 平均 F 为正，但 7 救回、8 损伤，净值 −1；训练非示例及 Test 对应净值仅 +5、+13。此结果推翻“这条单独 Value 更新在每个完整分区都提高正确数”的预冻结预期。它没有推翻 Value 在 K2 配合背景下较强的条件效应，也不能据此说 Value 无作用。

## Finding 3 — K×V 配合稳定，Query 调节有明显异质性

| 分区 | mean I1 | mean I2 | mean J | raw I1 正/负/零 | raw I2 正/负/零 | raw J 正/负/零 |
|---|---:|---:|---:|---|---|---|
| Train 非示例 | .309396 | .336266 | .026870 |896/171/27|957/124/13|557/473/64|
| Validation | .285187 | .324676 | .039489 |247/51/1|261/36/2|174/111/14|
| Test | .292795 | .317713 | .024917 |948/199/25|1003/149/20|617/493/62|

两个 Query 背景的 KV 交互均为正均值、正中位数，且精确 raw 正值超过半数；Validation/Test 的自然无 tie 子集仍成立。联合 K2V2 的正确数在每种 Query 背景均高于两个单独更新。

四条 Query 边的平均 F 改变：

| 分区 | K1V1 | K1V2 | K2V1 | K2V2 |
|---|---:|---:|---:|---:|
| Train 非示例 | .078063 | .090874 | .007132 | .046813 |
| Validation | .115594 | .135539 | .032957 | .092391 |
| Test | .057527 | .070326 | −.000169 | .037547 |

Test 的 K2V1 背景 gap 均值略负，但对应决策净救回 +27；其自然无 tie 均值仍为 −.001233。不能把 Query 说成无作用，或说它在所有背景都提高 gap。总体 J 为正也不代表每题 Q2 都增强 KV 配合：Test 有 493 题 raw J 为负、62 题为零。

令 H1=G121−G112、H2=G221−G212。Train 非示例、Validation、Test 的 mean H1 分别为 +.010169、+.068004、+.038774；mean H2 分别为 −.073574、−.034577、−.031721。平均 gap 的 Key-only/Value-only 排序随 Query 背景反转。H_i 同时等于两条对应条件边之差，是同一个代数对比，不能重复计算为两份独立证据。

## Finding 4 — 完整队列有重复规律，也有明确反例

冻结文件 SHA `59438c1694501f7bf5dba9d6547fe037258a9bf57bba58aa6ff3cbb21956a7d7`，落盘早于 Validation/Test 本轮执行。Validation/Test 是项目内 train-informed 前瞻比较，不是新独立盲测：此前 ARC 与其他干预结果已暴露。

| 预冻结检验 | Validation 全体 | Validation 自然无 tie | Test 全体 | Test 自然无 tie |
|---|---|---|---|---|
| P1：两交互均值/中位数正，raw 正值多数 |通过|通过|通过|通过|
| P2：各 Q 的 K-only/V-only gap 均值及净救回正 |**失败：V_Q1K1 净 −1**|通过|通过|通过|
| P3：联合正确数超过单独；K-only 正确数超过 V-only |通过|通过|通过|通过|

Validation 无 tie 对应 V_Q1K1 净值 +3，不能覆盖全体失败。次级 H1>0、H2<0、J>0 方向及预定自然 rescue/stable-correct 分组方向得到支持；没有据此追加 p-value 阈值或删除相反样本。

自然 rescue 组的 H1/H2/J 为正，stable-correct 组为负。Test rescue 378 题的 mean I1/I2 为 +.339313/+.451251；自然 damage 51 题为 +.030229/−.034314。分组标签来自自然 B1/B2，而非本轮干预结果。分组说明异质性，但本身不是独立因果人群。

可复核反例：

- Validation `Mercury_7138688` 的 V_Q1K1 平均非 gold gap 增加 .020833，但动态最强错误 margin 从 +.125 到 −.0625，正确变错误且两端无 tie。
- Test `MCAS_1998_4_19` 的 Value-only 正确、Key-only 错误，两个 Query 背景均如此。
- Test `ACTAAP_2010_7_14` 在 Q1 背景两个单独更新均错、联合才正确。
- Test `LEAP_2000_8_2` 在 Q1 背景两个单独更新都正确、联合错误。
- Validation `MCAS_2006_9_35` 在 Q2 背景 Key-only 正确、联合错误，终端 I2 仍为 +.041667。

上述是完整反例集合中的例子；全部 ID、候选分数和变化保留在完整结果与反例文件。它们阻止把正平均交互解释为每题纠错，或将正 gap 等同于更高正确率。

BF16 原始 gap 精确为零时，FP32 log-softmax 可留下极小非零。Test 的 I1/I2/J 分别有 22/18/60 题属于此情形；原生产预测与 raw target-logit 的 argmax/ties 全队列一致。本审阅不把微小非零解释为机制方向，也未修改原生产 scores。固定错误候选敏感性与完整候选分析均保留，不能用其中一个代替其余指标。

## 机制结论的有效边界

本轮支持一个具体的计算事实：同题 L18 的 K2 Key 与 K2 Value 配合，能在固定 K1 接收后缀中产生稳定的正平均候选竞争交互，Query 会调节该效应。这是有限干预定义下的因果条件效应，超出了自然态相关；个体效果仍可为负或为零。

它不能证明完整自然 K1→K2 改进唯一由此产生，不能把 K 直接等同于语义寻址、把 V 等同于答案内容，不能声称某个 token/head/path 必要充分。Q/K 变化同时包含方向与尺度，混合状态可能非自然；后续 norm/MLP/层会转换输入效应。理想实数 SDPA 在固定 Q 下满足内部 KV 差为 `(A2−A1)(V2−V1)`，但本轮未直接测量注意力权重 A，也未证明终端 gap 交互就是该项的线性读出。保存的 SDPA_R/a_R 与终端 gap 必须分别解释。平均 gold-vs-nongold gap 亦不能单独判定“增强 gold”还是“压制错误”。不报告某个分量解释了多少百分比的自然推理收益。

## 资源与复核索引

三分区 collector receipt elapsed 合计 10606.743 秒（约 2.946 小时）；逐题 CUDA 同步 wall 合计 10232.672 秒。前者不是全部实验端到端时长，后者不是纯 GPU kernel 时间；模型加载、独立审计、分析及部分落盘开销不在这些相同范围内。记录的逐题时间包括恢复时保留的已完成题时间，不包括未记录的失败尝试；receipt elapsed 的含义以各次运行记录为准。最大逐题 allocated 为 3388515328 bytes，reserved 为 5788139520 bytes；逐题 peak reset 不涵盖模型加载峰。GPU 与其他作业共享，不能声称专属 A6000 基准或模型吞吐上限；本轮没有 40 GiB 预留。

| 分区 | coverage SHA256 | 完整 rows 压缩文件 SHA256 |
|---|---|---|
|Train|d257487676f31d0d122a913caad2c85369629ce1acca21bdabb5b2de0603c381|a8ac7510ec4bbad1a5b8e745a8d41d523ac984f4e2e3bdbc8c0d11b06b5a635e|
|Validation|b4bf09b8a1d3c2f2c54740851e6482eb7a761e7091d70bff0d592d5f79df9f7c|9d7e5c4c29d56cd97cd33e7adb53204430b650bc64dc0edc9c6661b189e7a942|
|Test|6af57d83f0c58c5832eb810c3a22bfbf3086db22f31bb37e41c3303fd5111526|f2916b7f957c9af95d29462a03a3dd0b443a243d10c04f9bfd711f2134c16287|

本轮三题 probe receipt SHA `a31ee534d84f552e5c31986db97f82cca0ed4671611a16563708fa2c571a4459`；其独立 CUDA replay SHA `1d1dd243be5e548da7792193abb40d487186afd201f6d42935fef78902e853a6`。全队列结果入口为 `ROUND1_FULL_RESULTS.json`；条件、源码与来源见 `ROUND1_PROTOCOL.md`、`ROUND1_IMPLEMENTATION_AUDIT.json`、`ROUND1_GPU_PARITY.json`；完整统计、反例和资源分别见 `ROUND1_FACTORIAL_ANALYSIS.md`、`ROUND1_COUNTEREXAMPLES.md`、`ROUND1_RUNTIME_AND_MEMORY.json`。

审阅认可本轮科学结果与规定范围的完成，不授权自动执行任何后续研究。唯一下一问另见 `ROUND2_RECOMMENDATION.md`。

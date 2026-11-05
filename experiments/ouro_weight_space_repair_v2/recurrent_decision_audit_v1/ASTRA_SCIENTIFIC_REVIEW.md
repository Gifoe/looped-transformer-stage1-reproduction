# Experiment A：独立计算语义与假设审查

2026-11-05。Astra；CPU/source review only。新任务已替代旧修复方法目标，R34 不执行。本文不是 GPU 授权，不修改历史记录，也不提出新解码算法。

## 1. 独立核实与证据权限

已读用户附件、r10_core.py、r11_inference.py、r11_worker.py、r11_scores.py、r11_evaluate.py、r7_inference.py、R11 FINAL_METHOD_LOCK、records index 和完整 test selected JSONL。Root 的归档 SHA/原始来源审计是另一份证据；本文不把一次数值重算冒充完整恢复链审计。

独立 CPU 重算 1172 题，结果如下：

|协议|K1|K4|错→对|对→错|对→对|错→错|错→错且换答案|净增|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|原顺序 joint|634|1021|435|48|586|103|46|387|
|cyclic 平均|657|1036|411|32|625|104|33|379|

R33 draft 中 18 个继承账本逐文件 SHA 匹配，另加实际 r33_gpu_budget.json 的 used_seconds 一次：19 个账本合计 **20834.389865819365 秒**，原 21600 秒余额 **765.6101341806352 秒**。不重复加入累计 total 字段。本文没有新增 GPU 耗时。

这些是已经暴露的历史测试集上的描述性复现，不是新的独立验证。不得用它们筛选后续解码方法或调参。

## 2. 必须保持的分数定义

R10 的 joint prompt 将固定 25 个示例渲染为 Question、完整带字母候选、Answer: 和正确字母；当前题同模板列出所有候选。待评分 continuation 是一个空格加字母，不是候选答案正文。每个 continuation 独立 encode_pair；每个 token 的原生 BF16 head 转 FP32，全词表 log_softmax 后取 target，再按原 FP32 reduction 求和。不得用候选间 softmax 替代全词表归一化。

历史完整 test 中 18751 个 target 记录长度均为一个 token；这是此次实际审计发现，不能在新数据入口跳过 tokenization 验证。Joint native 分数不除原答案正文字符数。原 repair baseline 的字符分母不应混入该辅助任务。

cyclic_rotations 的 order[j] 是显示位置 j 对应的原内容下标，sum_scores **已经映射回原内容顺序**。Cyclic 最终分数为这些映回后的完整 log-score 向量的算术平均；不是多数投票，不是概率平均，不是再次映射。最终组合预测按历史 np.argmax 原内容顺序打破精确并列。

每个 rotation 的行为一致性则应先对显示字母次序 argmax，再映回内容。并列时二者不能混用：实际 K1 有 169 个 top ties，其中 60 次两种顺序选出不同内容；K4 对应 64 和 26。主行为表应说明采用显示顺序规则，另存历史 mapped argmax 敏感性表；不改变四项历史准确率的主复现规则。任何结论都不能把差异隐藏在浮点容差内。

原始分数、request hash、choice order、target hash、model revision、head/norm readout 与源码锁分开验证。不要调用会重写历史报告的 r11_evaluate.final_report。

## 3. 深度覆盖与可实现边界

r7_inference.head_logits 只请求 (1,4,5)，而 R11 辅助前向的实际深度为 4，因此保存的辅助读出只有 native_k1、native_k4、loopcd_k4_fixed；完整实际 test 记录亦仅含这三项。forward_states 曾返回中间 states，并不能证明它们被永久保存。r11_inference 的 cache 是函数内临时 logits cache，不是持久 K2/K3 state archive。

原 answer-body 576 状态缓存、R28 的 true-prefix 状态缓存都不是 joint-option letter prompt 的 ABI 等价缓存。服务器是否另有真正 joint K2/K3 数据仍需文件清单、输入和 state/head provenance 实核；不能由本地源码缺保存便断言任何服务器位置都没有。

若新采集被另行批准，应在原 depth4 一次前向读取真实 normalized states 1..4，逐位保持历史 head 的位置张量形状与 FP32 sum。两题 fresh K1/K4 exact parity 是最低入口；独立 depth2/3 的对照用于确认前缀深度读出，不用插值代替。Depth6 不等于已验证 depth4；K5/K6 不自动加入。

在所有 labels 实际单 token 且同输入的条件下，一次 joint forward 可读出全部候选字母的分数；仍先证独立 encode_pair 的 input/position 完全相同。循环 n 个不同输入，不能假设复用一次 forward。64 题成本至少依赖真实循环数、上下文长度与装载/校验；765.61 秒并不自动保证完成。先固定 seed2021 的训练64集合与采集范围，再做预算审核；不据题目效果删样本或缩减循环。

## 4. 统计与轨迹定义

以题目为抽样单位。每题所有 rotations、两深度及正确性一起配对重采样；不能把循环排列当独立题目。Exact McNemar 用 rescue/damage discordant counts。置信区间是该暴露样本上的描述性抽样不确定性，不消除已有复用问题。

每题 permutation 指标：内容预测的成对一致率、全部一致指示、最大票比例；另外报告显示字母一致性。3/4/5 个选项的题都各占一题权重。循环子群没有覆盖全部排列，不能叫完整 permutation invariance。

方差分别报告原始 mapped scores 与每 rotation 内减候选均值后的 scores，ddof=0；先各内容跨 rotation 方差，再对内容平均成一题指标。正确答案对最大错误答案 margin、排名一致性独立报告。原始方差混有共同 log-normalizer 偏移；中心化仍受 logit 尺度影响。

仅 K1/K4 时，首次正确/首次换答案只能说“在已观测深度中”；变化区间为 (1,4]，不能定位 K2 或 K3。预固定“相邻可用深度预测相同”为局部稳定标准，并标记 depth_gap；K1=K4 不是连续三步稳定。另列回溯性“此后所有已观测深度不再改变”，不能与在线早停标准混称。K1/K4 两端都一样也不排除内部曾翻转。缺失 K2/K3 输出 NA，不补成稳定。

## 5. H1/H2/H3 的可区分范围

三者不是互斥的内部解释，不能按一个准确率差给它们排名。

* H1（内容比较改善）：支持它的更强行为证据是同题在不同显示位置都改选正确内容，尤其 K1 全部一致地选错、K4 全部一致地选对。这种变化无法仅由每 rotation 的正数缩放或公共加常数解释，但仍可能来自 readout maturation，不能称已发现语义推理回路。
* H2（标签/位置一致性改善）：内容分歧→内容一致、显示标签偏好减弱，与正确性迁移的交叉表支持它。单纯循环平均提升、或者变成一致地答错，不等于语义更正确。H2 可以与 H1 同时成立。
* H3（早期 readout 尚未成熟）：原生早期字母分数错误、后期改善与该假设相容；仅用同一 head 的外显读出不能证明早期隐藏态没有正确内容，也不能独立证明问题仅在 head。需要因果/独立可读性实验才有区分力，本轮不训练 probe。

Root 独立复算的显示顺序规则：全部一致题 341→952；K4 的 952 一致题中 898 对、54 错；220 不一致题中 cyclic 有 138 对、82 错。因此稳定与正确性有关联，但既非充分条件也非必要条件，不形成可部署可靠性门控。

Root 的方差结果 raw .236396→.301099、centered .192308→.204844，与排名一致性上升不矛盾。共同偏移只能解释 raw 方差；正尺度增大能增加两种方差但保持每个 rotation 的 argmax，因此不能单独解释实际内容答案迁移。现有统计不能区分尺度、相对方向和跨题异质性各占多少，不应把方差反升包装成因果发现。

## 6. Experiment B 的具体触发观测

先产出全题交叉表：K1/K4 是否全 rotations 内容一致 × 一致时的内容正确性 × 原顺序及 cyclic 的 rescue/damage。特别分开“稳定错→稳定对”与“分歧→稳定对”，并保留稳定错→稳定错、对→错的反证。这是行为分解，不是只挑正例。

若稳定错→稳定对占据实质迁移，最直接的 B 问题是哪些循环阶段改变内容比较，且是否独立于显示标签位置；若主要迁移来自分歧→一致，B 应优先针对标签/位置对应的形成。两种观测共存时，应保留双机制解释，不强制唯一答案。没有事前效应阈值就报告 counts/配对区间，不事后创造“显著比例”门槛。

只有 K1/K4 时可提出区间级 B 假设，不能据此挑某一中间层/循环作已定位站点。真实 K2/K3 的固定训练64读出若可负担，才用于定位相邻变化区间，并同时检查纠错、损害和一致性；它仍不是内部因果结论。B 需要之后独立冻结干预与 matched null，不由本轮 test 图表选择部署方法或挑选有利题目。

当前最强结果是大量配对纠错与内容排列一致性的共同变化；这足以提出形成过程的因果问题，不足以宣称新理论、H1 胜出或所有错误源于标签。Experiment A 的 CPU 审计可以完整交付，即使追加模型预算不足。

Root 后续交叉复算（待 Sol 同表核对）：K1 全部一致错→K4 全部一致对为 47 题，反向为 1 题；411 个 cyclic rescues 的 pair-agreement 增/平/降分别 331/62/18，32 个 damages 对应 19/7/6；277 个 rescues 是非一致错→一致对。这使“47 个稳定端点之间仍改变内容”的现象成为更清晰的 B 动机，而不是把全部收益归因于减少分歧。增加一致性既可伴随损害，也不是每次纠错的必要条件。这些交叉数字本段明确来源于 Root，不能替代独立逐题审计；也不证明 47 题在未观测 K2/K3 一直稳定。

## A4 源码审查补充（采集前）

prepare/collect/supervise/a4_cpu_checks 的独立审查为 conditional pass；详细定义和条件见同目录 ASTRA_A4_SOURCE_REVIEW.md。真实四层读出、独立 depth2/3 state/fullV exact、全部64历史 K1/K4/Fixed exact 与固定seed无gold选择均满足采集语义。最终补齐 dataset/history receipt provenance 后须重跑一致的 CPU receipt、冻结 readback，再由 Root 决定启动。

轨迹预定义为首次正确、首次相邻变化、首次两相邻预测相同的确认点；另列至少两深度的持久不变后缀。首次局部稳定之后的相邻变化数与全部变化数/回到旧答案次数分开；未稳定记NA。局部稳定不代表持久稳定，回溯稳定不代表可在线早停。只有完整 clean64 且原始证据校验通过后才分析64轨迹；runtime拒绝只能报告parity-only，不完整不得分析部分64。500秒是有限执行上限，不是完成保证，也不授权扩展576。未有A4实际结果，本补充没有H1/H2/H3新结论。

## A4实际终态补充

现已读a4_ANALYSIS.json及Root独立root_a4_terminal_review.json：2题8rotations技术验收完成，16个独立depth2/3 state/fullV/LP exact；24body/72heads，269参数不变。固定64的保守forecast1084.6512221960922秒未通过runtime准入，64 forward启动数为0，没有开发效果。forecast是规划估计，不是下界或实际64耗时。实际54.93683083169162秒，20账累计20889.326696651056，余额710.6733033489436秒。技术parity不用于阶段效果分析；1172题仍只能定位K1→K4区间。最终H1/H2/H3与B动机解释见本目录hypothesis_evidence.md；未授权新GPU。

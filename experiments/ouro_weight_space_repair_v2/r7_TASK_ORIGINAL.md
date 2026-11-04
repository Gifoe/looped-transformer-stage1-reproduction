# Ouro R7：前缀条件、参考稳定的循环先验投影

你负责实际执行一次基于 R6 的深化实验。

目标不是继续增加启发式系数，而是提高先验校正的选择性：
保留 R6 已有的纠错，减少误伤，并救回更多 R6 仍然错误的题。

一、起点与约束

仓库：
https://github.com/Gifoe/looped-transformer-stage1-reproduction

固定起点：
81d1c83b79461bf69882a0ffd4cf39b38dc3ba36

现有实验：
experiments/ouro_weight_space_repair_v2/

新建隔离目录：
experiments/ouro_weight_space_repair_v2/r7_prefix_prior_projection_v1/

先阅读：
slcd_README.md
slcd_RELEASE.md
results/slcd_REPORT_ZH.md
scripts/slcd_r6_scores.py
scripts/slcd_r5_evaluate.py
results/slcd_r5_development_results.csv
results/slcd_r6_development_results.csv
configs/slcd_R6_FINAL_LOCK.json

使用原始 ByteDance/Ouro-1.4B：
revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1

保持：
原始 BF16 权重、GB10 环境、seed2021、
同一固定25-shot示例、lm-eval0.4.9 acc_norm、
完整候选续写和原始字符分母。

禁止训练、反向传播、权重更新、adapter、learned gate。
推理系数只能来自当前无标签前向分布。
允许用开发标签比较有限数量的方法。

保留所有历史代码、锁、结果和旧预算。

二、主要基线与最终门槛

完整 ARC-Challenge test：
Native K4=702/1172
Fixed=717/1172
Adaptive=716/1172
A1=726/1172
R6=736/1172

主要比较必须是新方法 vs R6。

先完整评估ARC，只有冻结主方法正确数>736，
才允许运行HellaSwag、OpenBookQA或其他基准。

751/1172是争取目标，不是保证结果。

不得根据ARC test结果修改公式、参考、系数或选择另一个赢家。

三、先复现，并避免重复已失败路线

R6：
A = F + Q4 - Q5
G = F - Q4
E0 = F_prior - P4
alpha6 = clip(
    dot(center(G),center(E0)) / dot(center(E0),center(E0)),
    0,1
)
R6 = A - alpha6*E0

零分母时alpha6=0。

F必须是原始token-logit Fixed LoopCD经过完整log_softmax
和候选累加得到的分数，不能用候选分数线性外推替代。

R5已测试全量先验扣除：
A1-晚期先验=678/1093；
A1-早期guidance先验=691/1093；
A1=690/1093；
R6=705/1093。

不要把这些已失败的全量扣除重新当作主方法。

先在discovery384+selection192上复现基线。
完整1093题的705先作为历史汇总；本轮候选冻结后打开517题
确认子集，再完成全量训练基线复建。

四、低成本瓶颈诊断

只在开发数据上：

1. 统计R6的alpha区间、分母尺度、纠错与误伤。
2. 枚举 A_i-a*E_i，a∈[0,1] 的候选交叉点：
   a=(A_i-A_j)/(E_i-E_j)。
   检查真实答案是否能够在这条路径上成为第一名。
3. 该oracle只用于判断系数空间和方向空间，
   不能成为实际推理算法。
4. 固定原始prompt，检查复制、排列或删除候选对R6的影响。
5. 审计缓存：聚合候选分数不能恢复完整词表投影。
   缺少目标位置状态或logits时，必须实际重新前向。

开发期间不读取旧ARC test逐题结果或新候选test结果。

五、冻结三个参考上下文

N0：现有字面量 "Answer:"。

N1：保留原始任务模板、原始示例和分隔符，
只把当前真实问题字段替换为 "N/A"。

N2：与N1相同，但替换为 "Unknown"。

精确操作数据字段，不使用可能误删示例的字符串替换。
参考中不得含gold、解释、答案频率或正确答案提示。

预测候选第t个token时，真实流与参考流都必须已经消费
相同的候选答案前缀文本c_<t。

按各上下文独立运行官方encode_pair。
验证完整候选和预测位置对齐，不强行共享不一致target IDs，
不裁剪候选，不修改字符分母。

初版对齐失败时整题回退冻结R6，并记录覆盖率；
数值错误、模型错误和缓存损坏不能静默回退，应修复或停止。

六、实现四个主配置

M0：冻结R6。

M1：多参考、候选级稳健投影。

G = F-Q4
E_j = F_prior_j-P4_j
Gc = center(G)
Ec_j = center(E_j)

alpha = clip(
    dot(Gc,mean_j(Ec_j)) /
    mean_j(dot(Ec_j,Ec_j)),
    0,1
)

score_M1 = A-alpha*mean_j(E_j)

J=1、参考=N0时，必须精确重现R6。

M2：单参考、逐token完整词表投影。
使用N0，执行下述M3的J=1版本。

M3：多参考、逐token完整词表稳健投影。

在真实问题和相同答案前缀下：

g = 0.5*(zQ4-zQ1)
zFixed = zQ4+g
p = softmax(zQ4)

对参考j：
e_j = 0.5*(zNj4-zNj1)

定义：
inner_p(a,b) = sum_v p(v)*a(v)*b(v)
center_p(a) = a-sum_v p(v)*a(v)

gc = center_p(g)
ec_j = center_p(e_j)
ebar = mean_j(ec_j)

alpha_t = clip(
    inner_p(gc,ebar) /
    mean_j(inner_p(ec_j,ec_j)),
    0,1
)

分母为零或不可数值分辨时，使用预先固定的数值规则置零，
不根据标签选择阈值。

zNew = zFixed-alpha_t*ebar

必须对zNew重新计算完整词表log_softmax，
再读取实际目标token的值。

完整累加并只除一次原始字符分母，得到FNew。

最终：
score_M3 = FNew+Q4-Q5

不要再次减去旧R6的alpha6*E0。

alpha_t在完整词表上计算，不允许只用gold、目标token或
几个候选token估计方向。它是受约束解析投影，不是训练参数。

本轮保持晚期项Q4-Q5不变，隔离早期先验校正的贡献。

七、必须通过的实现检查

1. 模型、LM head、norm和tokenizer未改变。
2. 无干预路径与原始BF16评分一致。
3. 保留原attention-mask行为；缓存优化必须先过parity。
4. 独立循环深度与共享轨迹读出一致。
5. M1单参考精确恢复R6。
6. token校正强制为零时FNew恢复Fixed，
   最终分数恢复A1，而不是R6。
7. logits加常数不改变概率或投影系数。
8. 多个相同参考等价于一个参考。
9. 删除或随机置换gold元数据不改变推理输出。
10. 固定prompt且对齐通过时，新增其他候选不改变
    M2/M3对原候选的评分；回退例外单独报告。
11. 不混淆score-level与token-logit-level的归一化项。

八、消融和自主迭代

首先比较M0/M1/M2/M3。

开发集上至少增加：
- 稳健分母 vs 仅对平均参考投影；
- 固定R6整题alpha、但逐token校正后重新归一化的桥接控制；
- 匹配修正后guidance加权范数的非方向缩放控制；
- 固定seed的参考方向置乱/匹配范数控制；
- 去掉Q4-Q5的消融。

只有开发证据确有需要时，增加一个固定的
0.5*R6+0.5*新方法混合，不进行大量混合系数扫描。

除历史对照外最多8种实质候选配置。
最多3轮开发修正，记录假设、实现、实际结果和下一步。
随机控制不能被当作可择优模型。

优先检查参考构造、前缀对齐、聚合损失和归一化影响，
不要靠扩大超参数网格追分。

九、数据使用、冻结与最终测试

discovery384+selection192用于开发。
候选定义冻结后，一次性使用517题confirmation。
官方validation299最多比较两个已固定候选。

这些数据历史上都已使用，不得称为新盲测或独立交叉验证。

主要选择依据为准确率和净纠错，margin仅作辅助。
预先固定排序规则，不能事后更换指标。

额外报告：
- 保留R6已有纠错的比例；
- 新损坏的R6正确题数量；
- 新救回的R6错误题数量。

冻结一个主候选，保存公式、参考、回退规则、源码、
数值常量和全部输入/模型/scoring哈希。

随后评估完整1172题，与Native、Fixed、Adaptive、A1、R6比较。
主要比较为新方法vs R6。

报告正确数、纠错/误伤、题级配对bootstrap、
exact McNemar及逐题记录。

<=736：停止外部基准，不换候选重测。
>736：允许同一冻结公式评测完整HellaSwag validation10042、
OpenBookQA test500；保持原0-shot协议，主要对照仍是R6。
外部结果不得用于重新调方法。

十、预算与交付

新增GPU-worker上限6小时，独立记账。
至少预留25%给完整ARC最终评测。
先CPU重放与数学单测，再小规模真实前向，再扩展开发。

流式、按预测位置分块计算词表统计，
不要保存全数据全部token的完整词表logits。

记录主上下文、全部参考、输出head、失败尝试及历史缓存成本。
单独测量无诊断hooks部署路径的时间与显存；
不能用共享K6采集时间宣称K5部署速度。

交付：
REPORT_ZH.md
README.md
SELF_ITERATION_LOG.md
baseline_parity.json
oracle_alpha_diagnostics.csv
reference_manifest.json
development_results.csv
ablation_results.csv
FINAL_METHOD_LOCK.json
arc_full_test_results.csv
paired_transitions.csv
compute_costs.csv
完整源码、执行日志和重放证据。

结论必须说明：
R6受限于什么；
token粒度与多参考各贡献多少；
是否保留旧收益并获得新纠错；
是否超过匹配幅度与置乱控制；
是否超过736、是否达到751；
额外计算是否值得；
ARC过门槛后是否迁移。

实际完成方法和实验，不以研究计划代替执行。
允许报告负结果，不允许测试后换赢家。
提交新分支并给出精确结果。
# E3 下一真实 source 干预 order v1

基于完整 discovery64，不看余 Train 新效应。候选仍是 LocalCausal 入口，不是已证自然必要路径或信息初始来源。

选择候选当前题 Q 区：实际 current tokens 剔除 R；P 作竞争来源。选择依据：Q 区 Wγ·O 诊断在 rescue 为 +.220/.265，damage 为 −.095/−.092。这是训练探索选择依据，不是因果结论。

固定原 K1 receiver/residual，分别固定 q=Q1/Q2，起点 K2,V2。每 q 新增四个主条件：

1. K_Q 回退 K1，V 保持 V2。
2. V_Q 回退 V1，K 保持 K2。
3. Q 区 K,V 联合回退第一轮值。
4. P 区 K,V 联合回退第一轮值，作为竞争来源。

命名必须区分 query q 与 token 区域 Q。全部使用实际 post-RoPE K 直接替换，不再应用 RoPE；完整 shape 调用同一原 SDPA 函数、scale、mask、GQA，只有 R 输出回写，非 R 逐位不变。后续 o_proj、local norm、residual、MLP 和后缀重新计算。

每 q 另加三个控制：

1. K_Q 回退方向，但逐 token/head 范数匹配原 K2。
2. V_Q 回退方向，但逐 token/head 范数匹配原 V2。
3. 保持 K2，将 Q 区 V2 按实际 token 顺序循环右移一位。该规则预固定、不看 gold，打乱 K–V 配对但保持 V 集合；不能称为自然无关来源。

合计每题 14 个新增 suffix。旧 Gq22 identity 与已验原端点可复用，但每题仍须零差桥接。空/零范数状态明确记 UNDEFINED，保留该题其他条件，不删除整题。

先做 CPU 独立完整后缀 oracle，再做原固定三题技术端点验收及独立 CUDA replay，之后冻结同一 discovery64 全部题实际采集。不扫 head/token，不启新 benchmark。

主要报告 q 各自 K/V/joint 的生产全候选 gap、固定错误候选 gap、动态 margin、决策变化、自然四组与自然 no-tie、范数与配对破坏控制、SDPA/a 变化。K+V 联合效应不能当作两个独立贡献的和，须报告相互作用。

这是 source intervention，不等于删除 I_Q；softmax 重算会影响所有位置。64 之外的 1030 个非示例 Train、Validation/Test 的正式验证预测必须在其 source 效果揭示前冻结。

先报告实际 forward 数与资源预算。原 Train 可以继续；技术窗口仅在本方逐题边界协调，不影响其他 GPU 作业。

E3 完成边界：即使本轮区域回退有局部效应，也只定位 L18 被访问状态/入口。仍须追踪这些状态的上游实际生成计算，并用路径保留/阻断连接到 L18；不能据区域加权值位置宣布 E3 已完成。

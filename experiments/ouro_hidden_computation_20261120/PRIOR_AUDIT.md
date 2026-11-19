# Ouro-1.4B 前序证据审计

审计范围为任务书指定的九项实验。以下数字保留每项原有提示、数据集、度量及干预条件。重算使用已保存逐题分数的 CPU argmax，以第一个最高分作历史基准并列裁决。没有重跑模型。各版本的原生层号从 L0 起，循环号从 K1 起。

## 历史 ARC-Challenge 协议

| 协议，官方 test 1,172 题 | 评分单位 | K1 | K2 | K3 | K4 |
| --- | --- | ---: | ---: | ---: | ---: |
| 25-shot Joint-Option 原顺序 | 单 token 字母累计 log-likelihood | 634 | 缺档 | 缺档 | 1,021 |
| 25-shot Joint-Option + Cyclic | 映回内容后的各旋转 log-score 算术平均 | 657 | 缺档 | 缺档 | 1,036 |
| 25-shot Standard Full-Answer | 整个正文续写的累计 log-likelihood | 435 | 596 | 653 | 657 |
| 25-shot Standard Full-Answer | 同一分数除 Python 正文字符数，acc_norm | 439 | 650 | 696 | 702 |
| 0-shot Standard Full-Answer | 整个正文续写的累计 log-likelihood | 392 | 537 | 571 | 589 |
| 0-shot Standard Full-Answer | 同一分数除 Python 正文字符数，acc_norm | 424 | 555 | 615 | 616 |

- **Joint 渲染。** 25 个固定示例的每个题干后显示全部候选，以渲染时的 ASCII A–E 编号，示例输出为正确字母。被评分题以同一格式结束在 `Answer:`，对一个前导空格加字母作全词表原生 log-softmax。历史父子归档核验每个有效字母恰为一个 token。候选正文字符数与得分无关。冻结示例 SHA-256 为 `8cf4b70e7c07c67a589560d5fcbc1fa8feb85a0e7af3b348fa21f147271419da`。 [R10 prompt 源码](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/f221e4ee1f3de1f35067fbaa9196efc2a0ce3d4c/experiments/ouro_weight_space_repair_v2/scripts/r10_core.py)
- **Cyclic 聚合。** 遍历每题自身的全部循环移位，含原顺序，每次题干与候选上下文均需原生计算。1,165 题为四选，4 题三选，3 题五选，共 4,687 个 prompt。评分时各旋转映回原始内容顺序后平均 log-score，再对聚合值做 argmax。逐旋转离散一致性则先在实际显示顺序裁决并列，再映回内容。旧记录在 K1、K4 分别有 169、64 个逐旋转最高分并列，直接用映回后的顺序会改变 60、26 个离散预测。 [Decision Audit 源码及报告](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/e5764391a4a16f32e81588cd65f4abd5db0bb70e/experiments/ouro_weight_space_repair_v2/recurrent_decision_audit_v1)
- **Full-Answer 渲染。** 每个示例形如 `Question: …\nAnswer: <正文>`。被评分题止于 `Answer:`，候选正文分开作为 teacher-forced 续写。累加所有候选 token 的 log-prob，记录 raw 和除以正文字符数的 acc_norm。前导分隔空格参与分词但与字符分母无关。25-shot 完整 K1–K4 来自 `results/slcd_original_test.jsonl`，SHA-256 `64ab2f3487251a6eb08f9c580cfa1689048e6e0759a701c26901dab9a7bcc0ca`。0-shot 数据来自 ARC Pareto 的逐题 jsonl。用这些记录重算了分母和上表每一格。 [25-shot 报告](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/f221e4ee1f3de1f35067fbaa9196efc2a0ce3d4c/experiments/ouro_weight_space_repair_v2/r11_ARC_REPORT.md) · [0-shot 源码](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/5557699b244a0a6981beeb540dc58491bf165096/experiments/ouro_arc_pareto_20261119/run_arc.py)
- **可比较范围。** 25-shot Joint 与 Standard 在示例内容是否列出选项、提示长度、目标空间和候选评分方式上同时变化。25-shot 与 0-shot 的归一分数可作跨协议描述，以匹配运行和长度对照识别示例效应。历史 test 已暴露。ARC Pareto 0-shot 原归一 K1→K4 的 validation 为 108、133、151、153/299，raw 为 95、125、135、133/299。扩展因果及潜在组织报告同一验证集上的 127、232、252、249/299 属于 `factory.py` 的带选项字母诊断协议，不能拿来替补无选项正文。 [验证集输入和逐题分数](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/5557699b244a0a6981beeb540dc58491bf165096/experiments/ouro_arc_pareto_20261119/results) · [扩展因果输入工厂](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/5502634381e8a0c532f0a91a9b5cafe8bd7617b4/experiments/ouro_expanded_causal_20261118/factory.py)

### A6000 的完整 Joint 轨迹（2026-11-11）

`resume-test-20261111` 是一次独立的 25-shot Joint-Option 原顺序采集。其 1,172 个唯一 ID 和 gold 与 R11 逐题匹配，覆盖 4,687 个候选评分请求及 18,748 个请求×循环结果，覆盖账本无缺失或失败。RTX A6000 的运行账本保存 BF16、原生 SDPA、无 KV cache、PyTorch 2.5.1+cu124，保存状态的同运行分数回放以零误差通过。直接按逐题预测重算，K1–K4 为 **626、953、1,000、1,020/1,172**。相邻救回/回退为 **378/51、81/34、33/13**。这是原顺序数据，循环排列不在此次采集范围内。

R11 端点归档本身仍只有 K1/K4。A6000 与 R11 在 K1 有 **28 题**、K4 有 **9 题**预测不一致，两端取并集 **36 题**。R11 的请求文本哈希与续写长度不足以确认完整 token 序列等价。完整四轮可用于已暴露测试集上的独立描述，逐题转移和首轮正确应使用同一份 A6000 四轮结果。 [运行与文件索引](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/3c13d4e8ba026a6b617b61ccf3a95c88e88a6251/experiments/ouro_latent_state_mechanism_v1/results/ROUND_1_OBSERVATION_INDEX.json) · [逐题结果、覆盖账本及端点对照](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/3c13d4e8ba026a6b617b61ccf3a95c88e88a6251/experiments/ouro_latent_state_mechanism_v1/results/orchestration_runs/resume-test-20261111)

`behavior_test.json` SHA-256 为 `73ebd816d931556da08a74e22c8c74a28eafb8ac75a34160cd17b3a0a909b366`，`coverage_test.json` 为 `df164143c5921971c54a314ef560b3e3cc8566ff13636a931690efe53943d5e1`。发布提交与本地文件逐字节一致。

## 九项实验及约束

1. **`ouro_weight_space_repair_v2`。** R11 已冻结的 robust cyclic 轴为 742/1,172，R6 为 736，救回 15、损害 9，差 +0.512 pp，95% 配对区间 [−0.256, +1.365] pp。静态 Native 轴数值 748/1,172，目标方法未建立循环差轴的机制优势。这里的 702 是 25-shot 正文的字符归一原生 K4，Fixed/Adaptive K4 是 717/716。历史归档的多轮完整正文记录给出上表四档，Joint 辅助只保存 K1、K4。 [冻结报告](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/f221e4ee1f3de1f35067fbaa9196efc2a0ce3d4c/experiments/ouro_weight_space_repair_v2/r11_ARC_REPORT.md)
2. **`recurrent_decision_audit_v1`。** 全 1,172 题原顺序 Joint 的 K1→K4 为 435 救回、48 损害，cyclic 为 411、32。所有旋转的内容 top-1 一致数为 341→952。直接从 9,374 条逐旋转记录重算后，47 题两端均全旋转一致但从错变对，反向 1 题。旋转间中心化分数方差没有可靠下降，因此只能将一致性提升解释为离散排序表现。A4 的 `a4_COMPLETE.json` 明确记 `parity_only_runtime_not_admitted`，两题八旋转做过深度 2/3 原生校验，冻结的 64 题行为采集未启动。相同端点允许中间回退。 [报告、覆盖和逐题记录](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/e5764391a4a16f32e81588cd65f4abd5db0bb70e/experiments/ouro_weight_space_repair_v2/recurrent_decision_audit_v1)
3. **`recurrent_candidate_intervention_v1`。** 24 个训练来源的 pilot 保持同一批 25 demo，原顺序 Joint 四档正确数 10、20、19、20，后两轮非单调。改变选项排列使内容重映一致性上升。含占位符的干扰项变化与自然错误替代所测相对响应方向不同，严格 BPE/形状匹配的自然替代只有 4 题，也不能把尺度、长度、语义贡献完全分开。B5 只有 4 道，同题 2×2 交叉 demo 字母/正文与目标字母/正文，但每格都展示选项。可识别输出协议失配和 raw/acc_norm 差异，未识别候选可见性的主效应。 [24 题报告与矩阵](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/70dd18f18ca9a2f93c95b80c39350427dbcd7688/experiments/ouro_weight_space_repair_v2/recurrent_candidate_intervention_v1)
4. **`ouro_expanded_causal_20261118`。** 关系任务的 90 组 B 线路在 K1/L5 注入来源，在 K2 阻断来源→映射使 K4 注入效应减 0.238 nats，98.33% 区间 [0.165, 0.319]，来源→查询/选项/读出单独阻断减 0.012 nats，区间含零。双阻断后写回同接收方开放条件的映射区恢复 0.144 nats。该组原生四世界全对仅 5/90，结论落在受控 margin。A 线等词元供体和重命名对照支持相关上下文可被搬运，抽象绑定与同名相符度仍共存。前序关系/逻辑生成器把场景与选项共同重排，r/d 真值固定为 A，绝对正确率只能带此限制引用。 [报告、干预和逐题结果](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/5502634381e8a0c532f0a91a9b5cafe8bd7617b4/experiments/ouro_expanded_causal_20261118)
5. **`ouro_latent_organization_20261118`。** 修正语义场景与标签独立打乱，三类任务各 1,000 组。未见结构中晚轮中间值可读性增益为关系 +4.92 pp、逻辑 +11.08 pp、算术 +0.50 pp，后者区间跨零且同组的 2,048D 算术控制仍为 50%。固定 K1 探针跨轮效能下降。90 组/任务的语义一维方向相对配幅随机均无稳定正效应。用不重叠的 60 个纠正标签的新关系组，K2 阻断使 K4 注入效应少 0.078 [0.041, 0.120] nats，恢复开放 K2 映射补回 0.040 [0.013, 0.070]。修正版支持前述路径方向，同时明确效应尺寸因场景不同不应合并。 [报告及 route_retest](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/951eb63e448968ee29d91c442def7c4965477e55/experiments/ouro_latent_organization_20261118)
6. **`ouro_latent_transitions_20261119`。** K1→K2 的 96D/全宽角色均值、Ridge、残差 MLP 和低秩专家比较了同结构与结构留出。全宽 Ridge 同结构的首选项一致率为关系/逻辑/算术 80.8/84.3/72.2%，新结构变为 28.8/71.8/44.9%，此统计预测的是模型首选项而非正确答案。关系、逻辑真实源区移植最强简单对照为零响应。算术非线性部分拟合同结构的真实响应，新结构失效，全部任务均未同时达到几何、行为和结构泛化门槛。 [执行报告及失败判据](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/2afc5a51129fc506fbb982c3914477712a511e9b/experiments/ouro_latent_transitions_20261119)
7. **`ouro_causal_composition_20261119`。** K1 边界有 S、M、SM 和 clean 实际替换，每任务 80 个结构留出组。K2 隐藏几何非加性已经出现，K2 有符号答案交互区间仍跨零。K4 三任务交互 +0.268/+0.398/+0.529 nats，但联合替换平均均未提高原共同答案 margin，逻辑非并列组指定选项转入 5、转出 23。SM 联合响应的 2×2 比较说明真实干预训练有助于拟合部分几何，但 Token-Aware 低维汇总未优于 Region-Mean 混合，冻结候选亦未在独立结构下超过最优简单对照。补偿四世界和非零交互支持非加性证据，无法判定组合值。 [报告、语义和联合响应](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/f525239970b70c7ce28fe9fe91d95859653096cc/experiments/ouro_causal_composition_20261119)
8. **`ouro_loop_computation_20261119`。** 使用四种独立结果的主 2×2、额外同 Y 和无关供体，1,056 组和 8,448 条自然轨迹已经排除了历史固定 A。带候选条件以字母评分，开放条件以备选正文评分，两因素同时变化。结构留出逻辑的最大原生候选排序增益，带选项时 K1→K2（27.6→53.4%），开放时 K2→K3（27.9→67.7%）。开放新结构逻辑的独立整轮 K2 来源→映射切断，在 K2 对比 +0.0006 [−0.0108, +0.0124] nats，在 K4 损失 0.254 [0.178, 0.332] nats，K2/L23 后同接收方局部恢复使 K4 回升 0.238 [0.178, 0.302]。四层窗口阻断又使 K2 映射区 S 的固定探针分数上升，说明读出增强可与行为损害并存。单独 24 道新结构逻辑问题的 K4 短自由生成仅 4/24，算术 0/24。既有分析未得到稳定逐题预测增益。报告没有给出专门阻断 K3 映射→读出以中和 K2 恢复的下游检验结果。 [远端最终报告及原始记录](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/3cd9d3bc66a1822d56dc9bc715662911dcd7fe97/experiments/ouro_loop_computation_20261119)
9. **`ouro_arc_pareto_20261119`。** 开放无候选的 0-shot 正文评分含 299 validation 和 1,172 test 的 Native K1–K4、三种 LoopCD K2–K4。完整 K4 test 中 Raw 以 Adaptive 608/1,172 最高，acc_norm 以 Hidden 637/1,172 最高，后者相对 Native +1.79 pp [0.26, 3.33]，这两项最高值均为已暴露 test 的描述。Native K3→K4 的 acc_norm 仅增 1 题，27 救回、26 损害。验证集 128+171 个不同题的 K4→K2 Q/K/V 移植中 Q 和 V 效应换向，复核均低于冻结浅轮 LoopCD。缓存时延用 24 个固定 validation 提示，测了实际 prefill 和一次 decode，提前退出用离线评分回放。 [冻结报告、时延及逐条分数](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/5557699b244a0a6981beeb540dc58491bf165096/experiments/ouro_arc_pareto_20261119)

## 可复用读出，不同成本

| 0-shot test 的实际档位 | Native raw/norm | Fixed raw/norm | Adaptive raw/norm | Hidden raw/norm |
| --- | ---: | ---: | ---: | ---: |
| K1 | 392/424 | — | — | — |
| K2 | 537/555 | 534/548 | 537/547 | 532/554 |
| K3 | 571/615 | 591/623 | 600/622 | 585/622 |
| K4 | 589/616 | 606/636 | 608/630 | 592/637 |

表中斜线连接两种各自满分 1,172 的正确计数。K1 与其自身对比的差项为零，新实验可报告为读出退化成 Native。上述 25-shot 归档只覆盖 Fixed 与 Adaptive 的 K4，不能从 K4 读出直接内插干预的 K2/K3 或 Hidden。LoopCD 比较须保持协议和 K 预算一致。多 token 自由生成中，改变读出会改变已经发出的 token，后续轮转会接收不同前缀，固定前缀下的内部轨迹不变这一结论只适用于控制同一前缀的比较。

本次核验的上游代码固定到 `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`。历史一部分运行使用 5090，一部分使用 4090。一些旧的运行要求 BF16 分数逐位校验，不同设备的分数不得仅凭总正确数接近就认定等价。 [官方模型实现](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)

# Ouro-1.4B 冻结反事实复核

这是上一研究第四轮的独立验证扩展。求解器、权重、逆解迭代次数、两套 ridge 权重与比较口径预先冻结。上一轮 FP32 K1→K2 的 10 次原生改选命中全部来自 ARC，受控结构仅 0/2。本轮把受控结构单独作为主要检验，并补测真实 K3→K4。

## 主要结果

48 个新受体组的来源/规则臂共 96 个。FP32 K1→K2 的原生模型改向目标 1 次、改向其他选项 4 次、保持 91 次、并列 0 次。Anderson 严格同时命中改选前后两个原生选项 0/1，原生保持时误报 2/91。原生原先正确且改向目标 0 次，Anderson 命中 0 次。可判目标改选来源组过少，无法据此估计稳定命中率。 此处事件稀少，区间仅描述当前面板上观察到的改选组。

完整受体和完整供体均答对的 4 个早轮臂中，只写回一个角色后有 0 个受体改向目标。

相对前缀 ridge，FP32 的按组平均 logit 响应 RMSE 改善为 0.09218，95% 区间 [0.05575, 0.1305]。BF16 对应区间 [-0.01478, 0.07449]。

| 新提示族 | 臂 | 原生改向目标 | 改向其他 | 保持 | 并列 | 原先正确且改向目标 | 双端命中目标改选 | 其中原先正确 | 保持时误报 |
|---|---|---|---|---|---|---|---|---|---|
| tableau | 24 | 1 | 1 | 22 | 0 | 0 | 0 | 0 | 1 |
| twohop_open | 36 | 0 | 3 | 33 | 0 | 0 | 0 | 0 | 0 |
| twohop_options | 36 | 0 | 0 | 36 | 0 | 0 | 0 | 0 | 1 |

纯 Anderson 依赖给定的原生分支函数。结果描述对真实编辑响应的重建，不能识别唯一的连续中间轨道。结构目标是程序从供体来源和规则算出。预测器只读编辑后的隐藏状态，目标标签仅用于计分。

## 数据与对照

3×3 新组 12 个，双跳开放提示 18 个，双跳列选项新格式 18 个。按既定种子与布局验收筛选，评分前冻结。开放提示族复用了模板，世界和语义表有隔离。32 道 ARC 验证题仅用于自然轨迹，对上一版嵌入拟合及验证选题独立，这批题曾出现在项目历史整分区评分中。

每个结构受体组包含来源、规则、两种等剂量随机、两种位置对照及两个 same-Y 臂。same-Y 从目标不变的非相关前缀替换生成自然供体，写回相同角色并匹配剂量。角色中的原始词义与期望答案保持。剂量缩放可能把状态移出自然分布，原始供体剂量、缩放和写回后实际剂量全部在 JSON 中。另有双臂缩放均≤4 的敏感性子集。

| 精度 | 终点轮 | same-Y 臂 | 缩放中位 | 实际剂量相对偏差 P95 | 语义臂输出 L2 中位 | same-Y 输出 L2 中位 | 缩放≤4 的 same-Y 臂 |
|---|---|---|---|---|---|---|---|
| float32 | K2 | 96 | 1.977 | 1.338e-07 | 69.44 | 41.78 | 96 |
| float32 | K3 | 48 | 1.12 | 1.171e-07 | 71.92 | 45.53 | 48 |
| float32 | K4 | 48 | 1.176 | 1.008e-07 | 65.85 | 36.13 | 48 |
| bfloat16 | K2 | 96 | 1.96 | 0.0002843 | 69.67 | 42.05 | 96 |
| bfloat16 | K3 | 48 | 1.121 | 0.0003975 | 72.2 | 46.49 | 48 |
| bfloat16 | K4 | 48 | 1.18 | 0.0002655 | 65.96 | 36.37 | 48 |

原生和所有预测器从真实同一轮初态起步。每类每个轮界检查 48 个分支边界和全词表输出对齐。本轮 CAUSAL 的 K3、K4 仅用评分前从每族哈希选出的 8 个组。下表 K2 也只用这同一批 24 组。

## 配对的跨轮外推

| 精度 | 终点轮 | 臂 | 原生改向目标 | 原先正确且改向目标 | Anderson 双端命中 | Anderson RMSE | Picard RMSE | 整轮 RMSE | 分支 RMSE | 前缀 ridge RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| float32 | K2 | 48 | 0 | 0 | 0 | 0.02151 | 0.02933 | 0.1599 | 0.149 | 0.07253 |
| float32 | K3 | 48 | 0 | 0 | 0 | 0.005553 | 0.01022 | 0.1039 | 0.2528 | 0.2122 |
| float32 | K4 | 48 | 2 | 0 | 1 | 0.003843 | 0.006295 | 0.0628 | 0.1947 | 0.1678 |
| bfloat16 | K2 | 48 | 1 | 0 | 1 | 0.05393 | 0.07422 | 0.1901 | 0.1704 | 0.1088 |
| bfloat16 | K3 | 48 | 0 | 0 | 0 | 0.06289 | 0.08069 | 0.1197 | 0.254 | 0.2162 |
| bfloat16 | K4 | 48 | 2 | 0 | 1 | 0.05337 | 0.05103 | 0.08005 | 0.1961 | 0.1654 |

## 原生事件和严格改选

并列时不算换选或保持。双端命中要求原生及预测的两端均无并列且各自选项一致。保持时误报要求原生两端相同、预测两端均可判但选项不同。合计分母包含原生走错目标的情形。随机与位置对照沿用对应语义臂的计分目标，只有 same-Y 具有保持原结构真值的语义解释。

| 精度 | 终点轮 | 干预 | 臂 | 原生改向目标 | 改向其他 | 保持 | 并列 | 原先正确且改向目标 | Anderson 双端命中 | 原先正确的双端命中 | 保持时误报 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| float32 | K2 | source_rule | 96 | 1 | 4 | 91 | 0 | 0 | 0 | 0 | 2 |
| float32 | K2 | random | 96 | 3 | 5 | 88 | 0 | 3 | 3 | 3 | 4 |
| float32 | K2 | unrelated | 96 | 18 | 17 | 61 | 0 | 8 | 9 | 6 | 9 |
| float32 | K2 | same_y | 96 | 0 | 3 | 93 | 0 | 0 | 0 | 0 | 5 |
| float32 | K3 | source_rule | 48 | 0 | 7 | 41 | 0 | 0 | 0 | 0 | 0 |
| float32 | K3 | random | 48 | 0 | 4 | 44 | 0 | 0 | 0 | 0 | 0 |
| float32 | K3 | unrelated | 48 | 4 | 17 | 27 | 0 | 2 | 4 | 2 | 1 |
| float32 | K3 | same_y | 48 | 0 | 3 | 45 | 0 | 0 | 0 | 0 | 0 |
| float32 | K4 | source_rule | 48 | 2 | 5 | 41 | 0 | 0 | 1 | 0 | 1 |
| float32 | K4 | random | 48 | 3 | 4 | 41 | 0 | 0 | 3 | 0 | 1 |
| float32 | K4 | unrelated | 48 | 6 | 12 | 30 | 0 | 4 | 6 | 4 | 0 |
| float32 | K4 | same_y | 48 | 0 | 6 | 42 | 0 | 0 | 0 | 0 | 0 |
| bfloat16 | K2 | source_rule | 96 | 2 | 4 | 84 | 6 | 0 | 1 | 0 | 1 |
| bfloat16 | K2 | random | 96 | 2 | 3 | 84 | 7 | 2 | 2 | 2 | 4 |
| bfloat16 | K2 | unrelated | 96 | 15 | 19 | 56 | 6 | 6 | 5 | 3 | 4 |
| bfloat16 | K2 | same_y | 96 | 0 | 2 | 88 | 6 | 0 | 0 | 0 | 2 |
| bfloat16 | K3 | source_rule | 48 | 0 | 7 | 41 | 0 | 0 | 0 | 0 | 0 |
| bfloat16 | K3 | random | 48 | 0 | 5 | 43 | 0 | 0 | 0 | 0 | 0 |
| bfloat16 | K3 | unrelated | 48 | 3 | 16 | 27 | 2 | 1 | 3 | 1 | 2 |
| bfloat16 | K3 | same_y | 48 | 0 | 2 | 46 | 0 | 0 | 0 | 0 | 0 |
| bfloat16 | K4 | source_rule | 48 | 2 | 5 | 35 | 6 | 0 | 1 | 0 | 0 |
| bfloat16 | K4 | random | 48 | 1 | 2 | 37 | 8 | 0 | 1 | 0 | 0 |
| bfloat16 | K4 | unrelated | 48 | 5 | 10 | 26 | 7 | 3 | 5 | 3 | 0 |
| bfloat16 | K4 | same_y | 48 | 0 | 2 | 39 | 7 | 0 | 0 | 0 | 0 |

## 完整供体提示对照

在原批次重放完整供体提示，保留所有语义条件，再与只写入指定角色的受体比较。这是在看到早轮目标改选稀少后补充的诊断，仍用冻结组，未按输出筛选。最后一列以 clean 和完整供体都答对的臂为分母来理解，区分原生能力不足与单角色编辑未传递足够信息。

| 精度 | 终点轮 | 样本族 | 供体臂 | 完整供体答对 | 两完整提示都答对 | 其中编辑受体到达目标 |
|---|---|---|---|---|---|---|
| float32 | K2 | tableau | 24 | 9 | 0 | 0 |
| float32 | K2 | twohop_open | 36 | 15 | 3 | 0 |
| float32 | K2 | twohop_options | 36 | 15 | 1 | 0 |
| float32 | K3 | tableau | 16 | 8 | 0 | 0 |
| float32 | K3 | twohop_open | 16 | 7 | 1 | 0 |
| float32 | K3 | twohop_options | 16 | 8 | 1 | 0 |
| float32 | K4 | tableau | 16 | 10 | 1 | 0 |
| float32 | K4 | twohop_open | 16 | 5 | 0 | 0 |
| float32 | K4 | twohop_options | 16 | 6 | 4 | 0 |
| bfloat16 | K2 | tableau | 24 | 9 | 0 | 0 |
| bfloat16 | K2 | twohop_open | 36 | 15 | 3 | 0 |
| bfloat16 | K2 | twohop_options | 36 | 15 | 1 | 0 |
| bfloat16 | K3 | tableau | 16 | 8 | 0 | 0 |
| bfloat16 | K3 | twohop_open | 16 | 7 | 1 | 0 |
| bfloat16 | K3 | twohop_options | 16 | 8 | 1 | 0 |
| bfloat16 | K4 | tableau | 16 | 11 | 1 | 0 |
| bfloat16 | K4 | twohop_open | 16 | 5 | 0 | 0 |
| bfloat16 | K4 | twohop_options | 16 | 6 | 2 | 0 |

## 响应向量和候选 logits

RMSE 是预测器自身 clean→edited logit 差与原生差之间的误差。方向余弦只对原生全宽响应非零的样本计算。配对差按来源组重采样，正值表示 Anderson 的平均 RMSE 更低。

| 精度 | 终点轮 | 干预 | 方法 | RMSE 中位 | 方向余弦中位 | 目标改选命中 | 最终命中指定目标 | 保持时误报 | 预测并列 |
|---|---|---|---|---|---|---|---|---|---|
| float32 | K2 | source_rule | Anderson | 0.02694 | 0.9788 | 0/1 | 33/96 | 2 | 0 |
| float32 | K2 | source_rule | Picard | 0.04388 | 0.9773 | 0/1 | 32/96 | 4 | 0 |
| float32 | K2 | source_rule | Full-loop | 0.1482 | 0.8016 | 0/1 | 40/96 | 14 | 0 |
| float32 | K2 | source_rule | Branch-flow | 0.1365 | 0.1784 | 0/1 | 40/96 | 7 | 0 |
| float32 | K2 | source_rule | Prefix ridge | 0.07924 | 0.427 | 0/1 | 34/96 | 0 | 0 |
| float32 | K2 | source_rule | Token ridge | 0.06968 | 0.4034 | 0/1 | 36/96 | 0 | 0 |
| float32 | K2 | source_rule | Persistence | 0.06968 | 0.4469 | 0/1 | 28/96 | 0 | 0 |
| float32 | K2 | random | Anderson | 0.04834 | 0.9758 | 3/3 | 36/96 | 4 | 0 |
| float32 | K2 | random | Picard | 0.1109 | 0.9696 | 2/3 | 35/96 | 3 | 0 |
| float32 | K2 | random | Full-loop | 0.1821 | 0.6316 | 0/3 | 41/96 | 13 | 0 |
| float32 | K2 | random | Branch-flow | 0.2235 | 0.1042 | 0/3 | 38/96 | 8 | 0 |
| float32 | K2 | random | Prefix ridge | 0.2318 | 0.01348 | 0/3 | 33/96 | 0 | 0 |
| float32 | K2 | random | Token ridge | 0.2276 | 0.01383 | 0/3 | 36/96 | 0 | 0 |
| float32 | K2 | random | Persistence | 0.2276 | 0.02061 | 0/3 | 28/96 | 0 | 0 |
| float32 | K2 | unrelated | Anderson | 0.8482 | 0.8362 | 9/18 | 41/96 | 9 | 0 |
| float32 | K2 | unrelated | Picard | 1.269 | 0.8191 | 9/18 | 41/96 | 8 | 0 |
| float32 | K2 | unrelated | Full-loop | 1.922 | 0.6749 | 6/18 | 44/96 | 8 | 0 |
| float32 | K2 | unrelated | Branch-flow | 1.227 | 0.3473 | 4/18 | 42/96 | 24 | 0 |
| float32 | K2 | unrelated | Prefix ridge | 1.396 | 0.02729 | 0/18 | 34/96 | 0 | 0 |
| float32 | K2 | unrelated | Token ridge | 1.398 | 0.03124 | 0/18 | 36/96 | 0 | 0 |
| float32 | K2 | unrelated | Persistence | 1.398 | 0.03552 | 0/18 | 28/96 | 0 | 0 |
| float32 | K2 | same_y | Anderson | 0.02391 | 0.9647 | 0/0 | 22/96 | 5 | 0 |
| float32 | K2 | same_y | Picard | 0.03585 | 0.9536 | 0/0 | 23/96 | 3 | 0 |
| float32 | K2 | same_y | Full-loop | 0.09343 | 0.6302 | 0/0 | 24/96 | 5 | 0 |
| float32 | K2 | same_y | Branch-flow | 0.1258 | 0.06685 | 0/0 | 26/96 | 6 | 0 |
| float32 | K2 | same_y | Prefix ridge | 0.101 | 0.2395 | 0/0 | 36/96 | 6 | 0 |
| float32 | K2 | same_y | Token ridge | 0.06687 | 0.2343 | 0/0 | 32/96 | 0 | 0 |
| float32 | K2 | same_y | Persistence | 0.06687 | 0.2348 | 0/0 | 28/96 | 0 | 0 |
| float32 | K3 | source_rule | Anderson | 0.005553 | 0.9999 | 0/0 | 19/48 | 0 | 0 |
| float32 | K3 | source_rule | Picard | 0.01022 | 0.9983 | 0/0 | 20/48 | 0 | 0 |
| float32 | K3 | source_rule | Full-loop | 0.1039 | 0.9361 | 0/0 | 20/48 | 3 | 0 |
| float32 | K3 | source_rule | Branch-flow | 0.2528 | 0.3605 | 0/0 | 18/48 | 1 | 0 |
| float32 | K3 | source_rule | Prefix ridge | 0.2122 | 0.6104 | 0/0 | 22/48 | 1 | 0 |
| float32 | K3 | source_rule | Token ridge | 0.1945 | 0.5859 | 0/0 | 22/48 | 0 | 0 |
| float32 | K3 | source_rule | Persistence | 0.1945 | 0.6609 | 0/0 | 16/48 | 0 | 0 |
| float32 | K3 | random | Anderson | 0.005078 | 0.9998 | 0/0 | 22/48 | 0 | 0 |
| float32 | K3 | random | Picard | 0.01156 | 0.9975 | 0/0 | 22/48 | 0 | 0 |
| float32 | K3 | random | Full-loop | 0.1216 | 0.7618 | 0/0 | 24/48 | 2 | 0 |
| float32 | K3 | random | Branch-flow | 0.3058 | 0.2485 | 0/0 | 17/48 | 5 | 0 |
| float32 | K3 | random | Prefix ridge | 0.2484 | 0.03864 | 0/0 | 22/48 | 0 | 0 |
| float32 | K3 | random | Token ridge | 0.253 | 0.04184 | 0/0 | 22/48 | 0 | 0 |
| float32 | K3 | random | Persistence | 0.253 | 0.04377 | 0/0 | 16/48 | 0 | 0 |
| float32 | K3 | unrelated | Anderson | 0.03114 | 0.9999 | 4/4 | 18/48 | 1 | 0 |
| float32 | K3 | unrelated | Picard | 0.07889 | 0.9975 | 4/4 | 20/48 | 2 | 0 |
| float32 | K3 | unrelated | Full-loop | 4.101 | 0.8852 | 1/4 | 21/48 | 6 | 0 |
| float32 | K3 | unrelated | Branch-flow | 7.603 | 0.6144 | 1/4 | 20/48 | 12 | 0 |
| float32 | K3 | unrelated | Prefix ridge | 8.585 | 0.04676 | 0/4 | 22/48 | 0 | 0 |
| float32 | K3 | unrelated | Token ridge | 8.588 | 0.06517 | 0/4 | 22/48 | 0 | 0 |
| float32 | K3 | unrelated | Persistence | 8.588 | 0.08179 | 0/4 | 16/48 | 0 | 0 |
| float32 | K3 | same_y | Anderson | 0.002335 | 0.9998 | 0/0 | 10/48 | 0 | 0 |
| float32 | K3 | same_y | Picard | 0.00814 | 0.9966 | 0/0 | 10/48 | 0 | 0 |
| float32 | K3 | same_y | Full-loop | 0.0557 | 0.7276 | 0/0 | 10/48 | 1 | 0 |
| float32 | K3 | same_y | Branch-flow | 0.163 | 0.1387 | 0/0 | 12/48 | 2 | 0 |
| float32 | K3 | same_y | Prefix ridge | 0.09814 | 0.3133 | 0/0 | 12/48 | 0 | 0 |
| float32 | K3 | same_y | Token ridge | 0.08578 | 0.2975 | 0/0 | 4/48 | 0 | 0 |
| float32 | K3 | same_y | Persistence | 0.08578 | 0.3351 | 0/0 | 6/48 | 0 | 0 |
| float32 | K4 | source_rule | Anderson | 0.003843 | 0.9999 | 1/2 | 16/48 | 1 | 0 |
| float32 | K4 | source_rule | Picard | 0.006295 | 0.9989 | 1/2 | 16/48 | 3 | 0 |
| float32 | K4 | source_rule | Full-loop | 0.0628 | 0.9708 | 0/2 | 23/48 | 2 | 0 |
| float32 | K4 | source_rule | Branch-flow | 0.1947 | 0.7512 | 0/2 | 15/48 | 1 | 0 |
| float32 | K4 | source_rule | Prefix ridge | 0.1678 | 0.7135 | 0/2 | 24/48 | 0 | 0 |
| float32 | K4 | source_rule | Token ridge | 0.1702 | 0.6897 | 0/2 | 20/48 | 0 | 0 |
| float32 | K4 | source_rule | Persistence | 0.1702 | 0.7953 | 0/2 | 26/48 | 0 | 0 |
| float32 | K4 | random | Anderson | 0.004585 | 0.9999 | 3/3 | 18/48 | 1 | 0 |
| float32 | K4 | random | Picard | 0.007153 | 0.9982 | 1/3 | 19/48 | 2 | 0 |
| float32 | K4 | random | Full-loop | 0.1257 | 0.804 | 0/3 | 25/48 | 1 | 0 |
| float32 | K4 | random | Branch-flow | 0.1888 | 0.692 | 1/3 | 15/48 | 3 | 0 |
| float32 | K4 | random | Prefix ridge | 0.2326 | 0.07074 | 0/3 | 24/48 | 0 | 0 |
| float32 | K4 | random | Token ridge | 0.2463 | 0.06228 | 0/3 | 20/48 | 0 | 0 |
| float32 | K4 | random | Persistence | 0.2463 | 0.06072 | 0/3 | 26/48 | 0 | 0 |
| float32 | K4 | unrelated | Anderson | 0.03022 | 0.9987 | 6/6 | 17/48 | 0 | 0 |
| float32 | K4 | unrelated | Picard | 0.06896 | 0.9977 | 5/6 | 17/48 | 2 | 0 |
| float32 | K4 | unrelated | Full-loop | 4.325 | 0.9007 | 2/6 | 22/48 | 5 | 0 |
| float32 | K4 | unrelated | Branch-flow | 3.293 | 0.8408 | 2/6 | 18/48 | 4 | 0 |
| float32 | K4 | unrelated | Prefix ridge | 10.95 | 0.05765 | 0/6 | 24/48 | 0 | 0 |
| float32 | K4 | unrelated | Token ridge | 10.95 | 0.07589 | 0/6 | 20/48 | 0 | 0 |
| float32 | K4 | unrelated | Persistence | 10.95 | 0.08993 | 0/6 | 26/48 | 0 | 0 |
| float32 | K4 | same_y | Anderson | 0.002731 | 0.9999 | 0/0 | 12/48 | 0 | 0 |
| float32 | K4 | same_y | Picard | 0.005218 | 0.9974 | 0/0 | 12/48 | 1 | 0 |
| float32 | K4 | same_y | Full-loop | 0.05485 | 0.8503 | 0/0 | 12/48 | 0 | 0 |
| float32 | K4 | same_y | Branch-flow | 0.09411 | 0.5674 | 0/0 | 14/48 | 3 | 0 |
| float32 | K4 | same_y | Prefix ridge | 0.1122 | 0.4613 | 0/0 | 14/48 | 0 | 0 |
| float32 | K4 | same_y | Token ridge | 0.08717 | 0.4443 | 0/0 | 10/48 | 0 | 0 |
| float32 | K4 | same_y | Persistence | 0.08717 | 0.5203 | 0/0 | 10/48 | 0 | 0 |
| bfloat16 | K2 | source_rule | Anderson | 0.08764 | 0.9724 | 1/2 | 31/96 | 1 | 3 |
| bfloat16 | K2 | source_rule | Picard | 0.09852 | 0.9692 | 0/2 | 31/96 | 0 | 3 |
| bfloat16 | K2 | source_rule | Full-loop | 0.1777 | 0.7911 | 0/2 | 38/96 | 13 | 7 |
| bfloat16 | K2 | source_rule | Branch-flow | 0.1524 | 0.1737 | 0/2 | 37/96 | 9 | 0 |
| bfloat16 | K2 | source_rule | Prefix ridge | 0.108 | 0.423 | 0/2 | 33/96 | 1 | 4 |
| bfloat16 | K2 | source_rule | Token ridge | 0.1033 | 0.4008 | 0/2 | 40/96 | 0 | 2 |
| bfloat16 | K2 | source_rule | Persistence | 0.1033 | 0.4441 | 0/2 | 28/96 | 0 | 0 |
| bfloat16 | K2 | random | Anderson | 0.1037 | 0.9714 | 2/2 | 36/96 | 4 | 2 |
| bfloat16 | K2 | random | Picard | 0.151 | 0.9635 | 2/2 | 36/96 | 4 | 0 |
| bfloat16 | K2 | random | Full-loop | 0.1771 | 0.6296 | 0/2 | 40/96 | 10 | 6 |
| bfloat16 | K2 | random | Branch-flow | 0.2142 | 0.1026 | 0/2 | 36/96 | 9 | 0 |
| bfloat16 | K2 | random | Prefix ridge | 0.2106 | 0.01314 | 0/2 | 33/96 | 0 | 4 |
| bfloat16 | K2 | random | Token ridge | 0.2016 | 0.01388 | 0/2 | 40/96 | 0 | 2 |
| bfloat16 | K2 | random | Persistence | 0.2016 | 0.02059 | 0/2 | 28/96 | 0 | 0 |
| bfloat16 | K2 | unrelated | Anderson | 0.7999 | 0.8323 | 5/15 | 42/96 | 4 | 3 |
| bfloat16 | K2 | unrelated | Picard | 1.244 | 0.8183 | 6/15 | 43/96 | 4 | 1 |
| bfloat16 | K2 | unrelated | Full-loop | 2.015 | 0.6733 | 4/15 | 43/96 | 6 | 8 |
| bfloat16 | K2 | unrelated | Branch-flow | 1.192 | 0.351 | 4/15 | 41/96 | 19 | 1 |
| bfloat16 | K2 | unrelated | Prefix ridge | 1.421 | 0.02704 | 0/15 | 34/96 | 0 | 2 |
| bfloat16 | K2 | unrelated | Token ridge | 1.42 | 0.03054 | 0/15 | 40/96 | 0 | 2 |
| bfloat16 | K2 | unrelated | Persistence | 1.42 | 0.03443 | 0/15 | 28/96 | 0 | 0 |
| bfloat16 | K2 | same_y | Anderson | 0.0758 | 0.9484 | 0/0 | 22/96 | 2 | 5 |
| bfloat16 | K2 | same_y | Picard | 0.09128 | 0.928 | 0/0 | 24/96 | 1 | 1 |
| bfloat16 | K2 | same_y | Full-loop | 0.1035 | 0.6127 | 0/0 | 23/96 | 5 | 6 |
| bfloat16 | K2 | same_y | Branch-flow | 0.1283 | 0.06417 | 0/0 | 26/96 | 6 | 1 |
| bfloat16 | K2 | same_y | Prefix ridge | 0.09092 | 0.2317 | 0/0 | 37/96 | 2 | 3 |
| bfloat16 | K2 | same_y | Token ridge | 0.0769 | 0.2265 | 0/0 | 32/96 | 0 | 2 |
| bfloat16 | K2 | same_y | Persistence | 0.0769 | 0.2219 | 0/0 | 28/96 | 0 | 0 |
| bfloat16 | K3 | source_rule | Anderson | 0.06289 | 0.9932 | 0/0 | 19/48 | 0 | 2 |
| bfloat16 | K3 | source_rule | Picard | 0.08069 | 0.9912 | 0/0 | 20/48 | 0 | 0 |
| bfloat16 | K3 | source_rule | Full-loop | 0.1197 | 0.9336 | 0/0 | 20/48 | 2 | 2 |
| bfloat16 | K3 | source_rule | Branch-flow | 0.254 | 0.3559 | 0/0 | 18/48 | 2 | 2 |
| bfloat16 | K3 | source_rule | Prefix ridge | 0.2162 | 0.61 | 0/0 | 22/48 | 0 | 0 |
| bfloat16 | K3 | source_rule | Token ridge | 0.221 | 0.5844 | 0/0 | 22/48 | 0 | 0 |
| bfloat16 | K3 | source_rule | Persistence | 0.221 | 0.6573 | 0/0 | 18/48 | 0 | 0 |
| bfloat16 | K3 | random | Anderson | 0.07217 | 0.9923 | 0/0 | 20/48 | 0 | 4 |
| bfloat16 | K3 | random | Picard | 0.06385 | 0.9882 | 0/0 | 21/48 | 0 | 2 |
| bfloat16 | K3 | random | Full-loop | 0.1488 | 0.7599 | 0/0 | 23/48 | 1 | 3 |
| bfloat16 | K3 | random | Branch-flow | 0.3164 | 0.2491 | 0/0 | 19/48 | 3 | 3 |
| bfloat16 | K3 | random | Prefix ridge | 0.2631 | 0.04 | 0/0 | 22/48 | 0 | 0 |
| bfloat16 | K3 | random | Token ridge | 0.2703 | 0.04228 | 0/0 | 22/48 | 0 | 0 |
| bfloat16 | K3 | random | Persistence | 0.2703 | 0.04356 | 0/0 | 18/48 | 0 | 0 |
| bfloat16 | K3 | unrelated | Anderson | 0.0734 | 0.9996 | 3/3 | 19/48 | 2 | 1 |
| bfloat16 | K3 | unrelated | Picard | 0.1045 | 0.997 | 3/3 | 20/48 | 2 | 0 |
| bfloat16 | K3 | unrelated | Full-loop | 4.108 | 0.8837 | 1/3 | 21/48 | 5 | 2 |
| bfloat16 | K3 | unrelated | Branch-flow | 7.554 | 0.6152 | 1/3 | 20/48 | 11 | 2 |
| bfloat16 | K3 | unrelated | Prefix ridge | 8.735 | 0.04595 | 0/3 | 22/48 | 0 | 0 |
| bfloat16 | K3 | unrelated | Token ridge | 8.73 | 0.0652 | 0/3 | 22/48 | 0 | 0 |
| bfloat16 | K3 | unrelated | Persistence | 8.73 | 0.08215 | 0/3 | 18/48 | 0 | 0 |
| bfloat16 | K3 | same_y | Anderson | 0.0639 | 0.9858 | 0/0 | 10/48 | 0 | 1 |
| bfloat16 | K3 | same_y | Picard | 0.06308 | 0.9742 | 0/0 | 10/48 | 0 | 1 |
| bfloat16 | K3 | same_y | Full-loop | 0.07643 | 0.7219 | 0/0 | 9/48 | 1 | 2 |
| bfloat16 | K3 | same_y | Branch-flow | 0.175 | 0.1384 | 0/0 | 12/48 | 2 | 2 |
| bfloat16 | K3 | same_y | Prefix ridge | 0.1255 | 0.3018 | 0/0 | 11/48 | 1 | 3 |
| bfloat16 | K3 | same_y | Token ridge | 0.08454 | 0.2911 | 0/0 | 4/48 | 0 | 0 |
| bfloat16 | K3 | same_y | Persistence | 0.08454 | 0.3322 | 0/0 | 6/48 | 0 | 0 |
| bfloat16 | K4 | source_rule | Anderson | 0.05337 | 0.9939 | 1/2 | 16/48 | 0 | 5 |
| bfloat16 | K4 | source_rule | Picard | 0.05103 | 0.9931 | 1/2 | 17/48 | 0 | 5 |
| bfloat16 | K4 | source_rule | Full-loop | 0.08005 | 0.9667 | 0/2 | 22/48 | 0 | 3 |
| bfloat16 | K4 | source_rule | Branch-flow | 0.1961 | 0.7536 | 0/2 | 15/48 | 1 | 3 |
| bfloat16 | K4 | source_rule | Prefix ridge | 0.1654 | 0.7112 | 0/2 | 24/48 | 0 | 0 |
| bfloat16 | K4 | source_rule | Token ridge | 0.1614 | 0.6825 | 0/2 | 20/48 | 0 | 0 |
| bfloat16 | K4 | source_rule | Persistence | 0.1614 | 0.7907 | 0/2 | 26/48 | 0 | 0 |
| bfloat16 | K4 | random | Anderson | 0.05103 | 0.9935 | 1/1 | 21/48 | 0 | 4 |
| bfloat16 | K4 | random | Picard | 0.05103 | 0.9894 | 1/1 | 18/48 | 0 | 6 |
| bfloat16 | K4 | random | Full-loop | 0.135 | 0.8031 | 0/1 | 24/48 | 0 | 2 |
| bfloat16 | K4 | random | Branch-flow | 0.2011 | 0.6881 | 0/1 | 15/48 | 3 | 2 |
| bfloat16 | K4 | random | Prefix ridge | 0.2516 | 0.0699 | 0/1 | 24/48 | 0 | 0 |
| bfloat16 | K4 | random | Token ridge | 0.247 | 0.06156 | 0/1 | 20/48 | 0 | 0 |
| bfloat16 | K4 | random | Persistence | 0.247 | 0.06095 | 0/1 | 26/48 | 0 | 0 |
| bfloat16 | K4 | unrelated | Anderson | 0.06994 | 0.9984 | 5/5 | 17/48 | 0 | 5 |
| bfloat16 | K4 | unrelated | Picard | 0.1016 | 0.9976 | 4/5 | 16/48 | 0 | 6 |
| bfloat16 | K4 | unrelated | Full-loop | 4.252 | 0.9 | 2/5 | 21/48 | 1 | 4 |
| bfloat16 | K4 | unrelated | Branch-flow | 3.268 | 0.8408 | 2/5 | 17/48 | 4 | 2 |
| bfloat16 | K4 | unrelated | Prefix ridge | 10.94 | 0.05778 | 0/5 | 24/48 | 0 | 0 |
| bfloat16 | K4 | unrelated | Token ridge | 10.93 | 0.07575 | 0/5 | 20/48 | 0 | 0 |
| bfloat16 | K4 | unrelated | Persistence | 10.93 | 0.08971 | 0/5 | 26/48 | 0 | 0 |
| bfloat16 | K4 | same_y | Anderson | 0.05103 | 0.9821 | 0/0 | 11/48 | 0 | 5 |
| bfloat16 | K4 | same_y | Picard | 0.05107 | 0.979 | 0/0 | 11/48 | 0 | 5 |
| bfloat16 | K4 | same_y | Full-loop | 0.07731 | 0.8368 | 0/0 | 13/48 | 0 | 2 |
| bfloat16 | K4 | same_y | Branch-flow | 0.1241 | 0.5486 | 0/0 | 13/48 | 1 | 4 |
| bfloat16 | K4 | same_y | Prefix ridge | 0.1231 | 0.4518 | 0/0 | 14/48 | 0 | 0 |
| bfloat16 | K4 | same_y | Token ridge | 0.1083 | 0.4417 | 0/0 | 10/48 | 0 | 0 |
| bfloat16 | K4 | same_y | Persistence | 0.1083 | 0.5132 | 0/0 | 10/48 | 0 | 0 |

| 精度 | 终点轮 | 比较法 | 平均差 | 95% 组 bootstrap 区间 | 来源组 |
|---|---|---|---|---|---|
| float32 | K2 | Picard | 0.07769 | [0.04041, 0.123] | 48 |
| float32 | K2 | Full-loop | 0.1924 | [0.1378, 0.2492] | 48 |
| float32 | K2 | Branch-flow | 0.1296 | [0.09222, 0.1702] | 48 |
| float32 | K2 | Prefix ridge | 0.09218 | [0.05575, 0.1305] | 48 |
| float32 | K2 | Token ridge | 0.08553 | [0.04941, 0.1234] | 48 |
| float32 | K2 | Persistence | 0.08553 | [0.04941, 0.1234] | 48 |
| float32 | K3 | Picard | 0.009381 | [0.006356, 0.01297] | 24 |
| float32 | K3 | Full-loop | 0.2494 | [0.1488, 0.3653] | 24 |
| float32 | K3 | Branch-flow | 0.4247 | [0.2944, 0.5763] | 24 |
| float32 | K3 | Prefix ridge | 0.4344 | [0.2787, 0.6078] | 24 |
| float32 | K3 | Token ridge | 0.4205 | [0.2632, 0.5971] | 24 |
| float32 | K3 | Persistence | 0.4205 | [0.2632, 0.5971] | 24 |
| float32 | K4 | Picard | 0.002874 | [0.001437, 0.004646] | 24 |
| float32 | K4 | Full-loop | 0.1433 | [0.09277, 0.2008] | 24 |
| float32 | K4 | Branch-flow | 0.2708 | [0.189, 0.3574] | 24 |
| float32 | K4 | Prefix ridge | 0.3234 | [0.224, 0.4362] | 24 |
| float32 | K4 | Token ridge | 0.3176 | [0.2186, 0.43] | 24 |
| float32 | K4 | Persistence | 0.3176 | [0.2186, 0.43] | 24 |
| bfloat16 | K2 | Picard | 0.08298 | [0.01273, 0.1728] | 48 |
| bfloat16 | K2 | Full-loop | 0.1261 | [0.05994, 0.194] | 48 |
| bfloat16 | K2 | Branch-flow | 0.05705 | [0.01143, 0.1028] | 48 |
| bfloat16 | K2 | Prefix ridge | 0.03052 | [-0.01478, 0.07449] | 48 |
| bfloat16 | K2 | Token ridge | 0.02501 | [-0.02019, 0.06928] | 48 |
| bfloat16 | K2 | Persistence | 0.02501 | [-0.02019, 0.06928] | 48 |
| bfloat16 | K3 | Picard | 0.01484 | [0.003478, 0.02596] | 24 |
| bfloat16 | K3 | Full-loop | 0.1967 | [0.09826, 0.3104] | 24 |
| bfloat16 | K3 | Branch-flow | 0.3678 | [0.2366, 0.5183] | 24 |
| bfloat16 | K3 | Prefix ridge | 0.3841 | [0.2304, 0.5593] | 24 |
| bfloat16 | K3 | Token ridge | 0.3712 | [0.2142, 0.5486] | 24 |
| bfloat16 | K3 | Persistence | 0.3712 | [0.2142, 0.5486] | 24 |
| bfloat16 | K4 | Picard | -0.006285 | [-0.01788, 0.005137] | 24 |
| bfloat16 | K4 | Full-loop | 0.09335 | [0.0469, 0.145] | 24 |
| bfloat16 | K4 | Branch-flow | 0.2253 | [0.1516, 0.3069] | 24 |
| bfloat16 | K4 | Prefix ridge | 0.2683 | [0.172, 0.3774] | 24 |
| bfloat16 | K4 | Token ridge | 0.2614 | [0.1658, 0.3696] | 24 |
| bfloat16 | K4 | Persistence | 0.2614 | [0.1658, 0.3696] | 24 |

## 自然状态与成本

自然输出用留出题原生全宽状态、原生候选 logits 和语义真值分别计分。候选一致数排除任一侧并列。原生自然改选是起始轮与终点轮之间的选项变化。时间为固定批次同步测量，含候选向量场求解，峰值为该窗口的 GPU allocated。Anderson 每个物理分支重算 8 次原生函数，原生只算 1 次。相同调用数的其他场仍有实现开销差别。

| 精度 | 终点轮 | 样本族 | 方法 | 题数 | 平均相对 L2 | token P95 | 更新余弦均值 | 最大候选 logit 差中位 | 同原生 | 答对 | 原生自然改选复现 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| float32 | K2 | tableau | Native | 12 | 0 | 0 | 1 | 0 | 12 | 2 | 5/5 |
| float32 | K2 | tableau | Anderson | 12 | 0.1589 | 0.2235 | 0.9691 | 0.276 | 11 | 2 | 5/5 |
| float32 | K2 | tableau | Picard | 12 | 0.1602 | 0.2018 | 0.9683 | 0.2836 | 11 | 2 | 5/5 |
| float32 | K2 | tableau | Full-loop | 12 | 0.3227 | 0.7495 | 0.8544 | 4.548 | 7 | 1 | 2/5 |
| float32 | K2 | tableau | Branch-flow | 12 | 0.8323 | 1.123 | 0.2313 | 1.054 | 7 | 0 | 2/5 |
| float32 | K2 | tableau | Prefix ridge | 12 | 0.5377 | 0.853 | 0.5734 | 0.8794 | 6 | 3 | 2/5 |
| float32 | K2 | twohop_open | Native | 18 | 0 | 0 | 1 | 0 | 18 | 6 | 11/11 |
| float32 | K2 | twohop_open | Anderson | 18 | 0.1699 | 0.2845 | 0.9645 | 0.5116 | 17 | 6 | 11/11 |
| float32 | K2 | twohop_open | Picard | 18 | 0.1767 | 0.285 | 0.9602 | 1.889 | 17 | 7 | 10/11 |
| float32 | K2 | twohop_open | Full-loop | 18 | 0.3069 | 0.6991 | 0.8681 | 1.966 | 12 | 6 | 7/11 |
| float32 | K2 | twohop_open | Branch-flow | 18 | 0.8273 | 1.019 | 0.184 | 6.215 | 8 | 8 | 3/11 |
| float32 | K2 | twohop_open | Prefix ridge | 18 | 0.423 | 0.8425 | 0.7304 | 6.691 | 7 | 7 | 2/11 |
| float32 | K2 | twohop_options | Native | 18 | 0 | 0 | 1 | 0 | 18 | 4 | 7/7 |
| float32 | K2 | twohop_options | Anderson | 18 | 0.166 | 0.2712 | 0.9665 | 0.1609 | 17 | 4 | 7/7 |
| float32 | K2 | twohop_options | Picard | 18 | 0.1746 | 0.2655 | 0.9616 | 0.2599 | 17 | 4 | 7/7 |
| float32 | K2 | twohop_options | Full-loop | 18 | 0.3111 | 0.7122 | 0.8651 | 6.301 | 14 | 5 | 4/7 |
| float32 | K2 | twohop_options | Branch-flow | 18 | 0.8309 | 1.041 | 0.1881 | 1.038 | 5 | 7 | 2/7 |
| float32 | K2 | twohop_options | Prefix ridge | 18 | 0.4498 | 0.8284 | 0.6952 | 0.9287 | 11 | 7 | 2/7 |
| float32 | K2 | arc | Native | 32 | 0 | 0 | 1 | 0 | 32 | 28 | 17/17 |
| float32 | K2 | arc | Anderson | 32 | 0.05767 | 0.0932 | 0.9953 | 0.1722 | 31 | 28 | 16/17 |
| float32 | K2 | arc | Picard | 32 | 0.09025 | 0.176 | 0.9886 | 0.5682 | 32 | 28 | 17/17 |
| float32 | K2 | arc | Full-loop | 32 | 0.2634 | 0.4729 | 0.9167 | 2.516 | 28 | 27 | 14/17 |
| float32 | K2 | arc | Branch-flow | 32 | 0.6203 | 0.8611 | 0.5124 | 3.572 | 20 | 20 | 11/17 |
| float32 | K2 | arc | Prefix ridge | 32 | 0.6341 | 0.8161 | 0.3394 | 4.125 | 14 | 14 | 6/17 |
| float32 | K3 | tableau | Native | 12 | 0 | 0 | 1 | 0 | 12 | 1 | 6/6 |
| float32 | K3 | tableau | Anderson | 12 | 0.009008 | 0.01263 | 0.9998 | 0.03403 | 12 | 1 | 6/6 |
| float32 | K3 | tableau | Picard | 12 | 0.02509 | 0.04281 | 0.9987 | 0.1204 | 12 | 1 | 6/6 |
| float32 | K3 | tableau | Full-loop | 12 | 0.2036 | 0.3253 | 0.9473 | 3.536 | 11 | 2 | 5/6 |
| float32 | K3 | tableau | Branch-flow | 12 | 0.8296 | 1.028 | 0.1561 | 9.021 | 7 | 1 | 4/6 |
| float32 | K3 | tableau | Prefix ridge | 12 | 0.652 | 0.891 | 0.2875 | 9.303 | 5 | 3 | 1/6 |
| float32 | K3 | twohop_open | Native | 18 | 0 | 0 | 1 | 0 | 18 | 7 | 10/10 |
| float32 | K3 | twohop_open | Anderson | 18 | 0.01061 | 0.01606 | 0.9998 | 0.01491 | 18 | 7 | 10/10 |
| float32 | K3 | twohop_open | Picard | 18 | 0.06904 | 0.1169 | 0.9899 | 0.06083 | 18 | 7 | 10/10 |
| float32 | K3 | twohop_open | Full-loop | 18 | 0.1958 | 0.3004 | 0.9473 | 2.169 | 17 | 7 | 9/10 |
| float32 | K3 | twohop_open | Branch-flow | 18 | 0.7766 | 0.9096 | 0.1619 | 10.78 | 7 | 7 | 4/10 |
| float32 | K3 | twohop_open | Prefix ridge | 18 | 0.6153 | 0.891 | 0.3112 | 5.174 | 10 | 7 | 5/10 |
| float32 | K3 | twohop_options | Native | 18 | 0 | 0 | 1 | 0 | 18 | 5 | 6/6 |
| float32 | K3 | twohop_options | Anderson | 18 | 0.01039 | 0.01663 | 0.9998 | 0.01268 | 18 | 5 | 6/6 |
| float32 | K3 | twohop_options | Picard | 18 | 0.06753 | 0.111 | 0.9908 | 0.1072 | 18 | 5 | 6/6 |
| float32 | K3 | twohop_options | Full-loop | 18 | 0.1991 | 0.3026 | 0.9474 | 4.238 | 15 | 5 | 3/6 |
| float32 | K3 | twohop_options | Branch-flow | 18 | 0.7805 | 0.9269 | 0.1691 | 11.37 | 9 | 8 | 3/6 |
| float32 | K3 | twohop_options | Prefix ridge | 18 | 0.6138 | 0.8844 | 0.3197 | 11.2 | 13 | 3 | 3/6 |
| float32 | K3 | arc | Native | 32 | 0 | 0 | 1 | 0 | 32 | 29 | 2/2 |
| float32 | K3 | arc | Anderson | 32 | 0.04187 | 0.0357 | 0.9942 | 0.04629 | 32 | 29 | 2/2 |
| float32 | K3 | arc | Picard | 32 | 0.06335 | 0.1315 | 0.9867 | 0.2575 | 31 | 30 | 2/2 |
| float32 | K3 | arc | Full-loop | 32 | 0.1583 | 0.2777 | 0.9428 | 0.4417 | 31 | 30 | 2/2 |
| float32 | K3 | arc | Branch-flow | 32 | 0.485 | 0.7201 | 0.5492 | 1.964 | 28 | 27 | 2/2 |
| float32 | K3 | arc | Prefix ridge | 32 | 0.6765 | 0.8333 | 0.1628 | 6.307 | 26 | 24 | 0/2 |
| float32 | K4 | tableau | Native | 12 | 0 | 0 | 1 | 0 | 12 | 2 | 1/1 |
| float32 | K4 | tableau | Anderson | 12 | 0.05051 | 0.01275 | 0.9828 | 0.03712 | 12 | 2 | 1/1 |
| float32 | K4 | tableau | Picard | 12 | 0.05297 | 0.03456 | 0.9811 | 0.05766 | 12 | 2 | 1/1 |
| float32 | K4 | tableau | Full-loop | 12 | 0.1101 | 0.1946 | 0.9404 | 0.5344 | 12 | 2 | 1/1 |
| float32 | K4 | tableau | Branch-flow | 12 | 0.5983 | 0.8601 | 0.3553 | 4.93 | 10 | 1 | 1/1 |
| float32 | K4 | tableau | Prefix ridge | 12 | 0.6851 | 0.8742 | 0.08681 | 2.523 | 9 | 2 | 0/1 |
| float32 | K4 | twohop_open | Native | 18 | 0 | 0 | 1 | 0 | 18 | 8 | 3/3 |
| float32 | K4 | twohop_open | Anderson | 18 | 0.05127 | 0.01303 | 0.9814 | 0.02249 | 18 | 8 | 3/3 |
| float32 | K4 | twohop_open | Picard | 18 | 0.06231 | 0.07343 | 0.9731 | 0.0751 | 17 | 7 | 3/3 |
| float32 | K4 | twohop_open | Full-loop | 18 | 0.1074 | 0.188 | 0.9443 | 0.4697 | 17 | 8 | 2/3 |
| float32 | K4 | twohop_open | Branch-flow | 18 | 0.4584 | 0.6143 | 0.4326 | 5.355 | 12 | 9 | 0/3 |
| float32 | K4 | twohop_open | Prefix ridge | 18 | 0.6832 | 0.9153 | 0.04897 | 2.245 | 11 | 6 | 0/3 |
| float32 | K4 | twohop_options | Native | 18 | 0 | 0 | 1 | 0 | 18 | 6 | 6/6 |
| float32 | K4 | twohop_options | Anderson | 18 | 0.04938 | 0.01528 | 0.9841 | 0.02449 | 18 | 6 | 6/6 |
| float32 | K4 | twohop_options | Picard | 18 | 0.06151 | 0.07569 | 0.9757 | 0.04202 | 17 | 6 | 6/6 |
| float32 | K4 | twohop_options | Full-loop | 18 | 0.1121 | 0.2062 | 0.9437 | 0.4514 | 14 | 5 | 3/6 |
| float32 | K4 | twohop_options | Branch-flow | 18 | 0.4647 | 0.6349 | 0.4438 | 4.193 | 13 | 6 | 3/6 |
| float32 | K4 | twohop_options | Prefix ridge | 18 | 0.6815 | 0.904 | 0.04728 | 1.376 | 10 | 9 | 2/6 |
| float32 | K4 | arc | Native | 32 | 0 | 0 | 1 | 0 | 32 | 28 | 3/3 |
| float32 | K4 | arc | Anderson | 32 | 0.06423 | 0.04657 | 0.9654 | 0.09661 | 32 | 28 | 3/3 |
| float32 | K4 | arc | Picard | 32 | 0.09162 | 0.1722 | 0.9302 | 0.3693 | 31 | 29 | 2/3 |
| float32 | K4 | arc | Full-loop | 32 | 0.1262 | 0.2341 | 0.8783 | 0.3241 | 30 | 30 | 1/3 |
| float32 | K4 | arc | Branch-flow | 32 | 0.449 | 0.6394 | 0.359 | 1.937 | 28 | 28 | 1/3 |
| float32 | K4 | arc | Prefix ridge | 32 | 0.6839 | 0.853 | 0.04974 | 6.336 | 29 | 28 | 2/3 |
| bfloat16 | K2 | tableau | Native | 12 | 0 | 0 | 1 | 0 | 12 | 2 | 5/5 |
| bfloat16 | K2 | tableau | Anderson | 12 | 0.1618 | 0.2252 | 0.968 | 0.3486 | 11 | 2 | 5/5 |
| bfloat16 | K2 | tableau | Picard | 12 | 0.1632 | 0.2103 | 0.9672 | 0.3535 | 11 | 2 | 5/5 |
| bfloat16 | K2 | tableau | Full-loop | 12 | 0.3232 | 0.7473 | 0.854 | 4.552 | 7 | 1 | 2/5 |
| bfloat16 | K2 | tableau | Branch-flow | 12 | 0.8329 | 1.12 | 0.2305 | 1.01 | 7 | 0 | 2/5 |
| bfloat16 | K2 | tableau | Prefix ridge | 12 | 0.5377 | 0.8519 | 0.5736 | 0.8828 | 6 | 3 | 2/5 |
| bfloat16 | K2 | twohop_open | Native | 18 | 0 | 0 | 1 | 0 | 18 | 6 | 11/11 |
| bfloat16 | K2 | twohop_open | Anderson | 18 | 0.1707 | 0.275 | 0.9641 | 0.6016 | 17 | 6 | 11/11 |
| bfloat16 | K2 | twohop_open | Picard | 18 | 0.177 | 0.2895 | 0.9602 | 2.062 | 17 | 7 | 10/11 |
| bfloat16 | K2 | twohop_open | Full-loop | 18 | 0.3071 | 0.6996 | 0.8685 | 2.109 | 12 | 6 | 7/11 |
| bfloat16 | K2 | twohop_open | Branch-flow | 18 | 0.8276 | 1.013 | 0.1833 | 6.443 | 8 | 8 | 3/11 |
| bfloat16 | K2 | twohop_open | Prefix ridge | 18 | 0.4233 | 0.8424 | 0.7308 | 6.986 | 7 | 8 | 2/11 |
| bfloat16 | K2 | twohop_options | Native | 18 | 0 | 0 | 1 | 0 | 18 | 4 | 7/7 |
| bfloat16 | K2 | twohop_options | Anderson | 18 | 0.1681 | 0.2726 | 0.9656 | 0.2266 | 17 | 4 | 7/7 |
| bfloat16 | K2 | twohop_options | Picard | 18 | 0.1781 | 0.271 | 0.96 | 0.2832 | 17 | 4 | 7/7 |
| bfloat16 | K2 | twohop_options | Full-loop | 18 | 0.3109 | 0.7138 | 0.8656 | 6.261 | 14 | 5 | 4/7 |
| bfloat16 | K2 | twohop_options | Branch-flow | 18 | 0.8318 | 1.042 | 0.1869 | 1.004 | 5 | 7 | 2/7 |
| bfloat16 | K2 | twohop_options | Prefix ridge | 18 | 0.4501 | 0.8313 | 0.695 | 0.8906 | 11 | 7 | 2/7 |
| bfloat16 | K2 | arc | Native | 32 | 0 | 0 | 1 | 0 | 30 | 28 | 16/16 |
| bfloat16 | K2 | arc | Anderson | 32 | 0.06081 | 0.09495 | 0.9948 | 0.1875 | 30 | 28 | 16/16 |
| bfloat16 | K2 | arc | Picard | 32 | 0.09342 | 0.1818 | 0.9879 | 0.625 | 30 | 28 | 16/16 |
| bfloat16 | K2 | arc | Full-loop | 32 | 0.2637 | 0.4757 | 0.9164 | 2.562 | 27 | 27 | 13/16 |
| bfloat16 | K2 | arc | Branch-flow | 32 | 0.6208 | 0.862 | 0.5116 | 3.719 | 19 | 20 | 10/16 |
| bfloat16 | K2 | arc | Prefix ridge | 32 | 0.634 | 0.8185 | 0.3396 | 4.062 | 13 | 14 | 5/16 |
| bfloat16 | K3 | tableau | Native | 12 | 0 | 0 | 1 | 0 | 12 | 1 | 6/6 |
| bfloat16 | K3 | tableau | Anderson | 12 | 0.01825 | 0.02467 | 0.9993 | 0.125 | 12 | 1 | 6/6 |
| bfloat16 | K3 | tableau | Picard | 12 | 0.02922 | 0.04694 | 0.9983 | 0.1875 | 12 | 1 | 6/6 |
| bfloat16 | K3 | tableau | Full-loop | 12 | 0.2039 | 0.3273 | 0.9471 | 3.531 | 11 | 2 | 5/6 |
| bfloat16 | K3 | tableau | Branch-flow | 12 | 0.831 | 1.03 | 0.1553 | 9.031 | 7 | 1 | 4/6 |
| bfloat16 | K3 | tableau | Prefix ridge | 12 | 0.6526 | 0.889 | 0.2871 | 9.281 | 5 | 3 | 1/6 |
| bfloat16 | K3 | twohop_open | Native | 18 | 0 | 0 | 1 | 0 | 17 | 7 | 9/9 |
| bfloat16 | K3 | twohop_open | Anderson | 18 | 0.02074 | 0.02816 | 0.9991 | 0.0625 | 17 | 7 | 9/9 |
| bfloat16 | K3 | twohop_open | Picard | 18 | 0.07438 | 0.1253 | 0.9882 | 0.125 | 17 | 7 | 9/9 |
| bfloat16 | K3 | twohop_open | Full-loop | 18 | 0.1957 | 0.2982 | 0.9465 | 2 | 15 | 6 | 7/9 |
| bfloat16 | K3 | twohop_open | Branch-flow | 18 | 0.7808 | 0.9093 | 0.1596 | 10.75 | 7 | 6 | 3/9 |
| bfloat16 | K3 | twohop_open | Prefix ridge | 18 | 0.616 | 0.8924 | 0.3095 | 5.094 | 9 | 6 | 4/9 |
| bfloat16 | K3 | twohop_options | Native | 18 | 0 | 0 | 1 | 0 | 18 | 5 | 6/6 |
| bfloat16 | K3 | twohop_options | Anderson | 18 | 0.02182 | 0.0288 | 0.999 | 0.0625 | 17 | 5 | 5/6 |
| bfloat16 | K3 | twohop_options | Picard | 18 | 0.06637 | 0.1078 | 0.991 | 0.125 | 18 | 5 | 6/6 |
| bfloat16 | K3 | twohop_options | Full-loop | 18 | 0.1994 | 0.3018 | 0.9472 | 4.25 | 15 | 5 | 3/6 |
| bfloat16 | K3 | twohop_options | Branch-flow | 18 | 0.7835 | 0.9328 | 0.1665 | 11.37 | 9 | 8 | 3/6 |
| bfloat16 | K3 | twohop_options | Prefix ridge | 18 | 0.6143 | 0.8799 | 0.3192 | 11.18 | 13 | 3 | 3/6 |
| bfloat16 | K3 | arc | Native | 32 | 0 | 0 | 1 | 0 | 31 | 29 | 1/1 |
| bfloat16 | K3 | arc | Anderson | 32 | 0.04855 | 0.04428 | 0.9922 | 0.125 | 31 | 29 | 1/1 |
| bfloat16 | K3 | arc | Picard | 32 | 0.06751 | 0.1319 | 0.9849 | 0.3125 | 31 | 29 | 1/1 |
| bfloat16 | K3 | arc | Full-loop | 32 | 0.1589 | 0.28 | 0.9421 | 0.4375 | 31 | 29 | 1/1 |
| bfloat16 | K3 | arc | Branch-flow | 32 | 0.4846 | 0.7198 | 0.5498 | 1.969 | 28 | 27 | 1/1 |
| bfloat16 | K3 | arc | Prefix ridge | 32 | 0.6768 | 0.8341 | 0.1625 | 6.312 | 25 | 24 | 0/1 |
| bfloat16 | K4 | tableau | Native | 12 | 0 | 0 | 1 | 0 | 12 | 2 | 1/1 |
| bfloat16 | K4 | tableau | Anderson | 12 | 0.05258 | 0.02205 | 0.9814 | 0.125 | 12 | 2 | 1/1 |
| bfloat16 | K4 | tableau | Picard | 12 | 0.05478 | 0.03723 | 0.9798 | 0.09375 | 12 | 2 | 1/1 |
| bfloat16 | K4 | tableau | Full-loop | 12 | 0.1106 | 0.2028 | 0.9397 | 0.5469 | 12 | 2 | 1/1 |
| bfloat16 | K4 | tableau | Branch-flow | 12 | 0.5948 | 0.8565 | 0.3595 | 4.844 | 10 | 1 | 1/1 |
| bfloat16 | K4 | tableau | Prefix ridge | 12 | 0.6863 | 0.8755 | 0.08572 | 2.463 | 9 | 2 | 0/1 |
| bfloat16 | K4 | twohop_open | Native | 18 | 0 | 0 | 1 | 0 | 18 | 8 | 3/3 |
| bfloat16 | K4 | twohop_open | Anderson | 18 | 0.05356 | 0.02203 | 0.9799 | 0.0625 | 18 | 8 | 3/3 |
| bfloat16 | K4 | twohop_open | Picard | 18 | 0.06512 | 0.07821 | 0.971 | 0.125 | 18 | 8 | 3/3 |
| bfloat16 | K4 | twohop_open | Full-loop | 18 | 0.1086 | 0.1923 | 0.9426 | 0.4688 | 17 | 8 | 2/3 |
| bfloat16 | K4 | twohop_open | Branch-flow | 18 | 0.4603 | 0.6185 | 0.4315 | 5.344 | 13 | 8 | 1/3 |
| bfloat16 | K4 | twohop_open | Prefix ridge | 18 | 0.6819 | 0.912 | 0.05216 | 2.25 | 11 | 6 | 0/3 |
| bfloat16 | K4 | twohop_options | Native | 18 | 0 | 0 | 1 | 0 | 17 | 6 | 6/6 |
| bfloat16 | K4 | twohop_options | Anderson | 18 | 0.05166 | 0.02565 | 0.9826 | 0.0625 | 17 | 6 | 6/6 |
| bfloat16 | K4 | twohop_options | Picard | 18 | 0.06344 | 0.07659 | 0.9742 | 0.0625 | 17 | 6 | 6/6 |
| bfloat16 | K4 | twohop_options | Full-loop | 18 | 0.112 | 0.203 | 0.9435 | 0.4688 | 14 | 5 | 3/6 |
| bfloat16 | K4 | twohop_options | Branch-flow | 18 | 0.4641 | 0.6276 | 0.4443 | 4.188 | 12 | 6 | 3/6 |
| bfloat16 | K4 | twohop_options | Prefix ridge | 18 | 0.6823 | 0.9059 | 0.04695 | 1.438 | 9 | 8 | 1/6 |
| bfloat16 | K4 | arc | Native | 32 | 0 | 0 | 1 | 0 | 32 | 28 | 2/2 |
| bfloat16 | K4 | arc | Anderson | 32 | 0.06828 | 0.05081 | 0.9607 | 0.125 | 32 | 28 | 2/2 |
| bfloat16 | K4 | arc | Picard | 32 | 0.09 | 0.1727 | 0.9326 | 0.3125 | 31 | 29 | 1/2 |
| bfloat16 | K4 | arc | Full-loop | 32 | 0.1256 | 0.2312 | 0.8794 | 0.375 | 30 | 30 | 0/2 |
| bfloat16 | K4 | arc | Branch-flow | 32 | 0.4501 | 0.6388 | 0.3635 | 1.938 | 28 | 28 | 0/2 |
| bfloat16 | K4 | arc | Prefix ridge | 32 | 0.6837 | 0.8526 | 0.04988 | 6.312 | 29 | 28 | 1/2 |

| 精度 | 方法 | 同步墙钟秒中位 | 峰值分配 GiB P95 | Attention 中位 | MLP 中位 |
|---|---|---|---|---|---|
| float32 | causal/native | 0.1418 | 7.142 | 24 | 24 |
| float32 | causal/chart_anderson | 1.4 | 7.357 | 192 | 192 |
| float32 | causal/chart | 1.186 | 7.11 | 192 | 192 |
| float32 | causal/full_loop | 1.151 | 7.186 | 192 | 192 |
| float32 | causal/branch_flow | 1.177 | 7.168 | 192 | 192 |
| float32 | causal/prefix_ridge | 0.0009849 | 7.009 | 0 | 0 |
| float32 | causal/ridge | 0.0004595 | 6.994 | 0 | 0 |
| float32 | causal/persistence | 7.019e-06 | 6.98 | 0 | 0 |
| float32 | natural/native | 0.04915 | 5.697 | 24 | 24 |
| float32 | natural/chart_anderson | 0.5499 | 5.803 | 192 | 192 |
| float32 | natural/chart | 0.4156 | 5.715 | 192 | 192 |
| float32 | natural/full_loop | 0.3926 | 5.736 | 192 | 192 |
| float32 | natural/branch_flow | 0.4017 | 5.724 | 192 | 192 |
| float32 | natural/prefix_ridge | 0.0004219 | 5.666 | 0 | 0 |
| float32 | natural/ridge | 0.000226 | 5.657 | 0 | 0 |
| float32 | natural/persistence | 9.618e-06 | 5.647 | 0 | 0 |
| bfloat16 | causal/native | 0.04242 | 3.622 | 24 | 24 |
| bfloat16 | causal/chart_anderson | 0.6372 | 3.952 | 192 | 192 |
| bfloat16 | causal/chart | 0.421 | 3.697 | 192 | 192 |
| bfloat16 | causal/full_loop | 0.3413 | 3.687 | 192 | 192 |
| bfloat16 | causal/branch_flow | 0.41 | 3.748 | 192 | 192 |
| bfloat16 | causal/prefix_ridge | 0.000895 | 3.584 | 0 | 0 |
| bfloat16 | causal/ridge | 0.0004288 | 3.57 | 0 | 0 |
| bfloat16 | causal/persistence | 8.568e-06 | 3.541 | 0 | 0 |
| bfloat16 | natural/native | 0.02331 | 2.877 | 24 | 24 |
| bfloat16 | natural/chart_anderson | 0.3886 | 3.012 | 192 | 192 |
| bfloat16 | natural/chart | 0.2547 | 2.924 | 192 | 192 |
| bfloat16 | natural/full_loop | 0.1858 | 2.918 | 192 | 192 |
| bfloat16 | natural/branch_flow | 0.2493 | 2.937 | 192 | 192 |
| bfloat16 | natural/prefix_ridge | 0.0004266 | 2.876 | 0 | 0 |
| bfloat16 | natural/ridge | 0.0002376 | 2.869 | 0 | 0 |
| bfloat16 | natural/persistence | 1.085e-05 | 2.854 | 0 | 0 |

[冻结清单](results/FREEZE.json) · [汇总](results/SUMMARY.json) · [图 1](results/figures/figure_1_early_native_semantic_outcomes.pdf) · [图 2](results/figures/figure_2_causal_response_controls.pdf) · [图 3](results/figures/figure_3_paired_loop_transfer.pdf) · [上一轮六份报告与七图](../ouro_continuous_embedding_20261120/README.md)

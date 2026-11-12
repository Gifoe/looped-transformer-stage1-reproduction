# ORDER_E1_ADD_CONTROLS_v1

Astra 正式阶段执行令。控制依据只来自 Train，Validation/Test 不用于选择方向。原始实际协议、权重、BF16、SDPA、完整形状、25-shot、no-cache 与原评分不变。

## 受体与四条件

对象：固定 K1 receiver L18/index17/ut0 的 SDPA_R；q=1 端点为旧 G122，q=2 端点为旧 G222。记同 Query 背景旧实际 SDPA_R 为 S_qjk。每题对 q=1/2 各执行四条真实后缀条件：

1. **ID_q**：注入已审 S_q22；完整 logits、a_R、scores 与旧角逐位相等。
2. **ADD_q**：A_q=BF16(FP64(S_q21)+FP64(S_q12)−FP64(S_q11))。
3. **ADD_NORM_q**：将 FP64(A_q) 缩放至 ||FP64(S_q22)|| 后一次 cast BF16；目标范数、实际范数及误差均保存。
4. **UNRELATED_q**：有效移除向量 d_q=FP64(S_q22)−FP64(A_q)。从冻结 Train 非示例中选不同题、不同 input_hash 的同 q 向量 d_j，按 ||d_q||/||d_j|| 缩放，注入 BF16(S_q22−scaled_d_j)。donor 用 sha256(receiver split/id/q/固定 seed) 决定循环索引；遇同输入或零 donor，按已冻结顺序向后找，不按 gold 或效果选择。

若 d_q=0，相关条件应退化 identity 并核验。若 ADD 范数为零但目标非零，norm 条件明确记 UNDEFINED，保留该题主条件，不用 epsilon 伪造向量。若两者均零，ADD_NORM 按 identity 零向量处理并记录退化。

## Restore 与数值控制

FP64 构造 A_q+d_q 后 cast，须逐位等于 S_q22。固定三题在真实 GPU 对各 q 另跑 restore，完整词表与 ID/旧角零差。

全量每题保存 restore_tensor 位等校验；已经具有相同注入字节的 restore 不再重复 GPU。明确这是同输入恒等及实现验证，不是独立的自然机制恢复证据。

总计每题八个 K1 后缀前向；三题技术验收额外六个 restore。不重做旧八角 SDPA 组合，不用新结果重新选择基线。实际 QKV/A 重构和缺失阶段捕获可共享本批前向，但须单列实际采集范围；没有捕获的其他旧角中间状态不得声称已得到。

## 主要终点与保存

令 F 为某一固定标量评分，终端交互逐题恒等分解：

I_terminal = (F22−Fadd) + (Fadd−F21−F12+F11)。

分别对每个候选、固定错误候选 gap、全部错误候选题内平均 gap、动态最强错误 margin 计算。预测救回/损伤单独报告，不将 accuracy 作可加分解。

控制比较为 paired[(F22−Fadd)−(F22−Funrelated)] 与 F22−Fadd_norm，不用不稳定的 I 比值宣称解释百分比。无关来源匹配的是移除 delta 幅度，不声称同时匹配输出范数；ADD_NORM 是单独的尺度敏感性控制。

保存旧实际 SDPA 四角（或精确引用其既有字节与 SHA）、FP64 sum、BF16 ADD、cast 残差、有效 d、donor 身份与来源 SHA、实际注入 a_R、完整最终 logits 和 scores。实际 SDPA 交互 I_nat 包含有限精度效应，须与 FP64 理想分解及残差分开。F22−Fadd 是该人工受体中的条件因果差，不能直接称为自然 K2 必要性。

## 顺序与预算

固定原 A/B/C 三题只作技术验收，不解读新条件效果来选样本。source/model/input/真实 hook scope、ID/restore 端点及独立 CUDA 回放全部通过后冻结源码。

随后只运行完整 Train1119，25 个示例单列、1094 非示例为主要探索。完整审计及分析后，由 Astra 冻结 Validation/Test 判据，再执行 Validation299→Test1172。当前不启动 E3/E4 目标定位，不扫描全部层/head/token。

按三题实测的每题八条后缀、产物 bytes、CUDA allocated/reserved 与 CPU RAM 外推资源；共享 GPU 下的时间不是保证。无 40 GiB 预留，不停止其他 GPU 任务，不恢复 HF 大文件上传。无需等待额外无关工程即可进入本令的实现与技术验收。

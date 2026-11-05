# A4 采集源码独立审查

2026-11-05，Astra。只读审查 prepare_a4.py、collect_a4.py、supervise_a4.py、a4_cpu_checks.py 及 CPU plan/counts/receipt；未运行模型/GPU。

结论：**采集语义 conditional pass**。最终输入/源码哈希与 CPU receipt 一致、Root 资源审核和冻结 readback 完成后，可由 Root 单独决定启动；本文不授权启动。审查时 Sol 正补输入 provenance，当前 receipt 与更新 plan 的 SHA 尚未一致，不能使用旧 receipt 证明新稿。

## 已满足

1. 按 discovery384、selection192 的固定顺序，default_rng(2021).choice(576,64,replace=False)，再排序原下标。采集无基于正确性的重抽；计划与 worker 不携带 gold，不调用 argmax。
2. 真实 depth4 返回四个 normalized state，每个原生 BF16 head 转 FP32，再完整词表 log_softmax、target gather 和原 sum。不是从 K1/K4 插值构造 K2/K3。
3. 两个固定历史 parity 题的全部 8 个 rotations，独立 depth2/3 最后 state 与 head 对 depth4 相应读出作无损 hash exact，另核 target LL exact。
4. 全64的255 rotations，每个候选 K1/K4/Fixed 均对历史 R10 depth5 原生读出 strict equality。该 gate 覆盖全部新采集，而非只有两题。历史深度差若造成任何端点差异即失败，不放宽公差。
5. continuation 在当前 plan 全为一 token，positions 是 input 最后位置；同输入字母任务去重合法。完整 279 body calls、1092 native heads 的计数与计划一致；Fixed 的 log-softmax不额外算 native head。
6. 保存全部四深度选定位置 BF16 states；独立 parity 保存 FP32 fullV heads。269 参数前后相同、hook 数相同；未添加 hook、KV 或显式 mask。
7. 仅依据 parity runtime 决定是否有时间进入完整64，不依据结果表现。在进入64前拒绝时只能报告 parity-only；中途不完整不应分析部分64轨迹。

## 冻结前收口

Root 已要求的 discovery/selection、历史 score 与 COMPLETE receipt 的 provenance 哈希应纳入，并重跑 CPU 检查。建议完整路径还 assert head_calls==1092，避免日后计数漂移。分析必须先验证 COMPLETE、supervisor clean、raw/NPZ SHA、64 ID 与每rotation完整schema/有限性，再读 gold；模型产出本身不构成分析入口授权。

轨迹定义事前固定，原顺序 joint 和 cyclic 分开：

* first_correct_K：实际 K1..4 中首次预测正确；没有则 NA，回溯性统计。
* first_change_K：首次 k≥2 且 y[k]≠y[k−1]；没有则 NA。
* first_local_stable_confirmed_K：首次 k≥2 且 y[k]=y[k−1]，起点另存 k−1。这不保证之后不变。
* persistent_stable_start_K：至少包含两个实际深度的最早不变后缀起点；若只有 K4 单点则 NA。该定义需要未来轨迹，仅用于回溯描述。
* later_reversals：首次局部稳定确认之后，所有相邻预测变化的数量；尚未局部稳定则 NA。另存全部相邻变化数，以及返回过去选项的次数，不混称。

若 Sol 采用等价但名称不同的定义，需在效果出现前固定并明确分母/NA。两端历史 K1/K4 表仍标 gap3，不能因新64具有连续深度便给原1172插入中间状态。

本次64是固定、已多次暴露的训练开发样本上的描述性轨迹。任何结果都不能外推全1172的具体纠错循环分布，也不证明隐藏态因果角色。500秒上限在剩765.61秒内，但 runtime forecast 不是完成保证；Root 负责最终可完成性与进程/账本审查。

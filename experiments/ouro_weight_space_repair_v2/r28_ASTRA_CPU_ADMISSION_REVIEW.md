# R28 Astra CPU 准入审查

状态：CPU evidence review only；没有新 forward、模型加载、GPU 或方法/门槛修改。本文件不代替 Root 的源码审查、公示冻结与明确 START。

## 决定

**现有 CPU 证据解决了这24题的 true-prefix 输入构造与 exact-input 去重可行性问题。建议保留原拟议一次 hard1500秒预算，进入源码/合成验收；不保证该时限内完成。** 不需要缩 cohort、改回 teacher-forced 数值或变更主方法。

已读 configs/r28_TRUE_PREFIX_TASK_INVENTORY.json，以及独立 results/r28_ROOT_PREFIX_GRAPH_VERIFICATION.json。独立收据绑定 manifest SHA256 `a70f50e70abf8af9a9af52d31271e8f8b7936c5ebe9214edfb3da2d37c3cf9f0`：24题、96候选、548 prefix uses、389 reference groups、2366任务；UQ=421，U0=U1=U2=389（每个N1/N2含 native/CF 两任务）。总输入2186066 token，depth-weighted9216897。143 source 位置、已消费参考文本、输入 tokenizer roundtrip、native/CF 同输入、任务哈希与依赖均核验通过。完整 target-map 身份核验没有报告边界反例。

因此此前“是否需要未来字符才能完成参考位置对齐”的执行疑问，在这份固定任务图范围内已经消除。跨题共享来源于完全相同的已知输入，不是内容近似或未来长度泄露。该收据不证明任意未来输入皆同样可行；独立审计明确限定为保存图，原数据提取与 lm-eval 源码 parity 仍由 generator CPU receipt 支撑，不能扩大成已独立重跑全部数据提取。

## 预算判断

按已提出、未调参的 planning 公式：

`T=.405*421+.0739125*389+.650*(389+389)=704.9569625秒`。

`1.25*T+100=981.196203125秒`，满足先验 CPU 准入<=1400秒。hard1500相对该估算约留518.8秒；soft1485相对估算约留503.8秒。退出仍为 soft1485 TERM /1499 KILL /余1秒 reap，内部 reserve5不改。全部 worker wall 含加载、哈希、检查、IO和退出计费。

这只是基于旧每题四候选成本迁移的工程预估：prefix 长度、串行小调用、hook同步、fullV处理和流式管理可能改变耗时。25%系数及100秒不是实测置信上界；不能声称必然完成。无另行预热/模型 profile；不完整时不解封 partial 效果、不拼接补跑。继承16 ledger 总18930.428655151278，余额2669.571344848722；1500秒上限在余额内，名义剩1169.571344848722，任何真实超时仍完整计费。完整后续576+517+299+1172仍未获预算或执行授权。

## 编程完成后的关键运行时审查点

1. **真的执行保存的前缀**：模型每次 input IDs、总长度、depth、单位置 head `[1,1,2048]` 与任务图绑定。不能偷偷将原 full-candidate 输入或旧 Q cached states 接回；target 只能在完整 fullV方向和系数确定后 gather。
2. **donor 与 clock 正确**：CF 每层 loop4取对应 native forward 的同长度完整块，不能取已受上游干预污染的当前 CF V；前三 loop、未改行、身份 donor、capture 无扰动必须按原合同 exact。所有24层调用计数/时钟、scorer invalidate、finally cleanup 与参数前后哈希必须完整。
3. **流式复用不串数据**：native/CF 不共用错误缓存项；每个 reference group 的两个 donor、fullV和系数只服务其依赖的 Q。复用以完整 input/mode/depth/source-map键确定，不取决于题号、候选遍历或 gold。每个548 use恰好入账一次，96候选完整恢复；真实 duplicate replay 检查仍需运行时证据。
4. **匹配数值而非历史假 parity**：独立 NumPy/Torch formula、退化/零效应、完整词表 log_softmax、原字符归一化及原 late LL差分别验收。true-prefix baselines 从实际新 readout重建；历史 R6/M3数值不同不能伪装成实现错误，也不能归功于来源干预。两个 R6保护集、strong 与 cost-review布尔值保持原设计独立。
5. **先完整性后效果**：全部任务及 required evidence、controls norm-matching、无 unmatched、资源/参数/清理均完整后才读取标签与 argmax。任何 runtime 失败只给 invalid/incomplete结论；不添加“先看一部分再决定”的新分支。

这些是原设计所需证据的落实，不是追加方法候选、效果门槛或额外模型任务。待源码及合成 receipt 到位再逐项静态审查；当前 CPU PASS 仅允许继续无 GPU 准备。

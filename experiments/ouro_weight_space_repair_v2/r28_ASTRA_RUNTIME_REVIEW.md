# R28 Astra runtime 最终合同审查

状态：静态审查，conditional pass；无 GPU/模型执行。此审查未发现阻断正确性或完整性错误，不是 START 授权。最终 draft 和实际 tokenizer-only preflight 在本次读取时尚未落盘，因此其最终哈希绑定仍待 Root readback。

审读 worker、math_v2、hooks、common、preflight、stream、supervise、analyze、prepare，并核对 MATH_HOOK_CHECKS_V2 / STREAM_CHECKS 两份 CPU receipt。两份 receipt 的所有 source_sha256 均由本次只读 Get-FileHash 核验，与当前源码一致。Math CPU报告 FP32输出差0、FP64诊断最大差2.7755575615628914e-16；stream合成覆盖完整任务/位置buffer与先完整性后gold。本次没有重新执行这些测试，证据是现有收据及其绑定源码。

## 合同一致性

- 主方向确为 `.5*(n4[1:]-b4)`，只使用 N1/N2；旧 robust projector/64eps/FP32输出未换。G-matched、词表null、LOO及新增 residual-guidance matched_scale 同源，零h显式Fixed。matched_scale已进入 strong，cost-review未改变。
- 每个模型输入从冻结 task 的 prefix_pool+consumed_ids重建并核hash/长度；未接旧 teacher-forced Q states。readout 是单位置BF16 head→FP32。所有完整fullV方向/系数和log_softmax先完成，target仅用于随后gather。实际head dtype/shape有运行时断言。
- native donor绑定input/context/source位置；各层CF只在loop4取相同输入的native donor，前3完整序列exact，局部其他行rawexact。NativeDonor绝不从CF构造。全143来源含singleton映射钳制符合 Root preimplementation定义；singleton是自身native donor，不意味着受上游干预后的CF当前行必须不变。身份replay检验整个钳制路径。
- 2366基础任务、778CF、5附加验收共2371 forwards；共享reference组处理与Quses完整。FP32位置buffer拒绝重复/缺失，恢复原token顺序后一次sum，字符分母及native Q4−Q5保持。匹配R6来自同primitive实际分数；历史R6明确另列。
- hook先安装后新construct scorer，每次configure invalidate；clock严格枚举24层×depth，finally清理及269参数前后核验。代表性raw向量/缓存均以dtype、shape、原始bytes哈希绑定，analyzer实际重算NPZ哈希与finite。
- analyzer先完成源/预算/参数/完整任务/控制/分数重建审查，unmatched阻止effects解封，之后才gold和argmax。strong同时检验两种R6救伤及旧收益，LOO稳定和全部五个controls；cost-review只作独立非性能资格。
- 单次输出guard、公示protocol gate、16旧ledger继承、1485TERM/1499KILL/1500hard及reserve5均在源码；内部reserve使接近1480时停止新任务。所有实际wall以supervisor计账，分项同步计时不再叠加到账本。OS退出不能被宣称绝对保证1秒内完成；真实overrun已设ineligible并完整计费。

## 冻结前保留条件

Root须核对最终 tokenizer-only preflight PASS、最终draft绑定当前11源码/两CPU receipt/任务图/设计amendment与全部依赖；将draft转正式锁后做公开原文readback，另发明确START。若生成draft期间源码变更，应刷新相应CPU receipt和审查绑定，不能沿用本次哈希结论。

可选1588 native任务/6773单位置BF16 state缓存约27.74MB，不增加模型调用；导出IO仍在cap内。其index声明未来复用另审，与本次合同一致，不授予后续自动利用权限。

本次 source-bound input/backend证据不被描述为独立HF调用拦截；源码、冻结依赖、任务输入断言与实际结果验收共同构成证据。实际GPU identity/first3/参数/重复输入等exact检查仍可能失败，失败按原invalid/incomplete处理，不补跑、不调方法。

结论：**当前已读实现可条件通过，允许准备公开冻结；最终preflight/draft/readback条件满足后仍由Root单独决定START。** 无新增方法、门槛、样本或模型条件。

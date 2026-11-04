# R20 execution-only amendment：Astra科学与工程审查

结论：条件性同意一次公开的执行修订E1；不把它称作新方法，不声称原no-retry合同允许自动重试。原合同已经使R20 v0结束且不合格。Root须在E1运行前明确记录此合同修订及其依据；这是一项新的执行授权，保留原全部记录。

本审查只读取r20_EXECUTION_REPORT.md、r20_EXECUTION_VERIFIED.json的执行审计和原hook源码，未读取partial scores、准确率、预测比较或标签关联效果。v0完成19/24、373位置、168.55203619925305秒，最终权重hash/peak未完成；故不能把它判成科学方法失败或正面候选。

## 为什么此修订可以接受

提出修改的依据是确定可见的host同步与fingerprint开销，而不是19题的准确率。原每个V hook执行bool(cuda_tensor.all())会等待CUDA结果；每次注册/移除都刷新fingerprint，而这些注册/移除之间没有forward。将断言的CPU读取合并、把没有forward间隔内的invalidate合并，原则上不改变干预输出或统计方法。没有证据证明它们是主要瓶颈，亦不能预告24题一定在180秒内完成。

这仍是看过运行成本之后的执行适配，不是完全未经修改的事前实验；报告须如实呈现两次执行及两次成本。由于不改变公式、样本、门槛且此前没有效应选择，它不产生从多个准确率结果择优的同类问题。样本此前已反复开发使用的局限依然存在。

## 唯一允许差异

1. 新scripts/r20_e1_*、configs/r20_e1_*、results/r20_e1_*、logs/r20_e1_*命名空间。原R20源码、lock、publication、partial、supervisor和ledger全部保持字节不变；保存继承hash。
2. 每次活跃hook仍执行原clone、索引tensor构造、index_fill_以及原选中值全零比较与all计算。将生成的CUDA布尔scalar保存至本次forward专属列表，不在每hook读取CPU。在verify(loops)中检查每层调用次数，并核对布尔scalar数量=physical_layers*loops，再一次stack(...).all().item()断言。不得只抽查某层/循环/位置，不得删除原零值计算。off-hook期列表须为空；configure重置列表前必须确认上一forward已经verify成功，不能覆盖未检验结果。
3. 全部hook注册完成后invalidate一次；全部卸载后invalidate一次；期间绝无forward。注册部分失败时仍须finally移除已成功注册hook并invalidate。configure/closure变化后仍每次invalidate；不得绕过原_check_model。
4. 算法文件r20_math.py及旧评分源码原样引用并hash固定，不复制后偷偷修改。cohort顺序、span、控制窗口、PRNG、α、控制、输入、dtype、mask/backend、字符分母、晚期项、全部parity和最终all-parameter hash要求不变。必要namespace导入/路径变化显式列在差异清单。

## 执行前验收与禁止事项

CPU测试旧/新hook输出逐元素exact、off返回同tensor、每层调用次数、全部断言覆盖数量、configure清空语义、异常清理、原有hook恢复。测试至少包括多个loop和不同合法span，以及故意非零检测标量能使合并断言失败；不能只测正常路径。保持原数学测试receipt并核对引用hash。Root/Sol公开最终E1源码与lock后再启动。

不得新增独立GPU profile、热身worker、放宽reserve、减少hash、减少控制、分批拼接19+5或减小输入。E1完整24题从头跑。只允许这一工程差异；不能在未完成后再优化第三版。CPU测试失败可在任何GPU启动前修复实现，但不能借此改变设计。

## 完整性与效果解封顺序

完整24完成、supervisor正常退出且无timeout/overrun、最终全部权重hash一致之前，不计算任何准确率、argmax、纠错/损害或控制胜率。

完整性通过后，先由单独无gold、无argmax的只读parity程序对齐v0全部19题/全部候选/全部已存位置。比较所有共有可复算的数值：candidate sums/scores/late、所有alpha与norm指标、unmatched、token数、字符分母、reference map及关键输入/位置hash等；数值差必须严格0。时间、内存、namespace、协议hash等执行元数据预先排除，并列出穷尽的比较字段清单。不能只比较新/旧方法预测相同，不能放公差。保存per-field count/max_abs_diff及通过/失败receipt，不显示性能比较。

只有E1全部24完成且19题重叠的exact parity全部成功，才按原R20 gate进行一次完整24题效果分析。任何数值差、缺项、timeout、incomplete或overrun：停止、不可晋级、不分析部分效果、不另开修复worker。19题exact不能证明其余5题所有硬件行为，但结合CPU输出等价及相同代码足以支持本执行修订，不夸大为全V跨执行验证。

## 时间和账本

E1仍从worker启动计时：170秒禁止新题并TERM，179秒SIGKILL，最后1秒reap；既有runtime reserve(5)保持原样。不能借合并断言延后预算检查。所有实际overrun全计费并禁止晋级。

十旧ledger累计18128.605543765705秒，余3471.394456234295秒（来自已公开交接，启动前再独立求和used_seconds）。新增第十一ledger单独记E1实际耗时，原账本不改、不重复加total_used_seconds。最多180秒E1后名义余量3291.394456234295秒，未给完整开发/ARC或扩预算授权。另两数据集仍暂停，heldout逐题仍关闭。

## 决策边界

同意的是一次有透明合同修订、完整重跑、严格数值重叠复核的工程性尝试。不是自动复活v0，也不是对180秒内完成或准确率增益的保证。若Root不愿公开修订no-retry条款，应维持v0停止，不可把E1悄悄归类为原运行的一部分。

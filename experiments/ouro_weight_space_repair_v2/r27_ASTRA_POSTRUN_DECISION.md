# R27 Astra执行后决定：有限V来源传输成立，未建立ARC修复机制

本文件为新执行后结论，不修改冻结proposal、执行合同或结果。已审读 `results/r27_EXEC_ANALYSIS.json` 与 `results/r27_ROOT_EXEC_VERIFICATION.json`。没有新增模型/GPU或ARC score；原目标仍未完成。

## 原门槛确实通过，不需要改写成功定义

R27完整33次forward。Phase A全部8个prompt在预定四label子空间中正确辨别映射；不是自由生成准确率。Phase B的唯一主假设V通过：4个clean V margin为1.25、2.25、0.625、0.25，全部从B的负margin跨过0；transport delta为5.8125、8.125、5.0625、5.375，clean相对固定V置乱的specificity为5.8125、3.3125、4.9375、2.75，全部严格正。

这次正面结果由预先指定干预与负控支持，不是事后选出K/V赢家。K clean没有跨0，specificity四项均负；K置乱有两项正margin也不能晋级，它是预先禁止替代主假设的控制现象。

Root独立raw向量审核通过：5项no-hook/capture比较、192项同表跨query source比较、72项前三loop全序列比较、40项identity state/head比较、NPZ全引用/hash/schema、完整资源账本及269组参数。source donor确实与后续query独立，同一份donor用于两种query，且前三loop保持原B；不是给每个query另做答案向量或全输入还原。off-source投影保持属于冻结hook运行时断言，未由第二模型调用独立再生；这一证据层级保持原样。

## 能支持的因果陈述

在这两张表、四个配对query、相同shape与四个预定source位置上，将native4最后loop、全部physical layers的source V投影从A原位重放到B，足以把两目标label的偏好转向A的对应答案，并优于保持该组donor向量多重集的固定位置置乱。不同query使用同一source donor却恢复不同label，反对仅增加一个固定label偏好的解释。

这是**对指定V投影干预的有限内容对应关系传输证据**。它不是该路径的必要性证明，没有穷举替代电路，也没有证明只有V负责查表；K clean失败不能推出K无关。K/V与后续attention/MLP有交互，V干预的下游效应并不限于V变量本身。

所有24层的末loop source组一起处理，尚未定位compact head/circuit。固定置乱保留向量多重集合，不匹配所有下游activation分布；只使用一个置乱与四实例，不能把效果推成普遍因果定理。它与既有ICL/activation patching现象相容，不包装成新的function-vector原理。

正面label质量范围约0.9478–0.9952只是事后描述，不是原门槛；不据此将集内8/8改写为已验证任意文本生成。synthetic label由规则给定，且答案对应关系明确存在于当前表中，未检验事实判断、无答案证据时的可靠性或ARC纠错。

## 它改变了下一步哪些判断

R25之后“尚无Ouro本模型的独立来源功能证据”这一局部缺口现在得到有限补充。可以将末loop source V视为一个**已有可操作功能证据的干预对象**，而不只是猜测哪个向量有用。

但它没有填补更强的两个缺口：实际ARC原prompt的哪组来源与当前问题答案有关；从该来源的因果响应怎样得到能保护R6旧收益且产生新纠错的无标签修正。R27没有构造或选择此部署估计器，不允许把synthetic答案、donor或margin传给ARC。对R26严格prefix执行的长度敏感性也没有全局解决，只在R27同shape和独立source条件内隔离了该混淆。

因此下一步可以审查一个具体跨任务转移假设，但不能直接宣称ARC路线已准入。尤其不能把同一旧投影公式中的axis换成“causal V”就称新机制，也不能用这次正面结果重开关闭的强度/层/head搜索。

## 资源与交付

实际56.29345832020044秒；第16ledger累计18930.428655151278/21600，余2669.571344848722秒。本文件不授权新job，不因正面功能结果扩大预算。R27功能验收完成且主screen通过；原ARC目标未达成。下一ARC部署候选的范围审查另见新文件 `r28_ASTRA_DESIGN_PROPOSAL.md`，不混同两阶段的成功标准。

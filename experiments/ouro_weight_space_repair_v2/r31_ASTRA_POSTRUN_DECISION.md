# R31 Astra postrun：关闭 Fixed/C 二元prefix选择路线

已读完整r31_ORACLE_ANALYSIS.json与clean supervisor：24题13 certified_reachable、11 certified_unreachable_strict_margin、0 undetermined；全部10个R6错误、唯一R6-only收益题均在不可达类。CPU总1.014999999984866秒，worker/analyzer退出0，supervisor完整且未超时；没有GPU工作或账本改动。正式解封依赖既有final_result join；已读 results/r31_ROOT_CERTIFICATE_VERIFICATION.json：Root不调用原math/inputs/solver，从原LP与图独立重构全部24的系数/包络、192端点及witness/simplex上界通过，solver身份与18账本一致。

**按事前规则，关闭对当前Fixed/C两个已观察分布做binary prefix开关的selector研究。** 即使gold oracle可为每个共享Q prefix选择其中之一，该有限family也不能在上述11题上实现>tau=1e-8的严格正确margin（含预定FP32/FP64舍入包络）。因此用这批C输出训练、猜测或构造更好无标签开关，没有新增纠错或保留唯一旧收益的容量依据。不能把selector失败改成扫强度/符号，也不继续花GPU测试该路线。

通用协议只认证strict margin>tau，本次实际证书更强：全部11个不可达题的保守上界严格小于0，因此tie也不能救回这11题；13可达题恰为A1原正确集合，跨题没有共享key。故该固定24/Fixed-C binary family的最大正确数确为13，不只是超过tau的证据不足。它只覆盖反复使用的固定24题、原char/late/true-prefix、当前Fixed/C完整分布、共享prefix binary选择；不覆盖任意连续logit修改、其他来源、非线性内部运算或全ARC总体。13可达题不能报告成新方法13/24性能，因为gold参与oracle；实际C仍是已关闭的13/24、0救1伤。

结合前一CPU分解：没有late抹掉已得正确排序的实例；结合R30多数margin不利：当前瓶颈不能再简单描述为“普遍方向正确，只缺控制器或力度”。下一研究若继续，必须引入新的可识别内容或计算机制，并另证价值。H2/H3保留为研究问题，不自动授权它们或恢复A/C。

18 GPU ledger仍20755.46525352029/21600，余844.5347464797087。目标未达；不以CPU容量审计冒充方法改进或全链验证。

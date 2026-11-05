# R31 已解封24题的后验阶段归因：CPU协议

这是postrun诊断，不是新方法、独立验证或GPU准入。执行前固定：读取完整R28/R30 SCORES/ANALYSIS/COMPLETE及Root收据，遍历全部24及全部候选。不得搜索late系数、替换winner、升control或读heldout。gold仅用于解释已解封结果。

对每候选固定读取 F=Fixed.sum_score/原char，P=primary.sum_score/原char，L=已保存late；核P+L等于已发布primary、F+L等于A1。分别记录P/F/no-late与最终的argmax（原先序tie），仅作结构诊断。对gold与每个wrong j，以存储binary浮点的精确Fraction重建pre-margin、late-margin与correction-margin，保证符号判定不靠任意容差。

预定主要反证：若全24没有任何“P正确而P+L错误”，则固定late veto不能解释任一实际错误预测；若有，则只证明算术上的veto案例，不能证明改late合法或会泛化。另报告F错误→P正确及反向、R6错误集的上述类别，防止将原生pre能力误称修正贡献。相同pair的correction增量在pre和post相同；late会改变竞争者或决策阈值，不能改变同pair方向符号。

不拟合参数，不输出新可部署分数，不跑模型。报告只能确定某个解释的有限支持或反证，不能把no-late oracle变成候选。

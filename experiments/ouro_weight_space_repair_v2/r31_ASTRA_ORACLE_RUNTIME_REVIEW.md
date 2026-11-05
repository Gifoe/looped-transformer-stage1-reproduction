# R31 Astra oracle CPU稿静审

结论：**conditional pass，未发现阻断证书正确性/完整性的错误。** 未运行真实LP/MILP、模型或GPU。审查不授权实际oracle；仍由Root冻结、readback后明确安排Sol执行。

审查绑定draft SHA256 `318ab529c043120ff38be2a926b47d17dbd6d4368f8945738b734ddac0f08d5d`，本次23个source/input哈希全部本地核对一致。重点阅读math/solver/inputs/analyze/worker/supervise，结合现有合成与输入收据；没有自行复跑真实输入求解。

- coefficients使用实际FP32每位置LP的精确Fraction、同Q prefix共享key及保存rounded late，不误用R31此前理想native late组件。零列删除不依赖gold。输入逐项核FP32可表示/finite，192端点被B包住。
- B使用精确gamma、n*u<1/2及FP32无溢出域约束，纳入subnormal绝对项与保守FP64末级误差。可达witness必须真实0/1，逐pair减B后>tau；独立analyzer从候选分数重算，不只重调用worker margin函数。
- 不可达使用归一为精确非负simplex的dual权重，独立计算盒约束支持函数上界，且把pair误差B加到常数项。该上界对所有二元解有效，甚至覆盖连续松弛；solver infeasible/optimal/status本身不决定分类。LP整数间隙落undetermined，符合方案。
- 全24完整后才独立认证；cross-question共享key被检测并限制联合解释。当前CPU图报告无跨题共享，单题witness可兼容，但仍是gold oracle而非部署selector。strict-margin不可达不排除原先序tie拿到正确。
- 180秒覆盖worker和analyzer，两个solver各3秒、单线程，175TERM/179KILL及overrun拒绝；final_result还要求干净supervisor与分析hash。18旧GPU ledger仅核hash/总额、不修改，模型/CUDA路径不进入计算。

无须新增方法或门槛。非阻断边界：普通MILP若返回近0/1而非精确0/1，此实现保守拒绝witness，可能增加undetermined而不会造成伪可达；不要运行后放宽。最终分析文件在supervisor清洁退出前状态明确为pending，报告必须经过final_result join，不能直接将该中间文件当最终结果。

Root最终需确认冻结版本与上述哈希一致、公示执行定义和CPU一次运行授权；如源码变化，刷新receipt/draft，不沿用本审查覆盖新版本。

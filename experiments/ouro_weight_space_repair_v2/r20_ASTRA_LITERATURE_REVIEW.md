# R20 Astra定向文献复核与唯一条件性设计

状态：DESIGN ONLY；尚未运行；保留r20_ASTRA_DESIGN.md v0原样。该文件替代v0的“没有可执行候选”判断，不撤销其旧路线停止结论。Root审查后才可交Sol执行。

## 修正后的决定

允许一次180 GPU-worker秒硬封顶、24题完整候选的小型证伪pilot：**以当前问题V路径消融构造结构匹配的参考增量**。不是改保护轴、度量或标量优化器；只替换先验参考的生成机制。它是有可检验假设的探索，不是已有收益证据，也不预授权完整开发或ARC。

v0的推理有一处过强：现有价值代理失败不足以排除不同机制的一次检验。下面的路径干预能够改变证据内容，因为它直接操纵当前问题信息传播，而不是只测自然向量相关。其对正确性的作用仍必须由配对评分检验，不能由“因果”一词推断。

## 定向查阅：四篇原始研究与一份固定模型源码

1. [Meng et al., Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)：对内部激活的干预可定位影响事实预测的计算。此处借用干预思路，不使用其权重编辑，亦不把GPT结果外推成Ouro证明。
2. [Geva et al., Dissecting Recall of Factual Associations](https://aclanthology.org/2023.emnlp-main.751/)：用信息流分析研究事实提取，支持检验问题信息的传播路径。其attention knockout与本设计的V置零不同；本设计不修改attention mask或重新归一化attention权重。
3. [Zhang and Nanda, Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042)：干预方式和指标会改变解释结果。因此不把一个向量变化当机制成立，必须比较等长位置消融和修正幅度控制。
4. [Hase et al., Does Localization Inform Editing?](https://arxiv.org/abs/2301.04213)：定位与有效修改可能脱节。本pilot分别报告路径效应和纠错效果，只有后者过门槛才有下一步。
5. [固定revision Ouro源码](https://huggingface.co/ByteDance/Ouro-1.4B/raw/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)，同时检查本地sources/pinned_model/modeling_ouro.py：v_proj在view/transpose前生成value，后续repeat_kv及attention_interface使用该value。可在v_proj输出注册辅助forward hook，避免替换attention实现。既有scripts/causal_hooks.py主要处理o_proj/MLP分支，不可将其当成已实现的V路径hook。

这是标准激活消融的项目内新应用，不声称新原理或文献首次。只查上述四篇研究，未进行开放式文献搜索或论文内容生成。

## 可证伪假设与严格解释边界

假设H：N0的文本/位置/演示缺失造成参考增量混杂；保持完整原prompt而消融当前question源位置的V输出，可以得到较有用的背景循环增量，使修复超过N0及位置/幅度控制，并保留R6旧纠错。

这不是“问题完全移除”或自然无问题世界。Q、K、softmax分母、当前问题位置残差、MLP及其后续传播仍可携带问题信息；早层V干预会改变后层K/Q。它识别的是**所有层及循环中当前question源位置V置零的联合干预效应**，不是固定K/Q条件下的纯直接效应，不宣称正确性语义分解。零值也可能是分布外扰动；控制只能部分反证位置/幅度解释，不能消除所有分布外疑虑。

原权重、原token位置、causal mask及attention backend不变；辅助激活改变属新的辅助参考构造。若原授权解释为连辅助激活也禁止，则是scope extension，Root须先解决范围；不可默默运行。本文按用户允许无训练forward方法且Root提出辅助值路径解释进行条件性设计。

## 唯一公式

记Q1,Q4为原主干全词表logits，B1,B4为辅助完整原prompt的question-V置零分支。所有位置只使用已消费候选前缀；辅助分支最多运行4循环。定义与R7完全同一数值顺序：

- FP32 g=.5*(Q4−Q1)，F=Q4+g，e=.5*(B4−B1)。
- FP64 p=softmax(Q4)，C_p(x)=x−sum(p*x)，gc=C_p(g)，ec=C_p(e)。
- D=sum(p*ec²)，T=64*eps64*max(abs(ec))²。
- D<=T时alpha=0，否则alpha=clip(sum(p*gc*ec)/D,0,1)。
- Z=FP32(F−alpha*ec)；alpha=0时逐位返回原FP32 F，不重新重排算术。
- 全词表log_softmax(Z)，按原候选target累加，仅用于评分；原字符分母归一化一次，再加未变的原Q4−Q5晚期项。

无R6再减、无候选空间锚点、无保护轴。全零/退化参考回到A1而不是R6，这是风险，不把退化性质当保留R6的保证。非有限值硬停止，不回退挑另一方法。继续沿用原native Fisher估计器是为隔离参考生成机制，不重新打开度量搜索。

## Hook及span精确合同

在所有physical layers的self_attn.v_proj输出张量[B,S,Hkv*D]上，clone后将预定义源位置置零；所有循环均施加，所有head/通道一致，无层/循环搜索。仅辅助分支激活。off-hook必须返回原tensor而非乘1；try/finally卸载并确认hook计数归零。原主干/N0/控制分别独立forward，不允许污染缓存。

当前question范围由原prompt构造器的结构字段边界产生，绝不全局字符串搜索替换。保存prefix-before-question、question原字段、suffix并断言拼接与历史prompt字节一致。对实际encode_pair输入建立token字符offset；凡与question字段非空字符区间有交集的token进入span（边界token可能含分隔符，必须报告）。不得改tokenizer参数或重新分词取代原输入。若offset无法与实际input_ids精确对齐则整个pilot停止，不手工猜位置。每候选的prompt部分须相同，span不得触及候选目标/未来token；记录全部25demo及当前question边界。

保持原uncached/no explicit mask路径。可按旧安全causal全文评分路径一次计算完整候选，但证明每预测位置仅依赖过去；若原backend路径不能给此保证则停止。不得引入KV优化。

## 固定控制（均不可晋级）

A. 原N0 M2、R6、A1、Fixed：原定义和exact parity。
B. 等长位置消融：从25demo文本区域中、与当前question span同token长度的全部合法连续窗口，以seed2021+稳定question ID的确定性PRNG均匀抽一个；窗口不得触及当前题或候选。逐题固定且所有候选共享。若不存在合法窗口则停止，不删题或改长度。得到同公式e_demo及最终修正c_demo。
C. B的幅度匹配版本：中心化c_demo后按当前p范数缩放，使其范数等于主修正c=alpha*ec，再用F−c_demo_matched执行全Vlog-softmax。主范数零则控制也零；主非零而控制零时无法匹配，pilot记不合格，不加epsilon伪造匹配。
D. 主完成修正c的固定词表置乱，重新p中心化并匹配其原p范数；沿用R18已审计退化规则。

B和C共享一次辅助forward。等长demo控制不能完全匹配语义/距离，不称完美因果对照；但主方法若不超过它，没有当前question特异性的正面证据。N1/N2不新跑，因为本设计不是三参考集成；原定义保留，不以标量缓存冒充其全V分布。

## 一次pilot、parity与停止

从已冻结R16的64题清单中，按seed2021确定性打乱后取前24题；不看标签、margin、旧纠错或长度选题。冻结具体ID、顺序、span及控制窗口后才能运行效果。仍是多次复用开发数据，不是独立验证。使用全部候选/全部位置，不能截短以赶预算。

先做CPU合成测试：零参考、非有限值、中心化gauge、范数匹配、候选扩展不变性。真实worker首题先off-hook和卸载后forward与缓存主Q1/Q4 fullV exact比对，同时N0/旧M2/R6等历史评分exact；hook期间确认输入token/position/mask/backend与原调用一致，权重hash前后相同。保存hook命中层/循环/位置数和零值断言。独立读取首位置B1/B4检查readout语义，不能把共享深度最后head误读成native浅循环。任何parity失败停止，不放宽公差。

pilot合格须全部满足：24题完整；对R6净增至少2、damage=0；样本中的R6相对Fixed独有纠错至少1题且全部保留（若0题则该保留检验不可识别，不通过）；严格高于旧M2及B/C/D全部控制；相对B与C分别至少存在1个主对/控制错的配对净优势；全数值及parity通过。报告各方法完整题级配对、exact McNemar、主与控制disagreement和纠错/损害。门槛仅为继续可行性审查，24题不能证明显著泛化。失败即停止，不扩大样本补救，不换层/符号/强度/span，不挑控制。

## 预算与下一步

唯一worker硬上限180秒（从进程开始，含加载、核验、退出）；外部supervisor强制deadline且保留退出余量，内层在170秒停止启动新题。不满24题为incomplete，不按部分成绩晋级、不另开worker补足。全部实际耗时记新增ledger，继承九旧账本原字节；不重置21600秒总cap。

粗估两条四循环辅助24题约2*24*1.3=62.4秒，N0约7.1秒，加加载、parity、head/control与IO预计可能进入180秒，但这不是已实测profile。主states复用不等于独立部署测速。允许在同一worker内记录首题实际成本并据剩余时间提前终止；不得先另开未计费profile。

完整后续不能承诺预算内：576题新增双辅助约1497.6秒，ARC若仅主方法一条辅助，既有约2400秒部署量级还须额外约1172*1.3=1523.6秒；合计已约5421.2秒，未含确认/审计，超过现有3639.95秒。该算式是保守规划近似，实际应重新profile并避免重复计项。只有pilot正面且Root确认机制值得继续，才提交具体全链预算review；正面结果不授权自动扩预算或开ARC。另两数据集继续暂停，517/299/test逐题不得读取。

结论：有一个可实现、不同于旧几何路线且可在180秒内尝试证伪的条件性候选；没有预先的准确率保证，也没有整条链预算授权。若Root不接受辅助激活范围、精确span不可实现或worker成本不足，回到No-Go而不是改公式。

## 最终事前收口（Root审查后、任何R20新特征前）

以下五项优先于上文对应旧表述；v0文件保持原样。本次仅修改设计，未计算24题基线或新方法。

1. **C控制唯一改为原始方向匹配。** 取d=C_p(e_demo)，主完成修正c=alpha*ec，C的修正为d*sqrt(sum(p*c²)/sum(p*d²))。主范数为零时C修正严格为零；主非零且d的平方范数<=64*eps64*max(abs(d))²时记unmatched并令pilot不合格，不加epsilon、不另换方向。B仍使用自己的alpha_demo和完整实际输出。理由：原completed版本会将alpha_demo=0的系数截断混入方向可识别性，C现在隔离原始参考方向与干预幅度，不依赖alpha_demo非零。C不clip缩放比例，使用完整log-softmax；不将控制晋级。D维持原completed主修正置乱规则。

2. **coverage admission。** 先按无标签固定规则冻结24题ID、顺序与协议hash，再仅CPU读取这些开发题既有R6/Fixed基线正确性，计算旧R6独有纠错覆盖数。若为零，直接admission No-Go，不启动GPU、不重抽、不增样；不因coverage结果改协议。若非零，pilot须全部保留。它是选择后可行性检查，不是对新方法结果筛样；仍属重复开发探索。

3. **180秒总封顶的执行时间线。** t=0为worker进程启动，t=170禁止启动新题并请求终止；若未退出发送TERM，最多9秒grace；t=179强制SIGKILL，剩余1秒用于reap/记账，不使用180+15等额外宽限。supervisor须按单调时钟实施。任何调度等因素造成的实际overrun全部计入账本且禁止分析晋级，不另开worker补跑。未完成24题即incomplete。

4. **现有scorer guard必须保留。** OuroCachedScorer._fingerprint包含hooks；register、unregister及hook闭包的有效状态变化之后均调用scorer.invalidate()。使用原_run且use_cache=False，保留原调用/no explicit mask/attention backend。实际model.model HF forward可执行v_proj hook，但不能绕过fingerprint guard或修改旧scorer源码。每候选控制span更新同样属于closure状态变化，应invalidate。新独立hook模块可实现上下文管理与finally清理；旧缓存只在干净原主干状态读取。

5. **scope与成本纠正。** Root确认该辅助V路径干预属于当前冻结权重、无训练forward授权，不再要求用户重复许可。上文‘若辅助激活也禁止’条件已由Root解除。旧M2约2400秒包含N0参考，new aux替换N0，所以完整ARC粗估应为2400+1172*(1.3−.29565)=3577.0982秒，而非3923.6秒；576题双辅助粗估1497.6秒，合计约5074.70秒，尚未含pilot/确认/额外核验，仍高于当前余额3639.95秒。不同profile口径及四循环成本意味着这些只是规划估计，不能当实测。只有正面pilot之后才按实际加载、主干、辅助、head/control等拆分成本提交具体预算review；不自动开始完整开发/ARC。

最终合同：唯一question-V参考候选、B实际demo消融、C原始demo方向范数匹配、D主完成修正置乱；cohort先锁后coverage admission；最多一个180秒worker；所有原门槛、数据权限、停止条款仍有效。

研究问题：在固定P2及R背景的Q1→Q2切点替换效应中，切点之后第一物理层13写入R的attention残差增量是否对后缀决定具有可测的双向因果作用？此为基于已揭示三split ROUND_3的新机制定位计划，不是新独立验证，也不预设该路径足以解释自然K1→K2。

实验单位与来源：全部2590题，train1119另列25示例/1094非示例，val299/test1172各自报告，不按效果筛选。每题同原input IDs、25shot、P/Q/R位置、token评分与自然/ROUND_2/ROUND_3来源链。继续BF16、eval、SDPA、batch1、use_cache=False且past实际为None、labels=None，total_ut_steps=2。切点为ut1的physical layer index11输出。动态重建并核验同题自然K1/K2切点状态，构造B1r=F21r、B2r=F22r，r=1,2；两条均为实际hybrid完整前向。无效果/相同Q/a_R不是失败，不剔除。

唯一待测操作：在ut1 physical index12的input_layernorm_2输出（源文件line414，attention分支归一化后、line415残差加法前），只替换真实R位置a_R。U1r=B1r接收来自同题B2r实际捕获的a2r；U2r=B2r接收同题B1r的a1r。保留receiver x_R及所有非R分支输出；MLP和所有后续层在receiver改变后的状态上完整原生重算。不得替换self_attn原始输出、重复norm、缓存捏造donor或借用别题prefix。donor标为B1r/B2r hybrid捕获，不能标natural K1/K2。

关键可识别前提：两条baseline在layer13输入P与x_R逐位相同；实际mask、positions、RoPE、cache状态一致，模型/source/runtime均冻结；完整切点只有Q位置允许不同。该前提只定位首次R写入的来源集合，不将Q状态解释为独有问题语义，a_R差异含Q键值与attention归一化竞争。后续效应仍是混合状态干预。

每题保存四baseline、四identity（各receiver用自身真实capture作为donor）、四跨轨迹patch的完整候选scores/预测/top tie与BF16全词表readout logits；保存四a_R BF16 tensor及shape/hash、实际source/receiver输入与切点hash、loop/layer/site、P/Q/R位置、模型环境和有效mask/RoPE/cache绑定的capture receipt。receipt只能在完整forward成功、最终输出finite、前后来源核验成功且目标访问恰一次后发出。donor注入前重验hash与轨迹；异常清理hooks并恢复total_ut_steps，无成功receipt。允许共享存储同一份实际相同tensor但不得以声明代替逐位验真。

数值门槛：四baseline逐题桥接已coverage的F211/F221/F212/F222完整BF16 logits与所有候选分数零差；no-output-change capture与no-hook baseline零差；四self-identity与对应baseline零差；patch位置外输出逐位未改，x_R未改。若a1r与a2r逐位相同，两向patch必须与对应baseline零差。对production评分shape/device/dtype/order做原有native-head桥接，保存后独立CUDA全候选zero-gap回放。任何不满足停止，不能放宽epsilon或丢题。

结构锚：CPU完整后缀oracle覆盖24层两轮、非均匀norm、真实R位置、同题matched hybrid、attention输出后norm、MLP在receiver重算、来源错题/错loop/错层/错分支/错input拒绝，以及失败清理。真实最长题probe涵盖全部条件、分支实际调用与资源；保留最终物理层C-only零读出负控。同一layer13 C-only MLP替换只能要求该层R输出不变，不能要求最终R零效应；R的MLP也不是负控。两attention/MLP分支替换绝不当作全K1锚。

预定义指标：对每个非gold候选j计算原生产gap，并令E_r=gap(B2r)−gap(B1r)，N_r=gap(B2r)−gap(U2r)，S_r=gap(U1r)−gap(B1r)，残余E_r−N_r与E_r−S_r。主指标先题内全部非gold等权再题间等权；固定K1最强wrong版本同时报告。候选集合/固定wrong永不重选。全分数与各预测变化、救伤、相反方向、零效应和分位数都保存。原score不变，另从保存BF16候选logits以精确有理数计算相同gap对比，处理log-softmax伪零。

前瞻判据：对各split各r分别报告。'双向平均增益贡献'只在mean(N_r)>0且mean(S_r)>0时成立；任一<=0明确否定该分量，不靠合并split或另一R背景挽救。该判据本身不证明忠实恢复完整输出。为检验轨迹输出转移，另定义每题全非gold向量距离 d(X,Y)=mean_j |gap_X(j)−gap_Y(j)|，L=d(B1r,B2r)，A=L−d(U1r,B2r)，B=L−d(U2r,B1r)。只有mean(A)>0且mean(B)>0才能称双向更接近对应反事实端点；报告A/B个体正负零与残余距离，不用比例或删L=0题。不把微小非零说成实质充分性；同时报告实际大小和原E，不能以任意比例阈值追认机制。若N/S正而距离不改善，只能称提高平均gold相对gap，不称恢复Q2计算。全部候选逐位或gap向量exact恢复的题数另列，不代替全体。

解释与后续选择：自然rescue/damage/stable-correct/stable-wrong及自然无tie/全部相关条件无tie只作完整敏感性，保留伤害例。a_R差异为零/替换无效不推翻所有attention，可能后续Q→Q→R、冗余或补偿；不能从单层结论外推全部层。若该首写入双向作用且端点距离改善，可下一轮预注册其后的R-MLP增量变换/交互；若不改善，优先重新界定为后续层路径问题，按预先固定层序定位，不按成功样本优化。禁止将attention权重相关性当机制。

执行安全：独立新目录和冻结source，不改原PQR/ROUND_2；同全局GPU锁、唯一受监督service，用户>20GiB空闲准入，加实测峰值余量，动态资源退出，清理仅自有进程。先CPU契约与静态审查→真实最长题probe-only→Astra审查绑定receipt/完整score回放→fresh probe精确复现后全三split。严格逐题恢复验证、不覆盖已完成证据；完整coverage后才分析。若需train先揭示，必须再事先新增真实时间修订，不默认为允许。只有验证和产物真实完成才记完成。

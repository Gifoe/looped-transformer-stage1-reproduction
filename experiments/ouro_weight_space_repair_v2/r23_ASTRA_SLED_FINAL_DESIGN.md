# R23最终设计补充：SLED型全词表配对演化，一次固定pilot

版本：事前设计v1，补充r23_ASTRA_PRIMARY_REVIEW.md；旧审查原样保留。当前没有新特征或新模型运行。选择这一项，kNN记忆路线不并行。

## 来源核验与决定

已读[NeurIPS2024官方论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/0939f13ffce3ff487509d902ddba4571-Paper-Conference.pdf)§2、Algorithm1及[官方sled_decoding.py](https://github.com/JayZhang42/SLED/blob/main/sled_decoding.py)。论文解析gradient为p−P_latent；代码在no_grad内作logit更新，无模型反传。公开代码包含正cos平方、topK估计、10个衰减步及尾logit压低；本设计不复制后面三项，必须称SLED-inspired适配，不能冒充官方复现。实施前锁定所读官方源码commit/文件hash。

**同意形成唯一固定pilot。** 新内容不是KL字样，而是由逐词表one-hot假设的解析梯度对齐构造一个新的概率目标，再生成整条非线性修正向量。它通常不在原g/e线性span中，也不受R21候选score凸包限制；不能据此直接推断能救回157题。R16只在既定e上优化标量，本设计改变修正方向的生成机制。没有学习参数/controller或backprop，因此符合这部分合同。

SLED把深度间演变当作有关正确性的信息；这是经验假设，尤其Ouro的loop不同于独立层。R22的步长衰减失败不等同否定这个逐词表目标，但也不给它正面保证。

## 唯一数学定义

每个当前已消费prefix，主Q及三个合法参考N0/N1/N2均取native1、2、3、4的全V logits z^X_k。Q4为成熟分布，不使用Q5/Q6估新方向；Q4−Q5晚期项原样保留。三个参考文本、25demo、独立encode_pair和prefix文本对齐严格沿用原R7。

对每个X∈{Q,N0,N1,N2}：

1. FP32原head logits提升FP64。p_k=softmax(z_k)，所有softmax完整词表、temperature=1。
2. d_k=z_k−z_4，随后**uniform中心化**d_k←d_k−mean_V(d_k)，k∈{1,2,3}。这是预先固定的logit gauge，不是重新搜索Fisher度量。任意各层公共logit平移不改变它。
3. 对每个词v计算

   a_kv = (d_k·p_k − d_kv) / (||d_k||_2 * sqrt((1−p_kv)^2 + Σ_{u≠v}p_ku²)),
   b_kv = max(a_kv,0)^2。

   全V同时计算，使用O(V)恒等式，不创建V×V one-hot矩阵。Σ_{u≠v}p²可用总和减p_v²，对浮点负舍入只截到0；首实际prefix用显式小维合成one-hot版本验证该化简。
4. 若||d_k||²<=64eps64*maxabs(d_k)^2，该层所有b=0。每个one-hot梯度范数平方若<=64eps64*max(1,max(p_k²))，对应b=0，记录退化数。所有非有限值立即停止；不靠epsilon制造方向。sum b==0时P_latent=p4，u_X=0；否则P_latent(v)=Σ_k b_kv / Σ_{k,v}b_kv，u_X=P_latent−p4。

所有词都参与目标估计与输出，无topK、tail mask、温度/层/步数搜索。只作**一次单位解析logit步**；单位步是本次固定选择，不是已优化值。P_latent并不被称为真实事实分布。

令Z_M3为原R7三参考robust token校正返回的FP32 logits（原g/e与robust alpha公式不变）。唯一主修正：

c = u_Q − (u_N0+u_N1+u_N2)/3；
Z_main = FP32(FP64(Z_M3)+c)。

若c逐元素严格0，直接返回Z_M3原tensor数值；然后重新全V log_softmax，完整候选target仅用于最后gather评分、原字符分母一次、加未变晚期项。不得再次减R6。目标是**在原prefix先验校正之上增加配对的逐词表演化**，不是用q-only teacher替代。u_X每项由当前无标签forward全V得出，不用答案目标token、gold或未来答案估方向。

三参考平均使更新对参考重命名/排列不变，且共同演化抵消；这只是数学不变性，不能先验保证参考稳定或旧R6收益。因此下面有严格参考敏感性门槛。

## 实现与退化边界

只用原uncached/no explicit mask/backend/model revision/BF16路径。主native1..6缓存可读，新增main2/3仅head；N0/N1/N2全V状态未保存，fresh native4 forward并读取其native1..4。新helper仅扩展head读出键，不修改旧源码。首题独立深度2、3 forward比对相应native fullV exact，防止循环head误读；主native2/3候选sum与原R2 exact。原M2/M3、N0/N1/N2历史baseline与alpha完整exact。最终全参数hash前后相同。

全V公式参考NumPy FP64与Torch FP64，小词表用显式one-hot cosine矩阵独立验证，logit误差<=2e−5，P_latent及c误差<=1e−10（冻结前合成确定，不看真实效果放宽）。原基线parity仍exact0。合成覆盖独立logit gauge、相同深度、全退化、三个参考完全相同、Q与各参考完全相同则c=0、参考排列不变、候选扩展不变、非有限拒绝及全V归一化。

全V目标可能过于弥散；记录P_latent有效支持数1/Σp²、与p4的TV、u_X/c的P范数与zero率。它们是事前诊断，不据此改成topK或放大步长。没有“看起来太小”后的补救参数。

## 固定控制与参考稳定检验

所有控制不可晋级：

A. 原M3（以及原M2、R6、A1/Fixed基线）。
B. q-only：Z_M3+u_Q，检验配对参考是否贡献价值。
C. reference-only：Z_M3−mean(u_N)，检验是否只是参考减法。
D. completed-c固定词表置乱，重新按pQ4中心化并匹配主c的P范数；主c同样先中心化用于匹配，中心化不改变主softmax。零主修正→控制零；主非零而置乱范数退化则unmatched、pilot不合格，不加epsilon。

额外三条leave-one-reference-out敏感性读出：**Z_M3仍固定三参考base**，只把c中的参考均值改为其余两个，用相同一步与全Vnorm评分。它们不是新方法候选，不比较择优。每条均须对R6净变化>=0且保留旧R6相对Fixed全部独有纠错；否则主方法不获“参考稳定”资格。这检查本次新项的参考依赖，不能证明所有prompt稳定性。

## 一次固定24题pilot及停止

沿用R20已冻结的24题同顺序、完整所有候选，不重抽；已有R6独有纠错1题。数据已反复开发，不称盲测。所有新公式、controls、敏感性规则及实现hash在新feature前公开。

主方法须全部满足：24完整；对R6净增>=2且damage=0；保留全部旧独有纠错；严格超过M3、M2和B/C/D；三个leave-one-out稳定性门槛全过；全部baseline/native/checks通过；无unmatched、无timeout/overrun。任一失败即停止，不改单位步、正负号、全V支持、reference数、层或cohort，不晋级控制。

无accuracy前提的CPU实现/合成检查先行。真实worker最多一次，hard300秒、soft285、SIGKILL299、reap1，原reserve5函数保留。加载、参考forward、head、检查/退出全部计入。没有额外profile/warmup worker；首题成本从同一worker计时，仅作是否继续完成24的成本估计，不解封部分效果。完整24+clean supervisor+全参数hash一致后才计算效果。若300秒不足则incomplete停止，不自动改执行次数。

## 粗成本与全链可行性

余额3127.7056477041915秒；本设计尚未用GPU。历史每题N0≈.29565秒、N1/N2各≈1.3秒，故24题参考约69.50秒；另有cached main heads、目标构造、控制、加载及审计，300秒是保守操作上限，不是测速保证。全V O(V)式避免稠密矩阵，但三参考多head可能仍昂贵。

后续粗成本必须如实分开：

- 全开发576仅三参考≈1667.89秒，再加head/目标/审计。
- confirmation517仅三参考≈1497.05秒，再加head/目标。
- validation299仅三参考≈865.80秒，再加head/目标。
- fullARC1172无主state，主5循环≈1.62秒/题+三参考≈2.89565秒/题，即至少≈5292.34秒，再加head/目标/审计。

四阶段粗下限已约9323秒，远超现余额；即便省去部分开发阶段，fullARC单独也超余额。此事实不能隐瞒成预算可完整达标。本轮只提出一次有新机制依据的有限pilot；只有正面结果、真实成本拆分及Root认为值得继续，才向用户提交具体额外预算review。没有正面结果前不索要泛泛预算，不能自动进入下一阶段。另两数据集和heldout逐题权限保持关闭。

## 最终决定

SLED适配比继续查稳定性proxy更直接：它给出可立即按原候选评分证伪的新全V更新，而不是只做相关性诊断。它的经验假设、Ouro迁移、全V目标弥散、旧R6收益与参考依赖都可能失败；控制和停止门槛会明确暴露这些失败。

优先Sol实现上述唯一固定候选并作CPU合成，Root与Astra静态审核后再决定是否授权300秒真实pilot。当前不GPU、不扩数据、不增总预算。若Root认为单pilot而完整链需未来追加预算不值得，则应在运行前停止；不要删参考或样本以偷偷使预算表好看。

## 最终事前归因与readout收口（Root审查，尚无新特征）

新增唯一控制E，不增加方法候选或forward。因为c=(P_latent^Q−mean P_latent^N)+(mean p4^N−p4^Q)，主方法的任何收益可能只来自已有概率对比而非latent目标内容。定义原始E方向 e_prob=mean(p4^N)−p4^Q；用pQ4中心化，并按主c中心化后的同一P范数缩放：

C(x)=x−sum(pQ4*x)，n_c=sum(pQ4*C(c)^2)，n_E=sum(pQ4*C(e_prob)^2)；
e_E=C(e_prob)*sqrt(n_c/n_E)；Z_E=FP32(FP64(Z_M3)+e_E)。

n_c严格0时e_E=0；n_c>0且n_E<=64eps64*maxabs(C(e_prob))²时记unmatched，整轮不合格，不加epsilon或换方向。D使用相同退化判据与主范数目标。所有范数匹配在FP32最终舍入前核验；各输出重新完整V log_softmax。E不可晋级。主方法必须严格超过E，并且主对E的配对净纠错>=1，否则不具备新增latent目标的归因资格。原所有其他gate继续保留。E无温度、符号或强度搜索。

原精确native2/3检查明确覆盖首题首候选的Q、N0、N1、N2四种上下文：每种分别独立运行native2与native3，共8个浅深度forward，与共享native4 forward/cache对应深度fullV逐元素exact比较；不是只检查Q或N0。它们均计入同一300秒worker，不另开profile。对应encode_pair与已消费prefix文本按原映射核验；所有原baseline、M2/M3和native sums/alpha仍exact0，不放容差。

此补充优先于此前控制清单与readout范围的简写。Sol现在只做实现及CPU synthetic；最终源码/协议审查和公开锁之前不运行真实特征。完整链超当前预算的事实保持，只有正面pilot后才形成具体预算请求。

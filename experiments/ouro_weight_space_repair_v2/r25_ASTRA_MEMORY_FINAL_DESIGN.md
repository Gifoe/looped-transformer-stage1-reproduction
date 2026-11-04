# R25 Astra：固定 fullV Hellinger 1NN 经验记忆修复

状态：设计合同，未提取 key、未计算真实 query、未运行模型。只允许 Sol 实现和合成检查，实际 pilot 须 Root 静态审核后授权。本文件不修改 R24 或其他冻结文档。唯一候选；不并行其他机制。

## 证据和决定

Root 报告 R24 完整 576 CPU 支持审计通过：全部合法 value 有 104 types / 143 occurrences；specific 为 90 types，来自 24 demos。R6 错误题中 specific 区分覆盖为 62/216，discovery 41/141、selection 21/75；wrong-nonanchor 背景为 433/1512。28.704% 与 28.638% 几乎相同，不能称语义富集或预计纠错率。此数字目前引用 Root 报告，最终锁须绑定 Root 独立复核收据。

允许一次有限 pilot 的理由仅是尚未检验的 key–value 配对：在模型预测分布相近的过去位置，真实观察到的后继 token 是否比保持同一检索几何和 value 频率的置乱关联更有用。它可被直接干预反证，区别于 R22 轨迹相关性和 R23 自生成目标。25 demos 不提供外部知识，Hellinger 也不是语义距离；成功先验弱。若主方法没有救回 R6 错误并保留旧收益，就关闭此具体路线，不改 k、距离、强度、value 子集或控制晋级。

## 1. 唯一合法 memory 与源码可行性

使用 R24 全部 143 个合法 occurrence，按原共享文本 token 位置升序；不只用 specific，不去重。每条为 (source token position i, demo ID, value token ID v_i)。value 在 demo answer 内且 i>0，key 预测位置是 i−1。固定 key 输入为原 25 demo 的逐字节共享前缀加固定 `Question:` cue，不含任何当前 question 或当前答案。

已核验 `scripts/ouro_cached_scoring.py::_run`：`use_cache=False` 时只调用 `model.model(input_ids=x, use_cache=False)`；随后对每个原生 normalized state 作 `index_select(1, selected)`，返回全部 recurrent depths 的所选状态。`r7_inference.forward_states` 使用同一 API，原 head 是 BF16 LM head 后转 FP32。因此设置原 depth=4，调用 `_run(key_tokens, positions=[i-1,...], use_cache=False)`，读取 states[3] 即可。无需新增 norm hook、mask、KV 或旧源码改动。禁止从既有 continuation cache/标量重建 demo keys。

提取前 CPU 验证 key 输入在所有 value 位置前后的 token IDs 与 R24 occurrence inventory 完全一致；原 24 问题 prompt 的共享区间亦逐 token 一致，禁止按字符串猜边界。record value i 和 key i−1，value 不得被误当成自身 key 输入状态。完整 key 文本包含后续 demo token，但原 causal mask 保证前驱位置不读它们；不修改此 mask。

运行中对同一个 key 输入额外原样 forward 一次，选取从 0 到最大 key 位置的全部状态，再 index-select 比较所有 key 的 native4 BF16 states 及 head 全 V logits exact。它检验 selected-position 管线，不宣称是独立的 causal-mask 证明。保留原 HF 调用参数/backend 审计。此重复 forward 和所有检查均计费。

## 2. 固定检索和数值合同

对每个 key 和当前 Q/N prefix，原 native4 BF16 head -> FP32 logits z -> FP64 softmax p -> sqrt 向量 a。不得直接用 h4 距离。对所有 memory occurrence：

    D_i = sum_v (a_X[v] - a_i[v])^2
    i_X = first argmin_i D_i       # 源位置升序，精确浮点相等取最早
    r_X[v] = 1[v == v_iX]

权重仅来自当前无标签完整 V 分布与固定合法过去 memory。禁止 topK、ANN、投影、温度、epsilon smoothing、按 target 查 memory。当前 target 只能在最终全 V log_softmax 后 gather。独立候选的未来 token、其他候选列表不进入 query 或 key；给定 prompt/已消费 prefix 加入其他候选须不改变输出。

实现使用 Torch FP64 直接逐元素平方差、沿 V 求和，逐 query、固定 key block=8；不以 `2-2 dot` 近似替换，不另造归一向量。块仅限制临时内存，不能裁剪词表。每个 key 的概率须有限、非负、归一；非有限/空 memory/非法 shape 立即停止。距离精确相等按源位置，不使用近 tie 容差挑 value。真实首个 Q、N0/N1/N2 query 各以独立 NumPy FP64 从相同 FP32 logits 重算全部 distances：最大绝对误差 <=1e-12 且 neighbor occurrence ID 完全一致；不一致停止且不解封效果，不切 CPU 赢家重跑。记录所有 query 的最小两距离差作描述，不据此筛样本或调算法。

## 3. 主公式：M3 上的单位 paired memory bridge

M3 直接复用原 `r7_math.token_logits(q1,q4,refs1,refs4)` robust 路径，不重写其 FP32 算术、退化规则或 alpha。数学记号 g=(zQ4−zQ1)/2、e_j=(zNj4−zNj1)/2、F=zQ4+g，p=pQ4，C_p(x)=x−<p,x>；原 alpha 是 clip(<C_pg,mean C_pe_j>/mean||C_pe_j||²_p,0,1)，按旧代码精确退化规则，Z_M3=F−alpha mean e_j。

    c = mean_j(r_Q-r_Nj), j in {0,1,2}
    Z_primary = FP32(FP64(Z_M3) + c)

若 c 全零，直接返回原 Z_M3 对象/数值，保证 exact baseline。三参考相同应走相同值的均值快捷分支；Q 和全部 N 检索相同 value 时 c 必须严格零。不额外 center 主 c、不调幅度、符号或步数。单位 logit bridge 不等于概率混合，也不是传统 kNN-LM 的忠实复现。

所有方法重新完整 V FP32 log_softmax，之后 gather 原 target、完整候选累加、仅一次原字符分母，原 Q4−Q5 late 项保持原实现。不能二次减 R6。N0=`Answer:`，N1/N2 是原 25demo/template 中当前 question 换成 N/A / Unknown；每个独立 encode_pair，并按相同已消费候选 prefix 文本映射位置。

## 4. 固定控制（均无晋级资格）

所有控制共用同一 key/query/neighbor，除 value 置乱外不重检索，不增加模型 forward。

* M3 原基线；M2、R6、A1、Fixed 原结果与实际重构 parity 对照。
* Q-only：Z_M3+r_Q；reference-only：Z_M3−mean r_N。
* Memory-direction-only：Z=c，再 log_softmax，加相同 late、字符归一。这是去除 M3 的有限 logit 能量消融，**不是**对 one-hot r 直接取 log 的生成概率；后者会产生 −inf，不能伪装可比的 memory-only baseline。报告名须明确为 memory_direction_only。
* Frequency：f(v)=memory 中 value 频率。对 Q/N 全用 f，paired 项 f−mean f 严格零，输出 exact M3。这是必过代数检查，不是另一个可选赢家；不能据此宣称消除了所有词频混杂。
* Value-shuffle：一次 `numpy.random.default_rng(2021).permutation(M)`，将 occurrence value 顺序置乱，key/source顺序不动。保存 permutation/hash、固定点与 value 未改变数；不重抽。主公式中的 r 换为 shuffled r，得到 c_S，输出 Z_M3+c_S。词频精确保留。
* Matched value-shuffle：只隔离内容方向与幅度。令 a=C_p(c)、b=C_p(c_S)，A=||a||²_p、B=||b||²_p、t=64 eps64 max|b|²。若 A=0，更新严格0；若 A>0 且 B<=t，标 unmatched、更新0，整轮不具晋级资格；否则更新 b sqrt(A/B)，加原 M3。不以 epsilon、换置乱或删位置修复 unmatched。norm error <=1e-10 为数值检查。raw shuffle 必须同时报告，不能只挑 matched。
* 三个 leave-one-reference-out：原 M3 base 不改，仅新 c 平均剩下两个 r_Q−r_Nj；不把 LOO 变成新 M3 或选择最好 reference。

置乱反证检验的是过去 key–value 关联是否有决策价值；单一 permutation 与 24 个重复开发样本不支持一般因果结论或显著性保证。

## 5. 合成、完整性与效应门槛

先冻结源代码、上述公式、memory inventory、R24 proof、原 R20/R23 同一 24 IDs、所有输入和 13 ledger hashes。不得按 support 或 R21 reachable 重抽；已有 cohort 包含 1 个 R6 相对 Fixed 的旧独有纠错。无 test/heldout/其他数据集读取。

CPU 合成必须覆盖：显式距离/全 V；NumPy/Torch；精确 ties/重复 keys/重复 values；value histogram；c=0 exact M3；frequency exact0；reference permutation/相同参考复制；候选扩展；target-free 函数签名；normmatch 与 unmatched；输入非有限失败。重复参考不一般保证均值不变，只检验复制全体或全部参考相同的情形。

模型执行须全部 24 完成、正常 supervisor、前后所有权重 hash 相同、original mask/backend/input 证据通过。首 Q 的 cached native4 与 fresh 原4depth full V exact；三个 N 的 first native4 与独立原4depth full V exact。M2/M3 alpha/logits/候选 scores 逐实际管线 exact 对齐已有同输入结果；Q1/Q4/Q5、原字符分母、candidate顺序/target签名及原 baseline 向量全部 exact。不得用 parity 公差掩盖 head 或 mapping 差异。

在所有完整性检查通过之前不读取新预测比较/partial accuracy。之后一次解封全部 24，并固定要求 **所有**：

1. primary 对 R6 rescue−damage>=2 且 damage=0；旧 R6 独有纠错全部保留。
2. primary 严格胜 M3、M2、Q-only、reference-only、memory_direction_only、raw value-shuffle、matched value-shuffle；与两种 shuffle 各 paired net>=1。
3. 每个 LOO 对 R6 net>=0 且全部旧独有纠错保留。
4. 无 unmatched、全部 numerical/parity 检查通过。

同时完整报告所有 controls/counts、配对 rescue/damage、全部 LOO、zero c 比例、neighbor/value 占用与 query/reference 相同率；这些描述不能触发方法修改。门槛是探索资源分配，不是统计验证。失败关闭本定义，不晋级控制，不换子组/sign/k/距离/value 子集。

## 6. 一次执行与完整链预算

余额由 13 个 ledger 各 used_seconds 加一次：已用 18641.79952767305 / 21600，剩 2958.2004723269492 GPU-worker 秒。CPU preparation 不动 GPU ledger。唯一新 pilot hard300，soft285停止开始新任务，SIGTERM285，SIGKILL299、最后1秒reap；内部 reserve5 保留。模型加载、key 两次forward、head、检索、baseline checks、收尾 hashes、退出全部计费，不另 warmup/profile。实际超时仍全计费；incomplete/overrun 不解封部分效果。此设计不授权补跑或扩大时限，任何后续变更必须公开新审查，不称原合同默认允许。

M=143，FP64 sqrt keys 存储约8*143*V bytes（V=65536时约75 MB，实际以配置为准）；精确检索 O(T*143*V)，另有 LM head 与三参考 forward。固定 key block8 控制 scratch；不因耗时换近似。缓存 Q state 只节约主干，不免 query head，也不提供 N1/N2 states。必须分别记录 key建立、key审计、Q head、N forward/head、检索、检查的实际时间，不宣称已实测可在300秒内完成。

完整后续必须先按 pilot 实测重新预算，不能用 pilot24外推声称成功。供资源审查的量级：旧三参考成本下，全576 cached-main 开发约1900秒起；额外64 confirmation约210秒起；val299含主forward约1470秒起；ARC1172主+三参考约5740秒起；总约9320秒起，尚未加 fullV 143-key 检索/新增审计。这些是历史成本粗估且开发/confirmation并非独立验证，不是获准读取未授权数据的清单。M3 的 N0 已包含于三参考成本，不能再叠加旧 M2 全成本重复计费。

剩余2958秒无法容纳完整链。只有本次固定 pilot 全部门槛通过，Root 才向用户提交具体预算与后续冻结方案；当前不索要泛泛预算、不自动运行其他集。若 pilot 失败，报告本机制在固定24上失败，不能把它泛化为所有经验记忆不可能。

## 交付给 Sol 的最小执行顺序

冻结/CPU合成 -> Root静态审核 -> 单独授权一次300秒 worker -> exact memory边界/原forward/key读出 -> 24题Q/N fullV检索与所有控制（先构造logits后target gather） -> 全部完整性与预算收尾 -> 单独一次label分析。设计阶段到此为止，未运行任何新真实 key/query 特征。

## v2 事前最终控制与预算澄清（优先于上述 v1 冲突处）

本补充在任何 R25 实现、key/query 或真实方法特征之前制定。保留以上 v1 文本作审查记录；不修改任何已锁 R24 文档。修改依据是离散 occurrence 置乱的结构性质，不是准确率、邻居或修正结果。

### 唯一控制变更及其证明

接受 Root 提议：**保留原 raw occurrence value-shuffle**；将 v1 对该置乱做 norm matching 的控制替换为独立命名的 **freq_bijection_matched**。两者是两个不同 null，不得称新控制是 raw occurrence shuffle 的 matched 版本。

设 n(v) 是全部143 occurrence中 value v 的次数。将104个已出现 value ID 按 n(v) 分层。固定构造：新建独立 `numpy.random.default_rng(2021)`（不共享 raw occurrence shuffle 的 RNG 消耗状态）；频次数值升序遍历，每层先将 value ID 升序排列，调用一次 rng.permutation 于该层数组，按排序前后位置定义 tau(v)。singleton 层仍按同一构造处理；未出现的词表 ID 映射到自身。保存全 V 映射/hash、每层大小、各层固定点及全部 occurrence value 未改变数。不重抽、不强制 derangement、不以固定点多少决定换 null。

在相同已选 neighbor occurrence 上，用 tau(v_i) 替换 value，得 r_X^tau 和 c_tau=mean_j(r_Q^tau-r_Nj^tau)。这是全 V 坐标置换 P_tau，因而 c_tau=P_tau c。每个 v 的新 occurrence 次数为 n(tau^{-1}(v))=n(v)，所以精确保留每个 token 的频率；双射不合并原本不同的 value，故 c_tau 全零当且仅当 c 全零，非零坐标数量及其值的多重集合不变。这里保留的是**置换后的零模式/全零资格**，不是声称零坐标仍位于相同词表 ID。

旧 raw occurrence permutation 并非 value-ID 双射，可把原本不同 neighbor 的 value 变为相同，从而使 c 非零而 c_S 为零；这是合法 raw null 结果，应原样评分。新控制避免此种碰撞导致的无法匹配，不保证在 p 加权范数下永不退化。

freq_bijection_matched 的公式：a=C_p(c)，b=C_p(c_tau)，A=sum p*a²，B=sum p*b²，t=64 eps64 max|b|²。A=0 时更新0；A>0 且 B<=t 时 unmatched、更新0、整轮不具晋级资格；否则更新 b sqrt(A/B)，加原 M3。原数值 norm error<=1e-10、不加 epsilon 的规则不变。全 V p 的有限精度下溢/极小质量仍可能触发退化，不能用数学正概率假定取消此检查。

主方法、Q-only、reference-only、frequency、memory_direction_only、LOO、raw occurrence shuffle 均不变。最终 gate 中的 `matched value-shuffle` 统一替换为 `freq_bijection_matched`；primary 仍须严格胜 raw occurrence shuffle 和 freq_bijection_matched，且分别 paired net>=1。不是放宽准确率门槛。

限制：频次单例层或置乱固定点可能使很多值不变，甚至整个 tau 为 identity；此时 matched null 可能等于主方法，严格胜出 gate 自然不能通过。不得删除该控制、改 permutation 或事后称 gate 无效。即使通过，也只支持相对于这一固定频次约束关联干预的证据，不能称对所有词频/语义混杂都已排除。

额外 CPU 合成必须验证：tau 全 V bijective；每个 value 直方图完全相同；包含重复 value、singleton frequency class、固定点和 identity tau 的情形；所有构造 query/reference one-hot 下 c_tau exact 等于坐标置换后的 c；全零资格/非零数/坐标值多重集合不变；原 raw occurrence shuffle 能构造 c 非零但 c_S 零的反例；p 极小质量下仍按既定 unmatched 停止。不能为了验证而运行真实 query，map 的构造只使用已冻结合法 memory histogram。

### 预算与样本规模纠错

v1 的“额外64 confirmation”错误缩小了原完整链要求，现明确撤销：confirmation 必须是原**完整517题**，不是64子样本；已有 main cache 不代表三参考和全 V检索免费。开发576、confirmation517、validation299、ARC1172的范围均不缩小。本补充不授权当前读取517/299/test逐题；仅修正未来完整链预算。

沿用 v1 的历史三参考粗估口径：576开发约1900秒起；517 cached-main confirmation按1900*517/576约1705.4秒起；299 validation约1470秒起；1172 ARC约5740秒起；合计约**10815.4秒起**，未含新增 key/检索/审计成本，也不是运行时保证。原9320估计撤销。实际耗时须在正面pilot后按拆分profile复核；余额2958.2004723269492与单次300合同不变，不能靠缩小confirmation使预算显得可行。

实际 pinned vocabulary为**49152**。FP64 sqrt-key数组为8*143*49152=56,229,888 bytes（约56.23 MB / 53.625 MiB）；撤销v1的65536例子。其他 logits、probabilities、scratch与模型驻留另计，不能将这个单数组容量当总显存。

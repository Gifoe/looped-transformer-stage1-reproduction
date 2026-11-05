# R33：逐词表共同概率增益消除

状态：exploratory设计，Astra选择唯一主候选，交Sol实现/CPU合成，Root审查后才能单独冻结/START。没有新模型执行。原18ledger20755.46525352029/21600，余844.5347464797087秒。R28/R30/R31结论不变，R32不执行。

## 三个可计算分支与唯一选择

**A，主候选：逐词表共同概率增益消除。** R6成功处理的是Fixed相对native4的概率增益中与参考共有的部分；旧prefix估计器用一个全局alpha移除一条方向，可能在某些坐标移除超过该词本身的增益、或把参考与问题相反的更新也耦合进去。A直接以同位置完整词表的实际log-prob增益构造逐坐标共同部分，只有Q和三参考同向的交集才扣除。它改变选择性而非更换C时钟/符号/强度，不需要自然QA donor。收益仍未知。

**B，absolute-density contamination responsibility。** 令p_F为Q literal Fixed分布，b为三参考literal Fixed分布算术均值；等先验污染模型下r_v=p_F(v)/(p_F(v)+b(v))，可定义Z=F+log r并重新全V归一化，zero-neutral p_F=b时分布回p_F。它使用绝对概率密度比，而非循环增益，完整支持且不训练；但不保证混合污染模型为真，可能只是PMI式偏爱问题词。R15否定的是候选汇总DeltaI作为价值标志，不是这项逐词表后验，但CAD的可靠上下文前提在ARC仍薄弱。本轮不执行B、不比较它挑赢家。

**C，menu-free自回答条件证据。** 用同模型原Q、无选项、greedy native4最多8个token/EOS产生r(Q)，N1同法产生固定r(N1)。辅助读出在保留25demo后固定插入`Possible answer: {r}\nQuestion: {original question}\nAnswer:`，接当前已消费前缀，得a/b fullV；可检验Z=F+logsoftmax(a)−logsoftmax(b)，当a=b严格F。方向来自独立生成内容而非候选菜单，区别于R11 joint候选teacher，也不直接改主prompt。风险是自我错误被放大，生成token虽不是真实未来答案仍须作为明确新辅助协议单独审查。必要负控是保持同一题但将r(Q)全部替换为r(N1)，及固定源内容打乱；完整解封后按free-answer恰等于原完整choice的子组描述wrong-self-answer损害，不能用该标签门控，也不能把未匹配字符串当错误真值。零匹配/打乱identity应如实报告。约192生成forward+最多842辅助prefixforward，真实长上下文可能接近/超过余844秒，且需新的编码/输出边界合同。本轮保留具体不同信息来源的fallback，不实施、不宣称新颖。

选择A是因为它直接改变旧prior removal的坐标选择性，CPU能证明边界约束，完整24可合法重用native缓存而不新增大批body。不是因为已保证它正确，也不是因B/C违法而全面否决。

## 四张证据卡及文献边界

1. R6候选级先验扣除确有旧收益，但菜单依赖，不能原样当prefix算法；R11 742/1172是候选空间结果，不用于挑此次方案。A借其Fixed−native4增益定义，不借其test逐题。
2. R15已测上下文偏好汇总标量无可靠价值；R20/R23/R25及A/C干预失败。A不是把这些标量阈值化；它仍可能失败，必须直接测纠错/损害而非norm。
3. R31只证明当前Fixed/C二元prefixfamily最大13，不约束新逐词表分布。A不使用oracle位选择，不训练controller。
4. [Calibrate Before Use](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf)以无内容输入估计输出偏置；[DExperts](https://aclanthology.org/2021.acl-long.522/)已采用expert/anti-expert产品；[NWCAD](https://aclanthology.org/2026.findings-acl.473/)的官方摘要给出no-context/uncertainty两阶段gate；[CoCoA](https://aclanthology.org/2025.emnlp-main.348/)官方摘要采用entropy gap、peakedness和distribution divergence。此次只对前两篇相关机制及后两篇官方摘要作有限核查，不是完整novelty certificate。“自适应、减少误伤、contrastive calibration”都已有；A待检验差异是逐vocabulary坐标共同增益集合，不是token-step单scalar不确定度。论文新颖性未建立。

## A的唯一精确公式

同一原true-prefix位置，取Q1/Q4及N0/N1/N2 native1/native4 FP32 readout，全部来自R28合法单位置BF16 state重放。先按原路径形成literal FP32：

```
gQ = float32(0.5*(q4-q1)); FQ = float32(q4+gQ)
gNj = float32(0.5*(nj4-nj1)); FNj = float32(nj4+gNj)
G = log_softmax_FP64(FQ) - log_softmax_FP64(q4)
Ej = log_softmax_FP64(FNj) - log_softmax_FP64(nj4), j=0,1,2
cplus[v]  = min(max(G[v],0), max(E0[v],0), max(E1[v],0), max(E2[v],0))
cminus[v] = min(max(-G[v],0),max(-E0[v],0),max(-E1[v],0),max(-E2[v],0))
c = cplus-cminus
p = softmax_FP64(q4)
h = C_p(c), using original shifted centering x-x[0]-sum(p*(x-x[0]))
Z = original FQ exactly if c is exactly all zero
    else float32(float64(FQ)-h)
```

**G/E明确定义为log p(literal Fixed)−log p(native4)**，不是raw loop4−loop1，也不是log p4−log p1。最接近R6实际候选guidance/prior的逐词表版本，但FP64估计器不冒充原FP32目标LL逐位相同。原FQ完全不改；不能先用normalized G重写FQ以消常数而悄悄改变baseline。

三reference min来自“各reference均支持的共同部分”假设，不试median/mean/子集/符号。reference顺序和重复同一reference不改变min，数学上gauge-free（输入literal logits各自常数平移不改其normalized增益）；实数gauge性质不承诺任意巨大FP32常数平移仍位级相同，原literal FQ计算的舍入单独保留。

若Q或任一reference增益在某词为零，该词c为零；方向不一致也为零；全零c严格原Fixed，完整评分严格A1。全部ref与Q相同，在精确算术下c=G，新分布退回native4（再加原late）；实际FP32输出只检查预声明舍入误差，不声称literal native4全V exact。非有限值invalid；不引入温度、epsilon floor、topK、拟合强度或可学习门控。

## 可证明选择性及不能承诺的内容

每个词上sign(c)与G相同，|c|<=|G|；若G>0则0<=G−c<=G，若G<0则G<=G−c<=0。这是**被去除共同增益后的未重新归一化坐标残差**约束，不是最终每词相对native概率增益不反号，也不是pairwise margin保护、正确性保证。

反例必须进CPU合成：native分布(.2,.3,.5)，Fixed分布(.4,.29,.31)，三参考native均(.1,.8,.1)、Fixed均(.2,.65,.15)。共同规则移除前两坐标的全部G、第三不移除；新未归一化质量(.2,.3,.31)，归一化后第二坐标约.37037，高于原native .3，即原第二词负G变成最终正概率增益。不得用局部不反号定理宣传最终do-no-harm。

A之所以可能改变错误方向：不再由少数全V高权重坐标决定一个全局alpha，对相反/不一致reference响应不作扣除；对一致词也不超过该词自身的已施加增益。它可能因此保留原投影错误扣掉的词，也可能保留错误偏置；这正是pilot要证伪的假设，不保证改进。

## 固定controls、reference稳定与原评分

同fullV上计算M2/M3/J2泛先验实际输出；另固定下列不可晋级controls：

- generic-matched：原raw generic `.5*(nj4-nj1)`三参考center_p均值方向，匹配主h的p范数，避免对照自己的alpha为0导致不能匹配。
- matched_scale：gc=C_p(gQ)，rho=||gc−h||p/||gc||p；gzero且hzero取1，gzero而h非零标unmatched。Zscale=float32(q4+rho*gQ)，主hzero显式原Fixed。rho不人为clip，匹配的是residual guidance范数，不必总是缩幅。
- vocabulary-null：seed2021固定词表置换h、center_p后匹配h范数；不得重抽。
- reference-consensus-matched：仅对E0/E1/E2取同样signed min，移除Q的逐坐标clamp，再center并匹配h范数。它检验Q本身的上限约束是否贡献，不能按结果替代主法。
- LOO_N0/LOO_N1/LOO_N2：分别删除一reference，同min规则。只作稳定性，不择优。

所有normmatch延续64eps退化规则：主norm零时Fixed；主非零而对照退化为unmatched，整轮不得strong或cost资格，不epsilon补救。输出全部重新FP32 fullV log_softmax，再target gather/原位置FP32一次sum/原字符分母一次；Q4−Q5 native late保持，禁止二次R6。候选增加/复制不改变当前prefix输出，缓存键不能带候选菜单。相同prefix不同nexttarget只在最终gather分开。

## 缓存ABI、零新CF、一次完整24

沿R30已执行的cache admission：1588 native任务、6773 BF16单posstates，原input IDs/depth/readpos、模型269参数、runtime/source/backend、[1,1,2048]head、NPZ rawhash全部绑定。没有旧KV或teacherforced cache，无R29 padding。全部readout通过同BF16 LMhead→FP32重放，head计算不是免费CPU恢复。

第一遍3597 heads，完整11类旧baseline每targetLP/sum/normalized/late/R6对R28标签免费SCORES strict0，全部9历史代表fullV exact；通过后第二遍3597heads流式计算新方法/controls。仅额外一个无hook fresh Q（原fixed exact_duplicate_Q_task）depth5，与完整cache states及已保存Q1/Q4/Q5 heads exact；不做source干预，body调用总**1**，body heads3，总7197heads。末尾再核旧baseline exact、269参数及cleanup。代表新fullV和数值诊断保存供Root独立复算，不能宣称所有历史fullV都被存储/核过。

CPU合成先做NumPy/Torch独立实现、原Fixed零分支exact、sharedgain区间定理、上述归一化反例、ref permutation/duplicate、LOO、candidate extension、gauge的实数边界、非有限拒绝、所有norm控制与故意unmatched。没有实际新head前向/效果前先冻结这些规则。

complete24+numeric+params+ledger+clean且无unmatched后唯一解封。strong保持对历史/matched R6各rescue>=2、damage0、两种旧独有收益全保留；严格胜M3、generic-matched、matched_scale、vocabulary-null、reference-consensus-matched；三LOO各对两种R6净>=0且旧收益保留。cost_review_eligible保持numeric/matched通过、对两R6净>=0和旧收益全保留，仅预算资格非成功。全部baseline/controls完整报告；不改门槛迎合1.3pp小效应，也不以小pilot失败否定所有校准。

## 资源与后续

R30实测两遍cache IO+head合计约22.75秒，load+前后hash约30.35秒，native-baseline归约约8.61秒，原main math约31.42秒；本轮无778 CF或额外hook检查，主数学/controls数量仍需实测。保守计划两遍head/IO30、load/hash35、native+main math60、onebody+代表/IO/退出25，合计150秒。建议一次hard200、soft185TERM、199KILL/reap1/reserve5；这是待Root审计的上限，不保证完成、不额外profile/补跑、不边跑扩时。完整actual记第19ledger，名义余额644.53，绝不重置cap。拟议并未授权GPU。

全链仍576+517+299+1172：这24之外没有同primitive完整native cache。已授权576图的native only任务10101Q+3*8669=36108；以R28实测Q约.642秒、N0约.109、N1/N2约.509/forward粗算，开发模型body本身约1.62万秒，再加head/检查。confirmation/validation/test不可用已暴露逐题调方法或缩水；用开发每题粗外推完整2564题量级约7.2万秒加overhead（仅规模预估，实际去重不同），远超当前余额。positive或cost资格后才做完整预算review，不自动申请或开始新6h。

## 一次重叠/证据挑战后的收口

重叠：A是calibration的估计器变体，不能将坐标min或reference一致性称新原理；需实际超R6/匹配控制后才谈贡献。证据：旧context偏好无价值、全V归一化可反转局部方向、三参考高度相关，均降低成功先验，但不使固定一次试验非法。修订已锚定原literal Fixed、选择最接近R6的实际概率增益、加入generic/scale/null/Q-clamp归因、明确全V归一化反例。当前唯一推荐进入CPU实现的就是A；B/C不并行、不自动fallback，不用pilot选三者赢家。

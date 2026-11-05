# R34：绝对密度责任重加权，唯一待审B

状态：Astra设计；仅交Sol CPU实现/合成，Root审核、独立公示readback和START之前不运行模型/head。R33 A关闭，不重调其signed-min/reference/强度；C menu-free self-answer仍不实施。19账累计20834.389865819365，余额765.6101341806352，原21600总cap不变。

## 为什么是不同假设，为什么仍可能失败

R33估计的是Fixed相对native4的循环概率变化交集。B使用**当前Fixed输出的绝对密度**与无问题参考Fixed密度的比值：一个词即使循环增益很小，若在参考中极常见但在当前问题中不受支持，B也会抑制；一个词虽有同向循环增益，若当前问题远比参考支持它，B不会因该增益共同而优先扣掉。这是不同信息对象，不由R33两个损害case选词或调参数。

假设是：一部分错误偏好来自无问题上下文的高密度输出，Q-vs-reference density可识别这种污染。它不等于“问题独特的词更正确”；ARC题干可能强化错误联想，正确答案也可能是通用词。R15的context偏好标量失败、CAD依赖可靠上下文的局限仍适用，不能当作已经获得truth evidence。

此B在R33运行前已列为absolute-density responsibility分支。现在确定一个固定公式做独立试验，既不把control升级，也不以A失败证明B会成功。

## 固定公式及数学域

沿原native readout构造literal FP32 Fixed：F=q4+.5*(q4−q1)，Fj=nj4+.5*(nj4−nj1)，所有乘/减/加保持原FP32顺序。N0/N1/N2和当前已消费prefix均原样，不换文本、不添加生成答案。

设 `a=log_softmax64(F)`，`aj=log_softmax64(Fj)`。对三参考的完整aj数组作**exact重复去重**，保留不同数组各一份；设集合J，令

```
log b = logsumexp_j(aj) - log(|J|)       # probability arithmetic mean
ell = a - logaddexp(a, log b)            # log r, r = pF/(pF+b)
p0 = softmax64(q4)
h = -C_p0(ell)                         # original shifted centering
Z = float32(float64(F) - h)
```

不存在训练参数、global alpha、温度、稀疏截断或搜索；责任模型的两个来源先验固定各1/2。新词表分布在精确算术下为`p_new(v)=pF(v)*r(v)/sum_u pF(u)*r(u)`。这不是从pF减lambda*b的硬密度剔除，保持完整支持；也不是外部专家或更强teacher路由。

**精确零限**：若a与log b完整数组exact相等，显式h=0、Z直接返回原F；不得靠浮点logaddexp自己抵消常数。其余场合若centered h exact全零也直接F。于是neutral分布严格原Fixed，完整评分严格A1。没有neutral threshold。

域：所有raw logits/literal F/normalized logp finite；pF=exp(a)、各参考exp(aj)、b=exp(log b)、p0均逐v严格>0且finite。任何FP64 underflow到0判numeric-domain invalid，不floor/重温度。main h范数必须finite且 `(norm2==0)==all(h==0)`，LOO同检；不允许正测度但范数乘积下溢假零。对照仍按原64eps unmatched，不把unmatched当zero main。

reference permutation、添加exact duplicate distribution不改变J及primary；数学上各logits的常数平移不改变normalized densities。浮点巨大平移可改变literal Fixed的舍入，不能称所有FP32平移位级不变；CPU将估计器的gauge性质和literal baseline分开测试。LOO按原三个reference编号各去掉一个后重新exact去重/求均值，不能挑赢家。

为不把参考输入顺序变成FP64求和顺序，去重后的aj在每个v坐标上按数值升序排列，再沿reference维logsumexp；这不改变算术平均的数学定义，但固定了有限精度归约次序。不按效果或语义排序，不需要下载整词表给CPU。exact duplicate判断针对完整normalized logp数组，不是相近/同argmax。该排序及去重规则同样用于LOO并进入合成测试。

## 有意义的选择性保证与限制

令r=p/(p+b)，p,b均为归一化正分布。由凸性或Cauchy可证`Zr=sum p²/(p+b)>=1/2`，且r<1，故精确算术下 `0<p_new(v)/p(v)<2`。高密度比的奖励被饱和：`ell=-softplus(-log(p/b))`；与线性PMI不同，不随很大的p/b无限奖励某词。

这个2倍界只约束**相对Fixed单词概率的上升**，不是下降下界、答案LL/长度归一化margin界、R6保护或准确率保证。一个参考支持的正确词仍可能被大幅降权。FP32最终输出只检查预声明误差内的数学实现，不把实数界冒称所有rounded词概率严格2倍。此处可证性质比“norm有影响”更直接地限定分布重排，但仍须实际救伤。

## 与已有方法的边界

[Calibrate Before Use](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf)已用content-free输出校准偏置；[CAD](https://arxiv.org/abs/2305.14739)强化有/无上下文的概率对比；[DExperts](https://aclanthology.org/2021.acl-long.522/)已有专家与反专家乘积。B是这些概率校准框架中的一个bounded likelihood-ratio重加权假设，不宣称新原理。[NWCAD](https://aclanthology.org/2026.findings-acl.473/)与[CoCoA](https://aclanthology.org/2025.emnlp-main.348/)已有防误伤/上下文适应性讨论，不能把“自适应”或“低误伤”本身当贡献。

不同于R33 normalized增益交集、R30来源clock、R16优化旧KL、R23潜在target估计、R15题级DeltaI门控；没有将任何已测失败公式调符号或alpha。与普通PMI最直接的区别是密度比的固定softplus饱和，必须用matched PMI控制检验，而不能仅与很差的随机null比较。

## 固定不可晋级controls与原评分

保留native/Fixed/A1/M2/M3/J2及双R6基线。共同fullV上构造：

1. G-matched：原三ref `.5*(nj4−nj1)`的center_p0均值方向，匹配主h范数，F减之。
2. matched_scale：gc=center_p0(.5*(q4−q1))，rho=||gc−h||p0/||gc||p0；主hzero显式F；gzero且h非零unmatched。其余沿R33，不clip rho。
3. vocabulary-null：固定seed2021 permutation主h，center并匹配主norm，F减之。
4. **PMI-matched**：定义 d=a−log b；为统一F−direction记号，取方向`−C_p0(d)`，匹配主h范数后F减之。它对比线性density-ratio与B的饱和选择性，替代R33专属consensus控制，不是新的可晋级方法。
5. 三LOO：N0/N1/N2各移除一次，其他公式完全不改。全部报告，不择优。

以上norm controls：主norm0全返回F；主非零而对照norm<=64eps*maxabs(direction)^2为unmatched，保留raw诊断并关闭资格，不epsilon或重抽。不得为了matching改变主h。

全部Z重新FP32完整fullV log_softmax，之后才gather实际target。原位置FP32 buffer一次sum/原字符分母一次/Q4−Q5 native late不变。nexttarget只参与最终取值，不参与a、b、r、h、norm、LOO。原prompt/25shots及candidate-extension不变；不输入菜单字段。主参考不需要新的forward。

## ABI、CPU合成与完整24执行

原R28 native cache仍是1588任务/6773 BF16单posstates；严格复用R30/R33 ABI校验和source依赖闭包，current weights269哈希、runtime/backend/config、inputIDs/depth/readpos、[1,1,2048]head、NPZ rawhash全部核。无旧KV、无R29 canonical、无新CF。

两遍各3597heads；第一遍完整11类旧baseline逐targetLP/sum/normalized/late/R6 strict0，九个保存fullV代表exact，然后唯一fresh Q无hookdepth5（原同代表）五state/三个head exact。第二遍生成B/controls并末尾再验旧baseline。总7197heads、1body，保留原数字检查前无gold/argmax顺序，任何失配invalid，不退回fresh全部native重跑。

CPU合成至少：neutral非平凡F rawexact、ref permutation/duplicate/三LOO、gauge实数测试、原literalF不被normalized表达式替换、posterior式与direct normalized-density独立一致、2倍实数界及相近线性PMI不同的例子、非有限/underflow两类拒绝、normmatching/unmatched、candidate extension、所有零限、全部scalar/pieces与late/新control完整性负例。不得用真实24新head做预先调参。

## 事前门槛与资源

完整24/clean/numeric/params/ABI/19旧ledger继承且无unmatched后唯一解封。strong：对历史和matched R6各rescue>=2、damage0、所有旧独有收益保留；严格胜M3、J2、G-matched、scale、vocabulary-null、PMI-matched；三LOO各对双R6净>=0且旧收益保留。cost_review_eligible仅全资格通过、对两R6净>=0且旧收益保留，仍不是性能成功或后续GPU授权。不因B分布性质放松这些门槛。

R33同cacheonly范式实际78.9246秒；B多一个logsumexp，少signed-min与其控制，没有额外body。规划120秒含load/hash/head/math/IO与退出余量，建议一次hard150/soft135TERM/149KILL/reap1/reserve5；待Root最终审定，非完成保证。总actual记第20ledger，200/150不随运行延长，无独立profile、无补跑。150后名义余额615.6101秒，仍不足完成完整链。

576+517+299+1172不得缩。除本24外缺原true-prefix native缓存；已有576 native-only36108任务、按实际R28速度约1.62万秒body的预算限制依旧。fullchain仍约数万秒，不以本次head-only78秒外推部署。只有原cost资格成立后，才另给完整实际预算review；不能自动扩6小时。

失败则关闭这个预定equal-prior soft-responsibility公式，不扫prior mixing/temperature/参考/符号，不把PMI控制升主法。C的新信息路线需独立设计和wrong-self-answer反证，不自动接力。当前推荐仅B进入CPU实现，理由是不同概率信息对象与可审的饱和选择性，而非已有性能承诺。

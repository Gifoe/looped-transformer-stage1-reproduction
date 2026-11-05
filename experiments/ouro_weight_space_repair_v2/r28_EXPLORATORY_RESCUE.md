# R28 exploratory rescue：来源配对反事实先验

状态：Astra design only，2026-11-05；没有新增模型执行、特征或 GPU 授权。应用 ccf-idea-optimizer exploratory 模式。本文保留所有旧文件，修正 r28_ASTRA_DESIGN_PROPOSAL.md 对研究空间的过宽排除：R19 的特定保护轴和 R14 的特定残差失败，不推出所有新来源先验加旧投影器都不可检验。实验收益仍完全未知。

## 决定与四张证据卡

选择 **A，完整等长 demo-answer V 块重排，只用于 N1/N2 的先验估计**，作为唯一拟议 ARC pilot。C 是机制不同的后备研究问题，不是 A 失败后的自动第二候选。B 暂不落实。当前定位为 CCF-A 方法研究的探索性机制实验，不是已具发表证据的创新方法。

1. **本模型功能证据**：R27 的固定 V 路径在四个 lookup recipient 上全部恢复对应标签，身份、source-query 独立、前三 loop、参数检查通过；K 主结果不支持同样解释。它支持 V 可以传对应内容，不支持自然 demo 贡献有害，也不支持 ARC 正确性。
2. **实际自然字段可执行性**：r28_ANSWER_BLOCK_INVENTORY.json 及 r28_ROOT_BLOCK_INVENTORY_VERIFICATION.json 已独立核验 25 个完整 answer 块、143 token。等长组循环允许 23 demo /106 token 移动，实际 98 token ID 改变；11/26-token 单例不动。不裁切、不 padding、不打乱块内顺序。该证据解决先前关于自然多 token 单位无法定义的具体问题；V 仍携带原问题及位置，不是纯 binding。
3. **先验估计仍有未检验来源**：R20/R23/R25 未救回旧 R6 错题，R21 六深度只显示 59/216 错题在该分数族中可达；这些不保证新估计器有效，也不排除它。R25 从 LM-head 分布取词面最近邻，与 A 的原生 contextual V 来源干预不是相同机制。
4. **最近工作约束新颖性**：[Feng–Steinhardt ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9d1b7fc578c0d2d6431fc26d736ecaf3-Abstract-Conference.html)研究实体/属性对应的因果绑定；[DeCoRe](https://arxiv.org/html/2410.18860v1)已用检索头干预的输出差做解码。因此不能把“干预后对比 logits”称新原理。[LPT](https://arxiv.org/html/2610.02383v1)训练第一递归写共享记忆的架构；[FlashLoop](https://arxiv.org/html/2609.29812v1)已做 training-free 状态复用、稀疏 attention 和 KV residual 压缩。C 不能宣称早期 V 复用首创。这些工作未在已读内容中建立本任务的 reference-only prior identification；新颖性仍需完整研究后判断。

## 三个分支及一次挑战后的修订

| 分支 | 来源与操作 | 可检验问题 | 当前缺口 |
|---|---|---|---|
| A，主线；从 R27 transfer/refine | N1/N2 全部等长 answer 块，在 loop4 每层 V 上按原 demo 顺序循环映射 | 问题缺失参考中的配对来源响应，是否比泛循环 guidance 更准确识别应扣除的先验？ | 来源响应同时包括上下文/位置效应；方向符号及效用未证 |
| B，factorial instrument | 自然 question/answer 字段独立操作，再计算两因素交互 double difference | 是否能区分内容主效应与配对交互？ | 自然字段的两个独立、长度匹配且不互相改变语义的操作尚未定义；四格差不自动成为纯交互因果解释，成本更高 |
| C，不同 fallback；source-clock invert | 同一次参考 forward，各层 loop1 capture 全部 answer V，loop4 同位置替换；K 与 Q 原生 | 来源随循环演化产生的贡献是否构成可扣除参考先验？ | 避免 A 的空间重排，却引入时间分布外干预；R27 未证明来源演化有害。必须独立事前设计，不能按 A 效果选符号/clock |

旧文档/脚本的定向 source-clock、loop1-to-loop4 V 搜索未发现已实施同方案；这是有限搜索，不是代码全局不存在证明。R20 是问题 V 消融，R27 是同一最终 loop 的 donor replay。

**重叠挑战**：A 不是新投影器，更不是新 activation patching；若贡献只能描述为“换个差向量”，论文价值不足。修订后研究主张限定为“由明确自然来源干预识别参考先验”的可证伪实证，而不是几何创新。

**证据挑战**：自然 source 的响应不等于应该减去；R27 仅表明通路能传内容。修订后不保护该轴、不将其增强为 truth signal，只在无真实问题的 N1/N2 上估计 nuisance 候选，保持固定非负有界投影，并强制比较旧泛 guidance、幅度匹配 null 和 R6 救伤。若没救伤收益，关闭此确定方案，而不是宣称全类因果先验不可能。

## 唯一 A 公式与原始数值约定

每个已消费前缀 t，Q 为原问题，N0 为 Answer:，N1/N2 保留原25示例/模板，仅当前 question 字段替换 N/A、Unknown。独立编码相同已消费文本，不读取下一 token、未消费答案、候选标签或 gold 来构造任何方向/系数。N0 没有来源块，因此不加入 A 的 J=2 平均，也不人为添加零参考稀释系数；只用于基线。

在 N1/N2 原生 native4 forward，capture 每物理层 loop4 的全部143个 source V。CF forward 前三 loop 完全原生；loop4 每层将每个目标 answer 块替换为同长度组内下一 demo 的 **native donor** V 块。方向按原 demo 顺序锁定；单例 identity；全部其他行保持。不能从已干预上游产生的 V 重新取 donor。K/RoPE/attention mask/backend/权重不变，绝不干预真实 Q。

令 q1,q4,q5 为 Q 原生 FP32 fullV readout，n_j4 与 b_j4 分别为 N_j 原生/CF readout，j=1,2：

```
g = float32(0.5 * (q4-q1)); F = float32(q4+g)
e_j = float32(0.5 * (n_j4-b_j4))
p = softmax_FP64(q4)
C_p(x) = (x-x[0]) - sum_v p[v]*(x[v]-x[0])
gc=C_p(g); ec_j=C_p(e_j); ebar=mean_j(ec_j)
D=mean_j sum_v p[v]*ec_j[v]^2
num=sum_v p[v]*gc[v]*ebar[v]
threshold=64*eps64*mean_j max_v(abs(ec_j[v]))^2
alpha=0 if D<=threshold else clip(num/D,0,1)
Z=F exactly if alpha==0 else float32(float64(F)-alpha*ebar)
```

这保持 r7_math 的 centered FP64 projection 与 FP32 输出，而不是减 raw uncentered 均值另造舍入路径。相同参考退化可按旧实现去重，输出必须逐项对照独立 NumPy 实现。非有限值直接 invalid，不用 epsilon 救。e 全零严格退回 Fixed token logits；添加原 late 后严格为 A1，而不是 R6/M3，故旧收益保留必须实测。

对 Z **重新完整 FP32 fullV log_softmax**，之后才 gather 实际 next token。候选 LL 按原字符分母一次归一化；晚期项仍为原生 Q4 与 Q5 的候选 normalized LL 差。不得把候选 LL 当词表 logits，不二次扣 R6。所有候选完整计分，tie 规则固定原顺序。

## 真 prefix 执行与匹配基线

R26 表明内容因果检查通过但不同总长度可能位级敏感。本 pilot 选择 **true-prefix primitive**：对每个位置只 forward 已知 Q/N prompt 加已消费 token/text，原无 KV/无显式 mask/scorer backend/精度不变，单位置 head 形状恒为 [1,1,H]。不使用含未来长度的旧 Q state 估系数，也不要求新 logits 与旧 teacher-forced logits 位级相同。

CPU 事前构造全部 query 输入：原 Q candidate encode_pair 的前缀切片必须对应已消费 token；N012 仅用已消费文本独立编码，并核对与原映射边界。若原映射需要未消费字符才能确定该位置，不得偷偷读 future，应在执行前报告 tokenizer 合同不成立，由 Root 决定是否修订；不启动模型。相同已知前缀必须生成同一输入/长度，无菜单依赖。

同 primitive 从实际 fullV 重新计算 native1/4/5、Fixed、A1、M2、M3、J2 native-guidance projection，以及候选级 R6（其本身允许菜单依赖，只是基线）。历史 R6/M3 留作外部数值对照，明确标注 primitive 不同，不能把执行方式变化算成 A 的因果收益。要求新方法同时保护历史与 matched R6。

可记忆化完全相同 input IDs/depth/mode/checkpoint/backend/source-map 键下的已完成 fullV 输出；这是结果复用，不是 KV cache。不能复用不同长度的 source donor 或按候选菜单计算缓存键。随机/重复遍历顺序 CPU 验证不变，首个真实 duplicate 另作 exact replay。Q5 一次得到所需 native depths；N0 native4；N1/N2 各 native4+CF4。所有 controls 从这些共同 forward 构造，不另采模型条件。

## 固定对照与归因边界

1. 原生两参考 guidance e_j^G=.5(n_j4-n_j1)，同 projector 的未匹配实际输出；以及旧 M3/M2。这是“新来源估计是否超越旧循环先验”的强对照。
2. **G-matched**：取上述 e_j^G 的 centered 均值方向，按主完成 correction h=alpha*ebar 的 p 范数缩放，F 减该方向。它隔离贡献方向与修正幅度，不要求对照自己的 alpha 非零。
3. **vocabulary-null**：seed2021 一次全词表 permutation，置换主 h 后 center_p 并匹配主 p 范数，F 减该向量。不是充分的语义 null，必须与1/2联合解释；不得因它很差宣称成功。
4. 两个 LOO 输出分别使用 N1/N2，同公式 J=1，不能择优晋级。逐参考响应/alpha/修正范数完整报告。

任一 norm-matched 控制：主 norm=0 时输出 F；主 norm>0 而对照 centered norm 达上述 FP64 退化判据时标记 unmatched，整轮不能获得 strong 或 cost-review eligibility，不重抽 permutation、不 epsilon、不换方向。

身份 donor 是执行验收而非额外候选：首 N1/N2 的 identity replay fullV/state exact；前三 loop exact、source map/其他 V 行 exact、hook 调用24层×4、参数前后哈希、cleanup、原 uncached input/mask/backend source provenance通过。capture 不改变输出；安装/卸载/模式变化遵守 scorer fingerprint invalidate。测试代码必须含故意错 clock/错 source/未 cleanup 的负例。按 R27 无损记录代表性的 raw V/state/fullV 以供独立复核。所有 integrity 先于读取 gold/argmax。

## 一次有限 pilot 与停止规则

固定原24题完整 cohort、所有候选，不重抽/删题；这是反复使用的探索开发样本，非独立验证。CPU 先冻结 source/input/cache 任务表、全部 controls、上述公式/精度和检查。新方案只一项。Root 源码审查、公示冻结与明确 START 前没有 GPU。

拟议单 job hard1500秒、soft1485 TERM、1499 KILL、余1秒 reap，内部 reserve5不改；不是授权执行。所有 load/hash/IO/检查/退出 wall 都记第17 ledger，继承16旧 ledger total18930.428655151278，余额2669.571344848722，cap21600不变。无另行 profile/warmup、无补跑、无 partial effect 解封。CPU 成本准入失败则暂停执行设计，不能缩 cohort 或把更弱 primitive 静默替换进来。

complete24、allparams、supervisor、原始证据/数值、全部 controls matched 合格后才一次解封效果：

- **strong screen**：相对历史及 matched R6 各 rescue>=2、damage=0，所有两种 R6-only-vs-Fixed 旧收益全部保留；准确数严格高于 matched M3、J2 native guidance、G-matched、vocabulary-null；两个 LOO 各对两种 R6 净救伤>=0并保留旧收益。每项都输出原计数/配对而非只布尔值。这个 screen 是高信号筛选，不是检验1.3pp小提升的充分功效设计。
- **cost_review_eligible**：仅在全部 numeric/integrity通过、无 unmatched、对两种 R6 净救伤>=0且各自旧收益全保留时成立。strong失败也可仅审实际完整后续成本，不叫 positive、method pass 或准许新 GPU。
- 其他为该固定方案 stage closed；不泛化为全部来源干预无效。不得调符号、强度、层、source范围、cycle、reference或提升 control。C 不自动启动。

## 可审预算：未实测，不保证速度

CPU 必须给 UQ,U0,U1,U2：按完整已知输入去重的 prefix 数。24题原记录548位置，因此各 U<=548。旧成本是每题四候选总额，不能误当单 forward：Q5约1.62s/q，N0约.29565，N1/N2各1.3。粗略每 forward 系数为 .405、.0739125、.325秒；A每个N1/N2需两次。因此 planning T=.405UQ+.0739125U0+.650(U1+U2)。这是长度分布迁移的近似，非保证上界。无去重 T<=974.8秒；采用预算估算1.25T+100检查/load/IO，约1318.5秒，落在拟议1500内。CPU 准入要求该估算<=1400秒，且整个cap<=余额；不能把这个经验公式称硬实时证明。任何超时按实际计费且不分析部分效果。

完整后续仍为 **576开发+517 confirmation+299 validation+1172 ARC test**，不缩成小 confirmation。用开发13417位置/576题、无去重同上述率，仅模型路径粗估：开发约23867秒；confirmation约21422秒；validation约12389秒；test约48564秒，合计约106242秒（29.5小时），另加检查/IO；参考 prefix 去重会降低，Q也有题内共享，但没有证据能降至余额。这里只做预算预判，不读取 heldout/test逐题或新模型 profile。正式预算应由 pilot 实测 component time+完整合法工作量清单重算；任一 cost_review_eligible 后仍须 Root/必要用户具体预算审查，不自动扩 cap。当前1500 pilot即使完成，原余额也不足据现估完成目标。

## 交付与下一安全行动

执行资源澄清（仍为未冻结设计）：1500 是提议的单次上限，最终 cap 在 CPU 任务计数和 Root 审查后事前冻结。N1/N2 固定 reference 在不同题上也能共享相同已消费前缀；U1/U2 必须全24跨题去重。可以按 reference-prefix 任务分组流式处理所有依赖的 Q，保留 scalar LL/diag 后释放 fullV，不必将所有词表向量长期保存。相同 source donor 只在完全相同输入长度/IDs下复用，不能从短 prefix 猜长 prefix 的位级相同。

现在可以交 Sol 做 **CPU-only 的 true-prefix 输入/去重任务清单、独立数学/源 hook 合成测试、费用准入**；不抽 V、不执行新 Q/N、不读取 heldout。A 的可行性已超过只有名称的猜想，但效用仍是待证假设。若 token 边界或资源准入不通过，具体 rescue 条件是修订并公开 prefix执行合同或申请可审的实际预算；不能用它伪装成科学否定，也不能偷偷退回旧数值路径。

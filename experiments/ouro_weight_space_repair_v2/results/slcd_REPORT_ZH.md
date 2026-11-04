# 原始 Ouro 权重的训练免费 LoopCD 改进：最终报告

冻结主方法 **R6 prior projection** 在完整 ARC-Challenge 上达到 **736/1172（62.7986%）**，超过用户要求的 62.5%（至少 733 题）。相对 Fixed 717 净增 19 题；相对已有 A1 726 净增 10 题。没有训练或编辑权重。全部工作保留在既有 V2 文件夹。

## Q1. 原始 LoopCD 为什么产生有害改答？

完整 ARC 上，Fixed 相对 Native K4 改答 95 题：纠错 41、损害 26、错到错 28；净增 15。它把早期到 K4 的 token-logit 差继续外推，这个方向同时包含有用和有害的排序位移。训练门控、几何投影和能量约束均被实际检验；单纯降低变化大小并未稳定解决问题。分数分解不能证明隐藏状态缺失、事实知识来源或某个神经元机制。

## Q2. 实现了哪些方法？

初始五轮注册 47 种含控制的配置：A1 完整候选分数晚期反转；A2 token-logit 晚期反转后重算完整 log-softmax；centered/raw 正交和平行引导；候选一致性/分差门控；最小混合；系数邻域和参考循环。用户提高目标后追加 R2–R6：Fisher late-residual 投影、匹配范数/余弦对照、分数复合、Fisher-RMS、精确 categorical-KL 约束、答案域先验以及四种部分/自适应先验修正。全代码与被否定分支保留。

原始 Fixed 在每个目标 token 上使用 z4+0.5(z4-z1)，再做全词表 log-softmax，完整累加候选似然；方法背景见 [Decoding Looped Transformers Better for (Almost) Free](https://arxiv.org/html/2610.02185v1)。A1 的候选级修正不同于在 token logits 上做相同形式的线性算术；二者的 log-partition 项通常依赖候选历史。所有归一化均只除一次相同答案字符数，不按 token 数，不截断候选。

## Q3. 自主迭代发现了什么？

初始正交/门控在小子集上上升，却未通过新增 517 题确认训练子集。R2/R4 的方向或信赖域操作没有超过已有 A1；R3 最好仅多1题，未获追加测试；R5 直接全强度减先验没有通过 +7 训练门槛。R6 改用当前候选集合上、有界且不用标签的投影系数，才获得较一致的训练净纠错。

| round | best | correct | net | test |
| --- | --- | --- | --- | --- |
| R2 | r2_fisher_signed | 689 | -1 | 未测试 |
| R3 | r3_matchednorm_scorelate | 691 | 1 | 未测试 |
| R4 | r4_kl_trust_region_scorelate | 687 | -3 | 未测试 |
| R5 | r5_a1_earlyprior | 691 | 1 | 未测试 |
| R6 | r6_prior_projection | 705 | 15 | 仅选中投影：736 |

训练比较的 A1 分母均为 1093，正确 690。R2–R5 没有追加测试集候选评估；R3 最好只多 1 题，也未消耗全量测试。R1 五轮加用户目标扩展后的五轮共十轮，不能把整个历史写成只有五轮。推理公式不使用 gold、样本频率或答案出现次数。

## Q4. 开发选择与最终公式是什么？

原始 train1119 排除全部固定25示例并去重后为1093；划分384/192/517。R1 只在前三个晋级方法上做一次官方 validation299，A1=173、Fixed=168、Adaptive=172。后续 R2–R6 使用已暴露的同一1093训练样本和五个固定切片，没有新的独立验证集。

R6 projection 训练705，对 A1690 为22纠错/7损害；三子集净 +6/+1/+8；五切片净 +2/+3/+2/+4/+4。half=698、matched magnitude=692、quarter=690。通过预先规定的净增至少7题且至少3/5切片非负门槛后，只冻结 projection。五切片是反复使用训练数据的描述性稳健性检查，不是独立交叉验证。

令 S 为完整候选 LL/字符数，F 为 token-logit Fixed 的同口径分数：

```text
A = F + (S4 - S5)
g = F - S4
e = F_prior - S4_prior
gc = g - mean(g); ec = e - mean(e)
alpha = clip(dot(gc, ec) / dot(ec, ec), 0, 1)
        (dot(ec, ec) == 0 时 alpha = 0)
score = A - alpha * e
```

先验上下文是字面量 `Answer:`，保留完整候选续写，不保留题干或25示例。该先验估计文本/格式偏好，不能称为已识别的示例偏置。alpha 依赖当前候选集合；新增或删除选项可能改变排序。clip 后也不保证严格正交。分数是排名分数，不是校准概率。

最终锁 SHA256 `6de605b885b4fb9ff72b273ecc0c0484e9fe72fc5d51d793b22367db879b7da7`，在测试前公开于 GitHub commit `496602d2416f8114c5227720235c26e5c14e488d`。根目录 `FINAL_METHOD_LOCK.json` 仍是历史 R1 锁；当前主方法使用 `configs/slcd_R6_FINAL_LOCK.json`，不得混淆。

## Q5. 完整 ARC 是否达到目标？

| method | correct | n | accuracy_percent | net_vs_fixed |
| --- | --- | --- | --- | --- |
| native4 | 702 | 1172 | 59.8976 | -15 |
| fixed | 717 | 1172 | 61.1775 | 0 |
| adaptive | 716 | 1172 | 61.0922 | -1 |
| incumbent_A1 | 726 | 1172 | 61.9454 | 9 |
| r6_prior_projection | 736 | 1172 | 62.7986 | 19 |

模型 ByteDance/Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`，原始 BF16，GB10，seed2021，lm-eval0.4.9 acc_norm；同一组固定25示例，完整1172题。原始 safetensors SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`。

所有1172原始上下文分数在完整输入/模型/协议兼容核验后复用；R6 此次上下文 GPU 重放覆盖为0。新增先验完整覆盖1172行、4687候选，其中4408次新推理、279缓存命中。不能说此次重新对1172个主上下文做了 GPU 评估。

## Q6. 纠错、新错误和失败案例是什么？

相对 Fixed：35纠错、16损害、净+19。相对 A1：23纠错、13损害、12错到错、1124不改答。alpha=0/interior/1 分别覆盖419/562/191题；interior纠错17损害11，alpha=1纠错6损害2。419题严格回退为 A1。均为冻结后的描述性分析，没有据此改系数。

35个对Fixed的纠错中，10个恢复Native K4原本正确但被Fixed改错的答案，25个修复Native K4与Fixed都错的题。16个损害中，6个撤销Fixed的有益纠错，10个把Native K4与Fixed都正确的题改错。因此本数据上的净增19来自Native K4原错子集（25-6）；原正确子集上恢复与新增损害相抵（10-10）。这仍是结果分类，不是神经因果机制。

源顺序第一个纠错案例是行星自转加快：A1 选“重力增强”，R6 改为“行星日变短”，gold margin 从 -0.053662 变为 +0.004431。第一个损害案例是实验改变车速：A1 正确选自变量，R6 改成因变量，margin 从 +0.015070 变为 -0.010395。全48改答案例及全部4687候选的 g/e/alpha/修正项保存在 [案例报告](slcd_R6_FAILURE_CASES.md)、[JSONL](slcd_r6_arc_changed_cases.jsonl)、[候选分数](slcd_r6_arc_candidate_scores.csv)。这些例子展示了排序翻转，并未识别隐藏状态因果链。

## Q7. 统计与计算成本是否支持结论？

R6 vs Fixed：+1.6212pp，配对50,000次题级 bootstrap95%CI [0.4266, 2.8157]pp，双侧 exact McNemar p=0.0109736。R6 vs A1：+0.8532pp，CI [-0.1706, 1.8771]pp，p=0.1324982。因此“达到用户整数目标”成立，“可靠超过 A1”尚不成立。

本轮额外 worker 累计 13565.214 秒（3.7681小时），未超过6小时，包含失败/中止尝试；详见 [逐次成本](slcd_R6_compute_costs.csv)。worker 时间包含其 CPU/模型加载部分，不是 CUDA event 纯 GPU 时间；旧 V1/V2 缓存的生成成本不计入本轮新增预算。连接恢复未重置账本。

数学逻辑深度为主上下文K5、先验K4；需要的词表头为主1/4/5、先验1/4。实际收集统一运行共享K6轨迹，外部主上下文/先验长度不同。没有独立部署路径测速，不能声称延迟、FLOP或token吞吐收益。

## Q8. 是否跨任务泛化？

| benchmark | baseline | correct | accuracy | R | D | net | delta_pp | CI | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HellaSwag | fixed | 7259/10042 | 72.286% | 216 | 188 | 28 | +0.279 | [-0.110, 0.667] | 0.1791 |
| HellaSwag | A1_g1p0 | 7259/10042 | 72.286% | 135 | 115 | 20 | +0.199 | [-0.110, 0.508] | 0.22942 |
| OpenBookQA | fixed | 219/500 | 43.800% | 14 | 6 | 8 | +1.600 | [-0.200, 3.400] | 0.11532 |
| OpenBookQA | A1_g1p0 | 219/500 | 43.800% | 10 | 4 | 6 | +1.200 | [-0.200, 2.800] | 0.17957 |

两个外部任务对 Fixed 的点估计均为正，但是否可靠须结合下表区间；不能据此宣称普遍优势。

两个任务均为全量官方 split：HellaSwag validation10042，OpenBookQA main test500；官方 lm-eval0.4.9 零样本 prompt，acc_norm，同一冻结公式，无重调。A1 是固定历史对照。HellaSwag 复用1956行原始主分数，重放其源顺序首/中/末三题的八种readout完全一致；其余8086主上下文为新推理，不能把三题parity写成1956题独立重放。OpenBookQA500主上下文均为新推理。先验使用经授权 ARC 缓存或原始模型新推理，无独立历史 prior baseline。

另一项独立CPU检查从固定raw parquet与官方task定义重新渲染全部10542行，prompt、候选顺序、gold、请求边界和字符分母逐行完全相等，见 `slcd_R6_MANIFEST_AUDIT.json`。完整候选在各上下文分别按官方 encode_pair 边界编码；连续文本相同、字符分母相同，target token IDs 可能因上下文边界不同而不同，不强制裁剪成相同 token 段。外部任务与 ARC25-shot 的绝对准确率不直接比较。配对统计是描述性结果，未声称多重比较校正后的全局显著性。

## Q9. 是否继续，以及有哪些研究价值？

ARC 的62.5%目标已完成。值得保留的是“晚期反转 + 部分答案先验校正”的可复核排序干预，以及几何/能量控制失败的完整证据。下一阶段是否投入应由外部完整结果和新、未反复使用的数据决定；不能凭训练切片或一次已暴露测试集的达标就宣称通用改进。当前不启动新的大规模训练、循环结构或隐藏状态因果推断。

基于本轮完整外部检验的决策：保留该方向进入新数据和新模型的预先冻结验证；目前证据仍不足以支持普遍替换 Fixed 或声称新机制。继续在同一 ARC 上搜索会进一步削弱外推价值。

当前 R6 是针对给定候选集合的完整答案排名规则。它尚未实现或验证自由生成版本，不能将多项选择题上的提高直接推广为 GSM8K 自回归推理改善。候选集合变化对投影系数的影响是后续可证伪的问题，本轮没有据测试结果设计新的候选集合或再筛方法。

投影和校准均有明确先例：[Adaptive Projected Guidance](https://arxiv.org/abs/2410.02416)、[PMI_DC / Surface Form Competition](https://aclanthology.org/2021.emnlp-main.564/)、[Calibrate Before Use](https://proceedings.mlr.press/v139/zhao21c.html)。本轮是对 recurrent guidance 的具体适配，未证明首创、优于80%可比方法或揭示新神经机制。R2–R5的其他主文献边界见 [相关工作核验](../astra/slcd_FOLLOWUP_RELATED_WORK.md)。

## 完整性、数据重叠与交付范围

ARC test、官方validation和 A1历史726均已暴露；R6开发虽未读取新逐题test反馈，也不是新的盲测。原训练指纹审计按不同问题文本计数；后验逐测试行审计发现 7 行与训练重叠、2 行与validation重叠，合计 9 行。主结果保留全部1172，不事后删题；附加敏感性如下，排除重叠后净纠错19仍保留，但这些子集也不能称盲测。

| scope | n | correct | net_fixed |
| --- | --- | --- | --- |
| official_all_1172 | 1172 | 736 | 19 |
| posthoc_exclude_training_question_overlap | 1165 | 733 | 19 |
| posthoc_exclude_train_and_validation_question_overlap | 1163 | 732 | 19 |

完整代码、冻结配置、逐候选分数、转移、成本、两份PDF/PNG和核验记录随分支交付。完整输入及原始JSONL/hidden-state缓存保留服务器；模型权重不复制。HellaSwag上游仓库访问现被屏蔽，公开包保留完整数值、输入哈希与固定HF来源，不重复分发原文或可逆token IDs；其余输入依各自来源许可与attribution处理，见 [数据说明](../data/slcd_EXTERNAL_SOURCE_NOTICE.md)。

独立CPU审计重算1172个预测、5860条转移、全部配对统计、4408先验cache条目、37496先验LL与4687 contextual target IDs，均一致。审计使用本地tokenizers0.22.2、服务器0.21.4，具体输入token IDs精确相等，不能概括成两版本普遍等价。外部完整核验范围见 `slcd_R6_EXTERNAL_ACCEPTANCE.json`；所有旧671文件size/mtime和原模型完整字节SHA另有最终保留审计。

复现和恢复说明见 [README](../slcd_README.md)。原始R1报告归档为 `slcd_R1_REPORT_ZH.md`，R1锁/数据不被改写成R6。图表由本脚本直接读取实际完整结果绘制。

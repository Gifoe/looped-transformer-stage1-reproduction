# Codex Research Task: Quality Direction Predictability in Training-Free Looped Transformers

## 0. Goal

**Determine whether beneficial and harmful recurrent updates in a frozen pretrained Transformer can be distinguished using inference-time observable signals, and whether a selective update policy can outperform both the original model and unconditional Damped Euler on unseen data.**

本轮属于研究可行性实验，不提出新架构。

核心问题：

**当 Damped Euler 改变模型预测时，我们能否在不知道正确答案的情况下，判断应该保留 Original 状态，还是接受 Euler 更新？**

目标是检验这种判断能力是否真实存在、能否跨样本泛化，以及收益是否超过额外推理成本。

不要为了获得正结果而人为选择样本、阈值或特征。

---

## 1. 现有工程与实验基础

使用仓库：

https://github.com/Gifoe/looped-transformer-stage1-reproduction

优先复用现有 Stage 1 实验：

`experiments/looped_stage1_trajectory_audit_seed2021_v1/`

需要检查：

- `results/qwen_A0_test.jsonl`
- `results/qwen_A3_test.jsonl`
- `data/qwen_test.jsonl`
- `scripts/qwen_audit.py`
- `scripts/qwen_recipe.py`
- `configs/protocol.json`

通过样本 ID 对齐 A0 和 A3，不允许仅凭行号配对。

新建独立目录：

`experiments/loop_quality_direction_predictability_seed2021_v1/`

不要修改、覆盖或重新生成旧实验的原始记录。

### 使用模型

`Qwen/Qwen3-4B-Base`

固定原来已经跑通的配置：

**A0：Original Qwen3**

正常执行全部 Transformer 层。

**A3：Damped Euler**

- Window：[15,16,17,18]
- Block-mode
- K=6
- α=1/6
- 不使用 Anchored β
- 冻结全部模型参数
- 与原实验完全一致的 checkpoint、tokenizer 和评分协议

本轮不改变 K、层窗口和 Euler 方法。

### 已知结果

600 道测试题：

| 方法 | Accuracy |
|---|---:|
| Original | 49.50% |
| Damped Euler | 52.33% |
| Ideal Oracle Selection | 54.83% |

其中：

- Wrong → Correct：32 题
- Correct → Wrong：15 题

这 600 道题已经参与过此前的研究讨论，因此**只能用作探索性 Discovery Set，不能被称为本轮新的独立确认测试集**。

---

## 2. Research Question 与假设

### RQ1：更新质量是否可预测？

仅观察 Original 和 Euler 的输出及隐藏状态变化，能否判断 Euler 是否会改善预测正确性？

### RQ2：哪种信息真正有用？

比较：

- Confidence changes
- Entropy changes
- Logit/score margin changes
- Distribution divergence
- Hidden-state changes
- Recurrent trajectory dynamics

判断是否有特征能够稳定区分有益与有害更新。

### RQ3：能否带来实际收益？

如果根据这些信号选择 Original 或 Euler，能否在独立数据上同时实现：

- 相比 Original 更高的 Accuracy。
- 相比始终使用 Euler 更高的 Accuracy。
- 更少的 Correct → Wrong。
- 尽可能保留 Euler 的 Wrong → Correct。

### RQ4：收益是否能够泛化？

检查训练于一个或多个任务上的判断规则，是否能迁移到其他 Benchmark，而非只记住特定数据集的统计特征。

---

## 3. Phase A：构建 Direction Quality Dataset

首先读取已有 600 题结果。

对每一道题，定义：

\[
g_i=\mathbb{1}[\text{Euler correct}]
-\mathbb{1}[\text{Original correct}]
\]

因此：

| g | 含义 | 期望动作 |
|---|---|---|
| +1 | Euler 纠正了 Original 的错误 | Accept Euler |
| -1 | Euler 破坏了 Original 的正确答案 | Reject Euler |
| 0 | 两者正确性相同 | 两者对 Accuracy 等价 |

注意 g=0 包括两种情况：

1. 两者都正确。
2. 两者都错误。

还需要区分预测是否发生改变。

优先分析：

`Original prediction != Euler prediction`

因为如果两者最终选择同一个答案，选择 Original 或 Euler 不会改变该样本的 Accuracy。

对预测改变的样本，分别统计：

- Beneficial correction
- Harmful degradation
- Both wrong but different predictions

不要只删除 g=0 样本后训练二分类器，再将二分类指标误认为整个测试集的决策性能。

### 数据检查

验证：

- 总样本数为 600。
- 原模型正确数为 297。
- Euler 正确数为 314。
- Wrong→Correct 为 32。
- Correct→Wrong 为 15。
- 两模型结果完全按 ID 对齐。
- 现有结果能重建原报告中的 Accuracy 和转移计数。

如果任何检查失败，应先定位问题，不继续后续统计。

---

## 4. Phase B：构建不依赖 Gold Answer 的特征

这是实验最重要的部分。

**所有用于推理决策的特征都必须在不知道标准答案的情况下计算。**

禁止把以下变量作为特征：

- Gold answer
- Correct / Incorrect
- Gold NLL
- Gold-answer log probability
- 任何由正确标签直接计算的差异
- 测试集标签统计
- Oracle action

这些变量只能用于离线监督和评估。

### Feature Group 1：Output Confidence

分别从 A0 和 A3 提取：

- Top-1 confidence
- Top-1 / Top-2 margin
- Output entropy
- Top-1/Top-2 candidate scores
- 最优候选与次优候选的 score gap

计算两模型对应指标的差值。

注意现有仓库中的 `choice_probabilities` 来自候选归一化分数的 Softmax，不是严格校准的真实答案概率。

需要区分 Choice-level 和 Full-vocabulary 指标。

### Feature Group 2：Distribution Change

计算：

\[
D_{\mathrm{KL}}(p_E\|p_O)
\]

以及：

- Reverse KL
- Jensen–Shannon divergence
- Total variation distance
- Top-1 是否改变
- 原模型 Top-1 候选在 Euler 下的排名
- Euler Top-1 候选在原模型下的排名
- 两模型相同候选的 score differences

全部指标在相同候选集合上计算。

### Feature Group 3：Hidden-State Dynamics

读取 A3 中已有的 `window` 记录。

包括：

- Input-to-output relative L2
- Input-to-output cosine similarity
- Input-to-natural-update distance
- 每一步更新的 relative L2
- 各次更新幅度的均值、最大值和末轮值
- 相邻更新幅度变化
- 更新幅度是否持续增加或下降

首先检查历史结果实际保存了哪些数据。

不能假设已有日志保存了完整隐藏状态向量。

如果某个特征需要新的 hidden-state 或 update-vector 记录，先评估必要性，再决定是否进行补充前向计算。

### Feature Group 4：Simple Context

可考虑：

- Prompt token length
- Candidate count
- Candidate token lengths
- Candidate character lengths

Dataset ID 可以用于分组统计及跨任务评估。

不要让模型仅依赖 Dataset ID 学习不同数据集的固定选择偏好，并把这种偏好误认为真正的方向判断能力。

### Feature Sanity Check

输出每个特征的：

- 有效样本数
- 缺失比例
- 均值和方差
- 异常值
- 与其他特征的相关性

对特征命名、归一化、来源进行记录。

---

## 5. Phase C：先验证可分性，不急于设计预测器

分析两个最关键的类别：

**Beneficial：g=+1**

**Harmful：g=-1**

先只用已有 600 题检查哪些特征能够区分这两个类别。

### 单变量分析

逐一测试：

- Confidence delta
- Entropy delta
- Choice KL
- JS divergence
- Margin delta
- Hidden L2
- Hidden cosine
- Update magnitude trend

报告：

- AUROC
- AUPRC（明确正类定义）
- Effect size
- Bootstrap 95% CI
- 两类特征分布图
- 最优方向是正相关还是负相关

对两类样本数量不平衡的情况进行明确处理。

由于目前只有 32 个 Beneficial 和 15 个 Harmful 样本，单变量 AUC 不确定性可能很大，不能根据点估计选择研究方向。

### 重要负对照

设置：

1. Random feature
2. Shuffled feature-label association
3. Original confidence only
4. Euler confidence only
5. Dataset-prior policy

这些对照用于判断特征组合是否真的超出简单置信度或任务先验。

如果特征无法区分 Beneficial 和 Harmful，应如实报告。

---

## 6. Phase D：构建最简单的选择策略

本阶段不训练 LLM，不加入新的 Transformer 模块。

最多比较以下策略。

### Policy 0：Always Original

所有题使用原模型。

### Policy 1：Always Euler

所有题使用 Damped Euler。

**这是新选择器必须超过的主要 Baseline。**

### Policy 2：Fixed Confidence Gate

根据 Original 和 Euler 的置信度变化，决定是否接受 Euler。

阈值只能在开发集选择。

### Policy 3：Fixed Margin / Entropy Gate

根据 margin 或 entropy 的变化选择。

### Policy 4：KL-based Gate

根据预测分布差异选择。

### Policy 5：Linear Quality Predictor

使用一个简单的 Logistic Regression 或正则化线性模型，根据上述特征预测：

\[
\widehat{\mathbb E}[g_i\mid f_i]
\]

当预计接受 Euler 的正确率收益大于零时选择 Euler，否则保留 Original。

可以采用更保守的正阈值，但必须通过开发集确定。

训练时允许使用开发集的正确答案生成监督标签；推理时绝不能使用标签。

禁止使用 MLP、Transformer、复杂集成模型或其他大型预测器。

线性模型应使用严格嵌套的验证流程选择正则化与阈值。

所有缺失值填补、特征标准化和类别权重，都只能在对应训练折拟合。

### Oracle（仅作为上限）

允许在报告中计算：

\[
\text{Oracle}(x)=
\begin{cases}
\text{Euler}, & g(x)>0\\
\text{Original}, & g(x)\leq0
\end{cases}
\]

但 Oracle 使用真实答案，不可部署，绝不能作为实际预测方法。

---

## 7. Phase E：评估真正的 Selective Update 收益

对每个策略，必须统计：

- Accuracy
- Accuracy − Always Original
- Accuracy − Always Euler
- Correct→Wrong 相对 Original
- Wrong→Correct 相对 Original
- Euler beneficial corrections retained
- Euler harmful degradations avoided
- Euler beneficial corrections rejected
- Euler harmful degradations accepted
- Euler acceptance rate
- Prediction disagreement rate

重点关注：

\[
\boxed{
\text{Net Gain over Euler}
=
\text{Harmful updates avoided}
-
\text{Beneficial updates rejected}
}
\]

如果避免了 10 次退化，却同时拒绝了 12 次纠错，说明该策略相对始终使用 Euler 反而下降。

不能只报告：

- 分类器 AUC
- Harmful update detection recall
- 某些特征的相关系数

必须以最终选择策略的真实 Accuracy 作为主要指标。

此外加入一个 Acceptance-rate-matched Random Gate，检查收益是否只是来自拒绝了一部分更新。

统计使用问题级 paired bootstrap（10,000 次）和 McNemar 检验。

---

## 8. Phase F：独立样本验证

这一步是决定本研究是否值得继续的关键。

已有 Stage 1 的 600 题属于 Discovery Set。不能在其上反复选特征、阈值后，再用相同 600 题宣称实现泛化。

请建立新的、独立的 Qwen 评测集。

### 数据要求

优先从以下任务中选择没有参与本轮规则开发的新样本：

- ARC-Challenge
- OpenBookQA
- MMLU 其他学科

首先检查 Stage 1 以及已有 extension 中的所有 manifest，避免重用已被 Qwen 策略开发使用的样本。

若原任务剩余样本不足，可以补充 ARC-Easy、SciQ 等选择题 Benchmark，但要将跨任务测试单独报告，不可与同分布测试混为一谈。

建议新样本规模为 1,000–2,000 道；根据剩余数据和首批吞吐自主调整。

必须在查看其正确性标签对应的 A0/A3 结果之前，冻结：

- 模型及循环设置
- 特征集合
- 特征预处理
- Policy
- Threshold
- 统计方法
- Primary endpoint

使用新数据时，继续运行完全相同的 Original Qwen 和 Damped Euler K=6。

只有确认原始模型与 Euler 的评分协议一致，才能应用预测器。

不得根据新测试集调节阈值或重新训练。

### Cross-dataset Transfer

还要尝试：

- 在 ARC 和 MMLU 开发样本上拟合，测试 OpenBookQA 或其他独立任务。
- 或在其他合理的训练/测试划分下测试跨任务迁移。

如果样本数量不足，应降低结论强度，而不是把训练数据重复用于测试。

---

## 9. 计算成本必须公平

这里有一个不能忽略的问题：

如果选择器需要同时观察 Original 和 Euler 的最终预测，实际系统就需要计算两套候选输出。

因此：

**Selective Gate 的计算成本不一定等于单独运行 Damped Euler。**

请区分：

1. Offline Selection Cost：直接使用已保存的 A0/A3 输出，不需要额外 GPU。
2. Actual Selection Cost：在线获得 Original、Euler 及选择特征所需的实际计算量。
3. Predictor Overhead：轻量预测器自身的额外时间。

首先使用现有结果完成离线分析。

对于独立测试，记录额外计算是否可以共享。例如 Euler 的第一次窗口执行与原始窗口计算具有重用机会，但后续 Transformer 层是否需要分别运行必须实测。

不要声称模型实现了 training-free acceleration，除非真正证明其 wall-clock 或 FLOPs 优势。

### 预算

优先复用原始数据，探索阶段不应消耗大量 GPU。

建议新数据验证的初始 GPU 上限为 5 GPU-hours，使用 RTX 5090 BF16；根据首批实测吞吐估算是否足够。

如果时间或显存不足，优先保持严格的独立测试和 Baseline 完整性，适当减少数据量。

不要增加训练规模。

---

## 10. 必须完成的图表

### Figure 1：Beneficial vs. Harmful Update Feature Distributions

比较 KL、Confidence delta、Entropy delta、Hidden L2 等指标在两类样本上的分布。

### Figure 2：Direction Predictability

报告单变量与 Linear Predictor 的 AUROC/AUPRC，包含置信区间和随机对照。

### Figure 3：Selective Update Accuracy

比较：

- Original
- Always Euler
- Confidence Gate
- KL Gate
- Linear Quality Predictor
- Oracle

区分 Discovery、Cross-validation 和真正独立测试结果。

### Figure 4：Correction Retention vs. Degradation Avoidance

横轴：保留的有益纠错比例。

纵轴：避免的有害退化比例。

标注最终 Accuracy，明确展示 Gate 的收益与代价。

### Figure 5：Cross-Benchmark Generalization

比较各策略在不同测试集上的独立结果，尤其关注策略是否只对某类题目有效。

---

## 11. Go / No-Go 判定

### GO：值得继续研究

至少需要满足：

1. 存在推理时可获得、能够可靠识别更新质量的信号。
2. 在未参与规则开发的测试集上，Selective Gate 超过 Always Euler。
3. 收益不仅来自接受率变化、任务先验或某个特定阈值。
4. 能够保留大部分有益纠错，同时避免足够多的有害退化。
5. 额外的计算成本可以接受，或者存在明确的低成本实现路径。
6. 与 TaH、TaH2、PoLar 等现有动态计算或质量预测研究相比，仍存在实质性研究差异。

如果仅能通过离线 Gold Label、Oracle 或测试集选参得到提升，不属于 GO。

### NO-GO：暂时放弃这一思路

例如：

- 各特征的预测能力接近随机。
- Linear Predictor 在新数据上不优于简单规则。
- Gate 避免退化的同时损失更多有效纠错。
- 收益仅在已有 600 题上出现。
- 跨任务明显退化。
- 相比 Always Euler 的增益很小，且额外计算成本较高。
- 已有方法能够以相同或更低成本取得同等结果。

如果出现这些情况，请直接给出 No-Go，不要通过引入更复杂的预测器继续优化。

---

## 12. 最终交付

项目目录至少包括：

- `README.md`
- `configs/`
- `scripts/`
- `data/`
- `results/`
- `figures/`
- `logs/`
- `tests/`
- `FINDINGS.md`
- `REPRODUCE.md`

所有预测、特征、数据划分和模型选择记录必须保留。

最终 FINDINGS.md 必须回答：

**Q1.** 我们是否能够在不知道 Gold Answer 的情况下预测 Euler 更新的有益或有害方向？

**Q2.** 哪一类特征最有用？与置信度、KL 等简单指标相比有什么实质性改进？

**Q3.** 现有 600 题的 54.83% Oracle 上限中，实际可实现多少？

**Q4.** Gate 是否真正超过 Always Euler，而不仅仅是超过 Original？

**Q5.** Gate 在不同任务上是否稳定？跨任务泛化如何？

**Q6.** 这些收益是否足以抵消额外计算成本？

**Q7.** 如果只使用原模型状态而不预先计算 Euler，是否有任何预测潜力？如果没有，请明确说明必须先支付完整 Euler 成本。

**Q8.** 当前研究是否只是已知的 Quality Prediction / Dynamic Halting 的另一种实现？

**Q9.** 如果值得继续，下一步究竟应该研究动态 β、状态干预，还是更可靠的质量判别机制？

**Q10.** 最终 Go / No-Go 是什么？由哪些独立证据支持？

最后必须清楚区分：

- Oracle upper bound
- Discovery-set observations
- Cross-validated predictions
- Independent held-out performance
- Real inference overhead

不要将探索性结果包装成独立确认结果。

---

## 13. 执行要求

请自主执行完整实验，而不是仅写实验计划。

首先完成已有 600 道题的数据检查和特征分析。然后构建简单选择器，通过交叉验证确定规则，并在新的未参与开发的数据集上进行最终测试。

在新测试集标签解封之前，冻结所有方法及超参数。

本轮严格遵循：

**Correctness → Feature Predictability → Selective Accuracy → Cross-Dataset Generalization → Compute Cost → Research Decision**

不要提前设计 Adaptive Anchored Euler 模块，也不要添加新的可训练神经网络。

唯一目标是回答：

**我们是否真的能从循环过程中观察到的信号，判断模型正在变好还是变坏，并据此取得可泛化的净性能收益？**
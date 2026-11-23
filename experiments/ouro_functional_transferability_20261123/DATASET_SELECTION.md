# 训练来源与独立确认集

ARC-only 与 CSQA-only 各使用 1,119 道原始官方训练题。混合候选集合由 560 道 ARC-Challenge 与 559 道 CommonsenseQA 组成，总量也为 1,119。所有抽取按固定命名空间与原始题号的 SHA‑256 排序，答案和模型分数不参与排序。确认题不进入当前训练或发现集。原始题号、候选顺序、候选文字及原始答案标签保存在 JSONL。 [冻结清单](data/MANIFEST.json) · [构建器](dataset_transfer.py)

| 用途 | ARC | CommonsenseQA |
| --- | --- | --- |
| 拟合 | 完整 ARC-Challenge 官方 Train，1,119 | 官方随机划分 Train 按固定顺序取 1,119 |
| 理论发现 | ARC-Easy 官方 Validation 的 200 道互斥题 | 官方 Train 留出的 200 道互斥题 |
| 独立确认 | ARC-Easy 官方 Validation 的 299 道互斥题 | 官方 Train 留出的 299 道互斥题 |
| 历史对照 | ARC-Challenge Validation 299、Test 1,172 | 作者官方 Validation 1,221 |

ARC-Challenge 的历史 Parquet 与官方 Hub 文件的容器哈希不同。对固定修订 `210d026faf9955653af8916fad021475a3f00453` 的三个官方文件逐行比较后，核对了全部 2,590 行的原始顺序、题号、题干、选项文字、标签及正确答案。共同的候选前缀为 `Question: {question}\nAnswer:`，每个候选续写以一个空格加原选项全文计分。已用实际 Ouro tokenizer 与历史 `run_arc.encode`、`minimal.label_prompt` 逐项核对全部人口的候选 token 数及字母生成题面。 [AI2 ARC 官方修订](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453) · [CommonsenseQA 官方随机划分数据校验](https://huggingface.co/datasets/tau/commonsense_qa/blob/d90214517943631510b39b7d2503b7ea93ec09c2/dataset_infos.json)

## 历史泄漏审计

已从历史结果校验 256 个 ARC 开发题、1,172 个 ARC-Challenge Test 题与 1,221 个 CommonsenseQA 作者 Validation 题的实际身份。八样本 ARC-Easy 的历史运行对官方 Test 全部 2,376 个例子打分。归档 `doc_id` 按同修订官方 Test 行序映射回原始题号，候选与正确位置逐题相同。历史演示来源是 ARC-Easy Train，审计保守排除其中全部 2,251 道题。另以全部 ARC-Challenge 数据、历史 OpenBookQA Validation/Test 与作者 CSQA Validation 作排除来源。

ARC-Easy Validation 的 570 题中排除 1 个与审计来源规范化题干相同的例子。CSQA 官方 Train 的 9,741 题中排除 1 个同池规范化题干重复。四个发现／确认集和现有训练集、被审计历史的原始 ID、规范化题干、规范化题干加候选三种交集均为零。规范化使用 Unicode NFKC、大小写折叠和分隔符统一，能够验证已列明语料的精确文本排重。此审计无法判定改写等语义近重复，也不涉及基础模型预训练数据。上游 Ouro 论文曾用 10-shot 评估 CSQA，现有资料无法恢复实际示例 ID。这里的 CSQA 独立性指本轮和所列历史项目产物中的拟合、选型及评估隔离。ARC 独立确认是在 ARC-Easy 上的同领域难度变化。历史 ARC-Challenge Test 和作者 CSQA Validation 分开报告为复测。 [重叠审计与来源哈希](data/MANIFEST.json) · [上游历史审计](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/a07698897bfa705cf3d31deaa2d8c00050c5c84c/experiments/ouro_better_20261120/iteration2/DATA_LINEAGE.md)

## 完整训练池的候选成本

| 训练池 | 问题 | 候选 | 候选输入 token | 答案监督 token |
| --- | ---: | ---: | ---: | ---: |
| ARC | 1,119 | 4,476 | 155,721 | 25,944 |
| CSQA | 1,119 | 5,595 | 117,937 | 9,472 |
| ARC + CSQA | 1,119 | 5,035 | 139,201 | 18,225 |

该表使用完整语料一次无 padding 的 token 清点。等问题数与等更新步数保留了实际候选数、题长和答案长度的差异。实际 GPU 行为和前向预算以训练逐批台账为准。用相同参数另行重建数据后，冻结清单的 SHA‑256 仍为 `6036073abbfbf3fbd098914a124a2d8e690c6e59b6f1d47b48eb0e515803a31b`。

## 第二轮来源与新确认集

OpenBookQA-main 有 4,957 道官方 Train 题，QASC 有 8,134 道。排除官方 Validation/Test、主实验已使用题目、两个训练来源的精确重叠及历史 QASC 确认题后，OpenBookQA 剩 4,772 道、QASC 剩 7,129 道。核对了历史 QASC 384 道已评估题的原始 ID，并沿原审计的重复组件排除相关题目。QASC 历史和近重复组件合计排除 491 个 ID。当前程序先排重并确定 1,119 个 ID，再为这些 ID 读取训练答案。QASC 的 `fact1`、`fact2`、`combinedfact` 和 `formatted_question` 列未读取或进入评分。OpenBookQA 训练子集的 4,476 个候选含 85,345 个输入 token 与 13,957 个答案监督 token。QASC 的 8,952 个候选含 138,446 个输入 token 与 16,774 个答案监督 token。 [扩展拟合冻结](data/optional/MANIFEST.json) · [OpenBookQA 官方来源](https://huggingface.co/datasets/allenai/openbookqa/tree/8c4888010de9b23a2d136ebd31d5f5ea9f3d90fa) · [QASC 官方来源](https://huggingface.co/datasets/allenai/qasc/tree/a34ba204eb9a33b919c10cc08f4f1c8dae5ec070)

第二轮另封存主轮未分配的全部 70 道干净 ARC-Easy Validation 题。从作者 CSQA Train 中排除 1,618 个已用 ID 和一个重复题干后，在 8,122 道合格题中按新命名空间抽取 299 道。第二轮确认与主轮发现、确认、拟合和新增拟合来源的题号及规范化题干均无交集。这两个确认集在第二轮预测生成前冻结，独立于第一轮观察用于继续研究的决策。ARC-Easy 第二轮人口只有 70，区间精度须按此总体解释。 [第二轮确认冻结](data/optional/ROUND2_MANIFEST.json)

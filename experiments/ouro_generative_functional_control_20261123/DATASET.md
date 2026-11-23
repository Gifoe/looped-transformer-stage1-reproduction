# 数据来源、去重与格式

题目和回答统一为 instruction、context、response、task_type、dataset、sample_id。单个官方训练用样本最多被选一次。多个来源的规范化题干或指令一致时，只保留第一个来源。冻结固定种子的采样、标准化与选项置换。混合阶段分清等骨干计算量的筛选池与扩展后的完整池，分别保存文件 SHA、来源 SHA、实际题数和 token 长度分位数。

| 来源 | 完整池实际有效样本数 | 数据和许可来源 |
| --- | ---: | --- |
| CommonsenseQA | 6,000 | 作者 Train，排除归档确认、开发题及新预留折。 [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa) |
| OpenBookQA | 4,000 | 官方 main Train，排除验证和测试划分。 [OpenBookQA](https://github.com/allenai/OpenBookQA) |
| SQuAD 1.1 | 10,000 | 官方 Train，验证集按文章标题分为开发和确认。CC BY-SA 4.0。 [固定版本](https://huggingface.co/datasets/rajpurkar/squad/tree/7b6d24c440a36b6815f21b70d25016731768db1f) |
| GSM8K | 6,000 | 官方 main Train，约一半仅最终数字、另一半保留原始完整解题过程。MIT。 [固定版本](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866) |
| No Robots | 4,000 | 官方 Train，保留最后一轮完整回复与前序上下文。CC BY-NC 4.0。 [固定版本](https://huggingface.co/datasets/HuggingFaceH4/no_robots/tree/e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b) |
| SmolTalk | 10,000 | Magpie Ultra、Rewrite、Summarize、Constraints 四子集，作者数据说明为 Apache 2.0。 [固定版本](https://huggingface.co/datasets/HuggingFaceTB/smoltalk/blob/5feaf2fd3ffca7c237fc38d1861bc30365d48ffa/README.md#license) |

完整池共 40,000 条。SmolTalk 分别取 Magpie Ultra 4,000、Rewrite 2,000、Summarize 2,000、Constraints 2,000。Magpie Ultra 第一分片筛得 1,558 条，补用官方编号 01 和 03 两个分片。`SCALE_LOCK.json` 记录具体增补 ID 和来源字节指纹。`DATA_INDEX.json` 记录最终生成池及各评测集合的样本 ID、任务类型、题数和 SHA-256。

池内规范化题干或指令有 40,000 个独立键。No Robots 的两个不同指令共享一个上游 `prompt_id`，所以训练日志按标签计为 39,999 个不同 `sample_id`。数据索引另存这两行各自的行号及提示哈希。两行均来自官方训练集，未重复采样。使用固定原料独立重建后，共有 35 个数据和审计文件逐字节一致。

CSQA、OpenBookQA 在训练前随机置换候选标签并同步更新金标，精确划分为 70% 完整选项文字和 30% 大写字母。原有候选评分器以不带字母的候选原文作为辅助 ranking。SQuAD 保留供模型读取的 passage，在训练使用第一个正式答案，在评分计算合法参考答案的最大 EM/F1。GSM8K 保留原题与原答案的链接，使用严格最终标识。No Robots 与 SmolTalk 优先保留完整助手答案，采用同一套原生文本角色提示。

训练过滤完整原文和 Ouro 分词长度。回答加 EOS 最多 240 个 token，完整输入最多 640 个 token。超过上限的记录整条排除。SmolTalk 先以字符长度排除明显过长记录，再以真实分词复核。Magpie 首分片只使用达到上限的样本，再用独立官方分片扩展。不会以重复条目填补配额。

源模型的预训练语料不透明。去重结论覆盖现有研究记录和实际训练池，不能证明预训练从未接触公开 Benchmark。发布训练来源及采样索引，不重新分发六来源的训练文本合集。

确认前以规范化题干或指令比对两组训练池，合并后有 40,372 个独立键。近重复采用至少 12 词、词四元组集合 Jaccard ≥0.8，候选索引略去常见四元组。OpenBookQA Test 题 `7-334` 与 Train 题 `14-676` 的近重复相似度为 0.857，确认时排除该题。一并保留 ARC-Challenge Test 1,172 题、ARC-Easy Test 2,376 题和 CommonsenseQA 作者 Validation 1,221 题。CSQA 公开验证标签用作本轮完整确认，历史项目已经使用这一公开集合，因此只称为本轮训练和选型独立。两类非 MC 问答使用确定的 360 道 SQuAD 和 320 道 GSM8K，GSM8K 另有 96 道过程题。No Robots 与 SmolTalk 四子集在 K4 各用 64 条官方测试样本。低深度和外推深度使用锁定的较小子集。 [污染审计](data/filtered/CONTAMINATION.json) · [封存协议](results/FINAL_PROTOCOL.json)

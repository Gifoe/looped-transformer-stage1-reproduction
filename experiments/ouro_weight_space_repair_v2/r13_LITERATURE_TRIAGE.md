# 与当前候选想法重叠的已核验工作

此次仅核验以下一手摘要和题录，用于筛除把已知原则换名的提议；没有完成系统综述，也不据此声称尚无相似方法。

| 一手来源 | 对本实验的约束 |
|---|---|
| [Answer-level Calibration for Free-form Multiple Choice Question Answering，ACL2022](https://aclanthology.org/2022.acl-long.49/) | 已研究无上下文答案偏置及无监督校正；问题/先验LL之差本身不能作为新贡献。 |
| [Dynamic PMI-Guided Contrastive Decoding，Findings ACL2026](https://aclanthology.org/2026.findings-acl.1212/) | 已使用输入变换构造动态背景分布；换N/A参考或引入动态PMI不是充分新颖性。 |
| [The Direction of Ignorance，arXiv2609.02959v2](https://arxiv.org/abs/2609.02959v2) | 已将先验依赖表示为方向投影并进行推理时干预；“先验有几何方向”不能单独作为发现。 |
| [Accept or Repair，arXiv2610.29727](https://arxiv.org/abs/2610.29727) | 已针对对比解码损害正确答案提出选择性接受/修复。该工作依赖passage支持检查，不能直接等同于本实验无外部证据的科学选择题。 |
| [From Context-Aware to Conflict-Aware，arXiv2606.10298](https://arxiv.org/abs/2606.10298) | 已研究上下文与先验信任的动态路由及固定外推/内插局限；再加一个置信度门控不足以证明新原理。 |

这些文献并未在本轮Ouro完整ARC设置下被复现，不能借用其数字充当我们的效果或成本。当前可研究的局部问题是：候选级R6已取得的纠错为何在prefix级替换中丢失，以及如何在合法全词表归一化下保留粗粒度收益并利用额外前缀信息。R13仅提供开发集分解证据，没有证明答案。

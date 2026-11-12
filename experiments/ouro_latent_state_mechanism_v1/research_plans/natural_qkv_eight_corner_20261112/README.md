# Round 1：Q×K×V 八角实验

完整 2,590 题八角采集、独立进程 CUDA 重放和三分区分析已完成；[Astra 科学审阅](ROUND1_ASTRA_SCIENTIFIC_REVIEW.md)给出结论与限制。[完整逐题结果](ROUND1_FULL_RESULTS.json)含八角候选 scores、target logits、预测、平分、fixed/dynamic/all-wrong gap 和来源索引；原始大 tensor 不在本 GitHub 包内。[正确数表](figures/EIGHT_CORNER_TABLE.md)、[条件边与交互图](figures/eight_corner_edges.png)及其[源 CSV/绘图代码](figures/README.md)可直接使用。Validation 全体 V 边 G112−G111 净正确数为 −1，预注册该项未通过；不能用正的平均 gap 掩盖。

九项指定输出：[protocol](ROUND1_PROTOCOL.md)、[implementation audit](ROUND1_IMPLEMENTATION_AUDIT.json)、[GPU parity](ROUND1_GPU_PARITY.json)、[full results](ROUND1_FULL_RESULTS.json)、[factorial analysis](ROUND1_FACTORIAL_ANALYSIS.md)、[counterexamples](ROUND1_COUNTEREXAMPLES.md)、[runtime/memory](ROUND1_RUNTIME_AND_MEMORY.json)、[Astra review](ROUND1_ASTRA_SCIENTIFIC_REVIEW.md)、[唯一下一问题建议](ROUND2_RECOMMENDATION.md)。三分区 `*_analysis/` 含 summary、逐题 rows 和 shard ledger；三份 `*_independent_cuda_coverage.json` 与 [combined audit link](ROUND1_COMBINED_AUDIT_LINK.json)记录覆盖。脚本位于本目录；[来源与校验清单](DELIVERY_MANIFEST.json)列出文件及SHA。

原自然 tensor SHA 在被删除文件上继承历史审计，并非本次重读；原metadata从保全ZIP恢复并逐项核SHA，实际branch R锚字节另行核验。CUDA由独立进程运行同一冻结实现，不是独立实现。Train非demo为探索，validation/test是有既往ARC暴露的项目内前瞻检查，不是盲测。G222为T12而非自然B2；终端gap交互不等于内部SDPA向量交互。新一轮仅提出建议，没有自动启动额外实验。HF大文件上传仍按用户要求暂停。

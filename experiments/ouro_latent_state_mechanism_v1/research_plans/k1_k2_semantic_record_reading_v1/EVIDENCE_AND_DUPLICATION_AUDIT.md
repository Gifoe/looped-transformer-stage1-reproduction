# 仓库证据与重复实验审计

核查基线：Gifoe/looped-transformer-stage1-reproduction，分支`research/ouro-latent-mechanism-20261107`，提交`fa9bb5bf1794cb11e7d0b504747c13c304fc5d00`。本次元数据完整覆盖research_plans的3153个条目、68个顶层目录；310个Markdown路径按Git blob SHA去重为271份正文，均从该固定提交取回并核对Git SHA。索引齐全不等于逐条独立科学复算；实际精读范围为下表所列报告及Astra审阅列出的核心代码。

当前结构化确认A/B/C摘要已直接读取并检查关键数值。完整神经前向/所有旧研究的bootstrap本轮没有重跑。旧审计是继承证据，Astra本轮参与科学设计而未独立复现神经实验。

| 已有证据族 | 已完成结果 | 下一轮去重约束 | 来源 |
|---|---|---|---|
| Q/K/V三因子合作 | 固定K1接收轨迹的同题K1/K2来源Q×K×V八角，覆盖既有Train/Validation/Test；K×V条件合作成立且Value-only存在失败 | 不再做无语义预测的新八角；旧Test属于历史已暴露证据，下一轮不访问Test。 | [natural_qkv_eight_corner_20261112/ROUND1_ASTRA_SCIENTIFIC_REVIEW.md](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/fa9bb5bf1794cb11e7d0b504747c13c304fc5d00/experiments/ouro_latent_state_mechanism_v1/research_plans/natural_qkv_eight_corner_20261112/ROUND1_ASTRA_SCIENTIFIC_REVIEW.md) |
| SDPA与后缀非线性 | E1分离实际SDPA交互和整个后缀非加性；E2probe仅R状态可解码且额外E轴混淆 | 不要将正margin、probe可解码或后缀非线性改称新语义产生/纯MLP因果机制。 | [loop_mechanism_e1_e5_20261112/E1_E5_SYNTHESIS/FINAL_RESEARCH_REPORT.md](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/fa9bb5bf1794cb11e7d0b504747c13c304fc5d00/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E1_E5_SYNTHESIS/FINAL_RESEARCH_REPORT.md) |
| 早读与后写 | Q9条件下E=L9失败；L=L10-16 bundle在既有自然纠错组通过嵌套确认；Q是整道题+选项+格式，不是单独questionstem | 不再只切分近似相同窗口；新问题必须有记录级语义预测。 | [loop_mechanism_e1_e5_20261112/E3_WRITER_4090/E3_PATH_PATCHING.md](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/fa9bb5bf1794cb11e7d0b504747c13c304fc5d00/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E3_WRITER_4090/E3_PATH_PATCHING.md) |
| 自然R Query/attention中介 | 旧Validation299 attention恢复22/29被阻断自然纠错；后续128为6/9且仅2严格控制特异，均值恢复CI跨0 | 这些是不同队列/不同标量，不能汇总成已解释大多数自然收益。 | [loop_mechanism_e1_e5_20261112/E4_E5_4090/E4_MEDIATION_ANALYSIS.md](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/fa9bb5bf1794cb11e7d0b504747c13c304fc5d00/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E4_E5_4090/E4_MEDIATION_ANALYSIS.md) |
| 自然Q×jointKV | 自然K2接收者Q-only、jointQKV和Q×KV已有；某些Query方向有效但总体并非唯一决定因素 | 仅再分Q/K/V不是新问题；语义索引、具体来源位置与两跳预测才是增量。 | [loop_mechanism_e1_e5_20261112/E4_E5_4090/E4_QUERY_FACTORIAL_RESULTS.md](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/fa9bb5bf1794cb11e7d0b504747c13c304fc5d00/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E4_E5_4090/E4_QUERY_FACTORIAL_RESULTS.md) |
| 循环边界到晚期计算 | 全序列边界状态影响L10-16 attention及L18双分支；MLP恢复在Validation未全部通过 | 不再做相同固定delta抑制/恢复；也不能据MLP失败断言无MLP作用。 | [k1_k2_latent_mechanism_completion/UPSTREAM_CAUSAL_TRACING.md](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/fa9bb5bf1794cb11e7d0b504747c13c304fc5d00/experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_latent_mechanism_completion/UPSTREAM_CAUSAL_TRACING.md) |
| 自然标签/选项重编码 | 22280自然输入变体；Train信号未复现固定Validation，等价检验也未通过 | late label重编码可作为新语义试验控制，不能重新宣传成标签不变性已成立。 | [latent_dynamics_continuation_20261113/E7_natural_reencoding/README.md](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/fa9bb5bf1794cb11e7d0b504747c13c304fc5d00/experiments/ouro_latent_state_mechanism_v1/research_plans/latent_dynamics_continuation_20261113/E7_natural_reencoding/README.md) |
| 三答案分离与新组合 | 四个不同cabinet；512heldout中86cross，其中68同时不同于原/供体自然预测；强抽象门槛失败 | 增加三答案或再做source boundary swap已重复。 | [k1_k2_computational_mechanism_v1/INTERMEDIATE_VARIABLE_PATCHING.md](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/fa9bb5bf1794cb11e7d0b504747c13c304fc5d00/experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_computational_mechanism_v1/INTERMEDIATE_VARIABLE_PATCHING.md) |
| 广域后继状态运输 | 保留原R，替换source_start:R后继状态得441/512；可能已携带预计算答案 | 不能把整G搬运后的正确性当局部组合发生处。 | [k1_k2_computational_mechanism_v1/INTERMEDIATE_VARIABLE_PATCHING.md](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/fa9bb5bf1794cb11e7d0b504747c13c304fc5d00/experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_computational_mechanism_v1/INTERMEDIATE_VARIABLE_PATCHING.md) |
| 单边实际K/V交换 | 预设9单边位置/类型在新确认中均未建立对照特异性；最多43/256对native34，unrelated40 | 无依据的单token/单head猜测和筛选失败路径不可再次伪装成新机制。 | [k1_k2_causal_edge_discovery_v1/FINAL_MECHANISTIC_CONCLUSION.md](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/fa9bb5bf1794cb11e7d0b504747c13c304fc5d00/experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_causal_edge_discovery_v1/FINAL_MECHANISTIC_CONCLUSION.md) |
| K1/K2时间切断 | capacity/native确认各+.653564，双向汇总相等及interaction0是结构恒等；延迟C01仅158/512，自然441 | 延迟能力与自然敏感性的语义区别保留，但数值不是两份独立证据。 | [k1_k2_temporal_causal_bridge_v1/TEMPORAL_2X2_RESULTS.md](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/fa9bb5bf1794cb11e7d0b504747c13c304fc5d00/experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_temporal_causal_bridge_v1/TEMPORAL_2X2_RESULTS.md) |
| 完整接收区域入口 | 自然K1背景map-only保留97.441%指定margin，431/512对441；其余区域未在确认重新全搜索 | 可复用wholemap位置；97%不是全部推理的解释比例。 | [k1_k2_temporal_causal_bridge_v1/RECEIVER_REGION_RESULTS.md](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/fa9bb5bf1794cb11e7d0b504747c13c304fc5d00/experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_temporal_causal_bridge_v1/RECEIVER_REGION_RESULTS.md) |
| map→H粗层路径 | controlledT下mid恢复+.593506/reverse损失+.593018；early弱，late不明确；random近似幅度匹配 | 没有直接测试nativeC11的对应实际读取；窗口不等于组成算法。 | [k1_k2_temporal_causal_bridge_v1/PATH_BLOCK_AND_RESTORE.md](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/fa9bb5bf1794cb11e7d0b504747c13c304fc5d00/experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_temporal_causal_bridge_v1/PATH_BLOCK_AND_RESTORE.md) |
| 训练外输出控制/优化 | CATC、critical、contextconsistent等均未建立完整预设utility门槛 | 本目标是机制，不再搜索系数/策略/准确率。 | [catc_training_free_loop_v1/FINAL_SCIENTIFIC_REVIEW.md](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/fa9bb5bf1794cb11e7d0b504747c13c304fc5d00/experiments/ouro_latent_state_mechanism_v1/research_plans/catc_training_free_loop_v1/FINAL_SCIENTIFIC_REVIEW.md) |

## 目标完成审计

| 目标要素 | 当前证据状态 | 仍需辨别 |
|---|---|---|
| K1→K2传递时机和路径 | 已有受控前缀→map→H及粗时间区间证据 | 自然完整背景中实际语义读取事件、谁在何时执行组合 |
| 第二轮如何参与关系组合 | 新旧文本/变量交换证明部分语义影响，查询输入合法完整 | map结果预计算、整表relay、H组合的竞争解释仍可产生相同观测 |
| 具体跨Token计算 | 现有KV路径和晚期分支定位 | 具体查询如何选择记录及地址/内容配对随两跳关系变化 |
| 不预设独立中间变量 | 新A/B/C使用真实高维KV，无低维probe替代 | 不把任何单位置或可搬运状态自动称变量/组合器 |
| Astra分析下一步 | 已实际委派Astra并进行两轮去重/识别性修订 | 最终冻结唯一可执行配置与判据 |
| 检查仓库避免重复 | 完整树/所有Markdown哈希索引及上表证据阅读 | 新协议每个测试须对应至少一个未完成的判别项 |

结论：完整研究目标未完成，保持active；“本轮实验完成”与“完整机制已回答”必须分开。已有成功结果不授权将代数null、未来因果不变性或对照弱化改称完整算法发现。

## 必须先修的原生背景算子风险

`bridge.py`在`closed.any()==False`时先return，nativeC11(mask7)下后续path分支不会执行。旧C(mask1)仍有效。新算子必须与S通道切断解耦并断言实际编辑(loop,layer,receiver,source)集合。原`full_path_block_H_norm_exact==Z`也依赖controlledT把其他S→H关闭，不能要求原生完整背景阻断G后必然等Z。self/noop必须bitwise；KV联合置换为实数代数null，GPU浮点数值漂移须单列实测而非当科学效果。

执行者：Sol审计与实现；Astra负责后续识别性分析。所有旧成果保留。

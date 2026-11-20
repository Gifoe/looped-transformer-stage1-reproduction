# 主张与证据

| 可检验主张 | 样本、指标与对照 | 结论及证据 |
| --- | --- | --- |
| 有序完整 pair 流在小 α 二阶匹配真实 AM | 解析展开、相同起点与丰富网格 | 解析成立。正负 bracket 分开验证。[推导](THEORY.md) [原始数值](results/LOCAL.json) |
| 两阶段 norm 流复现实数 RMSNorm | γ 全正，显式径向和对角流，FP64、FP32、BF16 | 端点验证通过，较低精度存在舍入。[审计](results/AUDIT.json) |
| 原生幅度下完整修正改善局部状态误差 | 状态相对 L2，完整 vs 基础、自身、顺序、反号、Taylor、前缀 Anderson | 相对基础场和两种单项有效。前缀 Anderson 更优。[图 2](results/figures/figure_2_unit_amplitude_pair.pdf) |
| 两种结构的 K4 原生联合响应平均为正 | 两族各 96 个新组，世界组配对 bootstrap，H11 目标异于三种已知世界 | 两族确认，幅度和答案命中因任务而异。[逐组数据](results/NATIVE_confirmation.json) [对照](results/CONTROLS.json) |
| 完整场提高自由推进中四世界交互的预测 | 每族 48 个新组，独立 K1→K4，因果前缀 Anderson 与离散 Taylor | 未通过 K2/K4 联合门槛。独立复核未支持。[冻结规则](results/FREE_METHOD.json) [预测](results/PREDICTIONS_validation.json) |
| 顺序项帮助解释末层原生 swap 的局部响应 | 两族共 48 组、在 K2/K4 各测一次，条件式前缀与末端 norm/head | 相对离散 Taylor 的配对误差更低，因果前缀 Anderson 误差最小。[局部原生真值](results/LOCAL_NATIVE_validation.json) |
| 顺序项是可恢复语义组合的机制 | 原生 BA 盲加三剂量，比较无补偿 BA、等范数随机和跨层方向 | 满剂量在两个 Loop 加大 BA 到 AM 的距离。没有达到特异性和补偿条件。[剂量图](results/figures/figure_5_native_compensation.pdf) |
| 两跳中间 b 对下游 D 有稳定可迁移表征 | 冻结词表头和同 b 不同 y / 同 y 不同 b 的 16 组匹配移植 | 证据不足以识别载体或推理顺序。[桥接实验](results/BRIDGE.json) |
| 连续场时间参数代表原生内在时间 | 两种光滑时钟的 fixed RK4，同一场、同一局部起点 | 网格加密支持时钟端点相容。材料未辨识内部时间或语义事件顺序。[时钟](results/CLOCKS_AND_BETA.json) |
| 通用准确率或推理成本改善 | 自然 ARC 96 题及算子/计时日志 | ARC 仅作自然轨迹复核，未建立性能优化。[ARC](results/ARC_NATIVE.json) [报告](REPORT_ZH.md) |


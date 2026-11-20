# 实验迭代

| 阶段 | 执行与裁决 | 证据 |
| --- | --- | --- |
| A | 从指定提交重算历史 63/96 与候选评分口径。实现配对有效场、解析 norm、BF16/FP32 对齐及前缀因果对照。全序列 Anderson 和 hybrid 记录为可泄漏对照，主表使用因果前缀 Anderson。 | [历史复核](BASELINE_AUDIT.md)、[审计](results/AUDIT.json) |
| B | 在单位幅度比较自身项、顺序项、完整、反号、离散 Taylor 与固定网格。RK4 在粗细网格与第二积分设置下交叉检查。完整项相对基础场的改善跨至四层，因果前缀 Anderson 的误差更小。 | [局部实验](results/LOCAL.json)、[短段](results/SHORT.json) |
| C | 两族各 2 组完成先导。导数查询成本据此缩减开发规模为每族预留组中的前 8 组。开发检查两个共享修订系数 0.5 和 0.25，预期缓和单位步高阶误差，失败条件为 K2/K4 未同时超过最强对照。选择一次原系数，无系数通过门槛。提前固定局部 L24、答案 token 支持与三档剂量。 | [预登记](PREREGISTRATION.md)、[自由推进锁](results/FREE_METHOD.json)、[局部锁](results/LOCAL_FREEZE.json) |
| D | 按失败分支运行均为独立组的 48+48 自由推进确认及 96+96 原生确认。局部 24+24 冻结组同时运行 AM→BA、AM 削弱、BA 定向补偿与两个同剂量匹配对照。桥接和自然题采用单独的评分。 | [预测](results/PREDICTIONS_validation.json)、[原生确认](results/NATIVE_confirmation.json)、[局部干预](results/LOCAL_NATIVE_validation.json) |
| E | 结束全轮场级机制主张。核对不同精度、独立结构、ARC、相同场的时钟重参数化和失败范围。事后诊断仅补齐精确状态误差与 ARC 选项重排，没有调节方法或更改主要分母。 | [中文报告](REPORT_ZH.md)、[数据校验](results/RELEASE_VERIFICATION.json) |

独立的 GPT-6.2 Sol High 子代理复核历史协议并实现回归脚本，主代理落实模型、评测和发布。Astra 在第一批数值审计完成后审查理论与访问权限，在开发结果形成后审查失败路线与局部判断。分离每个实验的预测和实际干预，并保存失败记录和运行日志。 Astra 还从冻结输出独立复算了 192 组原生对照、96 组独立自由误差和 48 组 L24 补偿的配对区间。

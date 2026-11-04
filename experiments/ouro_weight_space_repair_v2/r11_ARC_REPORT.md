# R11 完整ARC结果

冻结主方法 robust_cyclic_loop_axis：742/1172（63.3106%）。R6为736/1172；救回15、误伤9，净增6（+0.5119pp）。exact McNemar p=0.307456，20k配对bootstrap95%CI [-0.2560,1.3652]pp。

保留R6相对Fixed独有纠错26/35。达到751：False。

| 方法 | 正确/1172 | 救回 | 误伤 | 净变化 |
|---|---:|---:|---:|---:|
| native4 | 702 | 26 | 60 | -34 |
| fixed | 717 | 16 | 35 | -19 |
| adaptive | 716 | 23 | 43 | -20 |
| A1 | 726 | 13 | 23 | -10 |
| M0_R6 | 736 | 0 | 0 | 0 |
| robust_cyclic_loop_axis | 742 | 15 | 9 | 6 |
| robust_cyclic_loop_axis_prior_norm | 736 | 7 | 7 | 0 |
| robust_cyclic_loop_axis_shuffle_update | 709 | 17 | 44 | -27 |
| robust_cyclic_native_axis | 748 | 21 | 9 | 12 |
| joint1_only | 634 | 198 | 300 | -102 |
| joint4_only | 1021 | 326 | 41 | 285 |
| jointFixed_only | 1012 | 324 | 48 | 276 |
| cyclic_joint1_only | 657 | 193 | 272 | -79 |
| cyclic_joint4_only | 1036 | 335 | 35 | 300 |

主对比为R11对R6；其他匹配幅度、置乱及静态Native轴均为非晋级对照。按对应控制判断方向或循环差的独立作用，不能只比较弱随机对照。当前辅助轴含有用信息不等于证明其是语义。原R6主提示、完整候选及字符分母保持，但额外联合选项视图改变辅助模板；单独联合评分同列，若更强则不能声称R11具有整体方法优势。历史ARC/dev均已暴露，不是盲测或独立交叉验证，统计区间未校正跨轮研究者适应。

另外两个数据集按用户要求停止。模型权重冻结，无训练/控制器/adapter。部署路径的实际mainK5、N0priorK4及auxiliaryK4时间/峰值见r11_DEPLOYMENT_PROFILE.json；原始主/先验缓存采集和全部新worker成本分开披露，不把缓存重放当部署速度。测试后未改赢家。

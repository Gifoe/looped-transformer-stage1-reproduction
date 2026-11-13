# E6-A fusion results

## Historical homogeneous A6000 / Joint-Option Letter Scoring

2590 original questions; development Train excludes25 demonstrations (1094); Validation299 and exposed historical Test1172. All methods use the same archived full candidate scores. This is actual CPU evaluation of recovered native model outputs, not newly collected5090 model inference.

| Method | Track | Correct | vs K4 pp | Rescue/Damage | 95% paired pp CI | Mean loops |
|---|---|---:|---:|---:|---|---:|
| K1 | ZL | 626/1172 | -33.6177 | 41/435 | [-36.7747, -30.5461] | 1.0000 |
| K2 | ZL | 953/1172 | -5.7167 | 27/94 | [-7.5939, -3.9249] | 2.0000 |
| K3 | ZL | 1000/1172 | -1.7065 | 13/33 | [-2.8157, -0.5973] | 3.0000 |
| K4 | ZL | 1020/1172 | +0.0000 | 0/0 | [+0.0000, +0.0000] | 4.0000 |
| uniform34 | ZL | 1017/1172 | -0.2560 | 7/10 | [-0.9386, +0.4266] | 4.0000 |
| uniform234 | ZL | 1011/1172 | -0.7679 | 13/22 | [-1.7918, +0.1706] | 4.0000 |
| uniform1234 | ZL | 1010/1172 | -0.8532 | 18/28 | [-1.9625, +0.2560] | 4.0000 |
| weighted4 | ZL | 1018/1172 | -0.1706 | 12/14 | [-1.0239, +0.6826] | 4.0000 |
| probability | ZL | 1013/1172 | -0.5973 | 14/21 | [-1.6212, +0.4266] | 4.0000 |
| log_probability | ZL | 1011/1172 | -0.7679 | 13/22 | [-1.7918, +0.1706] | 4.0000 |
| rank | ZL | 1003/1172 | -1.4505 | 9/26 | [-2.4744, -0.5119] | 4.0000 |
| cal_weight_0012 | Cal | 1020/1172 | +0.0000 | 7/7 | [-0.5994, +0.6826] | 4.0000 |
| cal_weight_0013 | Cal | 1023/1172 | +0.2560 | 6/3 | [-0.2560, +0.7679] | 4.0000 |
| cal_weight_0113 | Cal | 1018/1172 | -0.1706 | 9/11 | [-0.9386, +0.5973] | 4.0000 |
| cal_weight_0124 | Cal | 1019/1172 | -0.0853 | 10/11 | [-0.8532, +0.6826] | 4.0000 |
| cal_weight_1114 | Cal | 1022/1172 | +0.1706 | 12/10 | [-0.5973, +0.9386] | 4.0000 |
| cal_weight_1124 | Cal | 1019/1172 | -0.0853 | 12/13 | [-0.9386, +0.7679] | 4.0000 |
| cal_weight_0122 | Cal | 1018/1172 | -0.1706 | 14/16 | [-1.1092, +0.7679] | 4.0000 |
| cal_weight_0101 | Cal | 1001/1172 | -1.6212 | 12/31 | [-2.7304, -0.5119] | 4.0000 |

Primary ZL weighted4 yields1018/1172 vs Native K4 1020, with12 rescues and14 damages. It fails the primary improvement objective. Other frozen Test rows are disclosed as exploratory, not selected as a new primary.

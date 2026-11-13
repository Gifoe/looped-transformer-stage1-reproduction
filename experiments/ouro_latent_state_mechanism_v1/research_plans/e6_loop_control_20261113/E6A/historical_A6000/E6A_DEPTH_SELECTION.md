# E6-A depth selection

## Historical homogeneous A6000 / Joint-Option Letter Scoring

2590 original questions; development Train excludes25 demonstrations (1094); Validation299 and exposed historical Test1172. All methods use the same archived full candidate scores. This is actual CPU evaluation of recovered native model outputs, not newly collected5090 model inference.

| Method | Track | Correct | vs K4 pp | Rescue/Damage | 95% paired pp CI | Mean loops |
|---|---|---:|---:|---:|---|---:|
| K1 | ZL | 626/1172 | -33.6177 | 41/435 | [-36.7747, -30.5461] | 1.0000 |
| K2 | ZL | 953/1172 | -5.7167 | 27/94 | [-7.5939, -3.9249] | 2.0000 |
| K3 | ZL | 1000/1172 | -1.7065 | 13/33 | [-2.8157, -0.5973] | 3.0000 |
| K4 | ZL | 1020/1172 | +0.0000 | 0/0 | [+0.0000, +0.0000] | 4.0000 |
| highest_margin | ZL | 1004/1172 | -1.3652 | 13/29 | [-2.4744, -0.2560] | 4.0000 |
| highest_entropy | ZL | 1010/1172 | -0.8532 | 13/23 | [-1.8771, +0.1706] | 4.0000 |
| vote | ZL | 1007/1172 | -1.1092 | 4/17 | [-1.8771, -0.3413] | 4.0000 |
| stability | ZL | 1007/1172 | -1.1092 | 4/17 | [-1.8771, -0.3413] | 4.0000 |
| conservative | ZL | 1019/1172 | -0.0853 | 2/3 | [-0.4266, +0.2560] | 4.0000 |
| rollback | ZL | 1020/1172 | +0.0000 | 0/0 | [+0.0000, +0.0000] | 4.0000 |

All non-native selectors above have paid the full four-loop output budget. Selecting an earlier output after K4 is not early exit.

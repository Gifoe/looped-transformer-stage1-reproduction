# R28 幅度控制完整性补充

状态：事前设计补充；没有新模型条件、GPU 或效果读取。不修改既有 draft、已审文件或 R27 发布记录；实施后须以新版本源码/CPU receipt/hash 纳入最终冻结。

**接受加入原 r7_math 的 matched_scale 控制。** G-matched 匹配主修正 h 的幅度但保留另一先验方向；vocabulary-null 匹配打乱后修正幅度。两者都不能单独排除“收益只是将原 guidance 整体缩小”的解释。新增控制匹配修正后 guidance 的 p 加权范数，直接检验这一替代解释。它是既有 R7 诊断的恢复，不是新的可晋级方法。

沿用主方法同一次 native/CF readout 和主 correction：p、g、gc=C_p(g)、h=alpha*ebar 均与主方法完全同源。

```
g2 = sum_v p[v] * gc[v]^2              # FP64
r2 = sum_v p[v] * (gc[v]-h[v])^2      # FP64
rho = sqrt(max(r2/g2,0)) if g2>0 else 1
Z_scale = float32(float64(q4) + rho*float64(g))
if h is exactly zero: Z_scale = original Fixed exactly
```

保持原 r7_math matched_scale 的运算精度、非负平方根与 gzero→rho1 规则，不改温度、不扫描缩放、不加入 epsilon。显式零修正分支保证 alpha=0或修正向量全零时严格退回原 FP32 Fixed，不因另一路算术形成伪差。完整 FP32 fullV log_softmax 后再 gather，沿用相同字符分母和 Q4−Q5 late LL 差。非有限值 invalid。

这里匹配的是 FP64 中心化、最终 FP32 logit 舍入前的 residual guidance 范数；不声称舍入后词表向量或实际 candidate LL 范数逐位匹配，也不要求 control 与主方法具有相同 correction norm。二者目标本来不同。

CPU 合成检查应覆盖：零 g、零 h、普通非共线 g/h、rho公式独立 NumPy/Torch一致、返回Fixed零分支exact、目标 token不参与、既有主输出不变。在非退化例核验 `||rho*gc||p == ||gc-h||p` 到预先的 FP64 数值精度；真实记录 g2/r2/rho及有限性，不能按效果改变规则。直接调用旧函数须传入与主方法相同 e_j=.5(native−CF) 的排列；不能误把 generic loop guidance 输入该 control。

**效果判定补充**：原 strong screen 的全部条件保持，并额外要求主方法正确数严格高于 matched_scale（等价 paired rescue−damage>=1）。该控制不可晋级、不可作为 fallback，不能据其结果调主方法强度。原 cost_review_eligible 条件完全不变，不把该字段改称性能成功。原完整性要求同样覆盖新增 control 的 scalar/diagnostic 输出。

无额外 forward；只增加每位置已有 fullV 上的 FP64归约、输出和评分，全部耗时仍计原一次 worker 预算。源码审查/新 CPU receipt 完成并由 Root 冻结前不执行 GPU。

# R34 事前补充：uniform-background责任控制

状态：新特征/head执行前的设计补充；主方法仍唯一B，原设计文件不改。无GPU授权，无额外forward或head，拟议150秒预算不变。

**接受加入此控制。** 当b(v)=1/V时，r(v)=pF(v)/(pF(v)+1/V)随pF(v)严格递增；再以pF*r归一化会相对偏向原本高概率的词。故B可能仅凭不含参考内容的自动锐化得到变化。generic loop norm、residual-guidance scale及PMI-matched都没有去掉reference内容、保留同一responsibility非线性，不能仅凭它们范数匹配就排除此解释。

沿主方法已有a=log_softmax64(literalFixed)、p0=softmax64(q4)，固定V=49152：

```
log_bU[v] = -log(V)
ell_U = a - logaddexp(a, log_bU)
h_U_raw = -C_p0(ell_U)
```

对h_U_raw应用原norm-matching到主h的p0范数，输出`Z_U=float32(float64(F)-h_U_matched)`，然后同FP32完整log_softmax、target gather、原位置sum/字符分母/late。没有新温度、参数、截断或参考子集。它是同一soft-responsibility非线性在无reference内容背景下的对照，不称与所有temperature锐化完全等价。

规则沿原合同：主h exact zero直接literalFixed；主非零、h_U_raw norm<=64eps64*maxabs(h_U_raw)^2时标unmatched，整轮不获strong或cost资格，不补epsilon。全部已有positive-domain/finite checks保持；uniform背景本身严格正，不绕过pF/p0域检查。不得重抽或换其他非信息背景。

CPU新增合成：非uniform pF时uniform责任会产生非零方向；uniform pF时该方向为常数/center零；主零分支rawexact；普通非共线例norm匹配；构造主非零但uniform方向退化时正确标unmatched。独立NumPy/Torch核验并刷新新源码/receipt/draft，原文件及旧版本保持。

**strong gate**：在原M3、J2、G-matched、matched_scale、vocabulary-null、PMI-matched六项之外，再要求primary正确数严格高于`uniform_background_matched`。它不可晋级、不可fallback，不按结果调整主法。cost_review_eligible的原逻辑不变，但与其他controls一样必须先满足完整numeric/matched资格；不因增加控制而降低性能门槛。

归因边界：若主方法不能严格胜此控制，本pilot没有证据把变化归于参考内容；若胜出，也只是参考信息相对这个固定无内容背景有有限增量，不证明正确语义或泛化。剩余reference形状与校准风险仍需完整后续验证。

论文层面的LoopCD+calibration与native4+同calibration对照仍是独立贡献归因问题，标记为未来diagnostic，不将其偷偷增加为本轮模型条件或效果门槛。本补充仅堵住已明确的uniform自信锐化解释，不改变B公式。

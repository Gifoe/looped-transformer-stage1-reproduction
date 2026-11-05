# B最终采集前科学源码审查

Astra，2026-11-05，assessment only。已读analyze.py、core.py、worker.py、CPU_CHECKS.json、intervention_protocol.json；Root负责独立prompt/target/预算与supervisor终核。未启动模型/GPU。

**科学语义PASS，无阻断。** 可进入最终freeze/readback，实际START仍由Root授权。不是已取得实验结果。

1. prediction先在显示次序argmax，再通过order_to_original映射。P1非selfinverse不被误用作inverse；expected_display显式用order.index(expected_original)。并列遵守显示次序。最终P0/P1历史端点另有exact gate。
2. I=original−modified，J(i,l)=I(i)−I(l)。J正值表示原始i对l的margin高于替换后：对gold-relative J，正值意味着替换降低正确项相对margin，不是“干预改善”。meanabs/RMS只有响应大小，不提供有益方向。
3. Q/N两侧用相同wrong槽位与同replacement；Q−N DID符号一致；每个相邻K1→2、2→3、3→4均输出。完整unchanged options/pairs及gold-relative方向同时保留，无事后挑stage或改符号。FP64 normalizer分解残差明确保留，未冒称与实际FP32 sum位级相同。
4. 24题placeholder/位置/NoQ与4题matched/B5/B6分母独立。Root已事前接受secondary仅前4；20题是未尝试，不是语义拒绝。moved-P0-content子集为预定描述性分母，仍保留全24主结果，不能用其替代整体结论。
5. B5 raw sum与原答案正文字符归一分列，协议指定fullanswer charnorm为主、raw诊断；不跨格式直接比较绝对LL。B6仅4题，输出所有K；K1已天花板时不能据无提升否定H3，简单任务阳性也不能估计ARC收益的H3占比。
6. worker gold-free递归约束，不作argmax或拟合；先全部P0历史exact才进入干预。analyzer先完成所有任务、向量、LP、参数、账本、source gates再反序列化constructor labels。partial/P0-only不会解封干预效果。历史端点strict equality与CPU FP64重算GPU FP32 logsoftmax的预定数值容差是不同用途，没有将后者替代历史parity。
7. Root实际CPU图为218body/872heads，预估523.9556秒；64方案498body/998.4133秒。都是规划估计，非耗时下界或成功保证；710.6733秒余额不变。完整执行或合法技术退出后才更新账本与解释。

结果解释仍受lexical/format/length confounds约束。Q/N question长度不同；matched相同shape不等于难度匹配；非零DID不自动证明语义比较或有益推理。若实际结果不足以区分H1/H2/H3，应明确保留混合解释并停止，不能强推Experiment C。

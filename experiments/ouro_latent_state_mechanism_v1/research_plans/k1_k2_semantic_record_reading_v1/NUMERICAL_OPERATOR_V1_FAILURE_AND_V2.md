# 数值定义、失败与修订

V1：原full-shaped SDPA直接置换实际post-RoPE两record的K/V。32旧world×6条件=192forward，noop/self64项full-vocab bitwise通过，但joint在[{'id': 'TB008_L1_donor_edit', 'arm': 2, 'loop': 2}]改变原生非并列预测，epsilon_gap=0.1875，状态FAIL_PERMUTATION_NULL。没有放宽该门槛或抹去失败。

V2：在实际原生SDPA输出O上加入由实际current Q/K/V完整因果softmax得到的FP64理想配对差值Delta，再一次BF16回写。V2R1仅补齐全部选中事件的实际o_proj差值诊断，重新224旧技术forward。原V2的224forward、源码和审计也保留。V2是明确的新estimand，并非V1原生BF16重算的逐位等价版本。

严格源冻结、self/joint构造身份、K/V构造等价、194前缀点、实际48次originalSDPA调用与8次事件均检查。CPU独立NumPy重新构造完整K/V交换softmax，没有复用生产配对公式；覆盖4旧world、160事件；完整head审计448；误差/尺度tau最大9.3760663e-05；native BF16与FP64重建perheadrow最大相对误差0.0041837186<1%。

该perheadrow误差界是4个旧技术world全部8个事件见证的范围。科学全体记录global-H重建误差、实际before/after范数及projected变化、所有前缀身份与调用集合；没有声称每个科学样本都保存/独立重算全部内部张量。actualBF16write不等于请求Delta，也不等于投影后norm；报告分别保存。

同一GPU的3个进程按完整basegroup分片执行确认；原冻结科学代码仅通过已记录AST适配器过滤qs及分片completion文件名。完整实际AST、wrapperhash、freeze绑定、rawSHA与全ID覆盖均保留。模型、输入、因子、门槛与算子定义未改变。

"""Build the Chinese research reports from audited measurements."""
import argparse
from pathlib import Path

from summarize import FAMILIES, load


NAMES={"tableau":"3×3 新结构","twohop":"双跳新结构","arc":"ARC 验证面板"}
METHODS=("A","B")
LEVELS=(16,32,64,128,256)


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|"]+
                     ["| "+" | ".join(map(str,r))+" |" for r in rows])


def val(x):
    return f"{float(x):.5g}"


def mean(d):
    return val(d["mean"])


def write(dest,name,*body):
    (dest/(name+".md")).write_text("\n\n".join(body)+"\n",encoding="utf-8")


def pair(s,stage,f,method,m,solver="euler"):
    return s[stage][f][method][f"{solver}{m}:{solver}{m*2}"]


def ep(s,stage,f,k):
    return s[stage][f]["endpoints"][k]["state"]


def render(root,out):
    s=load(root,"SUMMARY")
    native=load(root,"NATIVE_PARITY")
    runtime=load(root,"RUNTIME")
    e_rows=[]
    for stage in ("primary","holdout","precision_bf16","precision_fp32","reference"):
        for f in FAMILIES:
            for mode in METHODS:
                for m in LEVELS:
                    k=f"euler{m}:euler{m*2}"
                    pairs=s[stage][f][mode]
                    if k not in pairs:
                        continue
                    q=pairs[k]
                    e_rows.append((stage,NAMES[f],mode,f"{m}→{m*2}",mean(q["before_final_norm"]),
                                   mean(q["end"]),val(q["worst_time_mean"]),
                                   val(q["endpoint_p"]) if "endpoint_p" in q else "—",
                                   val(q["median_time_p"]) if "median_time_p" in q else "—",
                                   val(q["endpoint_token_max"]),val(q["logit_max"]["max"]),q["argmax_changes"]))
    rk_rows=[]
    for stage in ("primary","precision_fp32","reference"):
        for f in FAMILIES:
            for mode in METHODS:
                for k,v in s[stage][f][mode].items():
                    if k.startswith("rk4"):
                        rk_rows.append((stage,NAMES[f],mode,k,mean(v["before_final_norm"]),mean(v["end"]),
                                        val(v["worst_time_mean"]),v["argmax_changes"]))
    local=[]
    for f in FAMILIES:
        d=s["primary"][f]
        v=d["B_matched_branch_vs_accumulated"]
        local.append((NAMES[f],mean(v["mean_local"]),mean(v["last_global"]),v["max_local_branch"][0],
                      val(v["max_local_branch"][2]),d["largest_branch_defect"][0],val(d["largest_branch_defect"][1])))
    write(out,"NUMERICAL_CONVERGENCE",
          "# 数值收敛",
          "主面板固定 12 题，每个任务 4 题。3×3 和双跳从上轮结构留出中各抽取四个不同组，"
          "ARC 从上轮冻结面板的四类原生 K1/K2 正误转移中各选一题。独立复核有 8 题，"
          "其中 ARC 仍覆盖四类。FP32 精度面板对每类取一题，以相同 singleton 布局比较 BF16 和 FP32。",
          "Eₘ 是相同人工时间、有效 token 全宽相对 L2，分母为较细路径的状态范数。"
          "pₘ=log₂(Eₘ/E₂ₘ)，表中对样本均值计算。B 路径在每个分支边界及四分点比较，"
          "终点误差单列归一化前后。主表最坏 token 是同一终点全部有效 token 的最大相对误差。"
          "分母较小的 token 可能出现较大的比例。logit 使用原冻结候选，但它们的 argmax 只描述这个固定面板。",
          table(("阶段","队列","族","m","归一化前 E","终点 E","时间网格最大均值","终点 p","沿轨 p 中位数",
                 "最坏 token","最大候选 logit 差","预测变化数"),e_rows),
          "## Euler 与 RK4",
          "RK4 的一个微步调用四次向量场。对比包含同积分器相邻分辨率，以及同微步数和同前向调用数的交叉对照。"
          "在 BF16 量化场中，不能把 RK4 的名义四阶直接套到实际测量。FP32 列的误差也未免除舍入误差。",
          table(("阶段","样本","族","对比","归一化前","归一化后/整轮终点","时间网格最大均值","预测变化数"),rk_rows),
          "## 分支积分误差的传播",
          "局部误差用每个 B512 分支起点重启 B256，误差传播用自由运行的两条完整路径。"
          "两种对照对相同时间和相同分支计算，但起点控制不同。单次原生步缺陷以同一 B512 分支起点测量 h+rᵢ(h) 与分支积分终点的差。",
          table(("队列","48 分支局部 E 均值","整串累积 E","局部 E 最大分支","该局部 E","单步缺陷最大分支","该单步缺陷"),local),
          "[步长图](results/figures/figure_1_halving_trajectories.pdf) · "
          "[积分器图](results/figures/figure_2_euler_rk4.pdf) · [分支误差图](results/figures/figure_3_branch_errors.pdf)")

    fp_rows=[]
    fields=[]
    rounding=[]
    for f in FAMILIES:
        v=s["precision_fp32"][f]
        bf=s["precision_bf16"][f]
        for k in ("BF16_vs_FP32_K1","BF16_vs_FP32_native_K2","FP32_K2_own_vs_common_anchor",
                  "BF16_vs_FP32_A256","BF16_vs_FP32_B256_pre","BF16_vs_FP32_B256_post",
                  "B_norm_only_BF16_vs_FP32_same_input"):
            fp_rows.append((NAMES[f],k,mean(v["endpoints"][k]["state"])))
        for mode,q in v["same_input_field_errors"].items():
            fields.append((NAMES[f],mode,*[mean(q[k]) for k in ("state_cast_FP32",
                "bf16_vs_fp32_math_same_rounded_input","bf16_default_vs_math","fp32_default_vs_math")]))
        rounding.append((NAMES[f],mean(bf["rounded_A"]),mean(bf["rounded_B_pre"]),mean(bf["rounded_B"]),
                          mean(bf["rounded_A_displacement"]),mean(bf["rounded_B_displacement"])))
    write(out,"FP32_PRECISION_AUDIT",
          "# FP32 精度审计",
          "权重源是冻结 BF16 检查点。加载参数提升至 FP32，保存同一权重的数值，禁用 TF32 和 autocast，"
          "Q 线性投影、MLP 投影、Sandwich Norm 和边界 Norm 均通过实际 I/O dtype 钩子。"
          "真正 FP32 比较直接在 FP32 上做前向，主对照固定同一个 BF16 K1 的精确 FP32 表示作为起点。"
          "另从 FP32 的原生 K1 计算原生 K2 状态，分离锚点精度影响。模型参数哈希在测量前后不变。",
          table(("样本","比较","相对 L2"),fp_rows),
          "## 相同状态的向量场评估",
          "在 BF16 的实际轨迹检查点上，显式强制 SDPA math 后比较。"
          "第一列只在 FP32 前向中量化输入。第二列保持输入和 SDPA 后端相同，切换模型计算精度。"
          "最后两列在同一精度中比较默认 SDPA 与 math。"
          "A 的场评估比较 F 输出，B 比较分支 rᵢ 输出。对比为相对 L2，这些量没有直接可加性。",
          table(("样本","族","仅 BF16 输入舍入","同舍入输入 BF16↔FP32 math","BF16 默认↔math","FP32 默认↔math"),fields),
          "## 累加器对照",
          "在同一 BF16 向量场上，每一次状态加法后把结果舍入 BF16，再与 FP32 累加比较。"
          "末两列用于判断它是否仍从 K1 移动。",
          table(("样本","A256 终点变动","B256 预 Norm 变动","B256 后 Norm 变动",
                 "BF16 累加 A 距 K1","BF16 累加 B 预 Norm 距 K1"),rounding),
          "[参数哈希与开关记录](results/WEIGHT_PRECISION_AUDIT.json) · [精度与归一化图](results/figures/figure_7_norm_precision_finite_steps.pdf)")

    endpoints=("K1_vs_native_K2","A_vs_native_K2","B_pre_vs_native_pre","B_post_vs_native_K2",
               "A_vs_B_pre","A_vs_B_post","native_norm","B_norm")
    measures=[(NAMES[f],*[mean(ep(s,"primary",f,k)) for k in endpoints]) for f in FAMILIES]
    ablation=[(NAMES[f],mean(s["primary"][f]["A_norm_intervention"]["pre"]),
               mean(s["primary"][f]["A_norm_intervention"]["post"]),
               mean(ep(s,"precision_fp32",f,"FP32_A_vs_B_pre")),
               mean(ep(s,"precision_fp32",f,"FP32_A_vs_B_post"))) for f in FAMILIES]
    comm=[[NAMES[f],*[mean(s["dynamics"][f]["commutator_coefficients"][str(a)]) for a in (.02,.01,.005)],
           mean(s["dynamics"][f]["direction_cos"]),s["dynamics"][f]["negative_direction_pairs"]] for f in FAMILIES]
    reference_names=("A_RK128_vs_B_RK128_pre","A_RK128_vs_B_RK128_post","A_RK128_vs_native_common",
                     "B_RK128_vs_native_common","A_E512_vs_RK128","B_E512_vs_RK128")
    refs=[(NAMES[f],*[mean(ep(s,"reference",f,k)) for k in reference_names]) for f in FAMILIES]
    outcome=[(NAMES[f],*[s["primary"][f]["endpoints"][k]["changed_argmax"] for k in
              ("A_vs_native_K2","B_post_vs_native_K2","A_vs_B_post")]) for f in FAMILIES]
    write(out,"DISCRETE_VS_CONTINUOUS",
          "# 原生离散步与两种构造流",
          "本文简写 A 表示整轮 Euler/RK4，B 表示分支积分，原生更新单独记为 F。"
          "设 Rᵢ(x)=x+rᵢ(x)，N 为循环末端 RMSNorm。固定输入的原生一轮为 F=N∘R₄₈∘…∘R₁。"
          "A 的向量场为 F(x)−x，单位时间一次 Euler 更新就是 F。"
          "B 的 48 个局部向量场依次为 rᵢ，每个积分单位分支时间。"
          "每段只做一次单位步长 Euler 也复现 Rᵢ，最后再应用 N。B 的连续分段流末端还有一个离散 N。",
          "以真实数运算理解，固定因果掩码和位置后，Softmax、SiLU、线性映射和有正 ε 的 RMSNorm 产生局部 Lipschitz 的向量场。"
          "分支后置 RMSNorm 使 rᵢ 有界，整轮末端 RMSNorm 使 F 有界，因此 F(x)−x 至多线性增长。"
          "这足以推出每个已指定场的有限时间初值解存在且唯一，连续分段从各自前一段的终点重启。"
          "真实数结论不直接涵盖 BF16 量化场。量化门槛使实际 BF16 前向形成不连续的近似向量场。",
          "改变有限残差步为精细积分会引入自相互作用。单分支流的短时展开为"
          " x+τrᵢ(x)+(τ²/2)D rᵢ(x)rᵢ(x)+O(τ³)。单位 Euler 步只含线性项。"
          "顺序组合还有 D rⱼ(x)rᵢ(x) 一类跨项。对调相邻的小残差映射时，其主导差为"
          " α²(D r_M r_A−D r_A r_M)。本实验在共享起点用 α 的三个尺度计算等价差分，"
          "比值由两分支更新范数之和归一化。正式 B 轨迹始终保持原生顺序。",
          "不交换的证据只说明顺序组合会影响结果。即使分支可交换，单分支流与单次大 Euler 步仍可不同，"
          "A 的整轮场定义及 Norm 放置也足以让 A 与 B 表现不同。"
          "诊断中将 A 中场的 Norm 移除，再比较其原始终点及只在终点补 N 的状态。"
          "该干预不把各因素解释成可加的百分比分解。",
          table(("样本","K1→K2","A→K2","B→原生预 Norm","B→K2","A→B 预 Norm","A→B 后 Norm",
                 "原生 N 跳变","B 的 N 跳变"),measures),
          table(("样本","A 场去 N","去 N 后只在末端加 N","FP32 A→B 预 Norm","FP32 A→B 后 Norm"),ablation),
          table(("样本","交换系数 α=.02","α=.01","α=.005","同起点 Attention/MLP 方向余弦均值","余弦为负的层数 / 24"),comm),
          "## 同起点的 FP32 高阶参考",
          "A 和 B 的参考值使用 RK4 m=128，均由相同的 BF16 K1 经精确提升后开始。"
          "原生单步比较在 FP32 下对此同一 K1 直接求 F。对应 Euler m=512 的距离也一起报告。",
          table(("样本","A→B 预 Norm","A→B 后 Norm","A→原生 F(K1)","B→原生 F(K1)","A Euler→RK4","B Euler→RK4"),refs),
          "下面是主面板每类 4 题的候选首最大预测分歧数。未将这批人为分层题用于准确率推断。",
          table(("样本","A 对 K2","B 对 K2","A 对 B"),outcome),
          "L2 表的分母均为比较对的第二个状态。组合映射从右到左。",
          "跨 A/B 的同进度图用于描述路径差距。A 的整轮时间与 B 的分支进度无物理同一性。",
          "[官方实现](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)"
          " · [分裂误差的 Lie 元素展开](https://www.opuscula.agh.edu.pl/om-vol34iss2art3)"
          " · [原生距离图](results/figures/figure_4_native_and_continuous_endpoints.pdf)"
          " · [轨迹差异图](results/figures/figure_5_dynamical_disagreement.pdf)")

    geom=[]
    dyn=[]
    perturb=[]
    k3=[]
    for f in FAMILIES:
        for kind in METHODS:
            g=s["primary"][f][kind+"_geometry"]
            geom.append((NAMES[f],kind," / ".join(val(t) for t in g["peak_speed_time_by_item"]),
                         val(g["largest_turn_degrees"]),val(g["maximum_curvature"]),mean(g["net_over_length"]),
                         " / ".join(val(t) for t in g["token_peak_p90_minus_p10_by_item"]),
                         " / ".join(val(t) for t in g["fraction_tokens_with_global_peak_time"])))
        d=s["dynamics"][f]
        for dtype,speed_key,gain_key in (("BF16","relative_speed_t0_t1_t4","stability"),
                          ("FP32","relative_speed_fp32_t0_t1_t4","stability_fp32")):
            dyn.append((NAMES[f],dtype,*[val(x) for x in d[speed_key]]))
            for k,v in d[gain_key].items():
                perturb.append((NAMES[f],dtype,k,*[val(x) for x in (*v["A_t1_t4"],v["B_end"],v["A_max"],v["B_max"])]))
        k3.append((NAMES[f],*[val(d["K3"][k]) for k in ("A","B","cross")]))
    write(out,"DYNAMICS_ANALYSIS",
          "# 几何、稳定性及跨轮检查",
          "各轨迹报告有效 token 的原始 L2 速度、相邻割线转角，以及单位方向改变量除以平均弧长的离散曲率。"
          "A 观测步为 1/32，B 为每个分支 1/4 的人工刻度。B 的边界转角包含场切换，表中的 B 曲率是离散描述量。"
          "净位移与弧长的比衡量回转和抵消。token 峰值时间跨度和与全状态峰值重合率衡量同步性。",
          table(("队列","族","全态速度峰值时刻","最大转角度数","最大曲率","净位移/弧长","token 峰值时间 P90−P10","峰时重合率"),geom),
          "## 有限时间的扰动响应",
          "在每类代表题固定两个正态方向和两个相对幅度。扰动与未扰动轨迹共享真实算子和相同采样网格。"
          "增益为状态差范数除以实际初始剂量。A 延展到 T=4，B 贯穿一轮预 Norm 分段。"
          "FP32 复核使用相同的提升后 K1、随机种子和实际剂量。",
          "A 的相对场速率为 ||F(H)−H||/||H||。有限时间增长或减弱不能判定 Lyapunov 指数、吸引域或周期轨道。",
          table(("队列","精度","相对场速率 T=0","T=1","T=4"),dyn),
          table(("队列","精度","种子/剂量","A T=1 增益","A T=4 增益","B 预 Norm 增益","A 峰值增益","B 峰值增益"),perturb),
          "## K2→K3",
          "对每类一题从真正的原生 K2 开始，以捕获的 K3 掩码和位置重新构造两条 m=128 轨迹。"
          "同样检查全部 48 个真实分支边界。以下为终点相对 L2。",
          table(("队列","A→K3","B→K3","A→B"),k3),
          "[速度、转角与曲率图](results/figures/figure_6_speed_turn_curvature.pdf)")

    primary_ends=[(NAMES[f],mean(pair(s,"primary",f,"A",256)["end"]),
                    mean(pair(s,"primary",f,"B",256)["end"]),
                    mean(ep(s,"primary",f,"A_vs_native_K2")),
                    mean(ep(s,"primary",f,"B_post_vs_native_K2")),
                    mean(ep(s,"primary",f,"A_vs_B_post"))) for f in FAMILIES]
    timing=[(k,val(q["wall_s"]),val(q["max_allocated_bytes"]/2**30),q["loop_field_calls"],q["attention_calls"],q["mlp_calls"])
            for k,q in s["timed_scope"].items()]
    rk_quantities=[(NAMES[f],mean(pair(s,"reference",f,"A",64,"rk4")["end"]),
                    mean(pair(s,"reference",f,"B",64,"rk4")["end"]),
                    mean(ep(s,"reference",f,"A_RK128_vs_B_RK128_pre")),
                    mean(ep(s,"reference",f,"A_RK128_vs_B_RK128_post"))) for f in FAMILIES]
    fp_speeds0=[s["dynamics"][f]["relative_speed_fp32_t0_t1_t4"][0] for f in FAMILIES]
    fp_speeds4=[s["dynamics"][f]["relative_speed_fp32_t0_t1_t4"][2] for f in FAMILIES]
    fp_gain4=[q["A_t1_t4"][1] for f in FAMILIES for q in s["dynamics"][f]["stability_fp32"].values()]
    ranges=lambda x: f"{min(x):.3g}–{max(x):.3g}"
    write(out,"ODE_REFINEMENT_ZH",
          "# Ouro-1.4B 的连续动力学细化",
          "## 结果摘要",
          "整轮场和逐分支流的理想实数初值解各自存在且唯一，实测分辨率下的数值细化支持两条不同轨迹。"
          "分支流比整轮流需要明显更细的步长。原生 K2 与两种构造终点的偏离远大于各自最细两档的数值差异。",
          f"模型为 574fa66 冻结检查点，使用 {runtime['gpu']}。主面板 12 道，独立验证 8 道，"
          "真 FP32 对照 3 道。整轮 A、分支 B 分别跑到 m=512，"
          f"并通过 {len(native['rows'])} 个固定 batch 的 48 分支边界及"
          f" {sum(x['full_vocab_values_bitwise'] for x in native['rows']):,} 个全词表 logit 的逐位核对。",
          table(("样本","A E₂₅₆","B E₂₅₆","A512→K2","B512→K2","A512→B512"),primary_ends),
          "在真 FP32 的三个同起点样本中，RK4 的相邻分辨率差和跨定义差如下。"
          "相邻差是数值一致性测量，精确解误差仍未知。",
          table(("样本","A RK4 64→128","B RK4 64→128","RK4 A→B 预 Norm","RK4 A→B 后 Norm"),rk_quantities),
          "## 1. 能否形成数值稳定的连续动力学？",
          "能。先指定场和初值，再讨论收敛。真实数解释的两个场都具有局部 Lipschitz 性与至多线性增长。"
          "主面板的步长减半结果、独立面板和真 FP32 对照给出有限分辨率验证。"
          "B512 仍有可测细化误差，因此把它视为数值代理。 [收敛表](NUMERICAL_CONVERGENCE.md)",
          "## 2. 两种方法分别趋于什么？",
          "取理想实数场时，A 趋向由 F(H)−H 定义、从原生 K1 出发的整轮场轨迹。B 依次趋向 48 个单位分支时长的局部流，"
          "在整个分段流结束后应用终端 RMSNorm。BF16 结果描述量化场，"
          "FP32 结果保留检查点权重数值并去掉 BF16 前向舍入。 [FP32 审计](FP32_PRECISION_AUDIT.md)",
          "## 3. 两种极限是否相同？",
          "各自细化后仍有大幅路径间差距，FP32 同起点对照也保留差距。"
          "证据范围是这些提示和定义的高精度数值比较。",
          "两种离散化都能复现同一原生映射，仅凭 m=1 等价无法识别内禀连续场。"
          "向量场定义、有限残差更新的高阶自相互作用、跨分支的非交换性和末端归一化共同决定差异。"
          "同起点的小幅算子对照与移除 A 场归一化的干预分别验证了顺序依赖和 Norm 位置的影响。"
          "归一化前已经存在差异。 [离散与连续关系](DISCRETE_VS_CONTINUOUS.md)",
          "## 4. 原生 K2 与连续流有什么关系？",
          "同一个原生离散映射既是 A 的一次单位 Euler 步，也是 B 的逐分支单位 Euler 组合后经 RMSNorm 的映射。"
          "两种极限给出不同的单位时间轨迹与候选分数。这个实验能区分原生的有限幅更新与细化，"
          "固定面板的终点偏差无法判断一般任务能力对离散步骤的依赖程度。",
          "## 5. 哪些规律值得做机制研究？",
          "同起点的局部分支误差小于沿真实顺序传播后的累积误差。"
          "分支流净位移与弧长的比明显较低，小幅 Attention/MLP 交换系数在三个剂量下接近。"
          f"真 FP32 整轮流的相对场速率从 T=0 的 {ranges(fp_speeds0)} 降到 T=4 的 {ranges(fp_speeds4)}，"
          f"同批轨迹的扰动增益在 T=4 为 {ranges(fp_gain4)} 倍。"
          "减速与有限扰动放大可以同时出现。有限扰动筛查覆盖三个代表题与两个方向，"
          "未建立长期吸引子、周期或混沌结论。"
          "K2→K3 的小样本复核直接复用真实原生边界。 [动力学](DYNAMICS_ANALYSIS.md)",
          "## 实际成本",
          "表中为同步分段测量，包括路径采集开销。加载、离线摘要、绘图及部分对齐检查不计入计时窗。"
          "Attention 和 MLP 按一次 batched 调用记一次。真实 GPU 工作随增加微步而上升。",
          table(("精度","同步墙钟秒","峰值 allocated GiB","整轮 F 评估","Attention","MLP"),timing),
          "[对齐](results/NATIVE_PARITY.json) · [样本 manifest](results/MANIFEST.json) · "
          "[数值汇总](results/SUMMARY.json) · [复现与图表](README.md)")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("dest",type=Path)
    a=p.parse_args()
    a.dest.mkdir(exist_ok=True,parents=True)
    render(a.root,a.dest)

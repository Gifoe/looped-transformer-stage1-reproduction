"""Generate the six Chinese reports from the checked numerical records."""
import argparse
import json
from pathlib import Path


def n(v):
    return f"{float(v):.5g}"


def tab(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])


def save(dest,name,*sections):
    (dest/(name+".md")).write_text("\n\n".join(sections)+"\n",encoding="utf-8")


def med(q):
    return n(q["median"])


def render(results,out):
    u=json.loads((results/"SUMMARY.json").read_text())
    rec=u["natural"]["float32"]["K1_K2"]["pooled"]
    cf=u["causal"]["float32"]["K1_K2"]["pooled"]["semantic"]
    par=json.loads((results/"NATIVE_PARITY.json").read_text())
    arc=json.loads((results/"ARC_CAUSAL_MANIFEST.json").read_text())
    manifest=json.loads((results/"MANIFEST.json").read_text())
    semantic_groups=sum(len(q["causal"]) for q in manifest.values())
    recovery=("BF16 第一轮重放在随机对照写入处遇到索引 dtype 错误。修复只增加写入前的 dtype 转换，"
              "并保留原始冻结源码与已完成 FP32 数据的哈希。 [恢复记录](results/RECOVERY.json)"
              if (results/"RECOVERY.json").exists() else "")
    norm_rows=[]
    for family,q in u["normalization"].items():
        for k,v in q["norm_inverse"].items():
            norm_rows.append((family,k,f"{v['valid']}/{v['tokens']}",
                              med(v["relative_error"]) if v["relative_error"] else "—"))
    spec_rows=[(f,n(q["gamma_abs_min"]),q["zero_gamma"],q["negative_gamma"],med(q["radial"]),
              " / ".join(n(v) for v in q["spectrum_full"]["singular_values"]))
               for f,q in u["normalization"].items()]
    inv_rows=[(f,k,med(v),n(v["maximum"])) for f,q in u["normalization"].items()
              for k,v in q["inverse_unit_solver_errors"].items()]
    save(out,"MATHEMATICAL_FEASIBILITY",
         "# 连续嵌入的可行性",
         "固定因果掩码和 RoPE 后，原生整轮在实数模型中是 F=N∘R₄₈∘⋯∘R₁，Rᵢ(x)=x+rᵢ(x)。"
         "光滑同维 ODE 的有限时流在有可逆解的区域内是局部微分同胚。"
         "其导数沿轨迹满足变分方程，行列式是散度积分的指数，故方向保持。"
         "局部退化、非单射或反向不可延拓都会对相应的精确嵌入形成限制。自治场还有额外的流半群约束。",
         "即使已有一个非自治嵌入，也可用端点固定的光滑递增时间重参数 φ 替换它的时钟。"
         "新场 φ′(t)v(z,φ(t)) 保留单位时映射，所以仅有离散端点无法确定内禀时间。",
         "若所有 γ 非零且 ε>0，N 在实数算术下有像内逆。令 z=y/γ，"
         "则 x=√ε z/√(1−mean(z²))，像是 mean(z²)<1 的区域。"
         "N 的像有界，因此含 N 的 F 无法成为整个无界 Rⁿ 上一个正反向完备流的单位时双射。"
         "这并未排除限制到较小域的光滑嵌入、允许有限反向域的流或混合模型。"
         "去掉 γ 的逐坐标缩放后，径向与切向导数的比是 ε/(mean(x²)+ε)。",
         tab(("样本族","最小 γ 绝对值","零 γ","负 γ","径/切比中位数","整轮五维 Krylov 限制奇异值"),spec_rows),
         "Krylov 用真实 JVP 和 VJP 在 JᵀJ 上构建五维方向。表列限制到这些方向的奇异值，"
         "无法认证全空间的最小奇异值、行列式符号或单射。径向近退化另用解析式和真实 Norm 输出测量。",
         tab(("样本族","输出路径","判别式为正 token","有效反演相对误差中位数"),norm_rows),
         "FP32 和 BF16 使用相同冻结参数值以及原生可用 SDPA。BF16 原生 Norm 用真正 BF16 模块，"
         "FP32 输出再舍入 BF16 单独列出。原始文件保留径向微扰下输入不同、输出逐位相同的次数。"
         "这些碰撞属于有限精度的 Norm 算子。其结果不证明实数 F 的全局非单射。",
         "分支图 Ψₜ(x)=x+t rᵢ(x) 在 I+t Jᵢ 可逆的局部可求逆。若反演图能沿时间连续延拓，"
         "由 vᵢ(z,t)=rᵢ(Ψₜ⁻¹(z)) 得到局部非自治精确描述。下面在真实 unit 末态尝试反推起点，"
         "求解只接收目标和模型算子，真实 x 仅用于评分。Picard 的失败也可能来自求解器的收敛域。",
         tab(("样本族","迭代数/阻尼","起点反演误差中位数","最大"),inv_rows),
         "[原生源码](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)"
         " · [非自治可逆映射构造](https://pmc.ncbi.nlm.nih.gov/articles/PMC2650001/) · [原始谱与 Norm](results/)")
    local=[]
    for a,row in u["local"].items():
        for name,z in row["endpoint"].items():
            local.append((a,name,med(z),n(z["p95"]),n(z["maximum"]),med(row["direction"][name])))
    save(out,"MODIFIED_EQUATION",
         "# 修正方程与后向误差",
         "对一段真实残差 f=rᵢ，从原生分支输入 x 出发，以 x+α f(x) 作目标。"
         "流的二阶时间展开含 α² Jf·f/2。要让这个项抵消，取"
         " v₂=f−α Jf·f/2。三阶展开再加入 α²[D²f(f,f)/12+Jf(Jf·f)/3]。"
         "这依次令局部一致性误差在理想范围变为 O(α³)、O(α⁴)，量级不提供 α=1 的可靠保证。",
         "每类各一题、每题十个分支，共 30 个位置。Jf·f 和 Jf(Jf·f) 使用真实前向 AD JVP。"
         "D²f(f,f) 由固定方向的两个真 JVP 中心差得到，方向步长 0.0025。"
         "原生 SiLU 的嵌套前向 AD 后端缺少实现，因此三阶项含这一步有限差分。"
         "导数使用数学 SDPA，原生对齐使用原默认 SDPA，方法间的后端误差另有记录。"
         "RK4 与 Euler、RK4 网格自差均在固定分支另测。",
         tab(("α","场","末态误差中位","P95","最坏","更新方向余弦中位"),local),
         "同一题的分支是相关的。此表是局部诊断。48 个分支后候选 logits、方向及墙钟见重建报告。",
         "[Hairer 的 Backward Error Analysis](https://www.unige.ch/~hairer/preprints/hairer-pisa.pdf)"
         " · [局部误差图](results/figures/figure_2_modified_equation.pdf) · [完整汇总](results/SUMMARY.json)")
    order=u["order"]["loops"]
    o=[]
    for k,z in order.items():
        for a,v in z["scale"].items():
            o.append((f"K{int(k)-1}→K{k}",a,med(v["cos"]),med(v["relative"])))
    controls=[(k,med(z["transported_cos"]),med(z["random_cos"]),med(z["other_cos"]),
               *[z["flips"][j] for j in ("actual","norm_matched_random","norm_matched_other_layer")],
               f"{z['predicted_swap_flips']['exact']}/{z['predicted_swap_flips']['affected']}",
               f"{z['negative_pairs']}/{z['pairs']}") for k,z in order.items()]
    order_holdout=u["order_holdout"]["loops"]
    repeat=[(k,med(z["scale"]["0.03125"]["cos"]),med(z["scale"]["1.0"]["cos"]),
             med(z["transported_cos"]),med(z["random_cos"]),z["flips"]["actual"],
             z["flips"]["norm_matched_other_layer"]) for k,z in order_holdout.items()]
    phases=[(f,q["branches_bitwise"],"逐位一致" if q["forecast_bitwise"] else "有差异",
             n(max(q["relative_endpoint_difference"]))) for f,q in u["same_state_phase"].items()]
    save(out,"OPERATOR_INTERACTIONS",
         "# Attention 与 MLP 的非交换计算",
         "原序 AM 表示先 Attention 后 MLP。原序减倒序的小幅展开是"
         " α²(J_M r_A−J_A r_M)+O(α³)。实际替换采用倒序减原序，因而后缀 JVP 接受负号。"
         "真 JVP 与对称导数在相同 FP32 算子和边界检查。每个轮界三类各两题，每层共六题。",
         tab(("循环边界","α","方向余弦中位","幅度及方向合计相对误差中位"),o),
         "每次只在一个物理层交换 Attention/MLP，之后保留真实原生后缀，另跑同剂量随机方向及隔七层的剂量匹配交换方向。"
         "后缀 JVP 在原序输入的输出边界传输小幅 bracket 预测。预测改选只在原生干预和基线无并列、且真的改选时统计。",
         tab(("目标轮","后缀 JVP 余弦","随机余弦","替代层余弦","实际改选","随机改选","替代层改选",
              "JVP 命中实际改选","同点 A/M 余弦为负"),controls),
         f"按六题平均的末端相对 L2 排序，24 层跨轮 Spearman ρ={n(u['order']['layer_rank_correlation'])}。"
         "所测物理层和两次轮界支持局部方向关系。JVP 的单位步幅度误差和候选改变仍需单独看待。"
         "改选次数与替代层接近，无法由此把特定层认定为语义中介。连续流的顺序版本与状态依赖也保留在原始数据。",
         "预先按清单取每类两道独立留出题，直接复用同一算子干预。下表报告其复核。",
         tab(("目标轮","小步 bracket 余弦","单位步 bracket 余弦","后缀 JVP 余弦","随机余弦",
              "实际改选","替代层改选"),repeat),
         "固定相同真实 K2 输入状态，交换捕获自 K2 与 K3 的阶段上下文，检验无 cache 条件下是否有独立于状态的阶段影响。",
         tab(("样本","逐位相同的分支 / 48","Anderson 终点","最大相对终点差"),phases),
         "[交换效应图](results/figures/figure_3_attention_mlp_order.pdf) · [按分支误差图](results/figures/figure_4_branch_error_sources.pdf)")
    baseline=[]
    for dtype,loops in u["natural"].items():
        for boundary,v in loops.items():
            for m,q in v["pooled"].items():
                gate=n(100*q["actual_jump_fraction"])+"%" if "actual_jump_fraction" in q else "—"
                baseline.append((dtype,boundary,m,n(q["full_error"]["mean"]),med(q["full_error"]),n(q["full_error"]["p95"]),
                          n(q["full_error"]["maximum"]),n(q["token_error"]["p95"]),med(q["update_cos"]),
                          med(q["max_logit_error"]),f"{q['agreement']}/{q['n']}",
                          f"{q['native_flip_destination_correct']}/{q['native_flip_count']}",gate))
    development=[(k,n(v["full_error"]["mean"]),med(v["full_error"]),n(v["full_error"]["maximum"]),
                  n(v["actual_jump_fraction"]*100)+"%" if "actual_jump_fraction" in v else "—")
                  for k,v in u["dev"].items()]
    costs=[(k,med(v["wall_s"]),n(v["allocated_peak_bytes"]["maximum"]/2**30),med(v["attention_calls"]),
           med(v["mlp_calls"]),med(v["jvp"]),med(v["inverse_iterations"])) for k,v in u["method_cost"].items()
           if k.split("/")[-1] in u["dev"] or k.endswith("/prefix_ridge")]
    save(out,"CONTINUOUS_RECONSTRUCTION",
         "# 连续重建与自然留出",
         "原生 F、整轮场 F−x、逐分支无修正场、BEA2，以及普通 token 共享 ridge 都纳入对照。"
         "主近似是时间依赖分支场 v(z,t)=rᵢ(x)，其中 x 解 x+t rᵢ(x)=z。"
         "从分支起点做显式中点积分，在 t=.5 以当前中点状态为输入，用 6 次 Picard 或 Anderson 迭代求近似逆。"
         "Anderson 用有效 token 上的残差 Gram 矩阵求权重，系数和为 1。"
         "两者各有 8 次真实分支函数调用，包括初始斜率和最终残差。Anderson 另有矩阵求解开销。",
         "Hybrid 的逆残差相对 L2 门槛预先固定为 2×10⁻⁵。失败处保留原生离散步，"
         "打乱控制在每个样本保持相同离散步数并置换其物理位置。回退执行原生残差加法公式，"
         "Attention 复用候选求解器的 math SDPA，与原生默认后端可能存在数值差。",
         "纯近似路径全程无原生回退。图的分支曲线直接用候选自己的状态推进至下一分支，"
         "原生分支边界只用于评分。终端 RMSNorm 对所有分支构造单独应用。",
         "Ridge 以 12 个开发提示的 K1 token 拟合原生一轮状态增量。权重共享，固定后用于两个轮界。"
         "它在开发集的结果是训练重代入。原生线性插值依赖真实 K2，作为可视化的 oracle 中点对照；其已知终点不参与预测排名。",
         "逐 token ridge 无法把早期 token 编辑传到答案位置。第四轮看到这一结构性缺口后，"
         "补做了一个固定输入为本 token 和因果前缀均值的 ridge。它只用同一开发组拟合，"
         "保留同一惩罚比例，单独作为补充对照。纯 chart 和混合方法的结果保持原样。",
         tab(("开发方法","平均状态误差","中位","最坏","真实回退比例"),development),
         "结构留出与开发按提示组分离。8 道新 3×3、8 道新生成双跳和 8 道 ARC 验证题。"
         "同一留出题在两轮界配对。BF16 不计算 BEA 导数，独立测量其原生和无需求导的逆解。"
         "目标是复现原生下一轮，不将与原生的改选一致解释成准确率提升。",
         tab(("精度","边界","方法","平均 L2","中位 L2","P95","最坏","token P95","更新余弦","候选 logit 最大差中位",
              "候选一致","命中原生自然改选","实回退比例"),baseline),
         "CPU 调度、GPU kernel、序列布局及测量采集均进入同步墙钟。前向数是 batched 算子调用。"
         "JVP 和逆迭代另计，峰值是测试中实际 allocated。原生一步直接计算靶标，是更便宜的端点参照。",
         tab(("精度/方法","墙钟秒中位","峰值 GiB","Attention","MLP","JVP","逆迭代"),costs),
         "[原生边界对照](results/figures/figure_1_embeddings_and_native.pdf) · [新状态预测](results/figures/figure_5_untouched_state_prediction.pdf)")
    causal_rows=[]
    for dtype,loops in u["causal"].items():
        for boundary,by_family in loops.items():
            for family,arms in by_family.items():
                for label,data in arms.items():
                    for m,q in data.items():
                        causal_rows.append((dtype,boundary,family,label,m,q["outputs"],med(q["response_cos"]),
                            med(q["response_error"]),med(q["top3_overlap"]),med(q["logit_response_rmse"]),
                            med(q["content_aligned_logit_rmse"]),med(q["signed_margin_error"]),
                            q["undefined_response"],f"{q['exact_flips']}/{q['native_flips']}"))
    difference=[(k,n(v["delta"]),", ".join(n(j) for j in v["interval"]),v["clusters"])
                for k,v in u["causal_pairwise"].items()]
    save(out,"CAUSAL_VALIDATION",
         "# 反事实与跨轮验证",
         f"反事实与普通留出提示分组分离。{semantic_groups} 个受控语义组各有来源和规则供体，"
         f"另有 {len(arc)} 个 ARC 验证组。ARC 的供体在相同 token 布局中交换正确选项的位置，"
         "答案目标跟随新位置。每个原生 K1 或 K2 起点单独写入供体角色状态。"
         "同长度、同候选顺序、同批布局，保留等范数随机方向和无关 token 对照。"
         "只用未干预与干预的成对差向量评价预测，真实后缀与重建方法从同一编辑起点出发。",
         "报告全宽响应误差、逐 token 响应范数的 top-3 重合、候选 logit 差及目标 margin 变化。"
         "ARC 同时把重排后的字母槽位恢复到原选项内容坐标，再计算 logit 响应误差。"
         "只有原生未并列并真的改选时才计入命中原生改选的分母。"
         "每个提示组含两个轮界；这些不是独立重复。",
         tab(("精度","轮界","组","臂","方法","样本","响应余弦","响应误差","top3 重合","位置 logit RMSE",
              "内容 logit RMSE","目标 margin 误差","零响应","原生改选命中"),causal_rows),
         "配对差定义为比较方法的 logit 响应 RMSE 减去纯 Anderson。按接收提示聚类重采样，两轮与两个语义臂在同一组内配对。",
         tab(("比较","平均差","95% bootstrap 区间","组数"),difference),
         recovery,
         "[反事实图](results/figures/figure_6_counterfactual_responses.pdf) · [跨轮结果](results/figures/figure_7_cross_loop.pdf)")

    first=u["local"]["0.125"]["endpoint"]
    known=u["dev"]["hybrid"]
    pooled=u["natural"]["float32"]
    confirmation_phase=u["causal"]["float32"]["K2_K3"]["pooled"]["semantic"]
    a2=cf["chart_anderson"]
    a3=confirmation_phase["chart_anderson"]
    bf=u["natural"]["bfloat16"]["K1_K2"]["pooled"]["hybrid"]
    early_arc=u["causal"]["float32"]["K1_K2"]["arc"]["semantic"]["chart_anderson"]
    early_structure=[u["causal"]["float32"]["K1_K2"][f]["semantic"]["chart_anderson"]
                     for f in ("tableau","twohop")]
    main=[]
    for key,group in pooled.items():
        for name in ("full_loop","branch_flow","bea2","chart","chart_anderson","hybrid","ridge","prefix_ridge"):
            if name not in group["pooled"]:
                continue
            x=group["pooled"][name]
            c=u["causal"]["float32"][key]["pooled"]["semantic"][name]
            main.append((key,name,n(x["full_error"]["mean"]),med(x["full_error"]),n(x["full_error"]["maximum"]),
                         med(c["logit_response_rmse"]),f"{c['exact_flips']}/{c['native_flips']}"))
    save(out,"CONTINUOUS_EMBEDDING_ZH",
         "# 恢复 Ouro-1.4B 原生计算的连续描述",
         "**存在可验证的局部构造与可复验的算子关系。现有证据无法识别原生 Ouro 的唯一连续计算规律。**",
         "冻结检查点 574fa66，单卡 RTX 4090。FP32 前向关闭 TF32 和 autocast，"
         "权重由同一 BF16 权重数值精确提升，参数哈希核验一致。"
         f"正式先导通过 {len(par['bf16'])} 个 BF16 batch 的 48 边界及完整词表逐位对齐。"
         "新状态比较在 24 道独立留出上运行两个原生轮界。权重无训练，普通 ridge 只作端点对照。",
         "## 1. 原生映射能否连续嵌入？",
         "对局部可逆的残差图，时间依赖场可以精确恢复该图。实验由真残差函数、固定时间与当前状态近似求逆，"
         "并在新提示和编辑起点上测试。全轮同维自治光滑嵌入仍未建立。"
         "形式上的混合模型可沿分段局部流，保留必要的原生离散步及末端 Norm。 [数学可行性](MATHEMATICAL_FEASIBILITY.md)",
         "## 2. 阻碍来自哪里？",
         "分支从小幅扰动到单位步出现高阶项和局部反演误差。终末 RMSNorm 在实数像内可逆，"
         "但径向几乎被压缩，有限精度使部分输出的反解判别式失效。限制维数的谱估计无法认证整个 F 的最小奇异值。",
         "## 3. 哪些修正有效？",
         f"α=.125 时，一阶、BEA2、BEA3 的单分支中位相对误差依次为 "
         f"{med(first['order1'])}、{med(first['order2'])}、{med(first['order3'])}。"
         f"单位步 BEA3 的最坏误差仍为 {n(u['local']['1.0']['endpoint']['order3']['maximum'])}。"
         f"开发 hybrid 的平均末态误差为 {n(known['full_error']['mean'])}，但原生回退达 {100*known['actual_jump_fraction']:.1f}%。"
         f"BF16 留出的 K1→K2 回退比例为 {100*bf['actual_jump_fraction']:.1f}%。"
         "端点很准的混合构造依赖大量原生步骤。 [修正方程](MODIFIED_EQUATION.md) · [重建](CONTINUOUS_RECONSTRUCTION.md)",
         "## 4. 顺序与非交换性扮演什么角色？",
         f"两个轮界上经真实后缀传输的 bracket 方向余弦中位数为 "
         f"{med(order['2']['transported_cos'])} 和 {med(order['3']['transported_cos'])}，"
         "明显高于剂量匹配控制。实际改选率与等剂量替代层接近。数据支持本层交换对应的因果状态方向。 [算子实验](OPERATOR_INTERACTIONS.md)",
         "## 5. 是否跨输入、跨轮稳定？",
         f"开发阶段算子交换的层排序在两个轮界的 Spearman 约为 {n(u['order']['layer_rank_correlation'])}。"
         f"在六道独立复核题上，后缀传输方向余弦为 {med(order_holdout['2']['transported_cos'])} 和"
         f" {med(order_holdout['3']['transported_cos'])}。同一 K2 状态在两套无 cache 的阶段上下文上"
         "产生相同的分支输出与 Anderson 终点。"
         "独立自然及干预预测如下。均值、中位和最坏值均列出，以免由少量题掩盖累积误差。",
         tab(("轮界","方法","自然平均 L2","自然中位","自然最坏","反事实 logit 响应 RMSE 中位","原生改选命中"),main),
         "## 6. 因果预测能否超过旧连续化和端点拟合器？",
         f"在 K1→K2 的真 FP32 新状态上，纯 Anderson 的平均 L2 为 {n(rec['chart_anderson']['full_error']['mean'])}，"
         f"ridge 为 {n(rec['ridge']['full_error']['mean'])}。同一轮界，纯 Anderson 与 branch flow、ridge 的"
         f"语义干预 logit 响应 RMSE 中位数分别为 {med(cf['chart_anderson']['logit_response_rmse'])}、"
         f"{med(cf['branch_flow']['logit_response_rmse'])}、{med(cf['ridge']['logit_response_rmse'])}。"
         + (f"补充的跨 token 前缀 ridge 在相同轮界为 {med(cf['prefix_ridge']['logit_response_rmse'])}。"
            if "prefix_ridge" in cf else "") +
         "完整报告含跨轮、真实 ARC、聚类配对和未发生答案变化的样本。 [因果验证](CAUSAL_VALIDATION.md)",
         f"纯 Anderson 对严格可判的原生改选分别复现 {a2['exact_flips']}/{a2['native_flips']} 和"
         f" {a3['exact_flips']}/{a3['native_flips']}。这里匹配原生的干预响应，语义供体的正确答案仅用于评分。",
         f"K1→K2 的 ARC 选项重排命中 {early_arc['exact_flips']}/{early_arc['native_flips']}，"
         f"两个受控结构族合计命中 {sum(x['exact_flips'] for x in early_structure)}/"
         f"{sum(x['native_flips'] for x in early_structure)}。早轮合并结果由 ARC 主导。",
         "## 7. 哪些现象具有机制价值？",
         "小幅交换的符号和后缀传播方向能被同一组真实导数预测。"
         "这提供了比曲线平滑性更具体的原生机制证据。局部修正无法自动解决 48 个单位分支的误差传播，"
         "而高回退混合只能展示何处要保留原生离散更新。端点和编辑预测不能鉴别真实中间时间的轨道。",
         "四轮依次审查局部 BEA/可逆性、全层交换、开发求逆对照、独立及反事实验证。"
         "每轮由 Astra 复核已完成的实测后决定继续或限定主张。GPU 时间、显存、调用数和失败先导日志均随结果保留。",
         "[实验入口](README.md) · [全部汇总](results/SUMMARY.json)")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("results",type=Path)
    p.add_argument("dest",type=Path)
    args=p.parse_args()
    args.dest.mkdir(exist_ok=True,parents=True)
    render(args.results,args.dest)

"""Write the Chinese evidence report from verified numerical summaries."""
import argparse
from pathlib import Path

from native import load

F=(("openbookqa","OpenBookQA"),("arc_challenge","ARC-Challenge"))


def num(q,p=3):
    if q is None:
        return "—"
    return f"{q['mean']:+.{p}f} [{q['ci'][0]:+.{p}f}, {q['ci'][1]:+.{p}f}]"


def table(cols,rows):
    return "\n".join(["","| "+" | ".join(cols)+" |", "| "+" | ".join(["---"]*len(cols))+" |",
        *["| "+" | ".join(map(str,r))+" |" for r in rows],""])


def render(root):
    u=load(root/"SUMMARY.json")
    x=load(root/"EXTRA_SUMMARY.json")
    native=u["full_candidate"]
    body=x["body"]
    text=["# Ouro-1.4B 的早期前文状态与第二轮答案纠正",
          "", "## 结论", ""]
    fw=[native[f]["mean"]["ALL"]["contrasts"]["drop"]["own_base"] for f,_ in F]
    b=x["body_decision"]["pass_all"]
    text.extend([
      ("K2 第 1 层 Up 通过预登记的双向中介门槛。" if u["primary_decision"]["pass_all"] else
       "K2 第 1 层 Up 未通过预登记的双向中介门槛。")+
      f"只恢复清洁 Up 对 OpenBookQA 和 ARC 的全样本正确答案分差作用依次为 {num(fw[0])}、{num(fw[1])} nats/token。",
      ("前文状态在另一批题上通过双向行为判据。所有 K1 候选评分位置保持逐位不变。" if b else
       "独立前文交换保持了所有 K1 候选评分位置。下面报告它对 K2 连续分差、同题恢复、反向损伤和对照的检验。"),
      "", "## 1. K1 早期写入实际支持哪部分 K2 纠正", "",
      "固定冻结 Ouro-1.4B Base 574fa66。先冻结两个自然题队列各 384 道，再进行原生 eager FP32。"
      "OpenBookQA 依据已知历史 ID 和题干作排除。ARC 曾被项目进行历史整划分评估，此处保证本轮干预队列与前两轮无重复。"
      "上游编辑只删除 K1 第 1 层 Attention 写入的非首位提示 token。每题固定原生 K2 得分最高的错误对手，对 K1、K2 及全部反事实计算同一分差。"
      "主读出是完整候选平均词元对数似然，总分另保留。"])
    rows=[]
    for f,l in F:
        v=native[f]["mean"]
        p=v["WC"]; a=v["ALL"]
        y=a["advance"]["drop"]; z=p["advance"]["drop"]
        rows.append((l,a["n"],*a["native"]["clean"]["correct"],p["n"],v["CW"]["n"],p["discrete"]["native_correct_to_drop_wrong"],
                     num(y["K1_loss"]),num(y["K2_loss"]),num(y["increment_loss"]),num(z["increment_loss"])))
    text.append(table(("数据","n","K1 对","K2 对","错→对","对→错","删写破坏的纠正","K1 分差损失","K2 分差损失","全样本差分","纠正组差分"),rows))
    text.extend(["同支持半删、等范数坐标滚动、四种原生转换和逐题证据均保留。原生纠正组差分是依据自然输出分层后的条件效应。全样本差分和自由生成单独判断。","",
        "## 2. 同题的正向复位、反向替换与乘积剂量", "",
        "把清洁 Up 放入删写背景，或将删写 Up 放入清洁背景。保留对应真实 Gate、状态和剩余计算。两种配平分别在原始 Up 和真实 down 输入空间实现，"
        "后者匹配受体 Gate 加权后的实际剂量。每题的自体回放验证该接口逐位保持原生输出。"])
    rows=[]
    for f,l in F:
        v=native[f]["mean"]
        for pop in ("ALL","WC"):
            w=v[pop]
            for side,label in (("drop","正向"),("clean","反向")):
                z=w["contrasts"][side]
                rows.append((l,pop,label,w["n"],*(num(z[k]) for k in ("own_base","up_norm_control","product_norm_control","gate")),
                      w["arms"][side]["up"]["rescue"],w["arms"][side]["up"]["harm"]))
    text.append(table(("数据","分层","方向","n","相对原接收状态","相对 Up 剂量控制","相对乘积剂量控制","相对 Gate","救回","损伤"),rows))
    rows=[]
    for f,l in F:
        s=native[f]["mean"]["WC"]["discrete"]
        rows.append((l,s["native_correct_to_drop_wrong"],*(s["restored"][key] for key in (
            "up","up_norm_control","product_norm_control","gate","gate_and_up")),
            s["reversed"]["up"],len(s["both_direction_ids"]),len(s["both_directions_vs_all_controls_ids"])))
    text.append(table(("数据","删写致错的原生纠正","Up 救回","原始剂量","乘积剂量","Gate","联合","反向致错","同题双向","双向且两个对照排除"),rows))
    text.extend(["## 3. Up 由携带状态还是 K2 重新执行注意力产生", "",
       "在 K2 第 1 层，令 U(h,a) 为对入轮状态 h 和该层真实注意力写入 a 相加后执行 RMSNorm 与上投影。"
       "用两种真实混合量替换删写受体的 Up，仍保持其 Gate。本地 U(h,a) 重算与捕获的 Up 一致，"
       "下游原生重复执行。全提示与两种位置子集以同样方法复查。"])
    rows=[]
    for f,l in F:
        a=native[f]["mean"]["ALL"]["arms"]["drop"]
        n=native[f]
        q=x["support"][f]["ALL"]["background"]["drop"]
        rows.append((l,num(a["inherited"]["effect"]),num(a["fresh_attention"]["effect"]),
                     num(a["up"]["effect"]),num(q["earlier"]["versus_matched"]),num(q["readout"]["versus_matched"]),
                     num(n["geometry"]["effective_product_cosine"])))
    text.append(table(("数据","只恢复携带状态构造的 Up","只恢复 K2 注意力构造的 Up","完整 Up","前文 Up 相对配平","末位 Up 相对配平","两背景有效乘积方向余弦"),rows))
    text.extend(["## 4. 只交换前文时，K2 是否重新获得正确证据", "",
       "这是冻结的 384 道独立追加题，剔除 Up 队列和两轮历史自然题。"
       "对每个背景，逐位保持 K1 最后提示 token 和续写位置，验证全部候选 K1 log likelihood 完全一致。"
       "任何 K2 效应均通过实际共享网络重新传播。用同范数坐标滚动、只对齐范数、只对齐方向和只交换末提示 token 作为对照。"
       "以答对计数与置信区间同时判断。完整原生输入的交互关系不能从这些替换唯一归因到某个分支。"])
    rows=[]
    for f,l in F:
        allr=body[f]["ALL"]; w=body[f]["WC"]
        rows.append((l,allr["n"],*allr["native"]["clean"],w["n"],w["broken"],
            *(w["rescue"][k] for k in ("body","body_rolled","body_scaled","body_direction","boundary","whole_prompt")),
            w["reversal"]["body"],len(w["dual_ids"]),len(w["strict_dual_ids"])))
    text.append(table(("数据","n","K1 对","K2 对","原生纠正","删写致错","前文救回","同范数滚动","只范数","只方向","末 token","全提示","反向致错","同题双向","排除两个对照"),rows))
    rows=[]
    for f,l in F:
        for pop in ("ALL","WC"):
            q=body[f][pop]
            for b,label in (("drop","正向"),("clean","反向")):
                x0=q["tables"][b]
                rows.append((l,pop,label,q["n"],num(x0["body"]["effect"]),num(x0["body_vs_controls"]["body_rolled"]),
                      num(x0["body_vs_controls"]["body_scaled"]),num(x0["body_direction"]["effect"]),
                      num(x0["boundary"]["effect"]),x0["body"]["ood"]))
    text.append(table(("数据","转换","方向","n","前文影响","相对同范数滚动","相对只范数","仅方向","末位置","范数 >2×"),rows))
    text.extend(["## 5. 无候选进入模型的自由生成", "",
       "两个 Up 队列各前 128 题，使用固定简短答案提示。完整词表贪心生成至多 12 token，只评分首非空行完全匹配。"
       "六个生成条件均执行，解析失败留在全分母。此协议与完整候选读出不同。如果原生自由纠正很少，不能把候选修复描述为自由生成修复。"])
    rows=[]
    for f,l in F:
        p=u["free_decode"][f]
        rows.append((l,p["n"],*[(str(p["accuracy"][k])+"/"+str(p["unresolved"][k])) for k in (
            "K1","K2","drop","restore","reverse","product_control")],
            p["native_strata"]["WC"],p["native_wc_broken"],p["up_rescue_in_broken_wc"],p["strict_both"]))
    text.append(table(("数据","n","K1 对/未匹配","K2 对/未匹配","删写","Up 复位","Up 反向","乘积对照","自然纠正","删写破坏","Up 救回","严格双向"),rows))
    text.extend(["## 证据范围","完整候选似然说明实际模型的选择变化。句子同义自由回答和真正推理过程仍需不同的开放验证。"
                 "两个上下文支持位置可以改变原生共享函数的输出，效应并不等同于模型参数、层定义或循环编号变化。"
                 "预设配对自举区间按题计算。全部原始样本、协议哈希、五张图和数值校验在结果目录。","",
                 "[先前路由与内容实验](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/b1be92fc00c231533ffa69c9c69d81d5d43f3d01/experiments/ouro_route_content_20261121)"])
    return "\n".join(text)+"\n"


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("r",type=Path)
    p.add_argument("doc",type=Path)
    a=p.parse_args()
    a.doc.write_text(render(a.r),encoding="utf-8")


"""Append the independently verified direction experiment and render its figure."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from native import load
from report import F,num,table


def appendix(s):
    header=["## 6. 逐 token 同范数、同角距离的方向检验","",
        "完整上下文交换改变了部分状态范数。此复核在与此前所有干预队列不重叠的新题上，保持每个受体 token 的实际范数，"
        "用真实供体方向、半角位移和两个正交随机方向对照。随机条件的角距与落点范数逐 token 匹配，且所有候选 K1 输出逐位相同。",
        "两集均通过预设方向判据。" if s["cross_task_gate"] else
        "跨两数据集的预设方向判据未全部通过。每个数据集、方向和对照分开报告。"]
    rows=[]
    for f,name in F:
        a=s["families"][f]
        for pop in ("ALL","WC"):
            p=a["panels"][pop]
            for w,desc in (("drop","正向"),("clean","反向")):
                e=p["worlds"][w]
                rows.append((name,pop,desc,p["n"],num(e["arms"]["direction"]["effect"]),
                             num(e["arms"]["half"]["effect"]),num(e["versus"]["angle_control_0"]),
                             num(e["versus"]["angle_control_1"]),e["arms"]["direction"]["ood2x"]))
    header.append(table(("数据","转换","方向","n","自然转动影响","半角","相对随机 0","相对随机 1","大范数异常"),rows))
    rows=[]
    for f,name in F:
        w=s["families"][f]["panels"]["WC"]
        rows.append((name,w["n"],w["broken"],w["recovered"]["direction"],
             w["recovered"]["angle_control_0"],w["recovered"]["angle_control_1"],
             len(w["both_ids"]),len(w["strict_ids"])))
    header.append(table(("数据","原生纠正","删写破坏","方向救回","随机 0 救回","随机 1 救回","同题双向","超两个对照"),rows))
    header.append("这是边界保持的真实模型干预，支持范围限定在实际测得的分差与完整候选输出。自由生成在前述独立协议报告。")
    return "\n\n".join(header)+"\n\n"


def draw(data,dest):
    fig,ax=plt.subplots(1,2,figsize=(9,3.1),sharey=True,constrained_layout=True)
    for j,world in enumerate(("drop","clean")):
        p=ax[j]
        for f,(name,label) in enumerate(F):
            z=data["families"][name]["panels"]["ALL"]["worlds"][world]["arms"]
            for i,k in enumerate(("direction","half","angle_control_0","angle_control_1")):
                u=z[k]["effect"]; m=u["mean"]; a,b=u["ci"]
                p.errorbar(i+(f-.5)*.18,m,yerr=[[max(0,m-a)],[max(0,b-m)]],fmt="o",
                          color=("#216678","#9c573e")[f],capsize=3,ms=4,label=label if i==0 else None)
        p.axhline(0,color="#888",lw=.6)
        p.set(title="Restore" if j==0 else "Reverse",xticks=range(4),
              xticklabels=["Direction","Half-angle","Matched 0","Matched 1"])
    ax[0].set_ylabel("Oriented answer-margin effect (nats/token)")
    ax[1].legend(frameon=False,fontsize=8)
    for ext in ("png","pdf"):
        fig.savefig(dest/f"figure_6_matched_orientation.{ext}",bbox_inches="tight",dpi=240)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("results",type=Path)
    p.add_argument("report",type=Path)
    a=p.parse_args()
    q=load(a.results/"ANGLE_SUMMARY.json")
    draw(q,a.results/"figures")
    doc=a.report.read_text(encoding="utf-8")
    assert "## 6. 逐 token" not in doc
    a.report.write_text(doc.replace("## 证据范围",appendix(q)+"## 证据范围").replace("五张图","六张图"),encoding="utf-8")


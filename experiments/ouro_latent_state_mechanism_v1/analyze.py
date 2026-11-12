"""Full-cohort readout and exploratory latent transition audit.

Gold is opened HERE, after inference. Descriptive output is NOT a learned gate.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from core import read_jsonl, write_json


def records(path):
    items = list(read_jsonl(path))
    if len(items)!=len({x["id"] for x in items}): raise ValueError("Duplicate prediction ID")
    return items


def score_at(rec, kind):
    if kind=="native":
        return np.asarray(rec["rotations"][0]["scores_in_original_content_order"],dtype=float)
    if rec["cyclic_average_score_by_depth"] is None:
        return None
    return np.asarray(rec["cyclic_average_score_by_depth"],dtype=float)


def transition_matrix(preds, gold):
    result=[]
    for k in range(3):
        old = preds[k]==gold
        new = preds[k+1]==gold
        result.append(dict(from_loop=k+1,to_loop=k+2,
                           wrong_to_correct=int(not old and new),
                           correct_to_wrong=int(old and not new),
                           both_correct=int(old and new),
                           both_wrong=int(not old and not new),
                           answer_switched=int(preds[k]!=preds[k+1])))
    return result


def write_csv(path, rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--labels",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--expect-full-arc-test",action="store_true")
    p.add_argument("--exclude-fewshot",action="store_true",help="Exclude 25 demonstration IDs in train data")
    p.add_argument("--input-manifest",type=Path,default=None)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    gold={x["id"]:int(x["gold"]) for x in read_jsonl(a.labels)}
    ignore=set()
    if a.exclude_fewshot:
        if a.input_manifest is None:raise ValueError("--exclude-fewshot requires --input-manifest")
        ignore={x["id"] for x in read_jsonl(a.input_manifest) if x.get("in_fewshot_demonstrations",False)}
    rs=[r for r in records(a.predictions) if r["id"] not in ignore]
    if not rs:raise RuntimeError("No records")
    missing=set(x["id"] for x in rs)-set(gold)
    if missing:raise RuntimeError(f"Missing gold labels: {list(missing)[:3]}")
    report={"n":len(rs),"source":str(a.predictions),"native":{},"cyclic":{},
            "note":"Historical ARC test is descriptive/exposed. Select mechanisms on training data, freeze before held-out checks."}
    question_rows=[]
    statistic_rows=[]
    for kind in ("native","cyclic"):
        valid=[r for r in rs if score_at(r,kind) is not None]
        if not valid:continue
        counts=[0]*4
        transition=defaultdict(int)
        for rec in valid:
            y=gold[rec["id"]]
            scores=score_at(rec,kind)
            pred=np.argmax(scores,axis=1).tolist()
            if not 0<=y<scores.shape[1]:raise ValueError("Gold outside candidate list")
            for k in range(4):counts[k]+=int(pred[k]==y)
            for t in transition_matrix(pred,y):
                step=f"{t['from_loop']}->{t['to_loop']}"
                for key in ("wrong_to_correct","correct_to_wrong","both_correct","both_wrong","answer_switched"):
                    transition[f"{step}_{key}"]+=t[key]
            row={"id":rec["id"],"view":kind,"n":rec["n"],"gold":y,
                 "prediction_k1":pred[0],"prediction_k2":pred[1],"prediction_k3":pred[2],"prediction_k4":pred[3]}
            for k in range(4):
                rival=np.max(np.delete(scores[k],y))
                row[f"gold_margin_k{k+1}"]=float(scores[k,y]-rival)
                row[f"correct_k{k+1}"]=int(pred[k]==y)
            question_rows.append(row)
            if kind=="native":
                feats=[]
                for rotation in rec["rotations"][:1]:
                    for f in rotation["forwards"]:
                        st=f.get("stats")
                        if st is not None:feats.append(np.asarray(st["relative_l2_by_transition_and_layer"],dtype=float))
                if feats:
                    avg=np.mean(feats,axis=0)
                    label="wrong_to_correct" if pred[0]!=y and pred[1]==y else ("correct_to_wrong" if pred[0]==y and pred[1]!=y else ("both_correct" if pred[0]==y else "both_wrong"))
                    for t in range(3):
                        for layer in range(24):
                            statistic_rows.append(dict(id=rec["id"],group=label,transition=f"{t+1}->{t+2}",layer=layer,
                                                       relative_l2=float(avg[t,layer])))
        report[kind]={"n":len(valid),"correct_by_loop":counts,
                      "accuracy_by_loop":[v/len(valid) for v in counts],"transitions":dict(transition)}
    summary=[]
    for key in sorted({(r["transition"],r["layer"],r["group"]) for r in statistic_rows}):
        x=[r["relative_l2"] for r in statistic_rows if (r["transition"],r["layer"],r["group"])==key]
        summary.append(dict(transition=key[0],layer=key[1],group=key[2],n=len(x),mean_relative_l2=float(np.mean(x)),median_relative_l2=float(np.median(x))))
    write_csv(a.out/"question_transitions.csv",question_rows)
    write_csv(a.out/"latent_per_question.csv",statistic_rows)
    write_csv(a.out/"latent_by_layer_group.csv",summary)
    if a.expect_full_arc_test:
        if len(rs)!=1172:raise RuntimeError(f"Expected full ARC test 1172, got {len(rs)}")
        baseline=report["native"]["correct_by_loop"]
        report["historical_K1_K4_parity"]={"expected":[634,1021],"actual":[baseline[0],baseline[3]],
                                           "passed":baseline[0]==634 and baseline[3]==1021}
        if not report["historical_K1_K4_parity"]["passed"]:
            report["warning"]="Historical endpoints not reproduced. Investigate prompt/model/tokenizer/label ties before mechanistic claims."
        if report.get("cyclic"):
            c=report["cyclic"]["correct_by_loop"]
            report["historical_cyclic_K1_K4_parity"]={"expected":[657,1036],"actual":[c[0],c[3]],
                                                      "passed":c[0]==657 and c[3]==1036}
    report["latent_state_descriptives"]={"rows":len(statistic_rows),"grouped_rows":len(summary),
                                          "interpretation":"Exploratory correlations only, not a causal explanation."}
    write_json(a.out/"summary.json",report)
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=="__main__":main()

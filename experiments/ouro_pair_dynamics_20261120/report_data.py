"""Derive sample-paired assessments for the Chinese report without modifying sealed files."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze import END, collect_manifest, contrast, iq, load, measure, paired, phase_rows, sha, stat, winner
from artifact_json import write


def joint(root, lookup):
    f = root/"CONTROLS.json"
    ct = load(f)
    assert ct["prediction_sha256"] == sha(root/"PREDICTIONS_confirmation.json")
    controls = {r["group"]:r["conditions"] for r in ct["rows"]}
    out = {}
    for dtype in ("FP32", "BF16"):
        src = load(root/("NATIVE_confirmation_BF16.json" if dtype=="BF16" else "NATIVE_confirmation.json"))
        assert src["prediction_sha256"] == ct["prediction_sha256"]
        rows = src["rows"]
        out[dtype] = {}
        for family in ("all","tableau","twohop"):
            selected = [r for r in rows if family=="all" or family==r["family"]]
            panel = dict(n=len(selected), zero_k1=sum(r["initial"]["mixed_l2"]==0 for r in selected), loops={})
            for stage in END:
                data = defaultdict(list)
                interactions = []
                control_iq, control_margin = defaultdict(list), defaultdict(list)
                joint_margin = []
                lanes = np.zeros(4,dtype=int)
                norm_grid = []
                for r in selected:
                    true = r["methods"]["native"]
                    targets = lookup[r["group"]]["targets"]
                    scores = true[stage]["scores"]
                    interactions.append(iq(scores,targets[-1]))
                    joint_margin.append(contrast(scores,targets[-1])[3])
                    selected_win = [winner(z)==t for z,t in zip(scores,targets,strict=True)]
                    lanes += np.array(selected_win,dtype=int)
                    data["joint"].append(selected_win[-1])
                    if dtype=="FP32":
                        for key, entries in controls[r["group"]].items():
                            predicted = entries[stage]["scores"]
                            data[key].append(winner(predicted[3]) == targets[-1])
                            control_iq[key].append(iq(predicted,targets[-1]))
                            control_margin[key].append(contrast(predicted,targets[-1])[3])
                    token = lookup[r["group"]]["worlds"][0]["read"]
                    pre = true[stage.replace("/norm","/pre_norm")]
                    norm_grid.append([pre["token_mixed_norm"][token],true[stage]["token_mixed_norm"][token]])
                panel["loops"][stage] = dict(hits={k:sum(v) for k,v in data.items()}, lane_hits=lanes.tolist(),
                  iq=paired(interactions), zero_iq=sum(abs(v)<=1e-6 for v in interactions),
                  joint_iq_vs={k:paired(np.asarray(interactions)-v) for k,v in control_iq.items()},
                  joint_margin_vs={k:paired(np.asarray(joint_margin)-v) for k,v in control_margin.items()},
                  pre_post_norm_mean=np.mean(norm_grid,axis=0).tolist(),
                  joint_vs={key:paired(np.asarray(data["joint"],dtype=int)-np.asarray(values,dtype=int))
                            for key,values in data.items() if key!="joint"})
            out[dtype][family]=panel
    return out


def free(root, lookup, summary):
    phase=summary["decision"]["phase"]
    rows,pred,native=phase_rows(root,phase,lookup)
    out=dict(phase=phase, failed=pred["failures"], native_failures=native["failures"], n=len(rows), stats={})
    for stage in END:
        choice=summary["decision"]["choice"]
        v=[r["endpoints"][stage] for r in rows]
        wins={}
        for arm in ("base","taylor","prefix_anderson","zero"):
            wins[arm]=paired([abs(r["pred"][arm]-r["iq"])-abs(r["pred"][choice]-r["iq"]) for r in v
                               if arm in r["pred"] and choice in r["pred"]])
        target_margin={}
        raw={r["group"]:r for r in native["rows"]}
        for arm in (choice,"base","taylor","prefix_anderson"):
            effects=[]
            hits=0
            for item in pred["rows"]:
                if stage not in item["methods"][arm]:
                    continue
                y=lookup[item["group"]]["targets"][-1]
                p=item["methods"][arm][stage]["scores"]
                z=raw[item["group"]]["methods"]["native"][stage]["scores"]
                effects.append(abs(contrast(p,y)[3]-contrast(z,y)[3]))
                hits+=winner(p[3])==y
            target_margin[arm]=dict(mae=stat(effects),wins=hits)
        out["stats"][stage]=dict(against=wins,target_margin=target_margin)
    return out


def bf16(root, lookup):
    p=root/"BF16_PREDICTIONS.json"
    z=load(root/"BF16_NATIVE.json")
    d=sha(p)
    assert d==z["prediction_sha256"]==load(root/"BF16_PREDICTION_SHA.json")["sha256"]
    n={r["group"]:r for r in z["rows"]}
    pred=load(p)
    ans={k:defaultdict(list) for k in END}
    for r in pred["rows"]:
        y=lookup[r["group"]]["targets"][-1]
        for k in END:
            actual=iq(n[r["group"]]["methods"]["native"][k]["scores"],y)
            for method,values in r["methods"].items():
                if k in values:
                    ans[k][method].append((actual,iq(values[k]["scores"],y)))
    return dict(n=len(n),fails=pred["failures"]+z["failures"],
        loops={k:{m:measure(*zip(*q,strict=True)) for m,q in z.items()} for k,z in ans.items()})


def order(root, lookup, summary):
    p=load(root/"LOCAL_PREDICTIONS_validation.json")
    z=load(root/"LOCAL_NATIVE_validation.json")
    by={(r["group"],r["loop"]):r["methods"]["native"] for r in z["rows"]}
    result={}
    for k in (2,4):
        data=defaultdict(list)
        truth=[]
        for r in p["rows"]:
            if r["loop"]!=k:
                continue
            y=lookup[r["group"]]["targets"][-1]
            def m(q):
                return contrast(q["ba"]["scores"],y)[3]-contrast(q["am"]["scores"],y)[3]
            truth.append(m(by[r["group"],k]))
            for method, arms in r["methods"].items():
                data[method].append(m(arms))
        rows=[r for r in summary["order"]["validation"]["rows"] if r["loop"]==k]
        comps={}
        for label in ("BA_comp","BA_random","BA_other"):
            comps[label]={}
            for a in (.25,.5,1.):
                comps[label][str(a)]=paired([abs(q["truth"]["ba"])-abs(q["truth"]["ba"]+q["truth"][f"{label}/{a}"]) for q in rows])
        result[str(k)]=dict(margin={m:measure(truth,v) for m,v in data.items()},dose_gain=comps,
                            absolute_swap_iq=stat([abs(q["truth"]["ba"]) for q in rows]),
                            absolute_swap_margin=stat([abs(v) for v in truth]))
    return result


def diagnostics(root,lookup):
    short=load(root/"SHORT.json")["rows"]
    seg={}
    for depth in (2,4):
        seg[str(depth)]={method:{k:stat([x for r in short if r["span"]==depth and r["name"]==method for x in r[k]])
                         for k in ("pre_norm_error","norm_probe_error","direction","update_l2")}
                         for method in ("base","self","order","full","reverse","taylor","prefix_anderson")}
    rows=load(root/"CLOCKS_AND_BETA.json")["rows"]
    clock={}
    for m in (2,4,8):
        clock[str(m)]={}
        for c in ("slow_fast","fast_slow"):
            label=f"{m}/{c}"
            a, z = [], []
            for r in rows:
                for i,g in enumerate(r["groups"]):
                    target=lookup[g]["targets"][-1]
                    a.append(abs(iq(r["methods"][label]["records"][i]["scores"],target)-
                                 iq(r["methods"][f"{m}/linear"]["records"][i]["scores"],target)))
                z.extend(r["methods"][label]["distance_same_grid"])
            clock[str(m)][c]=dict(signed_iq_abs=stat(a),state_error=stat(z))
    beta={}
    for r in rows:
        key=f'{r["beta_s"]},{r["beta_r"]}'
        k=beta.setdefault(key,{"iq":[],"model_error":[]})
        for i,g in enumerate(r["groups"]):
            y=lookup[g]["targets"][-1]
            v=iq(r["native"][i]["scores"],y)
            k["iq"].append(abs(v))
            k["model_error"].append(abs(v-iq(r["methods"]["8/linear"]["records"][i]["scores"],y)))
    return dict(short=seg,clocks=clock,beta={k:{x:stat(y) for x,y in v.items()} for k,v in beta.items()})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    root=p.parse_args().out
    _, lookup=collect_manifest(root)
    s=load(root/"SUMMARY.json")
    write(root/"REPORT_STATS.json",dict(joint=joint(root,lookup),free=free(root,lookup,s),
              bf16=bf16(root,lookup),order=order(root,lookup,s),diagnostics=diagnostics(root,lookup)))


if __name__=="__main__":
    main()


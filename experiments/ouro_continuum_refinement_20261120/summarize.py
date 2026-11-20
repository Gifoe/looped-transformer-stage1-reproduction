"""Validate and summarize the GPU observations without changing them."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


FAMILIES=("tableau","twohop","arc")
STAGES=("primary","holdout","precision_bf16","precision_fp32","reference")


def load(root, name):
    return json.loads((root/(name+".json")).read_text(encoding="utf-8"))


def write(root,name,content):
    (root/(name+".json")).write_text(json.dumps(content,allow_nan=False,indent=2)+"\n",encoding="utf-8")


def stats(v):
    x=np.asarray(v)
    return {"mean":float(x.mean()),"min":float(x.min()),"max":float(x.max()),"n":int(x.size)}


def pair_dict(d,method):
    for v in d[method]["comparisons"]:
        a,b=v["left"],v["right"]
        yield f"{a[0]}{a[1]}:{b[0]}{b[1]}",v


def select(pairs, key):
    return dict(pairs)[key]


def comparison_data(d,method):
    rows={}
    for k,v in pair_dict(d,method):
        endpoint=v["endpoint"]
        subset=[float(r) for vec in endpoint["state"]["tokens"] for r in vec]
        interval=v["path"]["t"]
        full=v["path"]["state"]
        normed_path=[q["full"] for q in full]
        by_time=np.array(normed_path)
        rows[k]={"t":interval,"path_mean":by_time.mean(1).tolist(),"path_max":by_time.max(1).tolist(),
                 "before_final_norm":stats(normed_path[-1]),
                 "end":stats(endpoint["state"]["full"]),
                 "worst_time_mean":float(by_time.mean(1).max()),
                 "logit_max":stats(endpoint["max_candidate_logit_delta"]),
                 "argmax_changes":sum(endpoint["changed_argmax"]),
                 "direction_cos":stats(endpoint["direction_cos"]),
                 "endpoint_token_p95":float(np.quantile(subset,.95)),"endpoint_token_max":max(subset)}
    for k,r in rows.items():
        a,b=k.split(":")
        method_name="euler" if a.startswith("euler") else "rk4"
        ma,mb=int(a[len(method_name):]),int(b[len(method_name):]) if b.startswith(method_name) else -1
        key=f"{method_name}{mb}:{method_name}{2*mb}"
        if mb==2*ma and key in rows:
            right=rows[key]
            if r["end"]["mean"]>0 and right["end"]["mean"]>0:
                r["endpoint_p"]=float(np.log2(r["end"]["mean"]/right["end"]["mean"]))
            if r["before_final_norm"]["mean"]>0 and right["before_final_norm"]["mean"]>0:
                r["pre_p"]=float(np.log2(r["before_final_norm"]["mean"]/right["before_final_norm"]["mean"]))
            finer_by_time=dict(zip(right["t"],right["path_mean"],strict=True))
            visible=[(t,float(np.log2(p/finer_by_time[t]))) for t,p in zip(r["t"],r["path_mean"],strict=True)
                     if t in finer_by_time and p>0 and finer_by_time[t]>0]
            r["p_at_common_times"]={"t":[k for k,_ in visible],"p":[p for _,p in visible]}
            if visible:
                r["median_time_p"]=float(np.median([p for _,p in visible]))
    return rows


def geometry_stats(g):
    speed=np.asarray(g["speed"])
    mid=(np.array(g["t"][1:])+np.array(g["t"][:-1]))/2
    span=[]
    simultaneous=[]
    width=len(g["token_speed"][0])
    for i in range(width):
        dose=np.asarray([row[i] for row in g["token_speed"]])
        maxima=mid[dose.argmax(axis=0)]
        span.append(float(np.quantile(maxima,.9)-np.quantile(maxima,.1)))
        simultaneous.append(float(np.count_nonzero(dose.argmax(axis=0)==speed[:,i].argmax())/len(maxima)))
    return {"peak_speed_time_by_item":mid[speed.argmax(0)].tolist(),
            "peak_speed":stats(speed.max(0)),
            "largest_turn_degrees":float(np.degrees(np.arccos(np.clip(g["turn_cos"],-1,1))).max()),
            "maximum_curvature":float(np.max(g["curvature"])),
            "token_peak_p90_minus_p10_by_item":span,
            "fraction_tokens_with_global_peak_time":simultaneous,
            "net_over_length":stats(g["net_over_path"])}


def simple_endpoints(d):
    out={}
    for key,v in d["endpoints"].items():
        z=v["state"] if "state" in v else v
        out[key]={"state":stats(z["full"])}
        if "direction_cos" in v:
            out[key].update(direction_cos=stats(v["direction_cos"]),
                changed_argmax=sum(v["changed_argmax"]),max_logits=stats(v["max_candidate_logit_delta"]))
    return out


def verify_calls(group):
    for row in group["timings"]:
        a=row["run"].split("/")
        if len(a)!=3 or a[0] not in ("A","B") or a[1] not in ("euler","rk4"):
            continue
        n=int(a[2])*(4 if a[1]=="rk4" else 1)
        assert row["loop_field_calls"]==(n if a[0]=="A" else 0)
        assert row["attention_calls"]==row["mlp_calls"]==24*n
        assert row["wall_s"]>0 and row["peak_allocated_bytes"]>0


def summarize(root):
    raw={stage:{family:load(root,f"{stage}_{family}") for family in FAMILIES} for stage in STAGES}
    diag={f:load(root,f"dynamics_{f}") for f in FAMILIES}
    stable_fp={f:load(root,f"stability_fp32_{f}") for f in FAMILIES}
    orders={f:load(root,f"order_{f}") for f in FAMILIES}
    assert load(root,"NATIVE_PARITY")["status"]==load(root,"WEIGHT_PRECISION_AUDIT")["status"]=="PASS"
    manifest=load(root,"MANIFEST")
    summ={}
    for stage,panels in raw.items():
        summ[stage]={}
        for f,rows in panels.items():
            verify_calls(rows)
            if "items" in rows:
                assert rows["items"]==[w["item"] for w in manifest[f][stage]]
            entry={kind:comparison_data(rows,kind) for kind in ("A","B")}
            for k in ("A","B"):
                m=rows[k]["reference_m"]
                entry[k+"_geometry"]=geometry_stats(rows[k]["geometry"][f"euler{m}"])
            if "endpoints" in rows:
                entry["endpoints"]=simple_endpoints(rows)
            if "A_norm_intervention" in rows:
                entry["A_norm_intervention"]={k:stats(v["state"]["full"])
                    for k,v in rows["A_norm_intervention"].items()}
            if "B_local" in rows:
                branch_err=[(q["branch"],stats(q["native_vs_fine"]["full"])["mean"],
                                          stats(q["matched_paths"]["256"][-1]["full"])["mean"])
                             for q in rows["B_local"]]
                entry["largest_branch_defect"]=max(branch_err,key=lambda x:x[1])
                if stage=="primary":
                    k=select(pair_dict(rows,"B"),"euler256:euler512")["path"]["state"][-1]["full"]
                    entry["B_matched_branch_vs_accumulated"]={
                        "mean_local":stats([v[2] for v in branch_err]),"last_global":stats(k),
                        "max_local_branch":max(branch_err,key=lambda x:x[2])}
            if "terms" in rows:
                entry["same_input_field_errors"]={kind:{
                    key:stats([v[key][0] for name,v in rows["terms"].items() if name.startswith(kind+"/")])
                    for key in next(iter(rows["terms"].values()))} for kind in ("A","B")}
            if "rounded_A" in rows:
                entry["rounded_A"]=stats(rows["rounded_A"]["state"]["full"])
                entry["rounded_B"]=stats(rows["rounded_B"]["state"]["full"])
                entry["rounded_B_pre"]=stats(rows["rounded_B_pre"]["full"])
                entry["rounded_A_displacement"]=stats(rows["rounded_A_displacement"])
                entry["rounded_B_displacement"]=stats(rows["rounded_B_displacement"])
            summ[stage][f]=entry
    summ["dynamics"]={}
    for f,d in diag.items():
        levels=orders[f]["levels"]
        def gains(trials):
            return {k:{"A_t1_t4":[v["A_expansion"][j][0] for j in (8,32)],
                       "B_end":v["B_expansion"][-1][0],
                       "A_max":max(p[0] for p in v["A_expansion"]),
                       "B_max":max(p[0] for p in v["B_expansion"])} for k,v in trials.items()}

        summ["dynamics"][f]=dict(
            relative_speed_t0_t1_t4=[d["relative_field_speed"][j][0] for j in (0,8,32)],
            relative_speed_fp32_t0_t1_t4=[stable_fp[f]["relative_field_speed"][j][0] for j in (0,8,32)],
            stability=gains(d["trials"]),
            stability_fp32=gains(stable_fp[f]["trials"]),
            K3={k:v["state"]["full"][0] for k,v in d["K3"].items() if isinstance(v,dict) and "state" in v},
            commutator_coefficients={str(dt):stats([q["levels"][j]["coefficient"][0] for q in levels])
                                    for j,dt in enumerate((.02,.01,.005))},
            direction_cos=stats([q["shared_state_direction_cos"][0] for q in levels]),
            negative_direction_pairs=sum(q["shared_state_direction_cos"][0]<0 for q in levels))
    times=[v for stage in raw.values() for d in stage.values() for v in d["timings"]]
    times += [v for d in (*diag.values(),*orders.values(),*stable_fp.values()) for v in d["timings"]]
    wall={}
    for k in ("torch.bfloat16","torch.float32"):
        selected=[v for v in times if v["precision"]==k]
        wall[k]={"wall_s":sum(v["wall_s"] for v in selected),
                 "max_allocated_bytes":max(v["peak_allocated_bytes"] for v in selected),
                 **{label:sum(v[label] for v in selected) for label in
                    ("loop_field_calls","attention_calls","mlp_calls","native_loops")}}
    summ["timed_scope"]=wall
    return summ,raw,diag,orders


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    a=p.parse_args()
    s,*_=summarize(a.root)
    write(a.root,"SUMMARY",s)

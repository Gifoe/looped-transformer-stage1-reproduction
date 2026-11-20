"""Aggregate paired measurements. Missing cohorts are reported explicitly."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


FAMILIES=("tableau","twohop","arc")
TYPES=("float32","bfloat16")
LOCAL=("order1","order2","order3","chart_midpoint")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summary(v):
    a=np.asarray(v,dtype=float)
    return dict(n=int(a.size),mean=float(a.mean()),median=float(np.median(a)),
                p95=float(np.quantile(a,.95)),minimum=float(a.min()),maximum=float(a.max()))


def aggregate_local(pilot):
    rows=[r for d in pilot.values() for r in d["cases"]]
    out={}
    for idx,row in enumerate(rows[0]["rows"]):
        alpha=row["alpha"]
        groups=[r["rows"][idx] for r in rows]
        out[str(alpha)]={"endpoint":{k:summary([a["methods"][k]["state"][0] for a in groups]) for k in LOCAL},
                "direction":{k:summary([a["methods"][k]["direction"][0] for a in groups]) for k in LOCAL},
                "same_output_different_solver":summary([a["methods"]["order2_rk4_step_self_error"][0] for a in groups])
                    if "order2_rk4_step_self_error" in row["methods"] else None,
                "unit_chart_inverse_residual":summary([a["inverse_residual"][0] for a in groups]),
                "midpoint_inverse_residual":summary([a["midpoint_inverse_residual"][0] for a in groups])}
    return out


def norms_and_spectra(pilot,operators):
    out={}
    for f,a in pilot.items():
        b=operators[f]["numerical"]["2"]["norm"]
        vals={}
        for dtype in ("real_formula","actual_fp32","actual_bf16_norm","output_cast_bf16"):
            r=b[dtype]
            vals[dtype]=dict(tokens=r["tokens"],valid=r["valid"],relative_error=summary(r["error_if_valid"])
                             if r["error_if_valid"] else None)
        inv=[x for v in operators[f]["numerical"].values() for x in v["inverse"]]
        out[f]=dict(eps=b["eps"],gamma_abs_min=b["gamma_abs_min"],zero_gamma=b["zero_scale"],
                    negative_gamma=b["negative_scale"],radial=summary(sum(b["radial_over_tangent"],[])),
                    norm_inverse=vals,collision_probes=b["pairs"],
                    inverse_unit_solver_errors={k:summary([r for x in inv for r in x["solutions"][k]["state"]])
                         for k in ("6/1.0","24/0.5","24/1.0")},
                    spectrum_full=a["spectra"]["full_loop"],spectrum_norm=a["spectra"]["norm"])
    return out


def operators_summary(operators):
    out={}
    for depth in ("2","3"):
        content=[r for d in operators.values() for r in d["loops"][depth]]
        cells={}
        for i in range(3):
            a=[x["scales"][i] for x in content]
            cells[str(a[0]["alpha"])]=dict(cos=summary(sum((r["cosine"] for r in a),[])),
                  relative=summary(sum((r["relative"] for r in a),[])))
        affected=0
        exact=0
        for row in content:
            for z,edit,dz in zip(row["actual"]["ref_logits"],row["actual"]["logits"],
                                  row["transported_bracket"]["predicted_logit_delta"],strict=True):
                gold=np.argmax(edit)
                if gold!=np.argmax(z) and edit.count(max(edit))==z.count(max(z))==1:
                    affected+=1
                    prediction=(np.asarray(z)+dz)
                    exact+=int(gold==prediction.argmax() and np.count_nonzero(prediction==prediction.max())==1)
        out[depth]=dict(scale=cells,negative_pairs=sum(d<0 for x in content for d in x["direction"]),
           pairs=sum(len(x["direction"]) for x in content),
           swap_result=summary(sum((x["actual"]["error"]["full"] for x in content),[])),
           flow_swap_cos=summary(sum((x["flow_vs_discrete_swap_cos"] for x in content),[])),
           transported_cos=summary(sum((x["transported_bracket"]["cos"] for x in content),[])),
           random_cos=summary(sum((x["norm_matched_random"]["direction_vs_swap"] for x in content),[])),
           other_cos=summary(sum((x["norm_matched_other_layer"]["direction_vs_swap"] for x in content),[])),
           flips={k:sum(a!=b for x in content for a,b in zip(x[k]["pred"],x[k]["ref_pred"],strict=True))
                  for k in ("actual","norm_matched_random","norm_matched_other_layer")},
           layer_swap=[float(np.mean([y for x in content if x["layer"]==i for y in x["actual"]["error"]["full"]]))
                       for i in range(1,25)],
           predicted_swap_flips=dict(affected=affected,exact=exact),
           transport_logit_rmse=summary([float(np.sqrt(np.mean((np.array(a)-b)**2))) for x in content
                for a,b in zip(x["transported_bracket"]["predicted_logit_delta"],
                               x["transported_bracket"]["swap_logit_delta"],strict=True)]))
    r=spearmanr(out["2"]["layer_swap"],out["3"]["layer_swap"]).statistic
    return dict(loops=out,layer_rank_correlation=float(r))


def method_stats(rows,name):
    data=[r["methods"][name] for r in rows if name in r["methods"]]
    if not data:
        return {}
    def pull(k):
        return [v for x in data for v in x[k]]
    ret=dict(full_error=summary([v for x in data for v in x["error"]["full"]]),
             token_error=summary([v for x in data for row in x["error"]["tokens"] for v in row]),
             update_cos=summary(pull("update_cos")),max_logit_error=summary(pull("max_logit_error")),
             agreement=sum(a==b for a,b in zip(pull("pred"),pull("ref_pred"),strict=True)),
             n=len(pull("pred")))
    if all("anchor_logits" in r for r in rows):
        flip=0
        predicted=0
        for r in rows:
            for x,y,z in zip(r["anchor_logits"],r["methods"][name]["ref_logits"],
                             r["methods"][name]["logits"],strict=True):
                if x.count(max(x))==y.count(max(y))==1 and np.argmax(x)!=np.argmax(y):
                    flip+=1
                    predicted+=int(z.count(max(z))==1 and np.argmax(y)==np.argmax(z))
        ret["native_flip_count"]=flip
        ret["native_flip_destination_correct"]=predicted
    flags=[v for x in data for row in x.get("inverse",[]) for v in row.get("bad",[])]
    if flags:
        ret["raw_gate_flags"]=sum(flags)
        ret["branches"]=len(flags)
        ret["raw_gate_fraction"]=sum(flags)/len(flags)
        ret["actual_jump_fraction"]=sum(v for x in data for row in x["inverse"]
                                        for v in row["used_jump"])/len(flags)
        accepted=[err for x in data for row in x.get("inverse",[])
                  for bad,err in zip(row.get("bad",[]),row.get("native_branch_distance",[])) if not bad]
        ret["accepted_only_boundary_error"]=summary(accepted) if accepted else None
    paths=[[float(np.mean(y["native_branch_distance"])) for y in x["inverse"]]
           for x in data if x.get("inverse") and all("native_branch_distance" in y for y in x["inverse"])]
    if paths:
        ret["branch_path"]=np.mean(paths,axis=0).tolist()
    return ret


def paired_interval(values):
    data=np.array([np.mean(v) for v in values.values()])
    rng=np.random.default_rng(5742026)
    sample=data[rng.integers(len(data),size=(3000,len(data)))].mean(1)
    return dict(delta=float(data.mean()),interval=np.quantile(sample,[.025,.975]).tolist(),clusters=len(data))


def causal_stats(rows,method,lanes=(1,2)):
    data=[r["methods"][method] for r in rows if method in r["methods"]]
    if not data:
        return {}
    def flat(k):
        return [d[k][i] for d in data for i in lanes]
    responsive=[(d,i) for d in data for i in lanes if d["native_dose"][i]>1e-10]
    denom=[(d,i) for d in data for i in lanes if not d["native_ties"][0] and not d["native_ties"][i]]
    native_flips=[(d,i) for d,i in denom if d["native_predictions"][i]!=d["native_predictions"][0]]
    match=sum(d["predictions"][i]==d["native_predictions"][i] and
              d["predictions"][0]==d["native_predictions"][0] and
              not (d["ties"][0] or d["ties"][i]) for d,i in native_flips)
    return dict(response_cos=summary([d["response_cos"][i] for d,i in responsive]),
        response_error=summary([d["response_error"][i] for d,i in responsive]),
        undefined_response=(len(data)*len(lanes)-len(responsive)),
        native_response_dose=summary(flat("native_dose")),top3_overlap=summary(flat("top3_overlap")),
        logit_response_rmse=summary(flat("logit_response_rmse")),native_flips=len(native_flips),
        content_aligned_logit_rmse=summary(flat("content_aligned_logit_rmse")),
        exact_flips=match,evaluable_flips=len(denom),outputs=len(data)*len(lanes),
        signed_margin_error=summary([abs(d["target_margin_response"][i]-d["native_target_margin_response"][i])
                                    for d in data for i in lanes]))


def aggregate(root):
    pilot={f:load(root/f"pilot_{f}.json") for f in FAMILIES}
    operators={f:load(root/f"operators_{f}.json") for f in FAMILIES}
    operators_holdout={f:load(root/"operator_confirmation"/f"operators_{f}.json") for f in FAMILIES}
    phases=load(root/"operator_confirmation"/"PHASE_CONTEXT.json")
    dev=load(root/"DEVELOPMENT.json")
    natural={dtype:{f:load(root/f"natural_{dtype}_{f}.json") for f in FAMILIES} for dtype in TYPES}
    cf={dtype:{f:load(root/f"causal_{dtype}_{f}.json") for f in FAMILIES
               if (root/f"causal_{dtype}_{f}.json").exists()} for dtype in TYPES}
    prefix_files=[(dt,f,root/"prefix_control"/f"{dt}_{f}.json") for dt in TYPES for f in FAMILIES]
    prefix_done=all(p.exists() for _,_,p in prefix_files)
    prefix=[]
    if prefix_done:
        for dt,f,p in prefix_files:
            raw=load(p)
            prefix.append(raw)
            for original,new in zip(natural[dt][f]["loops"],raw["natural"],strict=True):
                assert original["loop"]==new["loop"]
                original["methods"]["prefix_ridge"]=new["data"]
            for original,new in zip(cf[dt][f]["runs"],raw["causal"],strict=True):
                assert original["loop"]==new["loop"] and original["items"]==new["items"]
                original["methods"]["prefix_ridge"]=new["data"]
    dmethods=list(dev["groups"]["arc"]["methods"])
    result=dict(local=aggregate_local(pilot),normalization=norms_and_spectra(pilot,operators),
                order=operators_summary(operators),
                order_holdout=operators_summary(operators_holdout),same_state_phase=phases,
                dev={k:method_stats(list(dev["groups"].values()),k) for k in dmethods})
    result["natural"]={}
    result["causal"]={}
    result["causal_pairwise"]={}
    for dt in TYPES:
        result["natural"][dt]={}
        result["causal"][dt]={}
        for depth in (2,3):
            name=f"K{depth-1}_K{depth}"
            ns=[d for f in FAMILIES for d in natural[dt][f]["loops"] if d["loop"]==depth]
            ms=[k for k in ns[0]["methods"]]
            result["natural"][dt][name]=dict(pooled={k:method_stats(ns,k) for k in ms},
                by_family={f:{k:method_stats([x],k) for k in ms} for f,x in zip(FAMILIES,ns,strict=True)},
                linear_oracle_midpoint=summary([v for x in ns for v in x["oracle_linear_midpoint"]["error"]["full"]]))
            result["causal"][dt][name]={}
            all_cf=[d for b in cf[dt].values() for d in b["runs"] if d["loop"]==depth]
            for f,rows in (("pooled",all_cf),*((f,[d for d in b["runs"] if d["loop"]==depth])
                                                             for f,b in cf[dt].items())):
                result["causal"][dt][name][f]={}
                for label,lanes in (("semantic",(1,2)),("random",(3,4)),("unrelated",(5,6))):
                    result["causal"][dt][name][f][label]={k:causal_stats(rows,k,lanes) for k in rows[0]["methods"]}
        controls=("branch_flow","full_loop","ridge","chart")
        for compare in (*controls,*(("prefix_ridge",) if prefix_done else ())):
            pairs=defaultdict(list)
            for q in cf[dt].values():
                for r in q["runs"]:
                    a=r["methods"]["chart_anderson"]["logit_response_rmse"]
                    b=r["methods"][compare]["logit_response_rmse"]
                    pairs[r["items"][0]].extend([b[i]-a[i] for i in (1,2)])
            result["causal_pairwise"][f"{dt}/chart_anderson_vs_{compare}"]=paired_interval(pairs)
    timed=[]
    for d in [*pilot.values(),*operators.values(),*operators_holdout.values(),dev,*prefix,*[x for p in natural.values() for x in p.values()],
                                                *[x for p in cf.values() for x in p.values()]]:
        timed.extend(d["timing"])
    result["cost"]={p:dict(wall_s=sum(q["wall_s"] for q in timed if q["dtype"]=="torch."+p),
        peak_bytes=max(q["allocated_peak_bytes"] for q in timed if q["dtype"]=="torch."+p),
        **{k:sum(q[k] for q in timed if q["dtype"]=="torch."+p) for k in
            ("attention_calls","mlp_calls","loop_field_calls","jvp","vjp","inverse_iterations")})
        for p in TYPES}
    stamps=defaultdict(list)
    for q in timed:
        stamps[q["dtype"]+"/"+q["run"].rsplit("/",1)[-1]].append(q)
    result["method_cost"]={k:{p:summary([v[p] for v in q]) for p in
          ("wall_s","allocated_peak_bytes","attention_calls","mlp_calls","jvp","inverse_iterations")}
          for k,q in stamps.items()}
    return result,pilot,operators,dev,natural,cf


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    args=p.parse_args()
    s,*_=aggregate(args.root)
    (args.root/"SUMMARY.json").write_text(json.dumps(s,indent=2,allow_nan=False)+"\n")

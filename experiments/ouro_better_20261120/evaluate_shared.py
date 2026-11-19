"""Frozen exact-generation, option-remapping and teacher-forced benchmark."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re
from collections import defaultdict
from itertools import permutations
import time

import numpy as np
from scipy.stats import binomtest
import torch

from train_shared import ARMS, DEPTHS, batch, load_cohort, load_runtime, make_view, merged, states, variants
from prepare import dump, key, sha

SINGLE_LETTER=re.compile(r"[ \t]*([A-Ea-e])(?:[.)])?[ \t]*")
LINEAR_NATIVE = {"native_k"+str(k) for k in DEPTHS}
LOOPCD = {f"loopcd_{v}_k{k}" for k in DEPTHS[1:] for v in ("fixed","adaptive","hidden")}


def parser(tok, vocab_id, option_order):
    raw=tok.decode([vocab_id],clean_up_tokenization_spaces=False,skip_special_tokens=False)
    m=SINGLE_LETTER.fullmatch(raw)
    pos=ord(m[1].upper())-65 if m else -1
    actual=option_order[pos] if 0<=pos<len(option_order) else None
    return actual,raw


def run_letter(model,tok,views,names):
    ordered=sorted(enumerate(views),key=lambda p:len(p[1]["ids"]))
    endpoints=[{} for v in views]
    for j in range(0,len(ordered),16):
        jobs=ordered[j:j+16]
        inp,rows,end,_,_=batch([v for i,v in jobs])
        h,r,_=states(model,inp,6,capture_raw=bool(LOOPCD&names))
        for name,logits in variants(model,[u[rows,end] for u in h],[u[rows,end] for u in r],names):
            chosen=logits.argmax(-1).tolist()
            for row,((i,v),token) in enumerate(zip(jobs,chosen)):
                exact, raw=parser(tok,token,v["order"])
                lp=logits[row,v["display"]].tolist()
                b=max(range(len(lp)),key=lambda a:lp[a])
                gold=v["gold_content"]
                lp_by_content=[lp[v["order"].index(q)] for q in sorted(v["order"])]
                endpoints[i][name]={
                    "exact":exact,"restricted":v["order"][b],
                    "decoded_subtoken":raw,"token_id":token,"logits_by_content":lp_by_content,
                    "gold_margin":lp_by_content[gold]-max(s for c,s in enumerate(lp_by_content) if c!=gold),
                }
    assert all(set(r)==names for r in endpoints)
    return endpoints


def run_content(model,tok,rows,views,names):
    jobs=[]
    result=[{n:[None]*len(r["choices"]["text"]) for n in names} for r in rows]
    for q,(row,view) in enumerate(zip(rows,views)):
        assert view["order"]==list(range(len(view["order"])))
        for c,text in enumerate(row["choices"]["text"]):
            full=tok.encode(view["prompt"]+" "+text,add_special_tokens=False)
            prefix=view["ids"]
            assert full[:len(prefix)]==prefix and len(full)>len(prefix)
            jobs.append((q,c,full[:-1],full[len(prefix):],len(text)))
    jobs.sort(key=lambda r:len(r[2]))
    for start in range(0,len(jobs),16):
        selected=jobs[start:start+16]
        width=max(len(ids) for _,_,ids,_,_ in selected)
        inp=torch.tensor([ids+[0]*(width-len(ids)) for _,_,ids,_,_ in selected],device="cuda")
        indices, positions, targets, owners=[],[],[],[]
        for i,(_,_,ids,continuation,_) in enumerate(selected):
            indices.extend([i]*len(continuation))
            positions.extend(range(len(ids)-len(continuation),len(ids)))
            targets.extend(continuation)
            owners.extend([i]*len(continuation))
        ix,pos,gold=[torch.tensor(x,device="cuda") for x in (indices,positions,targets)]
        h,r,_=states(model,inp,6,capture_raw=bool(LOOPCD&names))
        for name,z in variants(model,[x[ix,pos] for x in h],[x[ix,pos] for x in r],names):
            lp=z.log_softmax(-1).gather(1,gold[:,None]).squeeze(-1).cpu().numpy()
            sums=np.bincount(owners,weights=lp,minlength=len(selected))
            for (q,c,ids,tt,length),total in zip(selected,sums):
                result[q][name][c]={"sum":float(total),"characters":length,"tokens":len(tt)}
    assert all(all(all(c is not None for c in a) for a in r.values()) for r in result)
    return result


def run_arm(model,tok,rows,arm,names,out,checkpoint=None):
    a=[make_view(tok,r,"original") for r in rows]
    by_count=defaultdict(list)
    for r in rows:
        by_count[len(r["choices"]["text"])].append(r["id"])
    permutation_assignment={}
    for count, identifiers in by_count.items():
        canonical=tuple(range(count))
        choices=sorted((perm for perm in permutations(canonical) if perm!=canonical),
                       key=lambda p:key("permutation_catalog",*(str(v) for v in p)))
        for i,qid in enumerate(sorted(identifiers,key=lambda x:key("evaluation_permutation_assignment",x))):
            permutation_assignment[qid]=list(choices[i%len(choices)])
    b=[make_view(tok,r,"eval",permutation_assignment[r["id"]]) for r in rows]
    for x,y in zip(a,b):
        assert x["order"]!=y["order"] and x["gold_content"]==y["gold_content"]
        assert x["gold_token"]==x["content"][x["gold_content"]]
        assert y["gold_token"]==y["content"][y["gold_content"]]
    torch.cuda.synchronize()
    start=time.perf_counter()
    with merged(model,checkpoint) if checkpoint else nullcontext():
        letter=run_letter(model,tok,a+b,names)
        content=run_content(model,tok,rows,a,names)
    results=[]
    for x,y,va,vb,txt in zip(a,b,letter[:len(rows)],letter[len(rows):],content):
        results.append({"id":x["id"],"gold":x["gold_content"],
                        "original_prompt_digest":key("rendered_prompt",x["prompt"]),
                        "permuted_prompt_digest":key("rendered_prompt",y["prompt"]),
                        "original_order":x["order"],"permuted_order":y["order"],
                        "original_labels":x["display"],"permuted_labels":y["display"],
                        "original":va,"permuted":vb,"content_target":txt})
    torch.cuda.synchronize()
    dump(out/f"predictions.{arm}.json",results)
    print("evaluated",arm,len(rows),f"{time.perf_counter()-start:.1f}s",flush=True)
    return results


def summary(rows,name):
    metrics=defaultdict(int)
    for r in rows:
        g=r["gold"]
        a,b=(r[m][name] for m in ("original","permuted"))
        candidates=r["content_target"][name]
        norm=max(range(len(candidates)),key=lambda c:candidates[c]["sum"]/candidates[c]["characters"])
        raw=max(range(len(candidates)),key=lambda c:candidates[c]["sum"])
        for prefix,v,o in (("original",a,b),("permuted",b,a)):
            metrics[prefix+"_correct"] += v["exact"]==g
            metrics[prefix+"_invalid"] += v["exact"] is None
            metrics[prefix+"_restricted_correct"] += v["restricted"]==g
        metrics["same_content"] += a["exact"] is not None and a["exact"]==b["exact"]
        metrics["same_content_restricted"] += a["restricted"]==b["restricted"]
        metrics["both_exact_correct"] += a["exact"]==g and b["exact"]==g
        metrics["both_restricted_correct"] += a["restricted"]==g and b["restricted"]==g
        metrics["content_target_raw_correct"] += raw==g
        metrics["content_target_char_correct"] += norm==g
    return dict(metrics)


def compare(candidate,ck,baseline,bk,which):
    assert [r["id"] for r in candidate]==[r["id"] for r in baseline]
    base=np.array([r["original"][bk][which]==r["gold"] for r in baseline],dtype=bool)
    test=np.array([r["original"][ck][which]==r["gold"] for r in candidate],dtype=bool)
    a,b=int((~base&test).sum()),int((base&~test).sum())
    n=len(base)
    sample=np.random.default_rng(101).multinomial(n,[(n-a-b)/n,a/n,b/n],size=9999)
    ci=np.quantile((sample[:,1]-sample[:,2])*100/n,[.025,.975]).tolist()
    return {"n":n,"corrected":a,"damaged":b,"net":a-b,"pp":(a-b)*100/n,
            "paired_bootstrap95_pp":ci,
            "exact_one_sided_p": float(binomtest(a,a+b,alternative="greater").pvalue) if a+b else 1.0,
            "corrected_ids":[r["id"] for r,use in zip(candidate,~base&test) if use],
            "damaged_ids":[r["id"] for r,use in zip(candidate,base&~test) if use]}


def aggregate(results,dev_maxima=None):
    matrices={}
    for method,parsed in results.items():
        matrices[method]={k:summary(parsed,k) for k in sorted(parsed[0]["original"])}
    strongest={}
    for k in DEPTHS:
        n=[f"native_k{k}"]
        if k>1:
            n += [f"loopcd_{v}_k{k}" for v in ("fixed","adaptive","hidden")]
        strongest[str(k)]=max(n,key=lambda v:matrices["base"][v]["original_correct"])
    compare_summary={}
    for arm in ("base",*ARMS):
        for k in DEPTHS:
            cmd=f"native_k{k}"
            tags=[f"native_k{k}"]
            if arm=="base" and k>1:
                tags += [f"loopcd_{v}_k{k}" for v in ("fixed","adaptive","hidden")]
            for name in tags:
                compare_summary.setdefault(arm,{})[name] = {
                    q:compare(results["binding"],cmd,results[arm],name,q) for q in ("exact","restricted")}
    baseline=dev_maxima if dev_maxima is not None else strongest
    binding_total=[matrices["binding"][f"native_k{k}"] for k in DEPTHS]
    matched_ce=[matrices["ce"][f"native_k{k}"] for k in DEPTHS]
    exact=[(p["original_correct"]-matrices["base"][baseline[str(k)]]["original_correct"])
           for p,k in zip(binding_total,DEPTHS)]
    trained=np.mean([p["original_correct"]-q["original_correct"] for p,q in zip(binding_total,matched_ce)])
    invariant=np.mean([p["same_content"]-matrices["base"][baseline[str(k)]]["same_content"]
                       for p,k in zip(binding_total,DEPTHS)])
    checks={"positive_exact_gain_every_depth":min(exact)>0,
            "positive_mean_gain_vs_CE":bool(trained>0),
            "paired_content_agreement_increased":bool(invariant>0),
            "exact_gain_by_depth":exact,"mean_exact_gain_vs_CE_count":float(trained),
            "mean_content_agreement_gain_count":float(invariant)}
    verdict=all(v for k,v in checks.items() if isinstance(v,bool))
    return {"same_budget_strength":strongest,"standing_dev_baseline":baseline,"methods":matrices,
            "paired_binding_vs_all":compare_summary,"screen":checks,"passed":verdict}


def main():
    p=argparse.ArgumentParser()
    for k in ("model","data","manifest","weights","out","freeze"):
        p.add_argument(k,type=Path)
    p.add_argument("--cohort",choices=("dev","diagnostic","transfer"),default="dev")
    p.add_argument("--gate",type=Path)
    a=p.parse_args()
    frozen=json.loads(a.freeze.read_bytes())
    assert frozen["split_sha256"]==sha(a.manifest)
    for name,digest in frozen["code_hashes"].items():
        assert sha(Path(__file__).with_name(name))==digest
    manifest=json.loads(a.manifest.read_bytes())
    arm_files={arm:a.weights/f"adapter.{arm}.safetensors" for arm in ARMS}
    arm_shas={k:sha(v) for k,v in arm_files.items()}
    for k in ARMS:
        assert json.loads((a.weights/f"train.{k}.json").read_bytes())["checkpoint_sha256"]==arm_shas[k]
    gate=None
    if a.cohort!="dev":
        if a.gate is None:
            raise RuntimeError("Missing development gate")
        gate=json.loads(a.gate.read_bytes())
        if not (gate["passed"] and gate["execution_freeze_sha256"]==sha(a.freeze)
                and gate["adapter_hashes"]==arm_shas and gate["cohort"]=="dev"):
            raise RuntimeError("Development gate closed")
        if (a.out/a.cohort/"COMPLETE.json").exists():
            raise RuntimeError("Frozen cohort was already evaluated")
    root=a.out/a.cohort
    root.mkdir(parents=True,exist_ok=True)
    source="openbook" if a.cohort=="transfer" else "arc"
    freeze_source=manifest["source"]["openbookqa" if source=="openbook" else "arc"]
    assert freeze_source["sha256"]==sha(a.data)
    rows=load_cohort(a.data,source,manifest,a.cohort)
    model,tok,runtime=load_runtime(a.model)
    native_names=LINEAR_NATIVE | LOOPCD
    results={"base":run_arm(model,tok,rows,"base",native_names,root)}
    for arm in ARMS:
        results[arm]=run_arm(model,tok,rows,arm,LINEAR_NATIVE,root,arm_files[arm])
    evaluation=aggregate(results,gate["same_budget_strength"] if gate is not None else None)
    evaluation.update({"cohort":a.cohort,"execution_freeze_sha256":sha(a.freeze),"adapter_hashes":arm_shas,
                       "runtime":runtime,"input_rows":len(rows),
                       "predictions_sha256":{p.name:sha(p) for p in sorted(root.glob("predictions.*.json"))}})
    dump(root/"COMPLETE.json",evaluation)
    print("screen",a.cohort,evaluation["screen"],evaluation["passed"],flush=True)


if __name__=="__main__":
    main()

